"""
Rutas de trabajos de extracción, objeto individual e ImageNet.
"""

import asyncio
import base64
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from app.models.extraction_schemas import (
    AnnotationType,
    ExtractSingleObjectRequest,
    ExtractSingleObjectResponse,
    ExtractionJobStatus,
    ExtractionMethod,
    JobStatus,
)
from app.service_state import extraction_jobs, state

logger = logging.getLogger(__name__)


router = APIRouter()


def _db_row_to_extraction_job(db_job: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a DB job row to the extraction job dict format used by the list endpoint."""
    result_summary = db_job.get("result_summary") or {}
    total = db_job.get("total_items", 0)
    extracted = db_job.get("processed_items", 0)
    failed = db_job.get("failed_items", 0)
    progress = round(((extracted + failed) / total * 100), 1) if total > 0 else 0.0

    return {
        "job_id": db_job.get("id", ""),
        "type": "extraction",
        "job_type": "extraction",
        "status": db_job.get("status", "unknown"),
        "progress": progress,
        "created_at": db_job.get("created_at", datetime.now().isoformat()),
        "total_objects": result_summary.get("extracted", 0) + result_summary.get("failed", 0) if result_summary else total,
        "extracted_objects": result_summary.get("extracted", extracted),
        "failed_objects": result_summary.get("failed", failed),
        "current_category": db_job.get("current_item", ""),
        "output_dir": db_job.get("output_path", ""),
        "started_at": db_job.get("started_at"),
        "completed_at": db_job.get("completed_at"),
        "processing_time_ms": db_job.get("processing_time_ms", 0),
    }


@router.get("/extract/jobs", tags=["Object Extraction"])
async def list_extraction_jobs():
    """List all extraction jobs (active from memory, historical from database)."""
    jobs = []
    seen_ids: set = set()

    # First: active jobs from memory (most detailed real-time data)
    for job_id, job in extraction_jobs.items():
        # Convert JobStatus enum to string for JSON serialization
        status = job.get("status", "unknown")
        status_str = status.value if hasattr(status, 'value') else str(status)

        # Calculate progress
        total = job.get("total_objects", 0)
        extracted = job.get("extracted_objects", 0)
        failed = job.get("failed_objects", 0)
        progress = round(((extracted + failed) / total * 100), 1) if total > 0 else 0.0

        jobs.append({
            "job_id": job_id,
            "type": "extraction",  # Frontend expects 'type' field
            "job_type": "extraction",
            "status": status_str,
            "progress": progress,  # Add progress percentage
            "created_at": job.get("created_at", job.get("started_at", datetime.now().isoformat())),
            "total_objects": job.get("total_objects", 0),
            "extracted_objects": job.get("extracted_objects", 0),
            "failed_objects": job.get("failed_objects", 0),
            "current_category": job.get("current_category", ""),
            "output_dir": job.get("output_dir", ""),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "processing_time_ms": job.get("processing_time_ms", 0),
        })
        seen_ids.add(job_id)

    # Second: historical/completed jobs from DB not present in memory
    if state.db:
        try:
            db_jobs = state.db.list_jobs(service="segmentation", job_type="extraction", limit=50)
            for db_job in db_jobs:
                job_id = db_job.get("id", "")
                if job_id and job_id not in seen_ids:
                    jobs.append(_db_row_to_extraction_job(db_job))
        except Exception:
            pass

    return {"jobs": jobs, "total": len(jobs)}


@router.get("/extract/jobs/{job_id}", response_model=ExtractionJobStatus, tags=["Object Extraction"])
async def get_extraction_job_status(job_id: str):
    """Get the status of an extraction job."""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = extraction_jobs[job_id]

    # Calculate progress percentage
    total = job.get("total_objects", 0)
    extracted = job.get("extracted_objects", 0)
    failed = job.get("failed_objects", 0)
    progress = ((extracted + failed) / total * 100) if total > 0 else 0.0

    return ExtractionJobStatus(
        **{k: v for k, v in job.items() if k != "progress"},
        progress=round(progress, 1)
    )


@router.post("/extract/single-object", response_model=ExtractSingleObjectResponse, tags=["Object Extraction"])
async def extract_single_object(request: ExtractSingleObjectRequest):
    """
    Extract a single object for preview.

    Returns the extracted object as base64-encoded PNG with transparency.
    """
    start_time = time.time()

    try:
        # Load image
        image = None
        if request.image_base64:
            img_data = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        elif request.image_path:
            image_path = Path(request.image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {request.image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("Failed to load image")

        # Extract object
        result = await state.object_extractor.extract_single_object(
            image=image,
            annotation=request.annotation,
            category_name=request.category_name,
            use_sam3=request.use_sam3,
            padding=request.padding,
            force_bbox_only=request.force_bbox_only,
            force_sam3_resegmentation=request.force_sam3_resegmentation,
            force_sam3_text_prompt=request.force_sam3_text_prompt
        )

        processing_time = (time.time() - start_time) * 1000

        if not result["success"]:
            return ExtractSingleObjectResponse(
                success=False,
                processing_time_ms=processing_time,
                error=result.get("error", "Unknown error")
            )

        return ExtractSingleObjectResponse(
            success=True,
            cropped_image_base64=result["cropped_image_base64"],
            mask_base64=result.get("mask_base64"),
            annotation_type=AnnotationType(result["annotation_type"]),
            method_used=ExtractionMethod(result["method_used"]),
            original_bbox=result["original_bbox"],
            extracted_size=result["extracted_size"],
            mask_coverage=result["mask_coverage"],
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Single object extraction failed: {e}")
        return ExtractSingleObjectResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@router.post("/extract/imagenet", tags=["Object Extraction"])
async def extract_from_imagenet(
    root_dir: str,
    output_dir: str,
    padding: int = 5,
    min_object_area: int = 100,
    max_objects_per_class: Optional[int] = None
):
    """
    Extract objects from ImageNet-style directory structure.

    Expected structure:
        root_dir/
        ├── class1/
        │   ├── img001.jpg
        │   └── img002.jpg
        ├── class2/
        │   └── ...

    Uses SAM3 with class name as text prompt for segmentation.
    Runs asynchronously. Use GET /extract/jobs/{job_id} to track progress.
    """
    if not state.sam3_available:
        return {
            "success": False,
            "error": "SAM3 is required for ImageNet extraction but not available"
        }

    try:
        # Validate root directory
        root_path = Path(root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        if not root_path.is_dir():
            raise ValueError(f"Path is not a directory: {root_dir}")

        # Create job
        job_id = str(uuid.uuid4())

        # Count classes
        try:
            classes = [d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))]
            num_classes = len(classes)
        except Exception as e:
            raise ValueError(f"Failed to read root directory: {e}")

        extraction_jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "total_objects": 0,  # Will be updated as we discover images
            "extracted_objects": 0,
            "failed_objects": 0,
            "current_category": "",
            "categories_progress": {},
            "output_dir": output_dir,
            "errors": [],
            "extracted_files": [],
            "processing_time_ms": 0.0,
            "created_at": datetime.now().isoformat(),  # Track when job was created
            "started_at": None,
            "completed_at": None,
            "extraction_type": "imagenet"
        }

        # Persist ImageNet extraction job to database
        if state.db:
            try:
                state.db.create_job(
                    job_id=job_id,
                    job_type="extraction",
                    service="segmentation",
                    request_params={
                        "root_dir": root_dir,
                        "num_classes": num_classes,
                        "extraction_type": "imagenet",
                    },
                    total_items=num_classes,
                    output_path=output_dir,
                )
                state.db.update_job_status(job_id, "running", started_at=datetime.now())
            except Exception as e:
                logger.warning(f"Failed to persist ImageNet extraction job to DB: {e}")

        # Define extraction task
        async def run_imagenet_extraction():
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()

            def progress_callback(progress):
                # Update job progress
                current_class = progress.get("current_class", "")
                extracted = progress.get("extracted", 0)
                failed = progress.get("failed", 0)
                extraction_jobs[job_id]["current_category"] = current_class
                extraction_jobs[job_id]["extracted_objects"] = extracted
                extraction_jobs[job_id]["failed_objects"] = failed

                if state.db:
                    try:
                        state.db.update_job_progress(
                            job_id,
                            processed_items=extracted,
                            failed_items=failed,
                            current_item=current_class,
                        )
                    except Exception:
                        pass

            try:
                start_time = time.time()
                result = await state.object_extractor.extract_from_imagenet_structure(
                    root_dir=root_dir,
                    output_dir=output_dir,
                    padding=padding,
                    min_object_area=min_object_area,
                    max_objects_per_class=max_objects_per_class,
                    progress_callback=progress_callback
                )

                if result.get("success"):
                    extraction_jobs[job_id]["status"] = JobStatus.COMPLETED
                    if state.db:
                        try:
                            state.db.complete_job(
                                job_id,
                                "completed",
                                result_summary={
                                    "total_extracted": result.get("total_extracted", 0),
                                    "total_failed": result.get("total_failed", 0),
                                    "classes": result.get("classes", {}),
                                },
                            )
                        except Exception:
                            pass
                else:
                    extraction_jobs[job_id]["status"] = JobStatus.FAILED
                    if state.db:
                        try:
                            state.db.complete_job(job_id, "failed", error_message="ImageNet extraction returned failure")
                        except Exception:
                            pass

                extraction_jobs[job_id]["total_objects"] = result.get("total_extracted", 0) + result.get("total_failed", 0)
                extraction_jobs[job_id]["extracted_objects"] = result.get("total_extracted", 0)
                extraction_jobs[job_id]["failed_objects"] = result.get("total_failed", 0)
                extraction_jobs[job_id]["categories_progress"] = result.get("classes", {})
                extraction_jobs[job_id]["errors"] = result.get("errors", [])[:100]
                extraction_jobs[job_id]["processing_time_ms"] = (time.time() - start_time) * 1000

            except Exception as e:
                logger.exception(f"ImageNet extraction job {job_id} failed: {e}")
                extraction_jobs[job_id]["status"] = JobStatus.FAILED
                extraction_jobs[job_id]["errors"].append(str(e))

                if state.db:
                    try:
                        state.db.complete_job(job_id, "failed", error_message=str(e))
                    except Exception:
                        pass

            finally:
                extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Run in background
        asyncio.create_task(run_imagenet_extraction())
        logger.info(f"Started ImageNet extraction job {job_id} from {root_dir} ({num_classes} classes)")

        return {
            "success": True,
            "job_id": job_id,
            "status": "pending",
            "message": f"ImageNet extraction job queued. Processing {num_classes} classes."
        }

    except Exception as e:
        logger.error(f"Failed to start ImageNet extraction: {e}")
        return {
            "success": False,
            "error": str(e)
        }
