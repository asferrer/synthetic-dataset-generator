"""
Rutas de consulta y control de trabajos de etiquetado.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.extraction_schemas import (
    JobStatus,
    LabelingJobResponse,
    LabelingJobStatus,
    LabelingResultResponse,
)
from app.service_state import labeling_jobs, state
from app.services.labeling_images import _get_image_paths_from_output_dir
from app.services.labeling_processing import _process_labeling_job

logger = logging.getLogger(__name__)


router = APIRouter()


@router.get("/labeling/jobs", tags=["Labeling Tool"])
async def list_labeling_jobs():
    """List all labeling jobs (from memory and database)."""
    jobs = []
    seen_job_ids = set()

    # First, get jobs from memory (these are the most up-to-date)
    for job_id, job in labeling_jobs.items():
        status = job.get("status", "unknown")
        status_str = status.value if hasattr(status, 'value') else str(status)

        # Check if job can be resumed (has checkpoint and is failed/cancelled)
        can_resume = False
        if status in [JobStatus.FAILED, JobStatus.CANCELLED]:
            output_dir = Path(job.get("output_dir", ""))
            checkpoint_path = output_dir / "checkpoint.json"
            can_resume = checkpoint_path.exists()

        # Calculate progress
        total_images = job.get("total_images", 0)
        processed_images = job.get("processed_images", 0)
        progress = round((processed_images / total_images * 100), 1) if total_images > 0 else 0.0

        jobs.append({
            "job_id": job_id,
            "type": "labeling",  # Frontend expects 'type' field
            "job_type": job.get("job_type", "labeling"),
            "status": status_str,
            "progress": progress,  # Add progress percentage
            "created_at": job.get("created_at", job.get("started_at", datetime.now().isoformat())),
            "total_images": total_images,
            "processed_images": processed_images,
            "total_objects_found": job.get("total_objects_found", 0),
            "objects_by_class": job.get("objects_by_class", {}),
            "output_dir": job.get("output_dir", ""),
            "current_image": job.get("current_image", ""),  # For progress display
            "errors": job.get("errors", [])[:5],  # Include first 5 errors
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
            "processing_time_ms": job.get("processing_time_ms", 0),
            "can_resume": can_resume,
        })
        seen_job_ids.add(job_id)

    # Then, add jobs from database that are not in memory (e.g., interrupted jobs from previous runs)
    if state.db:
        try:
            db_jobs = state.db.list_jobs(service="segmentation", job_type="labeling", limit=50)
            for db_job in db_jobs:
                job_id = db_job.get("id")
                if job_id and job_id not in seen_job_ids:
                    # This job exists in DB but not in memory - likely from a previous run
                    output_path = db_job.get("output_path", "")
                    can_resume = False
                    if db_job.get("status") in ["interrupted", "failed"]:
                        checkpoint_path = Path(output_path) / "checkpoint.json"
                        can_resume = checkpoint_path.exists()

                    # Parse result_summary if available
                    result_summary = db_job.get("result_summary", {}) or {}

                    # Calculate progress
                    total_images = db_job.get("total_items", 0)
                    processed_images = db_job.get("processed_items", 0)
                    progress = round((processed_images / total_images * 100), 1) if total_images > 0 else 0.0

                    jobs.append({
                        "job_id": job_id,
                        "type": "labeling",  # Frontend expects 'type' field
                        "job_type": "labeling",
                        "status": db_job.get("status", "unknown"),
                        "progress": progress,  # Add progress percentage
                        "created_at": db_job.get("created_at", db_job.get("started_at", datetime.now().isoformat())),
                        "total_images": total_images,
                        "processed_images": processed_images,
                        "total_objects_found": result_summary.get("total_objects_found", 0),
                        "objects_by_class": result_summary.get("objects_by_class", {}),
                        "output_dir": output_path,
                        "current_image": db_job.get("current_item", ""),
                        "errors": [],
                        "started_at": db_job.get("started_at"),
                        "completed_at": db_job.get("completed_at"),
                        "processing_time_ms": db_job.get("processing_time_ms", 0),
                        "can_resume": can_resume,
                    })
        except Exception as e:
            logger.warning(f"Failed to fetch jobs from database: {e}")

    return {"jobs": jobs, "total": len(jobs)}


@router.get("/labeling/jobs/{job_id}", response_model=LabelingJobStatus, tags=["Labeling Tool"])
async def get_labeling_job_status(job_id: str):
    """Get status of a labeling job."""
    if job_id not in labeling_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = labeling_jobs[job_id]

    # Check if job can be resumed (has checkpoint and is failed/cancelled)
    can_resume = False
    status = job.get("status")
    if status in [JobStatus.FAILED, JobStatus.CANCELLED]:
        output_dir = Path(job.get("output_dir", ""))
        checkpoint_path = output_dir / "checkpoint.json"
        can_resume = checkpoint_path.exists()

    # Calculate progress percentage
    total_images = job.get("total_images", 0)
    processed_images = job.get("processed_images", 0)
    progress = (processed_images / total_images * 100) if total_images > 0 else 0.0

    # Get annotations count
    total_objects = job.get("total_objects_found", 0)

    # Build quality metrics if available
    quality_metrics = None
    if "quality_metrics" in job:
        from app.models.extraction_schemas import LabelingQualityMetrics
        qm = job["quality_metrics"]
        quality_metrics = LabelingQualityMetrics(
            avg_confidence=qm.get("avg_confidence", 0.0),
            images_with_detections=qm.get("images_with_detections", 0),
            images_without_detections=qm.get("images_without_detections", 0),
            low_confidence_count=qm.get("low_confidence_count", 0),
            total_detections=qm.get("total_detections", 0),
        )

    return LabelingJobStatus(
        job_id=job_id,
        job_type=job.get("job_type", "labeling"),
        status=status or JobStatus.QUEUED,
        total_images=total_images,
        processed_images=processed_images,
        progress=round(progress, 1),  # Percentage of completion
        annotations_created=total_objects,  # Frontend expects this name
        total_objects_found=total_objects,  # Keep for backwards compatibility
        objects_by_class=job.get("objects_by_class", {}),
        current_image=job.get("current_image", ""),
        output_dir=job.get("output_dir", ""),
        output_formats=job.get("output_formats", []),
        errors=job.get("errors", [])[:50],
        warnings=job.get("warnings", [])[:20],  # Include warnings
        quality_metrics=quality_metrics,  # Include quality metrics
        processing_time_ms=job.get("processing_time_ms", 0),
        created_at=job.get("created_at"),  # When job was created
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        can_resume=can_resume,
    )


@router.get("/labeling/jobs/{job_id}/result", response_model=LabelingResultResponse, tags=["Labeling Tool"])
async def get_labeling_result(job_id: str):
    """Get the result of a completed labeling job."""
    if job_id not in labeling_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = labeling_jobs[job_id]

    if job.get("status") != JobStatus.COMPLETED:
        return LabelingResultResponse(
            success=False,
            error=f"Job is not completed. Current status: {job.get('status')}"
        )

    # Load the COCO result
    output_dir = Path(job.get("output_dir", ""))
    coco_path = output_dir / "annotations.json"

    if not coco_path.exists():
        return LabelingResultResponse(
            success=False,
            error="Result file not found"
        )

    try:
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)

        # Build output files map
        output_files = {"coco": str(coco_path)}

        yolo_dir = output_dir / "yolo"
        if yolo_dir.exists():
            output_files["yolo"] = str(yolo_dir)

        voc_dir = output_dir / "voc"
        if voc_dir.exists():
            output_files["voc"] = str(voc_dir)

        return LabelingResultResponse(
            success=True,
            data=coco_data,
            output_files=output_files,
            summary={
                "total_images": len(coco_data.get("images", [])),
                "total_annotations": len(coco_data.get("annotations", [])),
                "categories": [c.get("name") for c in coco_data.get("categories", [])],
            }
        )

    except Exception as e:
        return LabelingResultResponse(
            success=False,
            error=str(e)
        )


@router.get("/labeling/jobs/{job_id}/previews", tags=["Labeling Tool"])
async def get_labeling_job_previews(job_id: str, limit: int = 10):
    """
    Get preview images for a labeling job.

    Returns base64-encoded preview images showing the annotations in progress.
    Useful for monitoring the quality of auto-labeling in real-time.
    """
    import base64

    # First check memory
    job = labeling_jobs.get(job_id)
    output_dir = None

    if job:
        output_dir = Path(job.get("output_dir", ""))
    elif state.db:
        db_job = state.db.get_job(job_id)
        if db_job and db_job.get("job_type") == "labeling":
            output_dir = Path(db_job.get("output_path", ""))

    if output_dir is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    previews_dir = output_dir / "previews"
    if not previews_dir.exists():
        return {
            "job_id": job_id,
            "previews": [],
            "total": 0,
            "message": "No preview images available yet"
        }

    # Get preview files sorted by modification time (most recent first)
    preview_files = sorted(
        previews_dir.glob("preview_*.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:limit]

    previews = []
    for preview_file in preview_files:
        try:
            with open(preview_file, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            previews.append({
                "filename": preview_file.name,
                "path": str(preview_file),
                "image_data": img_data,  # Base64 without prefix
                "timestamp": str(preview_file.stat().st_mtime),
                "size_kb": preview_file.stat().st_size / 1024,
            })
        except Exception as e:
            logger.warning(f"Failed to read preview {preview_file}: {e}")

    return {
        "job_id": job_id,
        "previews": previews,
        "total": len(preview_files),
        "output_dir": str(output_dir),
    }


@router.post("/labeling/jobs/{job_id}/resume", response_model=LabelingJobResponse, tags=["Labeling Tool"])
async def resume_labeling_job(job_id: str):
    """
    Resume a failed or interrupted labeling job.

    Loads the checkpoint and continues from the last processed image.
    Can resume jobs from database even after service restart.
    """
    job = None
    output_dir = None

    # First check memory
    if job_id in labeling_jobs:
        job = labeling_jobs[job_id]
        output_dir = Path(job.get("output_dir", ""))
    # If not in memory, try to load from database
    elif state.db:
        db_job = state.db.get_job(job_id)
        if db_job and db_job.get("job_type") == "labeling":
            output_dir = Path(db_job.get("output_path", ""))
            # Check if checkpoint exists before trying to reconstruct
            checkpoint_path = output_dir / "checkpoint.json"
            if not checkpoint_path.exists():
                return LabelingJobResponse(
                    success=False,
                    error="No checkpoint found. Job must be restarted from the beginning."
                )

            # Load checkpoint to get full job state
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)

                # Reconstruct job in memory from database and checkpoint
                request_params = db_job.get("request_params", {}) or {}
                labeling_jobs[job_id] = {
                    "job_id": job_id,
                    "job_type": "labeling",
                    "status": JobStatus.FAILED,  # Will be updated to PROCESSING
                    "total_images": db_job.get("total_items", 0),
                    "processed_images": checkpoint.get("last_processed_idx", 0),
                    "total_objects_found": sum(checkpoint.get("objects_by_class", {}).values()),
                    "objects_by_class": checkpoint.get("objects_by_class", {}),
                    "current_image": "",
                    "output_dir": str(output_dir),
                    "output_formats": request_params.get("output_formats", ["coco"]),
                    "errors": [],
                    "processing_time_ms": 0.0,
                    "started_at": None,
                    "completed_at": None,
                    # Reconstruct processing params from checkpoint/database
                    "_image_paths": _get_image_paths_from_output_dir(output_dir),
                    "_classes": request_params.get("classes", []),
                    "_task_type": request_params.get("task_type", "segmentation"),
                    "_min_confidence": request_params.get("min_confidence", 0.5),
                    "_min_area": 100,
                    "_max_instances": 100,
                    "_simplify_polygons": True,
                    "_simplify_tolerance": 2.0,
                    "_save_visualizations": True,  # Enable for resumed jobs
                    "_class_mapping": request_params.get("class_mapping"),
                    "_padding": 0,
                }
                job = labeling_jobs[job_id]
                logger.info(f"Reconstructed job {job_id} from database and checkpoint")
            except Exception as e:
                logger.error(f"Failed to reconstruct job from checkpoint: {e}")
                return LabelingJobResponse(
                    success=False,
                    error=f"Failed to reconstruct job: {str(e)}"
                )

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Only allow resuming failed/cancelled/interrupted jobs
    status = job.get("status")
    if hasattr(status, 'value'):
        status_str = status.value
    else:
        status_str = str(status)

    if status_str not in ["failed", "cancelled", "interrupted"]:
        return LabelingJobResponse(
            success=False,
            error=f"Job cannot be resumed. Current status: {status_str}. "
                  f"Only failed, cancelled, or interrupted jobs can be resumed."
        )

    # Check if checkpoint exists
    if output_dir is None:
        output_dir = Path(job.get("output_dir", ""))
    checkpoint_path = output_dir / "checkpoint.json"

    if not checkpoint_path.exists():
        return LabelingJobResponse(
            success=False,
            error="No checkpoint found. Job must be restarted from the beginning."
        )

    # Load checkpoint to get resume point
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        resume_from = checkpoint.get("last_processed_idx", 0)
    except Exception as e:
        return LabelingJobResponse(
            success=False,
            error=f"Failed to load checkpoint: {str(e)}"
        )

    # Reset job status and start processing
    job["status"] = JobStatus.PROCESSING
    job["errors"] = []  # Clear previous errors
    job["started_at"] = datetime.now().isoformat()
    job["completed_at"] = None

    # Start background task to resume
    async def run_resume():
        try:
            await _process_labeling_job(job_id, resume_from=resume_from)
            labeling_jobs[job_id]["status"] = JobStatus.COMPLETED
        except Exception as e:
            logger.exception(f"Resumed labeling job {job_id} failed: {e}")
            labeling_jobs[job_id]["status"] = JobStatus.FAILED
            labeling_jobs[job_id]["errors"].append(str(e))
        finally:
            labeling_jobs[job_id]["completed_at"] = datetime.now().isoformat()

    asyncio.create_task(run_resume())

    logger.info(f"Resumed labeling job {job_id} from image {resume_from}")

    return LabelingJobResponse(
        success=True,
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message=f"Job resumed from image {resume_from + 1} of {job.get('total_images', 0)}",
        total_images=job.get("total_images", 0)
    )


@router.delete("/labeling/jobs/{job_id}", tags=["Labeling Tool"])
async def cancel_labeling_job(job_id: str):
    """
    Cancel a running labeling job.

    Sets the job status to CANCELLED and allows it to be resumed later
    from the checkpoint.
    """
    # Check memory first
    if job_id in labeling_jobs:
        job = labeling_jobs[job_id]
        status = job.get("status")

        # Can only cancel running/processing/queued jobs
        if status in [JobStatus.PROCESSING, JobStatus.QUEUED]:
            job["status"] = JobStatus.CANCELLED
            job["completed_at"] = datetime.now().isoformat()

            # Update database if available
            if state.db:
                try:
                    state.db.update_job_status(job_id, "cancelled")
                except Exception as e:
                    logger.warning(f"Failed to update job status in database: {e}")

            logger.info(f"Cancelled labeling job {job_id}")
            return {
                "success": True,
                "job_id": job_id,
                "message": "Job cancelled successfully. Can be resumed from checkpoint.",
                "processed_images": job.get("processed_images", 0),
                "total_images": job.get("total_images", 0),
            }
        else:
            status_str = status.value if hasattr(status, 'value') else str(status)
            return {
                "success": False,
                "error": f"Job cannot be cancelled. Current status: {status_str}"
            }

    # Check database
    if state.db:
        db_job = state.db.get_job(job_id)
        if db_job and db_job.get("job_type") == "labeling":
            if db_job.get("status") in ["running", "pending"]:
                state.db.update_job_status(job_id, "cancelled")
                return {
                    "success": True,
                    "job_id": job_id,
                    "message": "Job cancelled in database."
                }
            else:
                return {
                    "success": False,
                    "error": f"Job cannot be cancelled. Current status: {db_job.get('status')}"
                }

    raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.post("/labeling/jobs/{job_id}/delete", tags=["Labeling Tool"])
async def delete_labeling_job(job_id: str, delete_files: bool = False):
    """
    Delete a labeling job from memory and database.

    Args:
        job_id: The job ID to delete
        delete_files: If True, also delete output files (default: False)

    Note: If the job is running, it will be cancelled first.
    """
    output_dir = None
    was_cancelled = False

    # First, try to cancel if running
    if job_id in labeling_jobs:
        job = labeling_jobs[job_id]
        output_dir = job.get("output_dir")

        if job.get("status") in [JobStatus.PROCESSING, JobStatus.QUEUED]:
            job["status"] = JobStatus.CANCELLED
            job["completed_at"] = datetime.now().isoformat()
            was_cancelled = True
            logger.info(f"Cancelled running labeling job {job_id} before deletion")

        # Remove from memory
        del labeling_jobs[job_id]
        logger.info(f"Removed labeling job {job_id} from memory")

    # Remove from database
    db_deleted = False
    if state.db:
        try:
            db_job = state.db.get_job(job_id)
            if db_job:
                if output_dir is None:
                    output_dir = db_job.get("output_path")
                state.db.delete_job(job_id)
                db_deleted = True
                logger.info(f"Removed labeling job {job_id} from database")
        except Exception as e:
            logger.warning(f"Failed to delete job from database: {e}")

    # Delete files if requested
    files_deleted = False
    if delete_files and output_dir:
        try:
            output_path = Path(output_dir)
            if output_path.exists():
                import shutil
                shutil.rmtree(output_path)
                files_deleted = True
                logger.info(f"Deleted output directory: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to delete output files: {e}")

    if job_id not in labeling_jobs and not db_deleted:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return {
        "success": True,
        "job_id": job_id,
        "message": "Job deleted successfully",
        "details": {
            "was_cancelled": was_cancelled,
            "removed_from_memory": True,
            "removed_from_database": db_deleted,
            "files_deleted": files_deleted,
        }
    }
