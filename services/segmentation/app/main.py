"""
Segmentation Service
====================
FastAPI microservice for semantic scene analysis and SAM3 segmentation.

Uses SAM3 (Segment Anything Model 3) for Promptable Concept Segmentation (PCS).
SAM3 released 2025-11-19 - segments all instances matching a text concept.

Model: facebook/sam3 (848M parameters)
Docs: https://huggingface.co/facebook/sam3

Endpoints:
- POST /analyze - Analyze scene regions
- POST /check-compatibility - Check object-scene compatibility
- POST /suggest-placement - Get placement suggestions
- POST /segment-text - Text-driven segmentation (SAM3 PCS)
- GET /health - Health check

Debug Endpoints (Explainability):
- POST /debug/analyze - Analyze with full debug visualization and logs
- POST /debug/compatibility - Check compatibility with decision explanation

Object Extraction Endpoints:
- POST /extract/analyze-dataset - Analyze dataset annotation types
- POST /extract/objects - Extract objects from dataset (async)
- GET /extract/jobs/{job_id} - Get extraction job status
- POST /extract/single-object - Extract single object (preview)

SAM3 Tool Endpoints:
- POST /sam3/segment-image - Segment single image with box/point prompt
- POST /sam3/convert-dataset - Convert bbox to segmentation (async)
- GET /sam3/jobs/{job_id} - Get conversion job status
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    analysis,
    debug,
    extraction,
    extraction_jobs,
    health,
    labeling_jobs,
    labeling_start,
    sam3_tool,
)
from app.startup import lifespan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create FastAPI app
app = FastAPI(
    title="Segmentation Service",
    description="Semantic scene analysis and SAM3 segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The registration order defines the route table order.
for _module in (
    health,
    analysis,
    debug,
    extraction,
    extraction_jobs,
    sam3_tool,
    labeling_start,
    labeling_jobs,
):
    app.include_router(_module.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info",
    )
