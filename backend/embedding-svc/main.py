import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import aiofiles

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from clip_embedder import CLIPEmbedder

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SHARED_DIR  = Path(os.environ.get("SHARED_DIR", "./shared_data"))
UPLOADS_DIR = SHARED_DIR / "uploads"
EMB_DIR     = SHARED_DIR / "embeddings"
CLIP_MODEL  = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
BATCH_SIZE  = int(os.environ.get("CLIP_BATCH_SIZE", 64))

for d in [UPLOADS_DIR, EMB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Pydantic response models
# -----------------------------------------------------------------------------

class JobSubmittedResponse(BaseModel):
    job_id: str = Field(..., example="a3f2c1d4-8b0e-4f2a-9c1d-123456789abc",
                        description="Unique identifier for the submitted job.")
    status: str = Field(..., example="pending",
                        description="Initial job status. Always 'pending' on submission.")
    files_received: Optional[int] = Field(None, example=3,
                        description="Number of image files received (image jobs only).")
    rows: Optional[int] = Field(None, example=150,
                        description="Number of text rows to embed (text jobs only).")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., example="a3f2c1d4-8b0e-4f2a-9c1d-123456789abc")
    status: str = Field(..., example="done",
                        description="Current job status: pending | running | done | error")
    error: Optional[str] = Field(None, example=None,
                        description="Error message if status is 'error', otherwise null.")

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model:  str = Field(..., example="openai/clip-vit-base-patch32",
                        description="CLIP model currently loaded in memory.")

# -----------------------------------------------------------------------------
# App & shared state
# -----------------------------------------------------------------------------
app = FastAPI(
    title="CapFlex — Embedding Service",
    description="""
Generates **CLIP embeddings** for images and text using `openai/clip-vit-base-patch32`.

Both modalities are projected into the same **512-dimensional space**, making their
embeddings directly comparable via cosine similarity.

## Workflow

1. Submit an embedding job via `POST /embeddings/images` or `POST /embeddings/texts`.
2. Poll `GET /embeddings/status/{job_id}` until status is `done`.
3. Download the resulting CSV via `GET /embeddings/download/{job_id}`.

The output CSV can be passed directly to the **Clustering Service** using the `embedding_job_id` parameter.
""",
    version="1.0.0",
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict] = {}
embedder: Optional[CLIPEmbedder] = None
executor = ThreadPoolExecutor(max_workers=2)


@app.on_event("startup")
def load_model():
    global embedder
    embedder = CLIPEmbedder(model_name=CLIP_MODEL, batch_size=BATCH_SIZE)


# =============================================================================
# Helpers
# =============================================================================

def _new_job() -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "output": None, "error": None}
    return job_id


def _run_image_embedding(job_id: str, image_paths: list[str], ids: list[str]):
    try:
        jobs[job_id]["status"] = "running"
        emb = embedder.embed_images(image_paths)
        out_path = str(EMB_DIR / f"{job_id}.csv")
        CLIPEmbedder.save_embeddings(emb, out_path, ids=ids, prefix="emb")
        jobs[job_id]["status"] = "done"
        jobs[job_id]["output"] = out_path
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)


def _run_text_embedding(job_id: str, texts: list[str], ids: list[str]):
    try:
        jobs[job_id]["status"] = "running"
        emb = embedder.embed_texts(texts)
        out_path = str(EMB_DIR / f"{job_id}.csv")
        CLIPEmbedder.save_embeddings(emb, out_path, ids=ids, prefix="emb")
        jobs[job_id]["status"] = "done"
        jobs[job_id]["output"] = out_path
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)


# =============================================================================
# Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health():
    """Returns the service status and the CLIP model currently loaded."""
    return {"status": "ok", "model": CLIP_MODEL}


@app.post(
    "/embeddings/images",
    response_model=JobSubmittedResponse,
    summary="Generate image embeddings",
    tags=["Embeddings"],
    responses={
        200: {"description": "Job accepted. Use the returned job_id to poll status."},
        400: {"description": "No files were uploaded."},
    },
)
async def embed_images(
    files: list[UploadFile] = File(
        ...,
        description="One or more image files (JPG, PNG, BMP, WEBP, GIF, TIFF). "
                    "Each file becomes one row in the output embedding CSV.",
    ),
):
    """
    Upload one or more image files to generate **CLIP image embeddings**.

    - Each image is encoded into a **512-dimensional L2-normalized vector**.
    - The output CSV will have columns: `id` (filename), `emb_0` ... `emb_511`.
    - Processing runs asynchronously. Poll `/embeddings/status/{job_id}` to track progress.

    **Supported formats:** JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP.
    """
    if not files:
        raise HTTPException(400, "No files uploaded.")

    job_id  = _new_job()
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for f in files:
        dest = job_dir / f.filename
        async with aiofiles.open(dest, "wb") as out:
            await out.write(await f.read())
        saved_paths.append(str(dest))

    ids  = [f.filename for f in files]
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_image_embedding, job_id, saved_paths, ids)

    return {"job_id": job_id, "status": "pending", "files_received": len(files)}


@app.post(
    "/embeddings/texts",
    response_model=JobSubmittedResponse,
    summary="Generate text embeddings",
    tags=["Embeddings"],
    responses={
        200: {"description": "Job accepted. Use the returned job_id to poll status."},
        400: {"description": "Specified text column not found in the uploaded CSV."},
    },
)
async def embed_texts(
    file: UploadFile = File(
        ...,
        description="CSV file containing at least one column with text strings.",
    ),
    text_column: str = Form(
        ...,
        description="Name of the column that contains the text to embed.",
        example="review_text",
    ),
    id_column: Optional[str] = Form(
        None,
        description="Optional column to use as row identifier in the output CSV. "
                    "If omitted, row index (0, 1, 2...) is used.",
        example="product_id",
    ),
):
    """
    Upload a CSV and generate **CLIP text embeddings** for a specified column.

    - Each text row is encoded into a **512-dimensional L2-normalized vector**.
    - CLIP's text encoder supports up to **77 tokens**; longer strings are truncated automatically.
    - The output CSV will have columns: `id`, `emb_0` ... `emb_511`.
    - Processing runs asynchronously. Poll `/embeddings/status/{job_id}` to track progress.

    **Tip:** Multiple text columns can be merged into one before uploading for richer embeddings.
    """
    job_id   = _new_job()
    csv_path = UPLOADS_DIR / f"{job_id}_input.csv"

    async with aiofiles.open(csv_path, "wb") as out:
        await out.write(await file.read())

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise HTTPException(
            400,
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").astype(str).tolist()
    ids   = df[id_column].astype(str).tolist() \
            if id_column and id_column in df.columns \
            else [str(i) for i in range(len(df))]

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_text_embedding, job_id, texts, ids)

    return {"job_id": job_id, "status": "pending", "rows": len(texts)}


@app.get(
    "/embeddings/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check embedding job status",
    tags=["Embeddings"],
    responses={
        200: {"description": "Current job status."},
        404: {"description": "Job ID not found."},
    },
)
def get_status(job_id: str):
    """
    Returns the current status of an embedding job.

    | Status    | Meaning                                      |
    |-----------|----------------------------------------------|
    | `pending` | Job queued, not yet started                  |
    | `running` | Embeddings are being generated               |
    | `done`    | Embeddings ready — call `/download/{job_id}` |
    | `error`   | Job failed — check the `error` field         |
    """
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    job = jobs[job_id]
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


@app.get(
    "/embeddings/download/{job_id}",
    summary="Download embeddings CSV",
    tags=["Embeddings"],
    responses={
        200: {
            "description": "CSV file with columns: `id`, `emb_0`, `emb_1`, ... `emb_511`.",
            "content": {"text/csv": {}},
        },
        400: {"description": "Job is not finished yet."},
        404: {"description": "Job ID not found."},
        500: {"description": "Output file missing on disk."},
    },
)
def download_embeddings(job_id: str):
    """
    Downloads the embedding CSV produced by a completed job.

    **Output format:**

    | id         | emb_0   | emb_1   | ... | emb_511 |
    |------------|---------|---------|-----|---------|
    | image1.jpg | 0.0312  | -0.0184 | ... | 0.0091  |
    | image2.png | 0.0187  | 0.0423  | ... | -0.0205 |

    - All vectors are **L2-normalized** (unit norm ~= 1.0).
    - This file can be passed directly to the Clustering Service using `embedding_job_id`.
    """
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, f"Job not ready. Current status: '{job['status']}'.")

    out_path = Path(job["output"])
    if not out_path.exists():
        raise HTTPException(500, "Output file not found on disk.")

    return FileResponse(
        path=str(out_path),
        media_type="text/csv",
        filename=f"embeddings_{job_id}.csv",
    )