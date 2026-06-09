import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import aiofiles
import math

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from capflex import (
    remove_label_column,
    preprocess_dataset,
    compute_cardinality_ranges,
    count_combinations_exact,
    run_parallel_search,
    analyze_pareto_exact,
    optimize_constrained_clustering,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SHARED_DIR  = Path(os.environ.get("SHARED_DIR", "./shared_data"))
UPLOADS_DIR = SHARED_DIR / "uploads"
EMB_DIR     = SHARED_DIR / "embeddings"
RESULTS_DIR = SHARED_DIR / "results"

for d in [UPLOADS_DIR, EMB_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Pydantic response models
# -----------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")

class JobSubmittedResponse(BaseModel):
    job_id: str = Field(..., example="b7e3a2f1-1c4d-4a8b-9f2e-abcdef123456",
                        description="Unique identifier for the clustering job.")
    status: str = Field(..., example="pending",
                        description="Initial status. Always 'pending' on submission.")
    input_type: str = Field(..., example="tabular",
                            description="Resolved input type: 'tabular' or 'embeddings'.")
    target_cardinality: list[int] = Field(..., example=[50, 50, 50],
                                          description="Target cluster sizes as provided.")
    delta: float = Field(..., example=0.1,
                         description="Flexibility tolerance applied to cardinality bounds.")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., example="b7e3a2f1-1c4d-4a8b-9f2e-abcdef123456")
    status: str = Field(..., example="done",
                        description="Current status: pending | running | done | error")
    error: Optional[str] = Field(None, example=None,
                                 description="Error detail if status is 'error'.")

class KneePointMetrics(BaseModel):
    solution_id: int   = Field(..., example=42,
                               description="Index of this solution in the explored pool.")
    silhouette: float  = Field(..., example=0.5821,
                               description="Mean silhouette score (cosine distance). Range: [-1, 1]. Higher is better.")
    AMI: Optional[float] = Field(None, example=0.7134,
                                 description="Adjusted Mutual Information vs ground truth labels. "
                                             "Range: [0, 1]. Only present when label_column was provided.")
    ILVC: int          = Field(..., example=4,
                               description="Instance-Level Violation Count: total absolute deviation "
                                           "in cluster sizes from the target cardinality.")
    CLVC: int          = Field(..., example=1,
                               description="Cluster-Level Violation Count: number of clusters whose "
                                           "size differs from the target.")
    CSVI: float        = Field(..., example=0.0267,
                               description="Cardinality Satisfaction Violation Index: weighted combination "
                                           "of ILVC and CLVC. Range: [0, 1]. Lower is better.")
    cardinality: str   = Field(..., example="49-51-50",
                               description="Actual cluster sizes in the final assignment.")

class ParetoSolution(BaseModel):
    solution_id: int
    silhouette: float  = Field(..., description="Silhouette score. Higher is better.")
    ILVC: float
    CLVC: float
    CSVI: float        = Field(..., description="Cardinality violation index. Lower is better.")
    AMI: Optional[float] = None
    cardinality: str

class ClusteringResultsResponse(BaseModel):
    job_id: str
    knee_point: KneePointMetrics = Field(...,
        description="Metrics of the selected optimal solution (knee point of the Pareto front).")
    pareto_front: list[ParetoSolution] = Field(...,
        description="All non-dominated solutions. Each maximizes Silhouette and minimizes CSVI.")
    n_pareto_solutions: int = Field(..., example=7,
        description="Number of solutions on the Pareto front.")
    combinations_explored: int = Field(..., example=450,
        description="Total cardinality combinations evaluated during the search.")
    combinations_possible: int = Field(..., example=1771,
        description="Theoretical maximum of valid cardinality combinations given delta.")

# -----------------------------------------------------------------------------
# App & shared state
# -----------------------------------------------------------------------------
app = FastAPI(
    title="CapFlex — Clustering Service",
    description="""
Soft clustering with **flexible cardinality balancing** using Mixed-Integer Linear Programming.

## Algorithm overview

CapFlex explores a pool of valid cardinality vectors (cluster size combinations) within
a user-defined tolerance `delta` around the target sizes. For each candidate, it solves
a **transport LP** to assign data points to clusters minimizing cosine distance.
Solutions are evaluated on a **Pareto front** (Silhouette vs CSVI) and the optimal
trade-off is selected via the **Knee Point** method.

## Input modes

| Mode | How to use |
|------|-----------|
| **Tabular CSV** | Upload a regular CSV. Preprocessing (imputation, encoding) is automatic. |
| **Embedding CSV** | Upload a CSV with `emb_0`...`emb_511` columns from CLIP. |
| **Embedding job** | Pass the `job_id` from the Embedding Service directly. |

## Workflow

1. Submit a job via `POST /clustering/run`.
2. Poll `GET /clustering/status/{job_id}` until status is `done`.
3. Inspect metrics and Pareto front via `GET /clustering/results/{job_id}`.
4. Download the labeled dataset via `GET /clustering/download/{job_id}`.
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
executor = ThreadPoolExecutor(max_workers=4)


# =============================================================================
# Helpers
# =============================================================================

def _new_job() -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "results": None, "output": None, "error": None}
    return job_id


def _load_input(csv_path, input_type, label_column, embedding_prefix):
    df = pd.read_csv(csv_path)
    true_labels = None
    metadata = None

    if label_column and label_column in df.columns:
        true_labels = df[label_column].values
        df = df.drop(columns=[label_column])

    if input_type == "tabular":
        input_data = preprocess_dataset(df)
    elif input_type == "embeddings":
        emb_cols = [c for c in df.columns if c.startswith(f"{embedding_prefix}_")]
        if not emb_cols:
            raise ValueError(
                f"No columns with prefix '{embedding_prefix}_' found. "
                f"Available: {list(df.columns)}"
            )
        meta_cols = [c for c in df.columns if not c.startswith(f"{embedding_prefix}_")]
        if meta_cols:
            metadata = df[meta_cols].reset_index(drop=True)
        input_data = df[emb_cols].astype(float).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown input_type '{input_type}'. Use 'tabular' or 'embeddings'.")

    return input_data, true_labels, metadata

def sanitize(obj):
    """Recursively convert NumPy types and NaN/Inf to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def _run_clustering(job_id, csv_path, input_type, label_column,
                    embedding_prefix, target_cardinality, delta, max_iter):
    try:
        jobs[job_id]["status"] = "running"

        input_data, true_labels, metadata = _load_input(
            csv_path, input_type, label_column, embedding_prefix
        )

        target_card = np.array(target_cardinality)
        n = len(input_data)
        k = len(target_card)

        X      = input_data.values.astype(float)
        X_norm = X / (np.sqrt(np.sum(X ** 2, axis=1, keepdims=True)) + 1e-10)

        ranges      = compute_cardinality_ranges(target_card, delta)
        max_teorico = count_combinations_exact(n, ranges)
        n_iter      = min(max_iter, max_teorico) if max_iter else \
                      min(10000, min(50 * (k ** 2), max_teorico))

        results_df = run_parallel_search(
            input_data, target_card, delta, true_labels, n_BA=n_iter,
        )
        X      = input_data.values.astype(float)
        X_norm = X / (np.sqrt(np.sum(X ** 2, axis=1, keepdims=True)) + 1e-10)

        if results_df is None or results_df.empty:
            raise RuntimeError("No feasible clustering solutions found.")

        analisis  = analyze_pareto_exact(results_df)
        if analisis is None:
            raise RuntimeError("Pareto analysis returned no results.")

        pareto_df = analisis["pareto_full"]
        knee_row  = analisis["knee_full"].iloc[0]

        best_card   = np.array([int(c) for c in knee_row["cardinality"].split("-")])
        final_model = optimize_constrained_clustering(
            X, X_norm, best_card, max_iter=100, seed=int(knee_row["saved_seed"])
        )
        if not final_model["valid"]:
            raise RuntimeError("Could not reconstruct final model.")

        p          = final_model["p"]
        centroids  = final_model["centroids"]
        cent_norm  = centroids / (np.sqrt(np.sum(centroids**2, axis=1, keepdims=True)) + 1e-10)
        dists      = 1 - (X_norm @ cent_norm.T)
        idx_mat    = (np.arange(n), p)
        a_i        = dists[idx_mat].copy()
        dists_copy = dists.copy()
        dists_copy[idx_mat] = np.inf
        b_i        = dists_copy.min(axis=1)
        denom      = np.maximum(a_i, b_i)
        final_sil  = float(np.nanmean(np.where(denom == 0, 0.0, (b_i - a_i) / denom)))

        real_counts = np.bincount(p, minlength=k)
        final_ilvc  = float(np.sum(np.abs(np.sort(real_counts) - np.sort(target_card))))
        final_clvc  = float(np.sum(np.sort(real_counts) != np.sort(target_card)))
        final_csvi  = 0.5 * (final_ilvc / n) + 0.5 * (final_clvc / k)

        final_ami = None
        if true_labels is not None:
            from sklearn.metrics import adjusted_mutual_info_score
            from sklearn.preprocessing import LabelEncoder
            final_ami = float(adjusted_mutual_info_score(
                LabelEncoder().fit_transform([str(x).strip() for x in true_labels]), p
            ))

        out_df = input_data.copy()
        if metadata is not None:
            for col in metadata.columns:
                out_df.insert(0, col, metadata[col].values)
        if true_labels is not None:
            out_df["true_label"] = true_labels
        out_df["cluster"] = p
        out_csv = str(RESULTS_DIR / f"{job_id}.csv")
        out_df.to_csv(out_csv, index=False)

        pareto_renamed = pareto_df.rename(
            columns={"s": "silhouette", "csvi": "CSVI"}, errors="ignore"
        )
        pareto_records = pareto_renamed[
            [c for c in ["solution_id", "silhouette", "ILVC", "CLVC", "CSVI", "AMI", "cardinality"]
             if c in pareto_renamed.columns]
        ].to_dict(orient="records")

        jobs[job_id]["status"]  = "done"
        jobs[job_id]["output"]  = out_csv
        jobs[job_id]["results"] = {
            "knee_point": {
                "solution_id": int(knee_row.get("solution_id", -1)),
                "silhouette":  round(final_sil, 4),
                "AMI":         round(final_ami, 4) if final_ami is not None else None,
                "ILVC":        int(final_ilvc),
                "CLVC":        int(final_clvc),
                "CSVI":        round(final_csvi, 4),
                "cardinality": "-".join(str(c) for c in real_counts),
            },
            "pareto_front":           pareto_records,
            "n_pareto_solutions":     len(pareto_records),
            "combinations_explored":  int(n_iter),
            "combinations_possible":  int(max_teorico),
        }

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
    """Returns the service status."""
    return {"status": "ok"}


@app.post(
    "/clustering/run",
    response_model=JobSubmittedResponse,
    summary="Launch a clustering job",
    tags=["Clustering"],
    responses={
        200: {"description": "Job accepted and queued."},
        400: {"description": "Invalid input — missing file/job_id, bad cardinality format, or column not found."},
    },
)
async def run_clustering(
    file: Optional[UploadFile] = File(
        None,
        description="CSV file to cluster. Can be a **tabular CSV** (raw features) or an "
                    "**embedding CSV** (columns `emb_0`...`emb_511` from the Embedding Service). "
                    "Mutually exclusive with `embedding_job_id`.",
    ),
    embedding_job_id: Optional[str] = Form(
        None,
        description="**job_id** returned by the Embedding Service. "
                    "When provided, the embedding CSV is read directly from the shared volume "
                    "without re-uploading. Mutually exclusive with `file`.",
        example="a3f2c1d4-8b0e-4f2a-9c1d-123456789abc",
    ),
    input_type: str = Form(
        "tabular",
        description="Type of input data:\n"
                    "- `tabular`: raw CSV — preprocessing (imputation, encoding) is applied automatically.\n"
                    "- `embeddings`: CSV with pre-computed CLIP embedding columns (`emb_0`...`emb_N`).",
        example="tabular",
    ),
    embedding_prefix: str = Form(
        "emb",
        description="Column prefix used to identify embedding columns in the CSV. "
                    "Default is `emb`, matching the output of the Embedding Service (`emb_0`, `emb_1`, ...).",
        example="emb",
    ),
    label_column: Optional[str] = Form(
        None,
        description="Name of the ground truth label column in the CSV. "
                    "When provided, **AMI** (Adjusted Mutual Information) is computed "
                    "to evaluate clustering quality against the known labels. "
                    "This column is excluded from the feature matrix.",
        example="Species",
    ),
    target_cardinality: str = Form(
        ...,
        description="Comma-separated **target cluster sizes**. Must sum to the total number of rows. "
                    "Example: `50,50,50` for 3 equally-sized clusters of 50 points each.",
        example="50,50,50",
    ),
    delta: float = Form(
        0.1,
        description="Flexibility tolerance for cardinality bounds. "
                    "Each cluster size is allowed to vary within `[target*(1-delta), target*(1+delta)]`. "
                    "Range: `[0.0, 1.0]`. Higher values allow more flexibility.",
        example=0.1,
    ),
    max_iter: Optional[int] = Form(
        None,
        description="Maximum number of cardinality combinations to explore. "
                    "If `null`, an automatic heuristic is used: `min(10000, 50 * k²)` "
                    "where `k` is the number of clusters.",
        example=500,
    ),
):
    """
    Launches a **CapFlex clustering job**. Accepts three input modes:

    **1. Tabular CSV** (`input_type=tabular`):
    Upload a regular CSV with numeric or categorical features.
    Preprocessing (mean imputation, label encoding, ID column removal) is applied automatically.

    **2. Embedding CSV** (`input_type=embeddings`):
    Upload a CSV with pre-computed embeddings (e.g., from the Embedding Service).
    Columns must follow the pattern `{embedding_prefix}_0`, `{embedding_prefix}_1`, ...

    **3. Embedding job_id** (`embedding_job_id=<uuid>`):
    Reference the output of a completed Embedding Service job directly.
    The embedding CSV is read from the shared volume — no re-upload needed.
    """
    if embedding_job_id is not None:
        csv_path   = EMB_DIR / f"{embedding_job_id}.csv"
        if not csv_path.exists():
            raise HTTPException(
                400,
                f"No embedding file found for job_id '{embedding_job_id}'. "
                "Ensure the embedding job completed successfully (status = 'done')."
            )
        csv_path   = str(csv_path)
        input_type = "embeddings"
    elif file is not None:
        uid      = str(uuid.uuid4())
        csv_path = str(UPLOADS_DIR / f"{uid}_input.csv")
        async with aiofiles.open(csv_path, "wb") as out:
            await out.write(await file.read())
    else:
        raise HTTPException(400, "Provide either 'file' or 'embedding_job_id'.")

    try:
        target_card_list = [int(x.strip()) for x in target_cardinality.split(",")]
    except ValueError:
        raise HTTPException(400, "target_cardinality must be comma-separated integers, e.g. '50,50,50'.")

    if len(target_card_list) < 2:
        raise HTTPException(400, "At least 2 clusters are required in target_cardinality.")

    job_id = _new_job()
    loop   = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, _run_clustering, job_id, csv_path, input_type,
        label_column, embedding_prefix, target_card_list, delta, max_iter,
    )

    return {
        "job_id": job_id, "status": "pending",
        "input_type": input_type,
        "target_cardinality": target_card_list,
        "delta": delta,
    }


@app.get(
    "/clustering/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Check clustering job status",
    tags=["Clustering"],
    responses={
        200: {"description": "Current job status."},
        404: {"description": "Job ID not found."},
    },
)
def get_status(job_id: str):
    """
    Returns the current status of a clustering job.

    | Status    | Meaning                                             |
    |-----------|-----------------------------------------------------|
    | `pending` | Job queued, not yet started                         |
    | `running` | Cardinality pool is being explored                  |
    | `done`    | Results ready — call `/results/{job_id}`            |
    | `error`   | Job failed — check the `error` field for details    |

    **Note:** Clustering can take from seconds to several minutes depending on
    dataset size, number of clusters, and `max_iter`.
    """
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    job = jobs[job_id]
    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}



@app.get(
    "/clustering/results/{job_id}",
    response_model=ClusteringResultsResponse,
    summary="Get clustering metrics and Pareto front",
    tags=["Clustering"],
    responses={
        200: {"description": "Full metrics and Pareto front for the completed job."},
        400: {"description": "Job is not finished yet."},
        404: {"description": "Job ID not found."},
        500: {"description": "Job failed — check status endpoint for error details."},
    },
)
def get_results(job_id: str):
    """
    Returns the full results of a completed clustering job:

    - **knee_point**: metrics of the selected optimal solution.
    - **pareto_front**: all non-dominated solutions (maximize Silhouette, minimize CSVI).
    - **combinations_explored / combinations_possible**: search coverage statistics.

    **Metrics explained:**

    | Metric      | Range     | Optimum | Description |
    |-------------|-----------|---------|-------------|
    | Silhouette  | [-1, 1]   | High    | Cluster compactness and separation (cosine distance) |
    | AMI         | [0, 1]    | High    | Agreement with ground truth labels (only if provided) |
    | ILVC        | [0, n]    | 0       | Total absolute deviation from target cluster sizes |
    | CLVC        | [0, k]    | 0       | Number of clusters with wrong size |
    | CSVI        | [0, 1]    | 0       | Combined cardinality violation index |
    """
    if job_id not in jobs:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    job = jobs[job_id]
    if job["status"] == "error":
        raise HTTPException(500, f"Job failed: {job['error']}")
    if job["status"] != "done":
        raise HTTPException(400, f"Job not ready. Current status: '{job['status']}'.")
    payload = sanitize({"job_id": job_id, **job["results"]})
    return JSONResponse(content=payload)


@app.get(
    "/clustering/download/{job_id}",
    summary="Download clustering output CSV",
    tags=["Clustering"],
    responses={
        200: {
            "description": "CSV with original features plus a `cluster` column (0-indexed) "
                           "and optionally a `true_label` column if labels were provided.",
            "content": {"text/csv": {}},
        },
        400: {"description": "Job is not finished yet."},
        404: {"description": "Job ID not found."},
        500: {"description": "Output file missing on disk."},
    },
)
def download_results(job_id: str):
    """
    Downloads the final dataset with cluster assignments.

    **Output columns:**
    - All original feature columns from the input CSV.
    - `true_label` — original ground truth label (only if `label_column` was provided).
    - `cluster` — assigned cluster index (0-based integer).

    **Example:**

    | sepal_length | sepal_width | ... | true_label | cluster |
    |-------------|-------------|-----|------------|---------|
    | 5.1         | 3.5         | ... | setosa     | 0       |
    | 6.3         | 3.3         | ... | virginica  | 2       |
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
        filename=f"clustering_{job_id}.csv",
    )