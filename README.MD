# CapFlex: Multimodal Soft Clustering with Flexible Size Constraints

> Interactive web platform for multimodal clustering under flexible cardinality constraints, supporting tabular data, images, and text documents.

[![Live Demo](https://img.shields.io/badge/demo-capflex.online-blue)](https://capflex.online/)
[![Video](https://img.shields.io/badge/video-demonstration-red)](https://bit.ly/4uI2jdR)
[![Algorithm](https://img.shields.io/badge/algorithm-source-green)](https://anonymous.4open.science/r/flexible_cardinality_clustering-5EB5/)

---

## Overview

**CapFlex** is an interactive system for multimodal clustering under flexible cardinality constraints. Unlike traditional clustering approaches that either ignore size constraints or enforce rigid bounds, CapFlex introduces **soft constraint optimization** allowing controlled deviations from user-defined cluster sizes while preserving clustering quality.

The system supports three input modalities:

- **Tabular data** — CSV files, automatically normalized within the pipeline.
- **Images / Text** — raw files from which 512-dimensional embeddings are generated via CLIP (`clip-vit-base-patch32`).
- **Pre-computed embeddings** — CSV files with previously computed embedding vectors.

---

## System Architecture

CapFlex follows a **microservices architecture** with independent, containerized services orchestrated via Docker Compose, deployed on a Google Cloud Compute Engine instance (e2-standard-4, 4 vCPU, 16 GB RAM).

| Service | Container | Port | Role |
|---|---|---|---|
| Nginx + React Frontend | `capflex_frontend` | `80` / `443` | Reverse proxy, static files, TLS termination |
| Embedding Service | `capflex_embedding` | `8001` | CLIP inference, embedding caching |
| Clustering Service | `capflex_clustering` | `8002` | CapFlex optimization |

Inter-service communication uses a **shared volume** (`shared_data`) for CSV exchange between the embedding and clustering backends.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, Nginx |
| Backend — Embeddings | Python, FastAPI, CLIP `openai/clip-vit-base-patch32` (HuggingFace) |
| Backend — Clustering | Python, FastAPI, MILP solver |
| Containerization | Docker, Docker Compose |

---

## Prerequisites

- [Docker](https://www.docker.com/) >= 20.x
- [Docker Compose](https://docs.docker.com/compose/) >= 2.x
- *(Optional)* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for accelerated CLIP inference.

> No local Python or Node.js installation is required. All runtime dependencies are encapsulated within Docker images.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/matt087/Web-CapFlex.git
cd Web-CapFlex
```

### 2. Build and start all services

```bash
docker compose up --build
```

### 3. Access the application

| Endpoint | Description |
|---|---|
| `http://localhost` | Web interface |
| `http://localhost:8001` | Embedding Service API |
| `http://localhost:8002` | Clustering Service API |

---

## Usage Workflows

### 1. Clustering from a tabular CSV

```bash
# Submit job
curl -X POST http://localhost:8002/clustering/run \
  -F "file=@iris.csv" \
  -F "input_type=tabular" \
  -F "label_column=Species" \
  -F "target_cardinality=50,50,50" \
  -F "delta=0.1"

# Response: {"job_id": "abc-123", "status": "pending", ...}

# Poll job status
curl http://localhost:8002/clustering/status/abc-123

# Retrieve metrics and Pareto front
curl http://localhost:8002/clustering/results/abc-123

# Download CSV with cluster assignments
curl -O http://localhost:8002/clustering/download/abc-123
```

### 2. Clustering from images (with embedding generation)

```bash
# Step 1 — Generate embeddings (embedding-svc)
curl -X POST http://localhost:8001/embeddings/images \
  -F "files=@img1.jpg" \
  -F "files=@img2.jpg" \
  -F "files=@img3.jpg"

# Response: {"job_id": "emb-456", "status": "pending", ...}

# Poll embedding job status
curl http://localhost:8001/embeddings/status/emb-456

# Step 2 — Run clustering using the embedding job_id
curl -X POST http://localhost:8002/clustering/run \
  -F "embedding_job_id=emb-456" \
  -F "target_cardinality=10,10,10" \
  -F "delta=0.1"
```

### 3. Clustering from a pre-computed embeddings CSV

```bash
# The user provides an embeddings.csv with columns emb_0, emb_1, ..., emb_511
curl -X POST http://localhost:8002/clustering/run \
  -F "file=@embeddings.csv" \
  -F "input_type=embeddings" \
  -F "embedding_prefix=emb" \
  -F "target_cardinality=16,17" \
  -F "delta=0.05"
```

### 4. Clustering from text (with embedding generation)

```bash
# CSV with a text column
curl -X POST http://localhost:8001/embeddings/texts \
  -F "file=@reviews.csv" \
  -F "text_column=review_text" \
  -F "id_column=product_id"

# Response: {"job_id": "txt-789", ...}

# Run clustering using the embedding job_id
curl -X POST http://localhost:8002/clustering/run \
  -F "embedding_job_id=txt-789" \
  -F "target_cardinality=30,30,30" \
  -F "delta=0.1"
```

---

## Interactive API Documentation (Swagger)

- Embedding service: http://localhost:8001/docs
- Clustering service: http://localhost:8002/docs

---

## `/clustering/run` Endpoint Parameters

| Parameter            | Type   | Description                                                     |
|----------------------|--------|-----------------------------------------------------------------|
| `file`               | File   | Tabular CSV or embeddings CSV                                   |
| `embedding_job_id`   | string | Job ID from embedding-svc (alternative to `file`)              |
| `input_type`         | string | `tabular` or `embeddings`                                       |
| `embedding_prefix`   | string | Column prefix for embedding columns (default: `emb`)           |
| `label_column`       | string | Ground-truth label column (optional, used to compute AMI)      |
| `target_cardinality` | string | Comma-separated target cluster sizes: `50,50,50`               |
| `delta`              | float  | Cardinality tolerance (0.0 – 1.0)                              |
| `max_iter`           | int    | Maximum number of configurations to explore (None = automatic) |

---

## Configuration

Environment variables per service (defined in `docker-compose.yml`):

**Embedding Service**

| Variable | Default | Description |
|---|---|---|
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | HuggingFace model identifier |
| `CLIP_BATCH_SIZE` | `64` | Inference batch size |
| `SHARED_DIR` | `/shared_data` | Shared volume mount path |

**Clustering Service**

| Variable | Default | Description |
|---|---|---|
| `SHARED_DIR` | `/shared_data` | Shared volume mount path |

---

## Repository Structure

```
Web-CapFlex/
├── backend/
│   ├── clustering-svc/        # CapFlex optimization microservice (FastAPI)
│   │   ├── Dockerfile
│   │   ├── capflex.py         # CapFlex algorithm implementation
│   │   ├── main.py            # FastAPI entrypoint
│   │   └── requirements.txt
│   ├── embedding-svc/         # CLIP embedding microservice (FastAPI)
│   │   ├── Dockerfile
│   │   ├── clip_embedder.py   # CLIP inference logic
│   │   ├── main.py            # FastAPI entrypoint
│   │   └── requirements.txt
│   └── README.md
├── frontend/                  # React + Vite application (served via Nginx)
│   ├── public/
│   ├── src/
│   ├── Dockerfile
│   ├── README.md
│   ├── eslint.config.js
│   ├── index.html
│   ├── nginx.conf             # Production Nginx configuration
│   ├── nginx.conf.local       # Local development Nginx configuration
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── docker-compose.yml
└── README.md
```

---

## The CapFlex Algorithm

CapFlex implements a constrained clustering pipeline with the following components:

- **Cardinality feasibility** — given a target size vector and tolerance δ, bounded intervals define the feasible cardinality space.
- **Random Search** — explores feasible cardinality configurations across the solution space.
- **Simulated Annealing** — tunes δ to identify stable quality–compliance trade-offs.
- **MILP assignment** — for each candidate capacity profile, instance assignments are solved exactly, producing crisp partitions that strictly comply with the selected capacities.
- **Pareto frontier** — solutions are evaluated jointly by Silhouette coefficient and cardinality violation metrics (ILVC, CLVC, CSVI), exposing the quality–constraint trade-off.
- **Evaluation** — when reference labels are available, Adjusted Mutual Information (AMI) is also reported.

---

## Live Deployment

- **Application:** https://capflex.online/
- **Video demonstration:** https://bit.ly/4uI2jdR

---

## Reference

If you use this software, please cite:

```
Vallejo-Huanga, D., Montenegro, M., Simbaña, B., Ferri, C., & Martínez-Plumed, F.
CapFlex: Multimodal Soft Clustering with Flexible Size Constraints.
ECML PKDD 2026 Demo Track.
```