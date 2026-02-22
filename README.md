# Cats vs Dogs MLOps Pipeline
## BITS Pilani – MLOps Assignment 2 (S1-25_AIMLCZG523)

An end-to-end MLOps pipeline for binary image classification (Cats vs Dogs).

---

## Project Structure

```
cats-dogs-mlops/
├── src/
│   ├── data/           # Data preprocessing & loading
│   ├── models/         # Model definition & training
│   ├── api/            # FastAPI inference service
│   └── tests/          # Unit tests (pytest)
├── notebooks/          # Exploratory notebooks
├── docker/             # Dockerfile & compose
├── k8s/                # Kubernetes manifests
├── .github/workflows/  # GitHub Actions CI/CD
├── scripts/            # Helper scripts
├── monitoring/         # Prometheus config
├── dvc.yaml            # DVC pipeline
├── .dvcignore
└── requirements.txt
```

---

## Milestones

| Milestone | Description | Tools |
|-----------|-------------|-------|
| M1 | Model Development & Experiment Tracking | Git, DVC, MLflow |
| M2 | Model Packaging & Containerization | FastAPI, Docker |
| M3 | CI Pipeline | GitHub Actions |
| M4 | CD Pipeline & Deployment | Kubernetes / Docker Compose |
| M5 | Monitoring & Logging | Prometheus, structlog |

---

## Quick Start

### 1. Clone & Setup
```bash
git clone <repo-url>
cd cats-dogs-mlops

# Install dependencies with uv
uv sync
```

### 2. Download Dataset (Kaggle)
```bash
kaggle datasets download bhavikjikadara/dog-and-cat-classification-dataset
unzip dog-and-cat-classification-dataset.zip -d data/raw/
```

### 3. Run DVC Pipeline (preprocess + train)
```bash
dvc repro
```

### 4. Launch MLflow UI
```bash
mlflow ui --port 5000
```

### 5. Run Inference API locally
```bash
uvicorn src.api.app:app --reload --port 8000
```

### 6. Build & Run Docker
```bash
docker build -f docker/Dockerfile -t cats-dogs-classifier:latest .
docker run -p 8000:8000 cats-dogs-classifier:latest
```

### 7. Run Tests
```bash
pytest src/tests/ -v
```

### 8. Deploy with Docker Compose
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 9. Deploy to Kubernetes
```bash
kubectl apply -f k8s/
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict cat or dog |
| GET | `/metrics` | Prometheus metrics |

### Example Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_cat.jpg"
```

Response:
```json
{
  "label": "cat",
  "confidence": 0.94,
  "probabilities": {"cat": 0.94, "dog": 0.06},
  "latency_ms": 23.4
}
```
