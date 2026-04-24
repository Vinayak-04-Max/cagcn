# Wafer Defect Dashboard (CAGCN)

A production-style wafer defect inference dashboard with:

- **Dark-themed web frontend** for sample selection and result visualizations
- **Node.js gateway/backend** for API orchestration and static hosting
- **Python FastAPI inference service** running TensorFlow model inference

The repository preserves the original research and data assets while providing a cleaner deployable app workflow.

## Dataset
- Wafer_Map_Datasets_1.npz
  
## Project structure

```text
.
|-- app.py                      # Legacy Streamlit app (reference)
|-- ml_api.py                   # FastAPI inference service
|-- server.js                   # Node entrypoint
|-- src/server/app.js           # Node API + static server
|-- public/                     # Dark-themed frontend (HTML/CSS/JS)
|-- experiment/                 # Trained model + calibration assets
|-- results/                    # Generated artifacts and outputs
|-- requirements.txt            # Python runtime dependencies
|-- package.json                # Node runtime scripts
`-- README.md
```

## Architecture overview

1. Browser UI loads at `http://localhost:3000`.
2. Frontend requests wafer samples from Node endpoint `GET /api/wafers`.
3. Node proxies requests to FastAPI (`ml_api.py`) for:
   - `GET /wafers/random`
   - `POST /predict`
4. FastAPI runs TensorFlow inference using:
   - `experiment/best_model.h5`
   - `experiment/alpha.npy`
   - `experiment/beta.npy`
   - `experiment/thresholds.npy`

## Prerequisites

- Python 3.10+ (recommended for TensorFlow compatibility)
- Node.js 18+ (required for built-in `fetch` in Node runtime)

## Local run

### 1) Start Python inference API

```bash
cd C:\Users\santo\OneDrive\Documents\wafer
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
uvicorn ml_api:app --reload --host 127.0.0.1 --port 8000
```

### 2) Start Node web server

```bash
cd C:\Users\santo\OneDrive\Documents\wafer
npm install
npm start
```

### 3) Open dashboard

- [http://localhost:3000](http://localhost:3000)

## API contract

### `GET /api/health` (Node)

Returns Node service status and active ML API target.

### `GET /api/wafers?count=10` (Node)

Returns random wafer sample maps from the Python inference service.

### `POST /api/predict` (Node)

Request body:

```json
{
  "wafer": [[0.0, 1.0], [1.0, 0.0]]
}
```

Response body:

```json
{
  "labels": ["Center", "Scratch"],
  "probabilities": [0.12, 0.88, 0.09, 0.21, 0.34, 0.11, 0.72, 0.03],
  "binary": [0, 1, 0, 0, 0, 0, 1, 0]
}
```

## Environment variables

- `PORT` (default: `3000`)
- `HOST` (default: `0.0.0.0`)
- `ML_API_BASE_URL` (default: `http://127.0.0.1:8000`)

## Notes

- `app.py` remains in the repo as a Streamlit reference implementation.
- Large dataset/model files are intentionally kept to preserve reproducible inference behavior.
