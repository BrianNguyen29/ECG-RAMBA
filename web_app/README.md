# ECG-RAMBA Web Application

A web-based interface for ECG analysis using the ECG-RAMBA deep learning model.

## Features

- 🩺 **12-Lead ECG Analysis**: Upload and visualize 12-lead ECG signals
- 🤖 **ECG-RAMBA Model**: Multi-label classification with 27 SNOMED classes
- 📊 **Probability Visualization**: See confidence scores for all predicted conditions
- 💡 **Medical Insights**: Automated explanations and recommendations
- 📜 **History Tracking**: Save and review past analyses

## Architecture

```
web_app/
├── backend/              # FastAPI Server (Python)
│   ├── main.py           # Entry point
│   ├── app/
│   │   ├── api/          # REST endpoints
│   │   └── core/         # Model loader, signal processing
│   └── requirements.txt
│
└── frontend/             # React + Vite (JavaScript)
    ├── src/
    │   ├── pages/        # Dashboard, History, Story
    │   ├── components/   # ECGGraph, DiagnosisReport
    │   └── services/     # API client
    └── package.json
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Trained ECG-RAMBA model weights in `models/` directory

### Backend Setup

```bash
cd web_app/backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn main:app --reload --port 8000
```

The API will be available at: http://localhost:8000

### Frontend Setup

```bash
cd web_app/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The UI will be available at: http://localhost:5173

### Using the Launcher (Windows)

```bash
cd web_app
run_app.bat
```

This will start both backend and frontend in separate terminals.

## API Endpoints

| Method | Endpoint              | Description                      |
| :----- | :-------------------- | :------------------------------- |
| GET    | `/api/models`         | List available model checkpoints |
| GET    | `/api/info`           | API and model information        |
| GET    | `/api/classes`        | List of SNOMED classes           |
| POST   | `/api/upload`         | Upload ECG file (JSON/CSV/MAT)   |
| POST   | `/api/predict`        | Run 12-lead ECG inference        |
| POST   | `/api/predict/simple` | Single-lead inference (demo)     |

### Example: Prediction Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "fold1_best.pt",
    "signal_data": [[...lead1...], [...lead2...], ...]
  }'
```

### Example: Response

```json
{
  "model_used": "fold1_best.pt",
  "top_diagnosis": "AF",
  "diagnosis_full_name": "Atrial Fibrillation",
  "confidence": 0.87,
  "predictions": [["AF", 0.87], ["PAC", 0.52]],
  "all_probabilities": {"AF": 0.87, "SNR": 0.23, ...},
  "explanation": "Irregular heart rhythm that can lead to blood clots.",
  "recommendation": "Consult cardiologist. May require anticoagulants."
}
```

## Supported File Formats

| Format   | Description                                                  |
| :------- | :----------------------------------------------------------- |
| **JSON** | `{"leads": [[...], [...], ...]}` or `{"lead_1": [...], ...}` |
| **CSV**  | 12 columns (one per lead) or single column                   |
| **MAT**  | MATLAB file with `val` or `signal` field                     |

## Development

### Backend Testing

```bash
cd web_app/backend
pytest
```

### Frontend Build

```bash
cd web_app/frontend
npm run build
```

Production files will be in `frontend/dist/`.

## Configuration

### Backend

- **CORS**: Configured in `main.py` for localhost ports
- **Model Path**: Automatically uses project root `models/` directory

### Frontend

- **API URL**: Set `VITE_API_URL` in `.env` for production

## Troubleshooting

| Issue                     | Solution                                      |
| :------------------------ | :-------------------------------------------- |
| "ECG-RAMBA import failed" | Ensure project root is in PYTHONPATH          |
| "Model not found"         | Check that `models/fold*_best.pt` exist       |
| CORS error                | Verify frontend URL is in `main.py` origins   |
| "mamba-ssm not found"     | Install with CUDA support or use CPU fallback |

## License

MIT License - See [LICENSE](../LICENSE)
