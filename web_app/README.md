# 🏥 ECG-RAMBA Clinical Dashboard

**A Professional Web Interface for Zero-Shot ECG Classification & Analysis**

[![React](https://img.shields.io/badge/React-18.2-61DAFB?style=flat&logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.0-646CFF?style=flat&logo=vite)](https://vitejs.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?style=flat&logo=tailwind-css)](https://tailwindcss.com/)

<div align="center">
  <img src="../reports/figures/Screenshot 2026-01-17 175832.png" alt="ECG-RAMBA Clinical Dashboard" width="100%"/>
</div>

---

## 📖 Overview

The **ECG-RAMBA Clinical Dashboard** is a state-of-the-art web application designed to bridge the gap between advanced deep learning research and clinical practice. It provides physicians with a high-fidelity interface to visualize 12-lead ECGs, run real-time AI inference, and utilize digital tools for precise diagnosis.

## ✨ Key Features

### 🖥️ Clinical Cockpit

- **High-Fidelity Rendering**: 500Hz sampling rate rendering with medical-grade grid systems (5mm/1mm).
- **12-Lead Visualization**: Standard layout for comprehensive heart rhythm assessment.
- **Focus Mode**: Interactive zoom and pan for detailed waveform inspection.

### 🛠️ Doctor's Toolkit

- **Digital Calipers**: Precision measurement tool for analyzing wave intervals ($\Delta t$) and amplitudes ($\Delta V$).
- **PDF Reporting**: One-click generation of A4 clinical reports containing patient info, ECG traces, and AI findings.
- **Dark Mode**: Optimized specific contrast modes for clinical environments (Dark/Light).

### 🧠 AI Integration

- **Real-Time Inference**: Powered by the **ECG-RAMBA (Bi-Mamba)** backend for sub-second classification.
- **Explainable AI (XAI)**: Grad-CAM attention maps overlay capabilities to show _where_ the model is looking.
- **Confidence Scoring**: Transparent probability distributions for 4 major diagnostic classes.

### ⚙️ Workflow

- **Patient Queue**: Drag-and-drop support for standard ECG formats (`.mat`, `.csv`, `.json`).
- **History Tracking**: Local storage of recent patient analyses for quick review.

---

## 🏗️ Architecture

The application follows a modern decoupled architecture:

```mermaid
graph LR
    User[Physician] -->|HTTPS| Frontend[React + Vite]
    Frontend -->|REST API| Backend[FastAPI Server]
    Backend -->|Inference| Model[ECG-RAMBA (Mamba2)]
    Backend -->|Storage| Cache[Local / SQLite]
```

### Technology Stack

| Component    | Technology       | Role                         |
| :----------- | :--------------- | :--------------------------- |
| **Frontend** | **React 18**     | UI Library                   |
|              | **Vite**         | Build Tool & Dev Server      |
|              | **Tailwind CSS** | Utility-First Styling        |
|              | **Recharts**     | High-performance D3 charting |
|              | **Lucide React** | Consistent Iconography       |
| **Backend**  | **FastAPI**      | High-performance Python API  |
|              | **Uvicorn**      | ASGI Server                  |
|              | **PyTorch**      | Deep Learning Inference      |
|              | **Mamba-SSM**    | State Space Model Backbone   |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js**: v18+
- **Python**: v3.10+ (with CUDA recommended)

### 1. Backend Setup (API)

```bash
cd web_app/backend

# Create virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Server
uvicorn main:app --reload --port 8000
```

> The API documentation will be available at: http://localhost:8000/docs

### 2. Frontend Setup (UI)

```bash
cd web_app/frontend

# Install node modules
npm install

# Start Dev Server
npm run dev
```

> The Dashboard will be accessible at: http://localhost:5173

---

## 📂 Project Structure

```text
web_app/
├── backend/                # FastAPI Server
│   ├── app/
│   │   ├── api/            # API Route Handlers
│   │   ├── core/           # Business Logic & Model Loader
│   │   └── models/         # Pydantic Schemas
│   ├── scripts/            # Utility Scripts (Seed, Check Health)
│   └── tests/              # Backend Unit Tests
│
└── frontend/               # React Client
    ├── src/
    │   ├── components/     # Reusable UI Components (ECGGraph, Calipers)
    │   ├── pages/          # Main Views (Dashboard, History)
    │   └── services/       # API Integration
    └── dist/               # Production Build Output
```

## 🤝 Contribution

This web application is part of the **ECG-RAMBA** research project.

- **Lead Developer**: Hai Duong Nguyen
- **License**: MIT
