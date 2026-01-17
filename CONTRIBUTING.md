# Contributing to ECG-RAMBA

We welcome contributions! Please read this guide before submitting.

---

## Getting Started

### Prerequisites

| Requirement | Version                                     |
| :---------- | :------------------------------------------ |
| Python      | 3.10+                                       |
| CUDA        | 11.8+                                       |
| PyTorch     | 2.0+                                        |
| GPU VRAM    | 10GB+ (20GB+ recommended for full training) |

### Setup

```bash
# Clone the repository
git clone https://github.com/BrianNguyen29/ECG-RAMBA.git
cd ECG-RAMBA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to all functions and classes
- Keep functions focused and under 50 lines when possible

### Documentation

- Update `README.md` if adding new features
- Add comments for non-obvious logic
- Document any new configuration parameters in `configs/config.py`

### Testing

Before submitting:

```bash
# Verify imports work
python -c "from src.model import ECGRambaV7Advanced; print('OK')"

# Run sanity check
python -c "from src.model import run_sanity_check; run_sanity_check()"
```

---

## Web App Development

If you are contributing to the Clinical Dashboard (`web_app/`):

### Backend (FastAPI)

- Located in `web_app/backend/`
- Run tests: `pytest web_app/backend/tests/`
- Add new endpoints in `app/api/endpoints/`

### Frontend (React)

- Located in `web_app/frontend/`
- Use Tailwind CSS for styling
- Components must be responsive and support Dark Mode

---

## Contribution Types

### 🐛 Bug Fixes

1. Open an issue describing the bug
2. Fork the repository
3. Create a branch: `git checkout -b fix/issue-description`
4. Make your fix
5. Submit a pull request

### ✨ New Features

1. Open an issue to discuss the feature first
2. Fork and create a branch: `git checkout -b feature/feature-name`
3. Implement with tests
4. Update documentation
5. Submit a pull request

### 📖 Documentation

- Improvements to README, docstrings, or comments are always welcome
- Create a branch: `git checkout -b docs/description`

---

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation as needed
3. Test your changes locally
4. Write a clear PR description explaining:
   - What changes were made
   - Why they were made
   - Any breaking changes

---

## Hardware Notes

### Training Requirements

| Configuration | GPU VRAM | Batch Size | Training Time (5 folds) |
| :------------ | :------- | :--------- | :---------------------- |
| Full          | 24 GB    | 192        | ~4 hours                |
| Reduced       | 12 GB    | 64         | ~8 hours                |
| Minimal       | 8 GB     | 32         | ~12 hours               |

### Inference Requirements

- Inference is significantly less demanding
- CPU-only inference is possible (but slow)
- Minimum 4GB VRAM for GPU inference

---

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person

---

## Questions?

Open an issue with the `question` label or contact the maintainers.

Thank you for contributing! 🎉
