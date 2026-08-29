# AI-Powered Pharmaceutical Complaint Management System

> **Note**: This repository contains both the original Jupyter notebook (`pharma-complaint-management.ipynb`) and a refactored Python package (`pharma_complaint_system/`).

## Quick Start

For the refactored, production-ready code:

```bash
pip install scikit-learn pandas numpy matplotlib gradio
python -c "from pharma_complaint_system import *; print('Package loaded!')"
```

See [pharma_complaint_system/README.md](pharma_complaint_system/README.md) for full documentation.

---

## Original Project

This project demonstrates an AI-powered customer complaint management system for pharmaceutical manufacturing. It uses machine learning to classify complaints by category and severity, combined with rule-based analysis for regulatory compliance.

### Features

- **Synthetic Data Generation**: Creates realistic pharmaceutical complaint data
- **ML Classification**: TF-IDF + Logistic Regression for category and severity prediction
- **Rule-Based Analysis**: Product detection, batch extraction, sentiment analysis, priority scoring
- **Interactive Dashboard**: Gradio web interface for complaint submission and tracking
- **CAPA Suggestions**: Automated Corrective and Preventive Action recommendations

### Files

- `pharma-complaint-management.ipynb` - Original Jupyter notebook with all functionality
- `pharma_complaint_system/` - Refactored modular Python package

### Requirements

```bash
pip install scikit-learn pandas numpy matplotlib gradio
```
