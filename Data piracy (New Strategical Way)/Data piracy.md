# Suicide Risk Prediction Project - Discussion Summary

**Project Title:** Suicide Risk Prediction Models with Clinical Explainability  
**Domain:** Machine Learning Operations (MLOps)  
**Sub-Domain:** Explainable & Deployable AI for Mental Health Risk Prediction  
**Team:** A17

## Project Overview
- Fine-tuned BERT/MentalBERT for suicide risk prediction from text
- Strong focus on Explainable AI (XAI) and model reverse engineering
- Privacy-preserving design to prevent data piracy
- Complete MLOps pipeline with on-device deployment

## Architecture & Workflow

### 1. Data Ingestion & Preprocessing
- Raw anonymized data (social media text / EEG signals)
- Multi-layer protection: Anonymization + Encryption + Differential Privacy
- Tokenization, cleaning, and synthetic data augmentation
- Secure version control using DVC + encrypted storage

### 2. Model Training
- Fine-tune MentalBERT (or BERT-base)
- Optional hybrid: MentalBERT + 1D-CNN
- Privacy-preserving training (DP-SGD via Opacus)
- Hyperparameter optimization with Optuna
- Experiment tracking with MLflow

### 3. Reverse Engineering & Explainability
- SHAP for local and global explanations
- Attention visualization (BertViz)
- Counterfactual analysis
- Extraction of linguistic patterns, emotional cues, and contextual features
- Key output: Clinically interpretable insights

### 4. Model Optimization
- Quantization (8-bit/4-bit)
- Pruning
- Conversion to ONNX and TensorFlow Lite

### 5. MLOps Pipeline
- CI/CD with GitHub Actions
- Model versioning and registry (MLflow)
- Automated testing (including privacy tests)
- Continuous monitoring and drift detection
- Docker containerization

### 6. Deployment
- Primary: On-device inference (TFLite / ONNX Runtime)
- Secondary: Secure API (FastAPI + authentication)
- Cross-platform support (Windows, Linux)

## Data Privacy & Anti-Piracy Measures
- Differential Privacy during training
- On-device inference (raw data never leaves device)
- Encryption at rest and in transit
- Anonymization + pseudonymization
- Secure MLOps with access controls and audit logs
- Model serving via authenticated APIs (no raw weights distribution)
- Model watermarking for ownership protection
- Federated learning option for future enhancements

## Tools Used
- **Programming:** Python
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **XAI:** SHAP, BertViz, Captum
- **Privacy:** Opacus
- **MLOps:** MLflow, Optuna, Docker, DVC, GitHub Actions
- **Deployment:** ONNX Runtime, TensorFlow Lite, FastAPI

## Key Benefits
- Accurate predictions with clinical explainability
- Strong protection against data piracy
- Production-ready MLOps practices
- Efficient on-device deployment
- Reproducible and scalable system

## Research Limitations
- Limited dataset diversity
- Post-hoc nature of XAI techniques
- Need for clinical validation
- Language dependency (primarily English)
- Residual risk of false positives/negatives

---

**Last Updated:** July 16, 2026  
**Status:** Ready for Review 1
