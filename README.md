# Adaptive Low-Latency Fraud Detection in Streaming Financial Systems with LLM-Augmented Explainability

## Project Overview
This repository contains the infrastructure, experimental framework, and codebase for an M.Tech research thesis focusing on fraud detection in streaming transactional environments. The research addresses the challenges of detecting fraudulent transactions in real-time under non-stationary conditions (concept drift) and providing human-interpretable, natural-language explanations for model decisions.

## Research Objective
The primary objective of this thesis is to develop, evaluate, and benchmark adaptive learning algorithms and retraining policies for fraud detection in credit card and online payment transaction streams. Specifically, this work aims to:
- Identify and adapt to concept drift (e.g., evolving fraud patterns and legitimate consumer behaviour shifts) in real-time streaming contexts.
- Benchmark adaptive retraining mechanisms against static models and traditional batch retraining strategies.
- Maintain low latency processing required for streaming financial systems.
- Augment model transparency by generating SHAP-based feature importance attributions translated into context-aware natural language explanations using Large Language Models (LLMs).

## Current Project Status
- **Project Status:** 🚧 In Progress
- **Note:** This repository is being developed incrementally as part of the thesis research. Features, experiments, and documentation will be added progressively as each research phase is completed.

## Repository Structure
```text
project_root/
├── .editorconfig         # Editor configuration for coding style
├── .env.example          # Environment variables template
├── .gitignore            # Git exclusion rules
├── LICENSE               # MIT License
├── pyproject.toml        # Tooling configurations (black, isort, pytest)
├── requirements.txt      # Infrastructure & development dependencies
├── configs/              # Model and pipeline configuration files (JSON/YAML)
├── datasets/             # Raw transactional datasets (git-ignored)
├── docs/                 # Research, architecture, and meeting documentation
├── experiments/          # Saved experimental configurations and results
├── logs/                 # Operational and training log outputs
├── models/               # Serialized model checkpoints and artifacts
├── notebooks/            # Jupyter Notebooks for exploratory analysis and prototyping
├── outputs/              # Subdirectories for visual, tabular, and metric outputs
│   ├── checkpoints/      # Model weight checkpoints
│   ├── figures/          # Generated plots and figures
│   ├── metrics/          # Performance evaluation logs (JSON/YAML)
│   └── tables/           # Generated tables (CSV/LaTeX)
├── reports/              # Drafts, notes, and figures for academic publications
├── scripts/              # Command-line scripts for batch executions
├── src/                  # Main Python source codebase
│   ├── adaptation/       # Retraining policies, incremental learning pipelines
│   ├── data/             # Batch and stream data ingestion/processing pipelines
│   ├── drift/            # Concept drift and covariate shift detectors
│   ├── evaluation/       # Evaluation metrics, performance trackers
│   ├── explainability/   # Model explanation modules (SHAP, LLM translation)
│   ├── models/           # Offline/online fraud detection model architectures
│   ├── streaming/        # Streaming transaction replay engines
│   ├── utils/            # Shared utilities (logging, configuration loaders)
│   └── visualization/    # Shared plotting and dashboard utilities
└── tests/                # Unit and integration test suites
```

## Technology Stack (Planned)
- **Core Language:** Python 3.10+
- **Data & Streaming:** Pandas, NumPy, Scikit-Learn, PySpark / River (incremental learning utilities)
- **Machine Learning & Deep Learning:** XGBoost, LightGBM, PyTorch (for neural streaming baselines)
- **Explainability:** SHAP (SHUp-ley Additive exPlanations)
- **LLM Integration:** Hugging Face Transformers (optional, for local/open-source LLM-based explanation generation)
- **Testing & Quality Assurance:** Pytest, Black, Isort, Flake8

## Dataset Overview
The project will explore adaptive fraud detection across three benchmark transactional datasets:
1. **IEEE-CIS Fraud Detection Dataset** (`ieee-fraud-detection`)
2. **PaySim Mobile Money Simulator Dataset** (`paysim`)
3. **ULB Credit Card Fraud Detection Dataset** (`ulb`)
