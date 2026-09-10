# Kaggle Artifacts Staging Directory

This directory is a local staging area for downloaded outputs and execution archives from Kaggle notebook runs.

## Intended Workflow
1. Run the notebook on Kaggle in `KAGGLE` mode.
2. Download the generated output archive (`*.zip` or `*.tar.gz`) from Kaggle.
3. Place the downloaded archive into this directory (`kaggle_artifacts/`).
4. Extract the archive contents locally for analysis and reporting.
5. Do NOT commit generated empirical artifacts or data archives to Git (this directory's contents are Git-ignored except this README and `.gitkeep`).
