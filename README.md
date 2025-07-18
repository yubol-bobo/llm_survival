# Survival Analysis of LLM Robustness to Adversarial Prompting

## Overview
This project investigates the robustness of large language models (LLMs) to adversarial prompting using survival analysis and count regression. We model the number of adversarial turns until failure and the time-to-event (failure) using both baseline and advanced (frailty/random effects) models. The pipeline is designed for reproducibility and extensibility, suitable for top AI conference submissions.

## Project Structure
```
Survival Analysis/
├── data_process.py           # Data processing and feature engineering
├── baseline_modeling.py      # Baseline models: Poisson, NegBin, CoxPH
├── advanced_modeling.py      # Advanced models: GLMMs, CoxPH with frailty
├── evaluation.py             # Metrics, diagnostics, statistical tests
├── visualization.py          # All key plots and interpretation
├── compare.py                # Systematic comparison of baseline vs advanced models
├── environment.yml           # Conda environment (Python dependencies)
├── processed_data/           # Processed static/long tables for each model
├── raw data/                 # Raw conversation and experiment files
└── README.md                 # This file
```

## Environment Setup
### Python Dependencies
```bash
conda env create -f environment.yml
conda activate survival_analysis
```

The project uses:
- Python 3.11
- Core packages: numpy, pandas, scikit-learn, matplotlib, tqdm
- Specialized packages: lifelines (survival analysis), sentence-transformers

## Data Processing
Run the following to process raw data and generate static/long tables:
```bash
python data_process.py
```
- Output: `processed_data/<model>_static.csv` and `processed_data/<model>_long.csv`

## Baseline Modeling
Fit baseline models (Poisson, Negative Binomial, CoxPH):
```bash
python baseline_modeling.py
```
- Outputs model summaries and saves splits for further analysis.

## Advanced Modeling
Fit advanced models (GLMMs, CoxPH with frailty):
```bash
python advanced_modeling.py
```
- Uses lifelines for Cox proportional hazards models with frailty terms.

## Evaluation & Visualization
Evaluate model fit, predictive accuracy, and diagnostics:
```bash
python evaluation.py
```
Generate all key plots:
```bash
python visualization.py
```

## Model Comparison
Compare baseline and advanced models (metrics, coefficients, plots):
```bash
python compare.py
```

## Reproducibility Checklist
- All code and dependencies are specified in `environment.yml` and this README.
- Random seeds are set for splits and modeling where possible.
- All scripts are modular and can be run independently or as a pipeline.
- Pure Python implementation with no external software dependencies.

## Notes & Recommendations
- For large datasets, consider batching or parallel processing in `data_process.py`.
- For publication, use the plots and tables generated in `visualization.py` and `evaluation.py`.
- For ablation or robustness studies, modify covariates in the modeling scripts.

## Contact
For questions or collaboration, please contact yubo li at yubol@andrew.cmu.edu. 