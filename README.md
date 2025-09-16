# 🧬 LLM Survival Analysis

Survival analysis framework for evaluating Large Language Model robustness in multi-turn conversations. Analyzes 12 LLMs across 40,000+ interactions to discover "drift cliffs" and vulnerability patterns.

## 🚀 How to Run

```bash
# Setup environment
conda env create -f environment.yml
conda activate survival_analysis

# Run complete analysis
python run_analysis.py

# Results saved to:
# - results/figures/baseline/    (13 baseline plots)
# - results/figures/advanced/    (14 advanced plots)  
# - results/outputs/             (statistical results)
```

## 📊 What It Does

- **Baseline Analysis**: Cox survival models + 13 visualizations 
- **Advanced Analysis**: Drift×model interactions + 14 visualizations
- **Output**: 28+ publication-ready plots + statistical results