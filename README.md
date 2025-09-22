# 🧬 LLM Survival Analysis

Survival analysis framework for evaluating Large Language Model robustness in multi-turn conversations. Analyzes 9 LLMs across 36,000+ interactions using 4 complementary modeling approaches (Cox PH, Interactions, AFT, RSF) to discover "drift cliffs" and vulnerability patterns.

## 🔬 Modeling Pipeline

**4-Stage Analysis Framework:**
1. **Baseline (Cox PH)**: Traditional survival analysis with conversation-level frailty
2. **Advanced (Interactions)**: Drift×model interaction effects (32 terms)
3. **AFT (Parametric)**: Accelerated failure time models (Log-Normal, Log-Logistic, Weibull)
4. **RSF (Ensemble)**: 🆕 Random Survival Forest with hyperparameter tuning

**Feature Engineering:**
- **53 Total Features**: 4 drift + 8 model + 6 subject + 3 difficulty + 32 interactions
- **Semantic Drift Metrics**: Prompt-to-prompt, context-to-prompt, cumulative, complexity
- **Model Comparisons**: 9 LLMs (claude_35, deepseek_r1, gpt_4o, etc.)
- **Domain Effects**: 7 subject clusters, 4 difficulty levels

## 🚀 Quick Start

```bash
# Setup environment
conda env create -f environment.yml
conda activate survival_analysis

# Run complete analysis
python run_analysis.py

# Results saved to:
# - results/figures/baseline/    (13 baseline plots)
# - results/figures/advanced/    (14 advanced plots)
# - results/figures/aft/         (8 AFT analysis plots)
# - results/outputs/baseline/    (Cox PH results)
# - results/outputs/advanced/    (Interaction results)
# - results/outputs/aft/         (AFT model results)
# - results/outputs/rsf/         (RSF ensemble results)
```

## 📚 Step-by-Step Tutorial

### Step 1: Environment Setup
```bash
# Create conda environment with all dependencies
conda env create -f environment.yml

# Activate the environment (ALWAYS use this for all commands)
conda activate survival_analysis

# Verify installation
python -c "import pandas, numpy, matplotlib; print('✅ Environment ready!')"
```

### Step 2: Data Preprocessing (Optional)
If you want to regenerate the processed data from raw conversations:

```bash
# Preprocess all models (generates drift metrics for all 8 rounds)
python src/data/preprocessing.py

# This creates:
# - data/processed/{model}/{model}_long.csv    (round-level data)
# - data/processed/{model}/{model}_static.csv  (conversation-level data)
# - data/processed/{model}/{model}.csv         (merged raw data)
# - data/processed/{model}/{model}.json        (conversation data)
```

**⚠️ Note**: Preprocessing takes ~2-3 hours for all 12 models and requires sentence-transformers embeddings.

### Step 3: Run Analysis Pipeline

#### Option A: Complete Analysis (Recommended)
```bash
# Runs all 4 modeling stages: baseline + advanced + AFT + RSF
python run_analysis.py

# Generates 28+ visualizations across 4 analysis types:
# - Baseline: Cox proportional hazards models (13 plots)
# - Advanced: Interaction effects and diagnostics (14 plots)
# - AFT: Accelerated Failure Time models (8 plots)
# - RSF: Random Survival Forest with feature importance
```

#### Option B: Run Individual Stages
```bash
# Baseline analysis only (Cox models + 13 plots automatically)
python run_analysis.py --stage baseline

# Advanced analysis only (interactions + 14 plots automatically)
python run_analysis.py --stage advanced

# AFT analysis only (survival acceleration + 8 plots automatically)
python run_analysis.py --stage aft

# RSF analysis only (ensemble method + hyperparameter tuning)
python run_analysis.py --stage rsf

# Visualization only (regenerate plots from existing results)
python run_analysis.py --stage visualization
```

### Step 4: View Results

#### Statistical Results
```bash
# View results directory
ls results/outputs/

# Key files:
# - baseline/complete_results.csv      (Cox model performance)
# - advanced/interaction_results.csv   (Interaction effects)
# - aft/model_performance.csv          (AFT model comparison)
# - aft/feature_importance.csv         (Risk factor rankings)
# - rsf/model_performance.csv          (RSF ensemble results)
# - rsf/feature_importance.csv         (ML-based importance rankings)
# - rsf/hyperparameter_results.csv     (RSF tuning results)
```

#### Visualizations
```bash
# View all generated plots
ls results/figures/

# Baseline plots (13 files):
ls results/figures/baseline/
# - hazard_ratios.png                  (Main risk factors)
# - model_comparison.png               (Model performance)
# - survival_curves.png                (Survival probabilities)
# - drift_dynamics.png                 (Risk evolution)

# Advanced plots (14 files):
ls results/figures/advanced/
# - interaction_effects.png            (Drift×model interactions)
# - residual_analysis.png              (Model diagnostics)
# - risk_stratification.png            (Risk group analysis)

# AFT plots (8 files):
ls results/figures/aft/
# - driver_dynamics_evolution.png      (Risk factors across 8 rounds)
# - aft_performance_comparison.png     (Model acceleration factors)
# - aft_feature_importance.png         (Survival insights)
```

### Step 5: Interpret Key Results

#### 🔍 Main Findings
1. **Critical Risk Factor**: Prompt-to-prompt drift (β = -15.16, p < 0.001)
   - Causes massive failure acceleration (AF ≈ 0.0000003)
   - Most dangerous in rounds 2-4

2. **Protective Factor**: Cumulative drift (β = +11.99, p < 0.001)  
   - Dramatically delays failure (AF ≈ 162,000)
   - Beneficial when managed properly

3. **Model Rankings**: Log-Logistic AFT achieves best performance (C-index = 0.8301)

4. **RSF Ensemble Results**: 🆕 Non-parametric validation of parametric findings
   - Feature importance without distributional assumptions
   - Model comparison: RSF vs Cox vs AFT
   - Hyperparameter optimization results

#### 📊 Key Plots to Examine
- `driver_dynamics_evolution.png` - Shows how risk factors evolve across 8 conversation rounds
- `aft_feature_importance.png` - Identifies most critical survival factors
- `hazard_ratios.png` - Shows relative risk impacts

#### 📋 RSF Results Files 🆕
- `rsf/model_performance.csv` - RSF ensemble metrics (C-index, OOB score, hyperparameters)
- `rsf/feature_importance.csv` - ML-based feature rankings (all 53 features)
- `rsf/hyperparameter_results.csv` - Grid search optimization results
- `rsf/model_comparison.csv` - RSF vs Cox vs AFT performance comparison

### Step 6: Troubleshooting

#### Common Issues
```bash
# Issue: "conda activate not found"
# Solution: Use conda run instead
conda run -n survival_analysis python run_analysis.py

# Issue: "NumPy compatibility error"  
# Solution: Ensure you're using the correct environment
conda activate survival_analysis
python run_analysis.py

# Issue: "No such file or directory: data/processed/"
# Solution: Run preprocessing first
python src/data/preprocessing.py
```

#### Performance Tips
```bash
# For faster testing, process subset of models:
# Edit src/data/preprocessing.py line 35:
# for model in ['claude_35', 'gpt_5']:  # instead of all models

# Run only specific stages for faster testing:
python run_analysis.py --stage baseline  # Just baseline + visualizations
python run_analysis.py --stage rsf       # Just RSF modeling only
```

## 📊 What Each Analysis Does

### Baseline Analysis (Stage 1)
- **Cox Proportional Hazards Models**: Traditional survival analysis with conversation-level frailty
- **Formula**: h_i(t|X_i(t), ν_i) = ν_i h_0(t) exp{β^T X_i(t)}
- **Features**: 4 drift + 8 model + 6 subject + 3 difficulty covariates
- **13 Visualizations**: Hazard ratios, survival curves, model comparison
- **Key Insights**: Main risk factors and baseline model performance

### Advanced Analysis (Stage 2)
- **Interaction Models**: Drift×model interaction effects
- **Formula**: Same baseline + γ^T I_i(t) where I_i(t) are 32 interaction terms
- **Features**: All baseline features + 32 drift×model interactions (44 total)
- **14 Visualizations**: Residual analysis, interaction plots, diagnostics
- **Key Insights**: Model-specific vulnerabilities and assumption testing

### AFT Analysis (Stage 3)
- **Accelerated Failure Time Models**: Time acceleration/deceleration analysis
- **Models**: Log-Normal, Log-Logistic, Weibull AFT (with/without interactions)
- **Features**: Same as advanced model (44 features)
- **8 Visualizations**: Driver dynamics, feature importance, model comparison
- **Key Insights**: How factors speed up or slow down conversation failure

### RSF Analysis (Stage 4) 🆕
- **Random Survival Forest**: Non-parametric ensemble method
- **Hyperparameter Tuning**: Optimized grid search with progress tracking
- **Features**: All 53 features (4 drift + 8 model + 6 subject + 3 difficulty + 32 interactions)
- **Model-Free**: No proportional hazards assumptions required
- **Key Insights**: Feature importance rankings and model comparison with parametric approaches