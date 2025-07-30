# 🔬 LLM Robustness Analysis: Individual Model Survival Analysis

## 📋 Project Overview hello

This project provides a **comprehensive individual model analysis** of Large Language Model (LLM) robustness using advanced survival analysis methods. The analysis evaluates **each LLM independently** with baseline Cox models, stratified/frailty analyses, and time-varying models to understand model-specific conversation breakdown patterns and coefficient profiles.

> **CARG is the proposed method in this study.** All models are evaluated on the same filtered set of conversations (only those with round_0 == 1, and only rounds 1–8 are analyzed) to ensure a fair, rigorous comparison of survival capability under adversarial interactions.

## 🎯 Key Features

- **🤖 Individual Model Focus:** Each LLM analyzed independently with dedicated modeling
- **📊 Individual Coefficients:** Model-specific hazard ratios, p-values, and interpretations
- **🧪 Stratified Analysis:** Subject and difficulty stratification per individual model
- **📈 Individual Comparisons:** Model-specific performance metrics and frailty effects
- **⏰ Temporal Analysis:** Turn-by-turn drift evolution and vulnerability windows
- **🔄 Advanced Time-Varying Models:** Three interaction types (P2P, Cumulative, Combined) for comprehensive drift analysis
- **📁 Clean Pipeline:** Individual model analysis from data to model-specific results
- **🎨 Individual Visualizations:** Per-model plots and comparative individual performance
- **🛡️ Survival Ranking by N_failures:** All results tables are ranked by the number of failures (N_failures, ascending), directly reflecting LLM robustness. C-index and AIC are included as supporting metrics.

---

## 🗂️ Project Structure

```
llm_survival_working/
├── modeling/
│   ├── advanced_modeling.py                # Individual model stratified/frailty analysis
│   ├── baseline_modeling.py                # Legacy baseline models (optional)
│   ├── time_varying_advanced_modeling.py    # Unified time-varying models (all types)
│   ├── time_varying_frailty_modeling.py    # Time-varying frailty models (per model)
│
├── utils/
│   ├── data_process.py                     # Data preprocessing & feature engineering
│   ├── extract_individual_model_coefficients.py # Individual model Cox analysis
│   ├── analyze_drift_by_turns.py           # Temporal drift analysis by turns
│   ├── further_analysis.py                 # Subject/difficulty clustering analysis
│   └── ...                                 # (other utility scripts)
│
├── environment.yml                         # Python environment specification
├── README.md                               # This guide
│
├── processed_data/                         # Processed datasets by model
│   └── <model>/                            # Model-specific processed files
│       ├── <model>_cleaned.csv             # Cleaned conversation data
│       ├── <model>_static.csv              # Static features per conversation
│       └── <model>_long.csv                # Long-format survival data
│
├── raw data/                               # Raw conversation & experiment files
│   └── <model>/                            # Raw data per model
└──
```

---

## ⚙️ Environment Setup

### 📦 Dependencies Installation

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate llm_survival

# Or install manually:
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
pip install lifelines sentence-transformers
```

### 🧪 Required Packages
- **Core Analysis:** `pandas`, `numpy`, `scikit-learn`
- **Survival Analysis:** `lifelines` 
- **Visualizations:** `matplotlib`, `seaborn`
- **NLP Features:** `sentence-transformers`
- **Utilities:** `tqdm`, `warnings`

---

## 🚀 Complete Analysis Pipeline

> **Note:** All scripts should be run from the project root directory for correct data loading. All models are evaluated on the same filtered set of conversations (round_0 == 1, rounds 1–8).

### **Phase 1: Data Processing**

```bash
python utils/data_process.py
```
- Processes raw conversation files from all models
- Extracts semantic drift features
- Generates cleaned, static, and long-format datasets per model

### **Phase 2: Count Models Analysis**

```bash
python modeling/baseline_modeling.py  # (optional legacy)
# or
python utils/run_enhanced_analysis.py # (if present)
```
- Fits Negative Binomial regression for each model
- Generates model performance rankings and reports

### **Phase 3: Advanced Subject & Difficulty Analysis**

```bash
python utils/further_analysis.py
```
- Analyzes performance by academic domain and difficulty
- Conducts ANOVA/statistical tests
- Generates key findings dashboards and survival visualizations

### **Phase 4: Individual Model Cox Analysis**

```bash
python utils/extract_individual_model_coefficients.py
```
- Fits Cox models for each LLM
- Outputs coefficient matrices, hazard ratios, p-values

### **Phase 5: Individual Model Stratified/Frailty Analysis**

```bash
python modeling/advanced_modeling.py
```
- Subject and difficulty stratification for each model
- Calculates frailty variance and stratified performance

### **Phase 6: Individual Cox Coefficient Extraction**

```bash
python extract_cox_coefficients.py
```
- Extracts coefficients, hazard ratios, p-values for each model
- Outputs comparison matrices

### **Phase 7: Temporal Drift & Time-Varying Analysis**

```bash
python utils/analyze_drift_by_turns.py
python modeling/time_varying_frailty_modeling.py
```
- Analyzes turn-by-turn drift and time-varying frailty effects for each model
- Outputs temporal statistics, rankings, and visualizations

### **Phase 8: Advanced Time-Varying Modeling (Optional)**

```bash
# Choose interaction type: p2p, cumulative, or combined
python modeling/time_varying_advanced_modeling.py --type p2p
python modeling/time_varying_advanced_modeling.py --type cumulative  
python modeling/time_varying_advanced_modeling.py --type combined
```
- **P2P**: Prompt-to-prompt drift interactions with adversarial/base contexts
- **Cumulative**: Cumulative drift interactions with context-specific effects
- **Combined**: All drift measures with comprehensive interaction modeling
- Generates model-specific time-varying coefficients and performance rankings

---

## 📋 Workflow Summary (One-Command Option)

```bash
python utils/data_process.py && \
python utils/run_enhanced_analysis.py && \
python utils/further_analysis.py && \
python utils/extract_individual_model_coefficients.py && \
python modeling/advanced_modeling.py && \
python extract_cox_coefficients.py && \
python utils/analyze_drift_by_turns.py && \
python modeling/time_varying_frailty_modeling.py && \
python modeling/time_varying_advanced_modeling.py --type combined
```

---

## 📊 Key Results & Outputs

- All outputs are saved in `generated/outputs/` (CSVs, JSONs, Markdown) and `generated/figs/` (visualizations).
- See `results.md` for a summary of findings and statistical interpretations.
- **All results tables are ranked by N_failures (ascending, fewer failures = better survival). C-index and AIC are included for context.**
- **CARG achieves the fewest failures and sets a new benchmark for LLM survival under adversarial interactions.**
- See `individual_model_coefficient_report.md` for detailed coefficient tables.

---

## ❓ Individual vs. Combined Modeling

- **This repository focuses on individual model analysis:**
  - Each LLM is analyzed independently for robustness, drift, and frailty effects.
  - All advanced and time-varying frailty modeling is performed per model.
- **Combined modeling (all LLMs together) is not performed by default.**
  - If you want to analyze all models together, you must modify the scripts to pool data and include a model indicator.

---

## 📚 Understanding the Results

- **N_failures (number of failures) is the primary metric for ranking LLM robustness.**
- **CARG is the most robust model by this metric.**
- **Prompt-to-prompt drift** is the dominant risk factor for conversation failure in all models.
- **Cumulative drift** is universally protective (adaptation effect).
- **Context drift** is model-dependent, sometimes highly significant.
- **Stratification and frailty** reveal unobserved heterogeneity by subject and difficulty.
- **Temporal analysis** shows adaptation and vulnerability windows across conversation turns.
- **Drift cliff analysis** reveals sharp, nonlinear increases in failure risk as semantic drift accumulates, especially for certain adversarial prompt types.
- **Advanced time-varying models** provide three different interaction frameworks for understanding drift effects across conversation turns.
- **Subject and difficulty analyses** show which content areas are most/least robust for each LLM.

---


