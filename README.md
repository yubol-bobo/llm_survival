# 🧬 LLM Survival Analysis

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

> **Tutorial: Analyzing LLM Robustness Through Survival Analysis**

A comprehensive framework for evaluating Large Language Model robustness in multi-turn conversations using survival analysis. This tutorial will guide you through analyzing 10 state-of-the-art LLMs across 40,000+ interactions to discover "drift cliffs" and strategic deployment insights.

## 🎯 What You'll Learn

- **Survival Analysis for LLMs**: Treat conversations as "lifespans" and model failure dynamics
- **Drift Cliff Discovery**: Find sharp vulnerability thresholds (up to 9 orders of magnitude risk increase)
- **Strategic Deployment**: Evidence-based recommendations for real-world LLM deployment
- **Publication-Ready Results**: Generate 20+ high-resolution plots and comprehensive analysis

## 📊 Key Discoveries Preview

| Model | Failures | Risk Level | Best Domain |
|-------|----------|------------|-------------|
| **CARG** | 68 | 🟢 Elite | STEM |
| **Gemini-2.5** | 78 | 🟢 Elite | Business |
| **GPT-4** | 134 | 🟡 Moderate | General |
| **Claude-3.5** | 453 | 🔴 Vulnerable | Humanities |

**Cliff Phenomenon**: GPT-4 shows 3.9M× baseline risk, Qwen-Max exhibits 1.1B× risk spikes!

---

## 🚀 Quick Start Tutorial

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/llm-survival-analysis/
cd llm_survival

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate survival_analysis

# Or use pip
pip install -r requirements.txt
```

### Step 2: Run Complete Analysis

```bash
# Run baseline survival models (5-10 minutes)
python scripts/run_baseline_analysis.py

# Generate all visualizations (3-5 minutes)
python scripts/generate_visualizations.py
```

**✅ That's it!** Check `results/figures/` for 20+ publication-ready plots and `results/outputs/` for detailed statistics.

### Step 3: Explore Results

```bash
# View key results
ls results/figures/          # High-resolution PDF plots
ls results/outputs/          # CSV analysis results
ls results/paper/           # LaTeX paper files
```

---

## 📈 Tutorial Walkthrough

### Understanding the Analysis Pipeline

```bash
# 1. Data Processing (automatic)
# - Loads 10 LLM datasets from data/processed/
# - Extracts semantic drift features using Sentence-BERT
# - Creates survival analysis format

# 2. Statistical Modeling (runs automatically)
# - Negative Binomial: Count total failures
# - Cox Proportional Hazards: Turn-by-turn dynamics  
# - Stratified Models: Domain/difficulty effects
# - Time-Varying Models: Complex interactions

# 3. Visualization Generation (runs automatically)
# - Performance rankings and trade-offs
# - Drift cliff phenomenon plots
# - Domain/difficulty heatmaps
# - Strategic archetype analysis
```

### Key Output Files

| File | Description | Use Case |
|------|-------------|----------|
| `model_performance_comparison_1.pdf` | Robustness ranking | Paper Figure 1 |
| `complete_cliff_cascade_dynamics.pdf` | Drift cliffs | Paper Figure 2 |
| `model_subject_clustering_heatmap.pdf` | Domain analysis | Deployment guide |
| `semantic_drift_effects.pdf` | Risk factors | Technical analysis |

---

## 🔧 Advanced Usage

### Custom Analysis

```python
# Generate specific visualizations
from src.visualization.core import create_model_performance_comparison
from src.visualization.cliffs import create_cliff_cascade_dynamics

create_model_performance_comparison()  # Performance plots
create_cliff_cascade_dynamics()       # Cliff phenomenon
```

### Individual Model Analysis

```bash
# Run time-varying models with interactions
python src/modeling/time_varying_advanced_modeling.py --type p2p
python src/modeling/time_varying_advanced_modeling.py --type cumulative  
python src/modeling/time_varying_advanced_modeling.py --type combined
```

### Custom Data

```python
# Add your own LLM data
# 1. Place conversation files in data/raw/your_model/
# 2. Follow the format: conversations_*.json + experiment_*.csv
# 3. Re-run the pipeline
```

---

## 📊 Understanding Results

### 1. Model Performance Hierarchy
- **Elite Performers** (68-134 failures): CARG, Gemini-2.5, GPT-4
- **Moderate Performers** (174-269 failures): Llama variants, Qwen-Max
- **Vulnerable Models** (344-453 failures): Claude-3.5, DeepSeek-R1

### 2. Drift Cliff Phenomenon
Models exhibit **extreme vulnerability thresholds**:
- **Stable Phase** (turns 1-3): Normal operation
- **Threshold Phase** (turns 3-5): Risk accumulation  
- **Cliff Phase** (turns 5-8): Catastrophic failure spikes

### 3. Strategic Deployment
- **Medical/Legal**: Use STEM-optimized models (CARG)
- **Business**: Leverage Gemini-2.5's domain strength
- **General Purpose**: GPT-4 offers balanced performance
- **Monitoring**: Track semantic drift for early intervention

---

## 🧪 Methodology Summary

### Survival Framework
- **Event**: Conversational breakdown/failure
- **Time**: Conversation turn (1-8)
- **Covariates**: Semantic drift measures, prompt types, domains

### Semantic Drift Measures
- **Prompt-to-Prompt (p2p)**: Distance between consecutive prompts
- **Context-to-Prompt (c2p)**: Distance from cumulative context
- **Cumulative (cum)**: Total semantic evolution

### Models & Metrics
- **Negative Binomial**: Total survival count
- **Cox Proportional Hazards**: Turn-by-turn risk
- **C-index**: Discrimination ability (higher = better)
- **Failures**: Primary robustness metric (lower = better)

---

## 📁 Repository Structure

```
llm_survival/
├── scripts/           # 🚀 Run these first
├── src/              # 🔧 Modular source code
├── data/             # 💾 Experimental datasets  
├── results/          # 📊 All outputs here
└── docs/             # 📚 Documentation
```

**Key Directories:**
- `scripts/`: Main execution scripts
- `results/figures/`: Publication-ready plots (PDF)
- `results/outputs/`: Statistical results (CSV)
- `results/paper/`: LaTeX paper files

---

## 🎯 Common Use Cases

### 1. Quick Analysis
```bash
# Get results in 10 minutes
python scripts/run_baseline_analysis.py
python scripts/generate_visualizations.py
```

### 2. Paper Figures
```bash
# Generate specific plots for publication
python -c "from src.visualization.core import *; create_model_performance_comparison()"
```

### 3. Custom Models
```bash
# Add your LLM data to data/raw/your_model/
# Re-run analysis
python scripts/run_baseline_analysis.py
```

### 4. Deployment Planning
```bash
# Check domain-specific heatmaps
open results/figures/model_subject_clustering_heatmap.pdf
open results/figures/model_difficulty_heatmap.pdf
```

---

## 🔍 Troubleshooting

**Environment Issues:**
```bash
# Reset environment
conda env remove -n survival_analysis
conda env create -f environment.yml
conda activate survival_analysis
```

**Missing Data:**
```bash
# Check data structure
ls -la data/processed/
# Should contain 10 model directories
```

**Import Errors:**
```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python scripts/generate_visualizations.py
```


---

## 🤝 Contributing & Contact

- **Contact**: Available upon acceptance

---

**🎉 Ready to discover LLM vulnerability patterns? Start with Step 1 above!**


