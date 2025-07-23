# Experiment Pipeline Design and Methodology

This document outlines the experimental pipeline for evaluating the consistency and robustness of Large Language Models (LLMs) in multi-turn adversarial interactions. Each step is designed to complement the others, providing a comprehensive and interpretable analysis of LLM behavior under challenging conditions. This structure is suitable for adaptation into the methods section of a scientific paper.

---

## 1. Data Sources and Preprocessing

- **Raw Data:**
  - Multi-turn conversation logs for 10 LLMs, each tested on a set of 700 questions.
  - Data includes user prompts, LLM responses, and metadata (e.g., prompt type, round number, correctness labels).
- **Preprocessing:**
  - Scripts in `utils/data_process.py` standardize, clean, and merge raw data from various sources.
  - Data is converted to both static and long formats for downstream modeling.

---

## 2. Filtering and Inclusion Criteria

- **Strict Consistency Setting:**
  - Only conversations where the LLM answered correctly at round_0 (`round_0 == 1`) are included.
  - Only follow-up rounds 1 to 8 are analyzed, representing 8 turns of adversarial interaction.
  - For each follow-up round, survival is defined as maintaining the original correct answer; swaying from the correct answer is marked as failure (0).
  - If an LLM maintains correctness for all 8 rounds, the conversation is censored (survived all rounds).
- **Purpose:**
  - This filtering ensures that only initially robust LLM responses are evaluated for consistency under continued adversarial pressure.

---

## 3. Baseline vs. Advanced Modeling: Key Differences

### 3.1 Baseline Modeling
- **Approach:**
  - Fits standard statistical models (Negative Binomial regression and Cox Proportional Hazards) to each LLM independently.
  - Treats all data points as independent, without accounting for hierarchical or repeated-measures structure.
- **Purpose:**
  - Provides interpretable, model-level summaries of robustness and survival, useful for high-level comparison across LLMs.
- **Limitation:**
  - Does not account for within-question or within-subject correlation, nor for unobserved heterogeneity.

### 3.2 Advanced Modeling (Mixed Effects)
- **Approach:**
  - Incorporates mixed-effects (random effects) into the regression and survival models.
  - Models random intercepts (and/or slopes) for clusters such as question, subject, or prompt type.
- **Purpose:**
  - Accounts for the hierarchical structure of the data (e.g., repeated measures for each question or subject).
  - Captures unobserved heterogeneity and improves the accuracy and generalizability of effect estimates.
- **Benefit:**
  - More realistic modeling of LLM performance, controlling for latent factors and dependencies in the data.

---

## 4. Motivation for Time-Varying Modeling

- **Why Not Just Baseline/Advanced?**
  - Baseline and advanced models assume that covariate effects are constant across all turns and do not explicitly model how risk or robustness changes as the conversation progresses.
  - In multi-turn adversarial settings, the risk of LLM failure may increase or decrease over time, and the impact of context or drift may accumulate.
- **Time-Varying Models:**
  - Allow covariate effects (e.g., drift, adversarial prompt type) to change at each turn.
  - Model the dynamic evolution of risk, providing a more nuanced understanding of when and why LLMs fail.
- **Insight:**
  - Essential for capturing temporal patterns and adaptation (or degradation) in LLM consistency over multiple rounds.

---

## 5. Rationale for Time-Varying Advanced Modeling

- **Why Add Further Complexity?**
  - Even time-varying models may miss important interactions or hierarchical effects (e.g., how drift interacts with specific prompt types, or how question-level effects modulate risk over time).
- **Time-Varying Advanced Models:**
  - Combine time-varying covariates with mixed-effects and interaction terms.
  - Model both the dynamic and hierarchical structure of the data, as well as higher-order interactions.
- **Purpose:**
  - To uncover subtle, context-dependent vulnerabilities and strengths in LLMs that are not apparent from simpler models.
  - To provide the most comprehensive, realistic assessment of LLM robustness in adversarial, multi-turn scenarios.

---

## 6. Visualization and Interpretation

- **Professional Plots:**
  - Bar plots, scatter plots, and coefficient visualizations summarize key metrics (C-index, drift, AIC, etc.) for each LLM.
  - Color coding and significance annotations highlight robust vs. vulnerable models and effects.
- **Integration:**
  - Visualizations and statistical summaries are integrated in `results.md` for clear communication of findings.

---

## 6. Complementarity and Insights

- **Pipeline Synergy:**
  - Data filtering ensures only meaningful, comparable cases are analyzed.
  - Baseline models provide interpretable, model-level summaries.
  - Advanced and time-varying models reveal dynamic, context-dependent vulnerabilities.
  - Visualizations synthesize complex results for actionable insights.
- **Overall Goal:**
  - To robustly characterize LLM consistency and drift in adversarial, multi-turn settings, informing both model development and deployment strategies. 