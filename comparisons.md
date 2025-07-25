# Comparison of Modeling Approaches for LLM Consistency in Multi-Turn Settings

This document provides a structured comparison of four modeling strategies used to evaluate the survival and consistency of Large Language Models (LLMs) in adversarial, multi-turn interactions. Each approach is assessed for its methodological strengths, limitations, and suitability for robust survival analysis.

---

## 1. Baseline Model

**Approach:**
- Fits standard statistical models (Negative Binomial regression, Cox Proportional Hazards) to each LLM independently.
- Assumes all data points are independent; does not account for repeated measures or hierarchical structure.

**Formula:**
- **Negative Binomial Regression:**
  ```
  time_to_failure ~ avg_prompt_to_prompt_drift + avg_context_to_prompt_drift + avg_prompt_complexity + C(subject_cluster) + C(difficulty_level)
  ```
  where `time_to_failure` is the count of survived turns for conversation, and covariates include drift measures, prompt complexity, and categorical subject/difficulty levels.
- **Cox Proportional Hazards:**
  ```
  prompt_to_prompt_drift + context_to_prompt_drift + cumulative_drift + prompt_complexity
  ```
  where the model includes all available drift measures and prompt complexity as covariates.

**Strengths:**
- Simple, interpretable, and computationally efficient.
- Provides high-level summaries and enables straightforward model-to-model comparisons.

**Limitations:**
- Ignores within-question or within-subject correlation.
- Cannot account for unobserved heterogeneity or latent factors.
- May underestimate uncertainty and overstate significance.

---

## 2. Advanced Model (Mixed Effects)

**Approach:**
- Extends baseline models by incorporating random effects (mixed effects) for clusters such as question, subject, or prompt type.
- Models the hierarchical and repeated-measures structure of the data.

**Formula:**
- **Individual Baseline Cox Model:**
  ```
  prompt_to_prompt_drift + context_to_prompt_drift + cumulative_drift + prompt_complexity
  ```
- **Subject Stratified Cox Model:**
  ```
  prompt_to_prompt_drift + context_to_prompt_drift + cumulative_drift + prompt_complexity
  ```
  with stratification by `subject_cluster` (strata=['subject_encoded'])
- **Difficulty Stratified Cox Model:**
  ```
  prompt_to_prompt_drift + context_to_prompt_drift + cumulative_drift + prompt_complexity
  ```
  with stratification by `difficulty` (strata=['difficulty_encoded'])

**Strengths:**
- Accounts for unobserved heterogeneity and latent group-level effects.
- Provides more accurate and generalizable effect estimates.
- Controls for dependencies in the data, reducing bias.

**Limitations:**
- More complex to specify and fit, especially with many random effects.
- Interpretation of random effects can be less intuitive.
- Still assumes covariate effects are constant over time.

---

## 3. Time-Varying Model

**Approach:**
- Uses Cox models with time-varying covariates to allow effects (e.g., drift, prompt type) to change at each turn.
- Models the dynamic evolution of risk as the conversation progresses.

**Formula:**
- **Time-Varying Cox Model (Frailty):**
  ```
  C(adv_id) + C(base_id) + C(turn_bin)
  ```
  where the model tries different combinations of adversarial prompt types, base prompt types, and turn bins as time-varying covariates.

**Strengths:**
- Captures temporal patterns and adaptation (or degradation) in LLM consistency.
- Provides nuanced understanding of when and why LLMs fail in multi-turn settings.
- Useful for identifying critical rounds or turning points.

**Limitations:**
- Does not account for hierarchical structure or unobserved heterogeneity.
- May miss latent group-level effects or repeated-measures dependencies.
- More complex than baseline, but less comprehensive than advanced time-varying models.

---

## 4. Time-Varying Advanced Model (Mixed Effects + Interactions)

**Approach:**
- Combines time-varying covariates with mixed effects and interaction terms.
- Models both the dynamic and hierarchical structure of the data, as well as higher-order interactions.

**Formula:**
- **Baseline Time-Varying Model:**
  ```
  C(adv_id) + C(base_id) + prompt_to_prompt_drift + context_to_prompt_drift + cumulative_drift
  ```
- **Interaction Time-Varying Model:**
  ```
  C(adv_id) * prompt_to_prompt_drift + C(base_id) * context_to_prompt_drift + cumulative_drift
  ```
- **Cumulative Interaction Model:**
  ```
  C(adv_id) * cumulative_drift + C(base_id) + prompt_to_prompt_drift + context_to_prompt_drift
  ```

**Strengths:**
- Most comprehensive and realistic modeling of LLM performance in adversarial, multi-turn scenarios.
- Captures subtle, context-dependent vulnerabilities and strengths.
- Accounts for both temporal dynamics and latent group-level effects.
- Provides the most accurate and generalizable estimates of survival and risk.

**Limitations:**
- Highest complexity; requires careful specification and more computational resources.
- Interpretation can be challenging, especially for higher-order interactions.
- May require larger sample sizes for stable estimation.

---

## Comparative Discussion and Recommendation

- **Baseline models** are useful for quick, interpretable summaries but risk oversimplifying the data and missing important dependencies.
- **Advanced models** improve accuracy by accounting for hierarchical structure, but still assume static effects over time.
- **Time-varying models** reveal how risk and covariate effects evolve, but may miss latent group-level effects.
- **Time-varying advanced models** offer the most complete picture, capturing both dynamic and hierarchical complexities, and are best suited for robust, real-world evaluation of LLM consistency in multi-turn adversarial settings.

**Recommendation:**
- For rigorous survival analysis of LLM consistency in multi-turn interactions, the **time-varying advanced model** is the preferred approach. It balances interpretability with realism, accounts for both temporal and hierarchical effects, and provides the most actionable insights for model development and deployment.

---

**Summary Table:**

| Model Type | Formula | Key Features | Complexity |
|------------|---------|--------------|------------|
| Baseline | `time_to_failure ~ drift_measures + C(subject) + C(difficulty)` | Simple, independent observations | Low |
| Advanced | `drift_measures` with stratification by subject/difficulty | Hierarchical structure, frailty effects | Medium |
| Time-Varying | `C(adv_id) + C(base_id) + C(turn_bin)` | Dynamic covariates, temporal evolution | Medium |
| Time-Varying Advanced | `C(adv_id) * drift + C(base_id) * drift + cumulative_drift` | Interactions, mixed effects, time-varying | High |

---

**Note:** All formulas are implemented using the lifelines library for survival analysis and statsmodels for count models. The specific variable names and interactions may vary slightly between implementations based on data availability and convergence requirements. 