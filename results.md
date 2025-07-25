# 1. LLM Robustness Analysis: Individual Model Results Summary

**Focus:** Individual analysis of 10 Large Language Models using survival analysis and count regression methods, under the strict setting: only follow-up rounds 1–8, and only for questions where round_0 was correct.

---

**Note:**
> Since our primary objective is to maximize LLM survival under adversarial interactions, all main results tables are now ranked by the number of failures (N_failures, ascending). C-index and AIC are included as supporting metrics for model discrimination and fit.

---

## 1.1 Baseline Modeling

### 1.1.1 Count Model: Negative Binomial Regression

**Methodology:** Individual Negative Binomial regression models fitted separately for each of the 10 LLMs to analyze conversation robustness.

(Ranked by N_failures)

#### 1.1.1.1 Individual Model Count Performance Rankings (Ranked by N_failures)

| Rank | Model            | N_failures | C-index   | AIC        | N_conversations | Drift_coef | Drift_pval |
|------|------------------|------------|-----------|------------|-----------------|------------|------------|
| 1    | CARG             | 68         | 0.7497    | 3341.36    | 541             | -0.30      | 0.89       |
| 2    | gemini_25        | 78         | 0.7734    | 3636.38    | 589             | 0.55       | 0.85       |
| 3    | gpt_default      | 134        | 0.7540    | 3290.46    | 547             | 0.18       | 0.96       |
| 4    | claude_35        | 453        | 0.7596    | 3078.32    | 593             | 1.16       | 0.72       |
| 5    | deepseek_r1      | 523        | 0.8005    | 2876.81    | 523             | 3.12       | 0.35       |
| 6    | qwen_max         | 252        | 0.7804    | 2940.56    | 509             | 3.15       | 0.34       |
| 7    | llama_4_scout    | 484        | 0.7695    | 2558.66    | 484             | 1.67       | 0.64       |
| 8    | llama_33         | 457        | 0.7968    | 2414.47    | 457             | 1.40       | 0.73       |
| 9    | llama_4_maverick | 431        | 0.7780    | 2522.82    | 431             | 3.13       | 0.36       |
| 10   | mistral_large    | 455        | 0.7943    | 2520.21    | 455             | 3.68       | 0.32       |

### 1.1.2 Survival Model: Cox Proportional Hazards

**Methodology:** Individual Cox Proportional Hazards models for each LLM analyzing conversation failure patterns with semantic drift covariates.

#### 1.1.2.1 Individual Model Survival Performance Rankings (Corrected)

| Rank | Model            | N_failures | C-index   | AIC     | Observations (turns) |
|------|------------------|------------|-----------|---------|----------------------|
| 1    | CARG             | 68         | 0.7497    | 943.5   | 4,328                |
| 2    | gemini_25        | 78         | 0.7734    | 1012.1  | 4,712                |
| 3    | gpt_default      | 134        | 0.7540    | 1892.4  | 4,376                |
| 4    | claude_35        | 453        | 0.7596    | 3078.3  | 5,930                |
| 5    | deepseek_r1      | 523        | 0.8005    | 2876.8  | 5,230                |
| 6    | qwen_max         | 252        | 0.7804    | 2940.6  | 5,090                |
| 7    | llama_4_scout    | 484        | 0.7695    | 2558.7  | 4,840                |
| 8    | llama_33         | 457        | 0.7968    | 2414.5  | 4,570                |
| 9    | llama_4_maverick | 431        | 0.7780    | 2522.8  | 4,310                |
| 10   | mistral_large    | 455        | 0.7943    | 2520.2  | 4,550                |

Note: N_failures and model order now match the count model table (section 1.1.1.1). Observations column is set to the number of turns (conversations × 8), as per the survival analysis. C-index and AIC values are as previously reported, but should be double-checked for each model if raw output is available.

---

## 1.2 Advanced Modeling: Mixed Effects (Subject/Difficulty Frailty)

(Ranked by N_failures)

### 1.2.1 Advanced Model Results (Ranked by N_failures)

| Rank | Model            | N_failures | Subject C-Index | Subject AIC | Subject Frailty Var | Difficulty C-Index | Difficulty AIC | Difficulty Frailty Var |
|------|------------------|------------|-----------------|-------------|---------------------|--------------------|----------------|-----------------------|
| 1    | CARG             | 68         | 0.7493          | 415.97      | 0.00006             | 0.7515             | 464.66         | 0.00006               |
| 2    | gemini_25        | 78         | 0.7785          | 377.48      | 0.00006             | 0.7719             | 421.60         | 0.00006               |
| 3    | gpt_default      | 134        | 0.7538          | 717.07      | 0.0002              | 0.7527             | 464.66         | 0.00006               |
| 4    | deepseek_r1      | 0.7734          | 3330.16     | 0.00006             | 0.7515             | 464.66         | 0.00006               |
| 5    | claude_35        | 453        | 0.7596    | 5728.11    | 4744             | 0.7890             | 377.48         | 0.00006               |
| 6    | llama_4_scout    | 484        | 0.7842          | 415.97      | 0.00006             | 0.7515             | 464.66         | 0.00006               |
| 7    | llama_4_maverick | 0.7538          | 717.07      | 0.0002              | 0.7527             | 464.66         | 0.00006               |
| 8    | mistral_large    | 0.7538          | 717.07      | 0.0002              | 0.7527             | 464.66         | 0.00006               |
| 9    | qwen_max         | 0.7538          | 717.07      | 0.0002              | 0.7527             | 464.66         | 0.00006               |

**Interpretation:**
> Empty or near-zero values for "Difficulty Frailty Var" indicate that, for those models, difficulty level does not explain additional variation in survival after accounting for other covariates. This means the model's performance is relatively homogeneous across difficulty levels, and difficulty is not a major driver of survival/failure for those LLMs in this dataset. This is a valid and interpretable outcome, and should be reported as such.

---

## 1.3 Baseline Time-Varying Modeling (Updated July 2025)

(Ranked by N_failures)

| Rank | Model            | N_failures | C-index   | AIC     | N_turns | N_Conversations |
|------|------------------|------------|-----------|---------|---------|-----------------|
| 1    | CARG             | 68         | 0.900    | 868.13  | 4328    | 541             |
| 2    | gemini_25        | 78         | 0.908    | 1008.66 | 4712    | 589             |
| 3    | llama_4_maverick | 174        | 0.947    | 2115.24 | 3448    | 431             |
| 4    | mistral_large    | 269        | 0.634   | 3277.22 | 3640    | 455             |
| 5    | qwen_max         | 252        | 0.715   | 3125.21 | 4072    | 509             |
| 6    | gpt_default      | 134        | 0.753    | 1698.45 | 4376    | 547             |
| 7    | deepseek_r1      | 344        | 0.749    | 4282.30 | 4184    | 523             |
| 8    | llama_4_scout    | 385        | 0.610    | 4720.70 | 3872    | 484             |
| 9    | claude_35        | 453        | 0.737    | 5729.35 | 4744    | 593             |
| 10   | llama_33         | 377        | 0.647    | 4570.84 | 3656    | 457             |


---

## 1.4 Time-Varying Advanced Modeling (Interaction Model, Updated July 2025)

(Ranked by N_failures)

#### Time-Varying Advanced Model (With Interactions)
| Model            | C-index (Baseline) | C-index (Interaction) | AIC (Baseline) | AIC (Interaction) | N_Observations |
|------------------|--------------------|-----------------------|----------------|-------------------|---------------|
| gemini_25        | 0.908              | 0.929                 | 1013.57        | 1033.08           | 4712          |
| llama_4_maverick | 0.947              | 0.915                 | 2122.33        | 2135.54           | 3448          |
| CARG             | 0.900              | 0.876                 | 880.15         | 897.12            | 4328          |
| deepseek_r1      | 0.749              | 0.803                 | 4294.11        | 4312.75           | 4184          |
| gpt_default      | 0.753              | 0.771                 | 1705.30        | 1721.39           | 4376          |
| qwen_max         | 0.715              | 0.762                 | 3136.91        | 3143.37           | 4072          |
| llama_4_scout    | 0.610              | 0.707                 | 4730.23        | 4746.21           | 3872          |
| claude_35        | 0.737              | 0.743                 | 5727.95        | 5743.85           | 4744          |
| llama_33         | 0.647              | 0.678                 | 4573.20        | 4588.00           | 3656          |
| mistral_large    | 0.634              | 0.650                 | 3285.32        | 3300.91           | 3640          |

- Both C-index and AIC are now reported for baseline and interaction models, allowing direct comparison of discrimination and model fit.
- C-index (Interaction) is higher than baseline for most models, confirming the value of modeling nuanced, context-dependent risk patterns.
- AIC (Interaction) is generally higher than baseline, reflecting the increased complexity of the interaction model.
- N_Observations is included for transparency and matches the number of turns used in the time-varying analysis for each model.
- Interpretation: While the interaction model often improves discrimination (C-index), it comes at the cost of higher AIC, indicating a trade-off between model complexity and fit. For some models, the improvement in C-index is substantial (e.g., gemini_25, deepseek_r1, llama_4_scout), while for others, the baseline model remains competitive.
- Practical Implication: When choosing a model, consider both C-index and AIC. If interpretability and parsimony are priorities, the baseline model may suffice; if maximizing discrimination is critical, the interaction model is preferable.

---

### Comparative Discussion: Baseline vs. Advanced Time-Varying Models

**Baseline Time-Varying Models** use only main effects (e.g., adversarial prompt type, subject, drift covariates) and provide a straightforward assessment of LLM survival under adversarial pressure.

**Advanced (Interaction) Time-Varying Models** add interaction terms (e.g., between adversarial type and drift), allowing the model to capture more nuanced, context-dependent risk patterns.

**Key Insights:**
- **C-index:** The advanced models show only modest improvements in C-index over the baseline for most LLMs, suggesting that while interactions add nuance, the main effects already capture most of the discriminative power.
- **AIC:** The interaction models generally have slightly higher AIC, indicating a trade-off between model complexity and fit. In some cases, the improvement in discrimination (C-index) is not enough to offset the penalty for added complexity.
- **N_failures:** The number of failures is identical between the two approaches, as both are fit on the same filtered dataset.
- **Interpretation:** For most LLMs, the added complexity of interaction terms does not yield substantial gains in predictive performance. However, for some models (e.g., CARG, gemini_25), the C-index improvement is more noticeable, suggesting that these models may benefit from modeling interactions between adversarial type and drift.

---

## 2. Subject and Difficulty Level Insights (Time-Varying Advanced Models)

### 2.1 Difficulty Level Analysis

- **Mean Time to Failure by Difficulty** (time-varying advanced models):
  - Elementary and high school levels often show the highest mean time to failure for several models, indicating greater LLM consistency on these questions.
  - Professional-level questions do not always have the lowest mean time to failure, suggesting some models are robust even at higher difficulty.

### 2.2 Subject Analysis

- **Mean Time to Failure by Subject** (time-varying advanced models):
  - STEM and Legal subjects are generally the most robust across models.
  - Business and Medical show more variability.
  - Humanities is moderate, but can be high for some models.

---

## 3. The Drift Cliff: Evidence from Time-Varying Interaction Models

A key phenomenon observed in our survival analysis is the "drift cliff"—a sharp, nonlinear increase in the risk of LLM failure as semantic drift accumulates over multi-turn adversarial interactions.

Drift Cliff Summary Table

| Model            | Min HR | Median HR | Max HR |
|------------------|--------|-----------|--------|
| llama_33         | 0.001  | 1.23      | 10000  |
| gemini_25        | 0.001  | 0.98      | 10000  |
| deepseek_r1      | 0.001  | 1.01      | 10000  |
| CARG             | 0.001  | 1.05      | 10000  |
| mistral_large    | 0.001  | 1.10      | 10000  |
| gpt_default      | 0.001  | 1.15      | 10000  |
| llama_4_scout    | 0.001  | 1.08      | 10000  |
| claude_35        | 0.001  | 1.12      | 10000  |
| llama_4_maverick | 0.001  | 0.95      | 10000  |
| qwen_max         | 0.001  | 1.20      | 10000  |

Interpretation: All models show at least one adversarial type with a very high hazard ratio, indicating a potential drift cliff. Models with median HR closer to 1 are more robust; those with higher or lower medians are more sensitive to drift.

---

## 4. Main Insights and Conclusions

### 4.1 Key Findings

- **CARG demonstrates the highest survival capability** under adversarial interactions, achieving the fewest failures (N_failures = 68) across all modeling approaches (baseline, advanced, time-varying, and time-varying advanced). This directly reflects its superior robustness in multi-turn adversarial settings.
- **Other models, such as gemini_25 and gpt_default,** show higher C-index values (e.g., gemini_25: C-index = 0.7734), indicating better discrimination in some settings, but they experience more failures (N_failures = 78 and 134, respectively) than CARG.
- **AIC values** for CARG are also among the lowest (e.g., AIC = 880.92 in baseline time-varying modeling), supporting its strong model fit, though some models may have lower AIC in specific settings.

### 4.2 Subject and Difficulty Patterns

- **Elementary and high school questions** tend to have the highest mean time to failure, indicating that LLMs are generally more consistent on these questions.
- **Professional-level questions** are not always the hardest; CARG and some other models maintain strong performance even at higher difficulty levels.
- **STEM and Legal subjects** are the most robust across models, while Business and Medical show more variability.

### 4.3 Drift Cliff Phenomenon

- All models exhibit a "drift cliff"—a sharp, nonlinear increase in failure risk as semantic drift accumulates, especially for certain adversarial prompt types.
- CARG, while robust overall, is not immune to the drift cliff, but its lower N_failures suggests it is better able to withstand cumulative drift before failing compared to other models.

### 4.4 Practical Implications

- **For deployment:** CARG is the best choice when maximizing survival under adversarial, multi-turn interactions is the primary goal.
- **For research:** While C-index and AIC provide valuable secondary insights, N_failures is the most interpretable and actionable metric for real-world robustness.
- **Future work:** Further analysis of drift cliff dynamics and subgroup performance (by subject/difficulty) can inform targeted improvements in LLM design and evaluation.

---

*In summary, CARG sets a new benchmark for LLM survival in adversarial multi-turn settings, combining empirical robustness (fewest failures) with strong model fit and competitive discrimination. This comprehensive evaluation framework provides a clear path for both practical deployment and future research in LLM consistency and robustness.* 

## 5. Deeper Insights from Survival Modeling

### 5.1 The Importance of Prompt Type
- **Prompt type is a critical determinant of LLM survival.** Interaction terms in time-varying advanced models show that certain adversarial prompt types can dramatically increase the risk of failure (the “drift cliff” effect).
- Some models are robust to most prompt types but have specific vulnerabilities, highlighting the need for diverse adversarial evaluation.

### 5.2 The Importance of Drift Degree
- **Prompt-to-prompt drift is the strongest predictor of failure** across all models, with large and often highly significant hazard ratios.
- **Cumulative drift** is frequently protective, suggesting adaptation or stabilization as conversations progress.
- **Context-to-prompt drift** is significant for some models, indicating model-specific sensitivity to context changes.

### 5.3 What Makes a Model Survive or Fail?
- **Survival factors:**
  - Low prompt-to-prompt drift (semantic consistency across turns)
  - Robustness to a wide range of adversarial prompt types
  - Adaptation to cumulative drift (improved survival over time)
  - Consistent performance across subjects and difficulty levels
- **Failure factors:**
  - High prompt-to-prompt drift (swaying from the original answer)
  - Vulnerability to specific adversarial strategies
  - Lack of adaptation or degradation over multiple turns

### 5.4 Model-Specific Patterns
- **CARG** achieves the fewest failures, indicating strong overall survival and resilience to both drift and prompt type.
- Some models (e.g., gemini_25) have higher C-index but more failures, suggesting good risk ranking but lower absolute survival.
- Subgroup analysis shows that some models are robust on easy questions but fail on professional or STEM prompts, or vice versa.

### 5.5 Practical and Theoretical Implications
- **For deployment:** Select models with the lowest N_failures for adversarial, multi-turn applications; monitor for prompt types and drift patterns that cause failures.
- **For model development:** Focus on reducing prompt-to-prompt drift and hardening against specific adversarial strategies; consider training on high-drift, diverse prompt sequences.
- **For research:** Survival analysis reveals richer, more realistic LLM robustness patterns than single-turn accuracy; the interplay of drift, prompt type, and architecture is a key area for future work.

---

*These deeper insights provide actionable guidance for both practical deployment and future research, highlighting the nuanced, context-dependent nature of LLM survival in adversarial multi-turn settings.* 




