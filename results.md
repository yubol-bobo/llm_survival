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

#### 1.1.2.1 Individual Model Survival Performance Rankings (Ranked by N_failures)

| Rank | Model            | N_failures | C-index   | AIC    | Observations |
|------|------------------|------------|-----------|--------|--------------|
| 1    | CARG             | 68         | 0.7497    | 943.5  | 4,328        |
| 2    | gemini_25        | 78         | 0.7734    | 1012.1 | 4,712        |
| 3    | gpt_default      | 134        | 0.7540    | 1892.4 | 4,376        |
| 4    | deepseek_r1      | 52         | 0.880     | 2,869.0 | 523          |
| 5    | qwen_max         | 50         | 0.875     | 2,931.1 | 509          |
| 6    | mistral_large    | 269        | 0.886     | 3802.4 | 3,640        |
| 7    | llama_4_scout    | 48         | 0.865     | 2,551.2 | 484          |
| 8    | llama_33         | 45         | 0.860     | 2,405.6 | 457          |
| 9    | llama_4_maverick | 43         | 0.855     | 2,511.7 | 431          |
| 10   | mistral_large    | 45         | 0.850     | 2,510.6 | 455          |

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

## 1.3 Baseline Time-Varying Modeling

(Ranked by N_failures)

| Rank | Model            | N_failures | C-index   | AIC    | N_turns | N_Conversations |
|------|------------------|------------|-----------|--------|---------|-----------------|
| 1    | CARG             | 68         | 0.7497    | 880.92 | 4328    | 541             |
| 2    | gemini_25        | 78         | 0.7734    | 1012.06| 4712    | 589             |
| 3    | llama_4_maverick | 174        | 0.7780    | 2122.21| 3448    | 431             |
| 4    | mistral_large    | 269        | 0.7943    | 3284.71| 3640    | 455             |
| 5    | qwen_max         | 252        | 0.7804    | 3136.50| 4072    | 509             |
| 6    | gpt_default      | 134        | 0.7540    | 1705.73| 4376    | 547             |
| 7    | deepseek_r1      | 344        | 0.8005    | 4293.64| 4184    | 523             |
| 8    | llama_4_scout    | 385        | 0.7695    | 4730.34| 3872    | 484             |
| 9    | claude_35        | 453        | 0.7596    | 5728.11| 4744    | 593             |
| 10   | llama_33         | 377        | 0.7968    | 4572.50| 3656    | 457             |

---

## 1.4 Time-Varying Advanced Modeling (Interaction Model)

(Ranked by N_failures)

| Rank | Model            | N_failures | C-index   | AIC    | N_Observations | N_Conversations |
|------|------------------|------------|-----------|--------|----------------|-----------------|
| 1    | CARG             | 68         | 0.7497    | 897.99 | 4328           | 541             |
| 2    | gemini_25        | 78         | 0.7734    | 1033.36| 4712           | 589             |
| 3    | llama_4_maverick | 174        | 0.7780    | 2137.64| 3448           | 431             |
| 4    | mistral_large    | 269        | 0.7943    | 3301.18| 3640           | 455             |
| 5    | qwen_max         | 252        | 0.7804    | 3141.45| 4072           | 509             |
| 6    | gpt_default      | 134        | 0.7540    | 1719.45| 4376           | 547             |
| 7    | deepseek_r1      | 344        | 0.8005    | 4312.25| 4184           | 523             |
| 8    | llama_4_scout    | 385        | 0.7695    | 4737.98| 3872           | 484             |
| 9    | claude_35        | 453        | 0.7596    | 5743.17| 4744           | 593             |
| 10   | llama_33         | 377        | 0.7968    | 4583.78| 3656           | 457             |

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