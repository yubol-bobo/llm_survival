# Survival Analysis of LLM Robustness to Adversarial Prompting

## Experiment Design

### Data and Units
- **Unit of analysis:** Multi-turn adversarial conversations between users and 11 LLMs (Claude-3.5, GPT-4, Gemini, LLaMA variants, etc.)
- **Dataset:** 7,700 conversations with 44,960 total turns across diverse prompt types
- **Outcome variables:** 
  - Count: Number of adversarial turns until first incorrect answer
  - Survival: Round index at first failure (right-censored if no failure)

### Predictors and Covariates
- **Core predictors:** Model type, initial prompt identifier, model×prompt interaction
- **Semantic drift measures:** 
  - Prompt-to-prompt drift (cosine distance between successive prompt embeddings)
  - Context-to-prompt drift (distance between accumulated context and new prompt)
  - Cumulative drift (rolling sum of prompt-to-prompt distances)
- **Complexity:** Prompt token length

### Statistical Models
- **Baseline:** Poisson regression (count), Cox Proportional Hazards (survival)
- **Advanced:** Mixed effects with conversation-level random effects (frailty models)
- **Data split:** 70/15/15 train/validation/test, stratified by model type

---

## Results

### Model Performance
- **Concordance index:** 0.873 (excellent discriminative ability)
- **Count model RMSE:** 2.74 with moderate overdispersion (variance/mean = 1.56)
- **Valid predictions:** 44,960 turn-level observations across 5,620 conversations

### Key Findings
| Predictor | Coefficient | Hazard Ratio | p-value | Interpretation |
|-----------|-------------|--------------|---------|----------------|
| Prompt-to-prompt drift | +18.30 | 8.9×10⁷ | <0.001 | Massive failure risk increase |
| Context-to-prompt drift | +3.00 | 20.1 | <0.001 | Moderate risk increase |
| Cumulative drift | -15.38 | 2.1×10⁻⁷ | <0.001 | Strong protective effect |
| Prompt complexity | -0.001 | 0.999 | <0.05 | Slight protective effect |

### Model Robustness
- **Baseline vs. Advanced:** Identical coefficients indicate minimal conversation-level clustering
- **Proportional hazards:** Significant violations (p < 0.001) suggest time-varying effects
- **Cross-validation:** Consistent results across train/validation/test splits

---

## Analysis

### Primary Insights

1. **Semantic Drift as Failure Predictor:** Prompt-to-prompt drift emerges as the strongest predictor of LLM failure, with a hazard ratio of 89 million. This validates the hypothesis that semantic inconsistency in adversarial sequences destabilizes model performance.

2. **Cumulative Drift Paradox:** Surprisingly, cumulative drift shows a protective effect (HR = 2.1×10⁻⁷). This counterintuitive finding suggests either: (a) models adapt to sustained drift patterns, or (b) survival bias where conversations with high early drift fail quickly, leaving only robust conversations in the analysis.

3. **Complexity-Robustness Trade-off:** More complex prompts slightly reduce failure risk, indicating that sophisticated prompts may be less effective for adversarial attacks than simple, semantically inconsistent ones.

### Model Validity
- **Excellent discrimination** (C-index = 0.873) demonstrates strong predictive power
- **Proportional hazards violations** indicate that covariate effects change over conversation turns, suggesting dynamic vulnerability patterns
- **Minimal clustering effects** show that conversation-level random effects are small, validating the baseline model approach

### Implications
- **LLM Safety:** Semantic drift monitoring could serve as an early warning system for adversarial attacks
- **Robustness Evaluation:** Turn-level survival analysis provides more nuanced assessment than aggregate failure rates
- **Defense Strategies:** Focus on detecting prompt-to-prompt inconsistencies rather than complex individual prompts

### Limitations
- **Time-varying effects:** Proportional hazards violations suggest model coefficients change over conversation length
- **Observational data:** Causal inference limited without randomized prompt assignments
- **Model heterogeneity:** Different LLMs may have distinct vulnerability patterns not fully captured by fixed effects

---

**Conclusion:** Semantic drift, particularly prompt-to-prompt inconsistency, is a powerful predictor of LLM adversarial failure. The survival analysis framework provides superior granularity over traditional aggregate metrics, revealing dynamic vulnerability patterns that inform both safety evaluation and defense strategies. 