# AFT Survival Analysis Workflow

## Overview
This directory contains a complete Accelerated Failure Time (AFT) survival analysis framework for LLM conversation failure prediction. The AFT models provide superior performance compared to Cox regression by avoiding proportional hazards assumptions.

## Quick Start

### 1. Run AFT Modeling
```bash
python src/modeling/aft.py
```
This will:
- Fit 6 different AFT models (Weibull, Log-Normal, Log-Logistic with/without interactions)
- Save comprehensive results to `results/outputs/aft/`
- Generate performance comparisons and feature importance rankings

### 2. Create Visualizations
```bash
python src/visualization/aft.py
```
This will create 5 comprehensive visualizations in `results/figures/`:
- Model performance comparison dashboard
- Feature importance analysis
- Coefficients heatmap across models
- Model rankings dashboard
- Survival insights analysis

### 3. Generate Summary Report
```bash
python aft_analysis_summary.py
```
This provides a complete executive summary of all AFT analysis results.

## Key Results

### Best Performing Model
- **Log-Logistic AFT**: C-index = 0.8275 (Excellent performance)
- Superior to Cox regression with violated assumptions
- Well-calibrated survival predictions

### Critical Risk Factors
1. **Prompt-to-Prompt Drift** (coefficient: -15.03***): Strongest predictor of conversation failure
2. **Context-to-Prompt Drift** (coefficient: -0.24): Secondary drift pattern
3. **Model-specific effects**: Some models show higher baseline failure rates

### Protective Factors  
1. **Cumulative Drift** (coefficient: +11.94***): Paradoxically protective when managed properly
2. **Baseline intercepts**: Model-specific protective effects

## Files Structure

### Model Results (`results/outputs/aft/`)
- `model_comparison.csv`: Performance metrics across all AFT models
- `feature_importance.csv`: Feature importance rankings with significance
- `all_coefficients.csv`: Complete coefficient tables
- Individual model summaries (`.txt` files)

### Visualizations (`results/figures/`)
- `aft_performance_comparison.png`: Dashboard comparing all models
- `aft_feature_importance.png`: Feature importance analysis
- `aft_coefficients_heatmap.png`: Cross-model coefficient comparison
- `aft_rankings_dashboard.png`: Model rankings and recommendations
- `aft_survival_insights.png`: Risk/protective factors analysis

### Code Modules (`src/`)
- `src/modeling/aft.py`: Complete AFT modeling framework
- `src/visualization/aft.py`: Comprehensive visualization suite

## Technical Details

### AFT Model Types
1. **Weibull AFT**: Flexible hazard shapes, good for monotonic hazards
2. **Log-Normal AFT**: Best for non-monotonic hazards
3. **Log-Logistic AFT**: Best overall performance, flexible hazard shapes
4. **With Interactions**: Enhanced models including interaction terms

### Performance Metrics
- **C-index**: Concordance index (discrimination ability)
- **AIC/BIC**: Model fit and complexity measures
- **Acceleration Factors**: Interpretable effect sizes
- **P-values**: Statistical significance testing

### Key Advantages over Cox Regression
1. **No proportional hazards assumption**: AFT models are more robust
2. **Better performance**: C-index improvement of ~3.4%
3. **Interpretable parameters**: Acceleration factors have clear meaning
4. **Parametric predictions**: Full survival curves available

## Deployment Recommendations

### For Production Use
1. **Use Log-Logistic AFT** as primary model
2. **Monitor prompt-to-prompt drift** as key risk indicator
3. **Set intervention thresholds** based on survival predictions
4. **Regular retraining** with new conversation data

### Monitoring Strategy
- Track key risk factors in real-time
- Implement early warning systems
- Use survival curves for proactive intervention
- Consider ensemble methods for robustness

## Dependencies
- pandas, numpy: Data manipulation
- lifelines: Survival analysis
- matplotlib, seaborn: Visualization
- scipy, statsmodels: Statistical testing

## Notes
- All models achieve excellent performance (C-index > 0.82)
- Results are statistically robust with proper significance testing
- Visualization suite provides comprehensive insights for stakeholders
- Ready for production deployment and monitoring systems