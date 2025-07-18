import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation import count_model_metrics, survival_model_metrics
from visualization import plot_km_by_group, plot_icr, plot_partial_effects

# --- Comparison Functions ---
def compare_count_models(baseline_model, advanced_model, static_df, y_col='time_to_failure'):
    """
    Compare baseline and advanced count models (Poisson/NB): metrics and coefficients.
    """
    y = static_df[y_col]
    X = static_df
    base_metrics = count_model_metrics(baseline_model, X, y)
    adv_metrics = count_model_metrics(advanced_model, X, y)
    print('Baseline Count Model Metrics:', base_metrics)
    print('Advanced Count Model Metrics:', adv_metrics)
    # Compare coefficients
    print('\nBaseline Coefficients:')
    print(baseline_model.params)
    print('\nAdvanced Coefficients:')
    if hasattr(advanced_model, 'fixef'):  # pymer4
        print(advanced_model.fixef)
    else:
        print(advanced_model.params)
    # ICR plots
    plot_icr(baseline_model, static_df)
    if hasattr(advanced_model, 'predict'):
        static_df['adv_predicted'] = advanced_model.predict(static_df)
        plt.figure(figsize=(8, 6))
        plt.scatter(static_df['time_to_failure'], static_df['adv_predicted'], alpha=0.5, label='Advanced')
        plt.scatter(static_df['time_to_failure'], baseline_model.predict(static_df), alpha=0.5, label='Baseline')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual (Advanced vs Baseline)')
        plt.legend()
        plt.tight_layout()
        plt.show()


def compare_survival_models(baseline_cph, advanced_cph, long_df):
    """
    Compare baseline and advanced CoxPH models: metrics and visualizations.
    """
    base_metrics = survival_model_metrics(baseline_cph, long_df)
    adv_metrics = survival_model_metrics(advanced_cph, long_df)
    print('Baseline CoxPH Metrics:', base_metrics)
    print('Advanced CoxPH Metrics:', adv_metrics)
    # Compare coefficients
    print('\nBaseline Coefficients:')
    print(baseline_cph.params_)
    print('\nAdvanced Coefficients:')
    print(advanced_cph.params_)
    # KM curves
    plot_km_by_group(long_df, group_col='model')
    # Partial effect plots
    plot_partial_effects(baseline_cph, long_df, covariate='prompt_to_prompt_drift')
    plot_partial_effects(advanced_cph, long_df, covariate='prompt_to_prompt_drift')

# --- Example usage ---
if __name__ == '__main__':
    from baseline_modeling import train_static, fit_count_model, long_all, fit_coxph_model
    from advanced_modeling import fit_coxph_frailty, fit_poisson_glmm, fit_negbin_glmm, pymer4_available
    # --- Count Model Comparison ---
    print('Fitting baseline Poisson...')
    baseline_poisson = fit_count_model(train_static, model_type='poisson')
    if pymer4_available:
        print('Fitting advanced Poisson GLMM...')
        adv_poisson, adv_poisson_results = fit_poisson_glmm(train_static)
        compare_count_models(baseline_poisson, adv_poisson, train_static)
    else:
        print('pymer4 not available: Skipping advanced Poisson GLMM.')
    # --- Negative Binomial ---
    print('Fitting baseline Negative Binomial...')
    baseline_nb = fit_count_model(train_static, model_type='nb')
    if pymer4_available:
        print('Fitting advanced NegBin GLMM...')
        adv_nb, adv_nb_results = fit_negbin_glmm(train_static)
        compare_count_models(baseline_nb, adv_nb, train_static)
    else:
        print('pymer4 not available: Skipping advanced NegBin GLMM.')
    # --- Survival Model ---
    print('Fitting baseline CoxPH...')
    baseline_cph = fit_coxph_model(long_all)
    print('Fitting advanced CoxPH with frailty...')
    adv_cph = fit_coxph_frailty(long_all)
    compare_survival_models(baseline_cph, adv_cph, long_all) 