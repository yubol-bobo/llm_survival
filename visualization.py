import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
# Remove problematic import - we'll implement partial effects differently

# --- Kaplan–Meier Curves ---
def plot_km_by_group(long_df, group_col='model', time_col='round', event_col='failure', censored_col='censored'):
    """
    Plot Kaplan–Meier survival curves by group (e.g., model or high/low drift).
    """
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    for name, grouped in long_df.groupby(group_col):
        kmf.fit(grouped[time_col], event_observed=grouped[event_col], label=str(name))
        kmf.plot_survival_function(ci_show=False)
    plt.title(f'Kaplan–Meier Curves by {group_col}')
    plt.xlabel('Round')
    plt.ylabel('Survival Probability')
    plt.legend(title=group_col)
    plt.tight_layout()
    plt.show()

# --- ICR Plots: Predicted vs Actual Failure Counts by Model ---
def plot_icr(count_model, static_df, group_col='model'):
    """
    Plot predicted vs actual failure counts by model (ICR plot).
    """
    static_df = static_df.copy()
    static_df['predicted'] = count_model.predict(static_df)
    summary = static_df.groupby(group_col).agg({'time_to_failure': 'mean', 'predicted': 'mean'}).reset_index()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=summary, x='time_to_failure', y='predicted', hue=group_col, s=100)
    plt.plot([summary['time_to_failure'].min(), summary['time_to_failure'].max()],
             [summary['time_to_failure'].min(), summary['time_to_failure'].max()],
             'k--', label='Ideal')
    plt.xlabel('Actual Mean Failure Count')
    plt.ylabel('Predicted Mean Failure Count')
    plt.title('Predicted vs Actual Failure Counts by Model')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Partial Effect Plots: Hazard Ratio vs Covariate ---
def plot_partial_effects(cph, long_df, covariate, values=None):
    """
    Plot partial effect of a covariate (e.g., prompt_to_prompt_drift) on hazard ratio.
    """
    if values is None:
        # Use percentiles for range
        vmin, vmax = np.percentile(long_df[covariate].dropna(), [5, 95])
        values = np.linspace(vmin, vmax, 10)
    
    # Create a simple hazard ratio plot
    plt.figure(figsize=(8, 6))
    
    # Get the coefficient for the covariate
    if covariate in cph.params_.index:
        coef = cph.params_[covariate]
        hazard_ratios = np.exp(coef * values)
        
        plt.plot(values, hazard_ratios, 'b-', linewidth=2, label='Hazard Ratio')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='HR = 1 (no effect)')
        plt.xlabel(covariate)
        plt.ylabel('Hazard Ratio')
        plt.title(f'Hazard Ratio vs {covariate}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, f'Covariate {covariate} not found in model', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'Partial Effect of {covariate} - Not Available')
    
    plt.tight_layout()
    plt.show()

# --- Interaction Heatmap: Model × Prompt₀ Effect Sizes (e.g., IRRs) ---
def plot_interaction_heatmap(count_model, static_df, model_col='model', prompt0_col='prompt0_id'):
    """
    Plot heatmap of Model × Prompt₀ interaction effect sizes (e.g., IRRs).
    """
    # Get IRRs (exp(coef)) for interaction terms
    params = count_model.params
    interaction_terms = [p for p in params.index if 'model_prompt0_interaction' in p]
    # Build a DataFrame for heatmap
    heatmap_data = []
    for term in interaction_terms:
        # Parse model and prompt0 from term name
        try:
            _, model_val, prompt0_val = term.split('[')[1].replace(']', '').split('_')
        except Exception:
            continue
        irr = np.exp(params[term])
        heatmap_data.append({'model': model_val, 'prompt0_id': prompt0_val, 'IRR': irr})
    heatmap_df = pd.DataFrame(heatmap_data)
    if heatmap_df.empty:
        print('No interaction terms found for heatmap.')
        return
    heatmap_pivot = heatmap_df.pivot(index='model', columns='prompt0_id', values='IRR')
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Model × Prompt₀ Interaction IRRs')
    plt.xlabel('Prompt₀ ID')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()

# --- Example usage ---
if __name__ == '__main__':
    from baseline_modeling import long_all, static_all, fit_count_model, fit_coxph_model, train_static
    # Example: KM by model
    plot_km_by_group(long_all, group_col='model')
    # Example: KM by high/low drift
    long_all['drift_group'] = pd.qcut(long_all['prompt_to_prompt_drift'], 2, labels=['Low Drift', 'High Drift'])
    plot_km_by_group(long_all, group_col='drift_group')
    # ICR plot
    count_model = fit_count_model(train_static, model_type='poisson')
    plot_icr(count_model, static_all)
    # Partial effect plot
    cph = fit_coxph_model(long_all)
    plot_partial_effects(cph, long_all, covariate='prompt_to_prompt_drift')
    # Interaction heatmap
    plot_interaction_heatmap(count_model, static_all) 