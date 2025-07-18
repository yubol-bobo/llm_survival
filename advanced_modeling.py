import pandas as pd
import numpy as np
import statsmodels.api as sm
from lifelines import CoxPHFitter

# For Poisson/NegBin GLMMs, use pymer4 (Python wrapper for R's lme4)
try:
    from pymer4.models import Lmer
    pymer4_available = True
except ImportError:
    pymer4_available = False
    print("pymer4 is not installed. Poisson/NegBin GLMMs require pymer4 and R with lme4 package.")

# Fallback: Python-only approximation using statsmodels
try:
    from statsmodels.regression.mixed_linear_model import MixedLM
    statsmodels_available = True
except ImportError:
    statsmodels_available = False

# --- CoxPH with Frailty (Gamma random effect per conversation) ---
def fit_coxph_frailty(long_df, duration_col='round', event_col='failure', cluster_col='conversation_id'):
    """
    Fit Cox Proportional Hazards model with frailty (random effect per conversation).
    Uses lifelines' cluster argument for robust SEs; true gamma frailty not yet in lifelines (as of 2024).
    """
    cph = CoxPHFitter()
    covariates = [
        'prompt_to_prompt_drift',
        'context_to_prompt_drift',
        'cumulative_drift',
        'prompt_complexity',
    ]
    # Select only the columns we need (covariates + duration + event + cluster)
    required_cols = covariates + [duration_col, event_col, cluster_col]
    df = long_df[required_cols].dropna()
    
    cph.fit(
        df,
        duration_col=duration_col,
        event_col=event_col,
        cluster_col=cluster_col,  # robust SEs by conversation
        show_progress=True,
        # Remove strata to avoid categorical issues
    )
    return cph

# --- Poisson GLMM (Python-only fallback) ---
def fit_poisson_glmm_python(static_df, response_col='time_to_failure', group_col='conversation_id'):
    """
    Fit approximate Poisson GLMM using log-transformed response and MixedLM (Python-only fallback).
    Note: This is an approximation, not a true Poisson GLMM.
    """
    print("DEBUG: Using updated fit_poisson_glmm_python function")
    if not statsmodels_available:
        raise ImportError("statsmodels is not available.")
    
    # Prepare data
    df = static_df.copy()
    df['log_response'] = np.log(df[response_col] + 1)  # +1 to handle zeros
    
    # Select only the columns we need and remove missing/infinite values
    required_cols = [response_col, 'avg_prompt_to_prompt_drift', 'avg_context_to_prompt_drift', 'avg_prompt_complexity', group_col]
    df = df[required_cols].copy()
    
    # Remove rows with missing or infinite values
    df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
    df = df.dropna()  # Remove rows with NaN
    
    # Recalculate log response after cleaning
    df['log_response'] = np.log(df[response_col] + 1)
    
    # Create proper group encoding (consecutive integers starting from 0)
    unique_groups = df[group_col].unique()
    group_map = {group: i for i, group in enumerate(unique_groups)}
    df['group_encoded'] = df[group_col].map(group_map)
    
    # Fit the model using explicit arrays instead of formula
    endog = df['log_response']
    exog = df[['avg_prompt_to_prompt_drift', 'avg_context_to_prompt_drift', 'avg_prompt_complexity']]
    exog = sm.add_constant(exog)  # Add intercept
    groups = df['group_encoded']
    
    print(f"DEBUG: Data shape after cleaning: {df.shape}")
    print(f"DEBUG: Any NaN in exog: {exog.isnull().any().any()}")
    print(f"DEBUG: Any inf in exog: {np.isinf(exog).any().any()}")
    
    model = MixedLM(endog, exog, groups=groups)
    results = model.fit()
    return model, results

# --- Poisson GLMM (random intercept per conversation) ---
def fit_poisson_glmm(static_df, response_col='time_to_failure', group_col='conversation_id'):
    """
    Fit Poisson GLMM with random intercept per conversation using pymer4 (requires R/lme4).
    """
    if not pymer4_available:
        raise ImportError("pymer4 is not installed. Please install pymer4 and R with lme4.")
    # Build formula: response ~ fixed + (1|group)
    formula = f"{response_col} ~ avg_prompt_to_prompt_drift + avg_context_to_prompt_drift + avg_prompt_complexity + (1|{group_col})"
    model = Lmer(formula, data=static_df, family='poisson')
    results = model.fit()
    return model, results

# --- Negative Binomial GLMM (random intercept per conversation) ---
def fit_negbin_glmm(static_df, response_col='time_to_failure', group_col='conversation_id'):
    """
    Fit Negative Binomial GLMM with random intercept per conversation using pymer4 (requires R/lme4).
    """
    if not pymer4_available:
        raise ImportError("pymer4 is not installed. Please install pymer4 and R with lme4.")
    # Build formula: response ~ fixed + (1|group)
    formula = f"{response_col} ~ avg_prompt_to_prompt_drift + avg_context_to_prompt_drift + avg_prompt_complexity + (1|{group_col})"
    model = Lmer(formula, data=static_df, family='nb')
    results = model.fit()
    return model, results

# --- Example usage ---
if __name__ == '__main__':
    from baseline_modeling import long_all, static_all
    # CoxPH with frailty
    print("Fitting CoxPH with frailty (clustered by conversation_id)...")
    cph_frailty = fit_coxph_frailty(long_all)
    print(cph_frailty.summary)
    # Poisson GLMM
    if pymer4_available:
        print("Fitting Poisson GLMM (random intercept per conversation)...")
        poisson_glmm, poisson_results = fit_poisson_glmm(static_all)
        print(poisson_results)
    elif statsmodels_available:
        print("Fitting approximate Poisson GLMM using Python-only fallback...")
        poisson_glmm, poisson_results = fit_poisson_glmm_python(static_all)
        print(poisson_results.summary())
    else:
        print("Neither pymer4 nor statsmodels available: Skipping Poisson GLMMs.") 