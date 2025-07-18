import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

# Directory containing processed data
PROCESSED_DATA_DIR = 'processed_data'

# Find all static and long tables
static_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*', '*_static.csv'))
long_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*', '*_long.csv'))

# Load and concatenate static tables
static_dfs = []
for f in static_files:
    df = pd.read_csv(f)
    static_dfs.append(df)
static_all = pd.concat(static_dfs, ignore_index=True) if static_dfs else None

# Load and concatenate long tables
long_dfs = []
for f in long_files:
    df = pd.read_csv(f)
    long_dfs.append(df)
long_all = pd.concat(long_dfs, ignore_index=True) if long_dfs else None

# --- Data Splitting ---
def random_split_static(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split static table into train/val/test with no stratification.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=random_state
    )
    return train, val, test

# Example usage for static table
if static_all is not None:
    train_static, val_static, test_static = random_split_static(static_all)
    print(f"Static split: train={len(train_static)}, val={len(val_static)}, test={len(test_static)}")
else:
    print("No static tables found.")

# --- Count Regression (Poisson/NB) ---
def fit_count_model(train_df, model_type='poisson'):
    """
    Fit a count regression model (Poisson or Negative Binomial) on the static table.
    """
    # Simplified formula without interaction term to avoid numerical issues
    formula = 'time_to_failure ~ C(model) + avg_prompt_to_prompt_drift + avg_context_to_prompt_drift + avg_prompt_complexity'
    if model_type == 'poisson':
        model = smf.glm(formula=formula, data=train_df, family=sm.families.Poisson()).fit()
    elif model_type == 'nb':
        model = smf.glm(formula=formula, data=train_df, family=sm.families.NegativeBinomial()).fit()
    else:
        raise ValueError('model_type must be "poisson" or "nb"')
    return model

# --- Survival Analysis (Cox PH) ---
def fit_coxph_model(long_df):
    """
    Fit a Cox Proportional Hazards model on the long table.
    """
    # Prepare data: one row per conversation x round, with time-varying covariates
    # Required columns: 'conversation_id', 'round', 'failure', 'censored', covariates
    cph = CoxPHFitter()
    # Simplified covariates - focus on continuous variables to avoid categorical issues
    covariates = [
        'prompt_to_prompt_drift',
        'context_to_prompt_drift',
        'cumulative_drift',
        'prompt_complexity',
    ]
    # Select only the columns we need (covariates + duration + event)
    required_cols = covariates + ['round', 'failure']
    df = long_df[required_cols].dropna()
    
    # Lifelines expects duration_col (time), event_col (failure)
    cph.fit(
        df,
        duration_col='round',
        event_col='failure',
        show_progress=True,
        # Remove strata to avoid categorical issues for now
    )
    return cph

# --- Mixed Effects Model Placeholder ---
def fit_mixed_effects_model(train_df):
    """
    Placeholder for mixed effects model (random intercepts for prompt0_id, random slopes for drift).
    """
    # This requires e.g. statsmodels MixedLM or pymer4 (R-style)
    # Example formula: time_to_failure ~ avg_prompt_to_prompt_drift + (1|prompt0_id)
    print("Mixed effects modeling not yet implemented. Use statsmodels MixedLM or pymer4.")

# --- Example usage ---
if __name__ == '__main__':
    # Count regression example
    if static_all is not None:
        model = fit_count_model(train_static, model_type='poisson')
        print(model.summary())
    # Survival analysis example
    if long_all is not None:
        cph = fit_coxph_model(long_all)
        print(cph.summary)
    # Mixed effects placeholder
    if static_all is not None:
        fit_mixed_effects_model(train_static) 