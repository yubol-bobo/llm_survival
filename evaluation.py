import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

# --- Count Model Evaluation ---
def count_model_metrics(model, X, y):
    """
    Evaluate count regression model (Poisson/NB):
    - AIC, BIC
    - Deviance residuals
    - RMSE of predicted vs actual
    - Overdispersion test
    """
    # Goodness of fit
    aic = model.aic
    bic = model.bic if hasattr(model, 'bic') else None
    deviance_resid = model.resid_deviance
    
    # Clean data for predictions - use only rows without missing values
    # Select only the columns that were used in the model
    model_cols = ['avg_prompt_to_prompt_drift', 'avg_context_to_prompt_drift', 'avg_prompt_complexity']
    X_clean = X[model_cols + ['model']].copy()
    
    # Remove rows with missing values
    valid_mask = ~(X_clean.isnull().any(axis=1) | y.isnull())
    X_clean = X_clean[valid_mask]
    y_clean = y[valid_mask]
    
    # Predictive accuracy
    y_pred = model.predict(X_clean)
    
    # Remove any remaining NaN predictions
    pred_mask = ~(np.isnan(y_pred) | np.isnan(y_clean))
    y_pred_clean = y_pred[pred_mask]
    y_clean_final = y_clean[pred_mask]
    
    if len(y_pred_clean) > 0:
        rmse = np.sqrt(mean_squared_error(y_clean_final, y_pred_clean))
    else:
        rmse = np.nan
    
    # Overdispersion test (mean vs variance)
    if len(y_clean_final) > 0:
        mean_y = np.mean(y_clean_final)
        var_y = np.var(y_clean_final, ddof=1)
        overdispersion = var_y > mean_y
        overdispersion_stat = var_y / mean_y if mean_y > 0 else np.nan
    else:
        mean_y = var_y = overdispersion_stat = np.nan
        overdispersion = False
    
    return {
        'AIC': aic,
        'BIC': bic,
        'RMSE': rmse,
        'Overdispersion': overdispersion,
        'Overdispersion_stat': overdispersion_stat,
        'Deviance_residuals': deviance_resid,
        'Valid_predictions': len(y_pred_clean)
    }

# --- Survival Model Evaluation ---
def survival_model_metrics(cph, df, duration_col='round', event_col='failure'):
    """
    Evaluate Cox PH survival model:
    - Concordance index (C-index)
    - Schoenfeld residuals (proportional hazards test)
    """
    # Clean data - use only the columns that were used in the model
    model_cols = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'prompt_complexity']
    required_cols = model_cols + [duration_col, event_col]
    df_clean = df[required_cols].copy()
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    
    if len(df_clean) == 0:
        return {
            'Concordance_index': np.nan,
            'Schoenfeld_pvalues': {},
            'Schoenfeld_test': None,
            'Valid_observations': 0
        }
    
    # Concordance index
    try:
        partial_hazards = cph.predict_partial_hazard(df_clean)
        # Remove any NaN predictions
        valid_mask = ~(np.isnan(partial_hazards) | np.isnan(df_clean[duration_col]) | np.isnan(df_clean[event_col]))
        if valid_mask.sum() > 0:
            c_index = concordance_index(
                df_clean[duration_col][valid_mask], 
                -partial_hazards[valid_mask], 
                df_clean[event_col][valid_mask]
            )
        else:
            c_index = np.nan
    except Exception as e:
        print(f"Warning: Could not compute concordance index: {e}")
        c_index = np.nan
    
    # Schoenfeld residuals test
    try:
        results = proportional_hazard_test(cph, df_clean, time_transform='rank')
        schoenfeld_p = results.summary['p'].to_dict()
        schoenfeld_test = results
    except Exception as e:
        print(f"Warning: Could not compute Schoenfeld test: {e}")
        schoenfeld_p = {}
        schoenfeld_test = None
    
    return {
        'Concordance_index': c_index,
        'Schoenfeld_pvalues': schoenfeld_p,
        'Schoenfeld_test': schoenfeld_test,
        'Valid_observations': len(df_clean)
    }

# --- Hypothesis Tests ---
def likelihood_ratio_test(model_restricted, model_full):
    """
    Likelihood ratio test for nested models (e.g., with/without semantic covariates).
    """
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_model - model_restricted.df_model
    p_value = sm.stats.chisqprob(lr_stat, df_diff)
    return {'LR_stat': lr_stat, 'df_diff': df_diff, 'p_value': p_value}

def wald_test(model, param):
    """
    Wald test for individual coefficient (param: string name of parameter).
    """
    est = model.params[param]
    se = model.bse[param]
    wald_stat = (est / se) ** 2
    p_value = sm.stats.chisqprob(wald_stat, 1)
    return {'Wald_stat': wald_stat, 'p_value': p_value, 'estimate': est, 'se': se}

# --- Example usage ---
if __name__ == '__main__':
    # Example: load model and data from baseline_modeling.py
    from baseline_modeling import train_static, fit_count_model, long_all, fit_coxph_model
    # Count model
    count_model = fit_count_model(train_static, model_type='poisson')
    y = train_static['time_to_failure']
    X = train_static
    count_metrics = count_model_metrics(count_model, X, y)
    print('Count model metrics:', count_metrics)
    # Survival model
    cph = fit_coxph_model(long_all)
    surv_metrics = survival_model_metrics(cph, long_all)
    print('Survival model metrics:', surv_metrics)
    # Example: likelihood ratio test (fit a restricted model first)
    # model_restricted = fit_count_model(train_static[['time_to_failure', 'model', 'prompt0_id']], model_type='poisson')
    # lr_test = likelihood_ratio_test(model_restricted, count_model)
    # print('Likelihood ratio test:', lr_test)
    # Example: Wald test for a coefficient
    # wald = wald_test(count_model, 'avg_prompt_to_prompt_drift')
    # print('Wald test:', wald) 