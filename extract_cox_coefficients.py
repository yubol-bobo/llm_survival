#!/usr/bin/env python3
"""
Extract Individual Model Cox Coefficients
=========================================
Extracts detailed Cox regression results for each individual model.
Focus on individual model analysis only.

Usage:
    python extract_cox_coefficients.py

Outputs:
    - generated/outputs/individual_cox_coefficients.csv
    - generated/outputs/individual_hazard_ratios_matrix.csv
    - generated/outputs/individual_pvalues_matrix.csv
"""

import pandas as pd
import numpy as np
import os
from lifelines import CoxPHFitter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CoxCoefficientExtractor:
    """Extract detailed Cox model results."""
    
    def __init__(self):
        self.cox_results = []
        
    def extract_cox_coefficients(self):
        """Extract Cox model coefficients for all models."""
        print("📊 EXTRACTING COX MODEL COEFFICIENTS")
        print("=" * 45)
        
        # Load processed data
        processed_dir = 'processed_data'
        if not os.path.exists(processed_dir):
            print("❌ No processed_data directory found!")
            return
            
        model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        
        for model_name in tqdm(model_dirs, desc="Fitting Cox models"):
            long_file = os.path.join(processed_dir, model_name, f'{model_name}_long.csv')
            
            if os.path.exists(long_file):
                try:
                    long_df = pd.read_csv(long_file)
                    cox_results = self.fit_detailed_cox_model(long_df, model_name)
                    
                    if cox_results:
                        self.cox_results.extend(cox_results)
                        print(f"✅ {model_name}: Extracted {len(cox_results)} coefficients")
                    else:
                        print(f"⚠️ {model_name}: Could not fit Cox model")
                        
                except Exception as e:
                    print(f"❌ Error processing {model_name}: {e}")
        
        # Save results for individual models
        if self.cox_results:
            cox_df = pd.DataFrame(self.cox_results)
            cox_df.to_csv('generated/outputs/individual_cox_coefficients.csv', index=False)
            print(f"\n✅ Individual model coefficients saved: {len(self.cox_results)} total coefficients")
            return cox_df
        else:
            print("❌ No individual model coefficients could be extracted")
            return None
    
    def fit_detailed_cox_model(self, long_df, model_name):
        """Fit Cox model and extract detailed coefficients."""
        try:
            # Available drift covariates
            drift_covariates = []
            for col in ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'prompt_complexity']:
                if col in long_df.columns:
                    drift_covariates.append(col)
            
            # Check for survival data columns
            duration_col = None
            event_col = None
            
            # Try different column names for duration
            for col in ['round', 'turn', 'time_step']:
                if col in long_df.columns:
                    duration_col = col
                    break
            
            # Try different column names for events
            for col in ['failure', 'event', 'censored']:
                if col in long_df.columns:
                    if col == 'censored':
                        # Convert censored to event (flip the values)
                        long_df['event'] = 1 - long_df[col]
                        event_col = 'event'
                    else:
                        event_col = col
                    break
            
            if not duration_col or not event_col or not drift_covariates:
                return None
            
            # Prepare data
            required_cols = drift_covariates + [duration_col, event_col]
            df_clean = long_df[required_cols].dropna()
            
            if len(df_clean) < 10 or df_clean[event_col].sum() < 2:
                return None
            
            # Fit Cox PH model
            cph = CoxPHFitter()
            cph.fit(df_clean, duration_col=duration_col, event_col=event_col)
            
            # Extract coefficients
            results = []
            summary = cph.summary
            
            for covariate in summary.index:
                coefficient = summary.loc[covariate, 'coef']
                hazard_ratio = np.exp(coefficient)
                p_value = summary.loc[covariate, 'p']
                ci_lower = summary.loc[covariate, 'coef lower 95%']
                ci_upper = summary.loc[covariate, 'coef upper 95%']
                hr_ci_lower = np.exp(ci_lower)
                hr_ci_upper = np.exp(ci_upper)
                
                results.append({
                    'Model': model_name,
                    'Covariate': covariate,
                    'Coefficient': coefficient,
                    'Hazard_Ratio': hazard_ratio,
                    'P_Value': p_value,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'HR_CI_Lower': hr_ci_lower,
                    'HR_CI_Upper': hr_ci_upper,
                    'Significance': self.get_significance_stars(p_value)
                })
            
            return results
            
        except Exception as e:
            print(f"Error fitting Cox model for {model_name}: {e}")
            return None
    
    def get_significance_stars(self, p_value):
        """Convert p-value to significance stars."""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"
    
    def create_cox_summary_table(self, cox_df):
        """Create a summary table of Cox results."""
        if cox_df is None or cox_df.empty:
            return
            
        print("\n📋 COX MODEL SUMMARY")
        print("=" * 30)
        
        # Pivot table for easier reading
        pivot_coef = cox_df.pivot(index='Model', columns='Covariate', values='Coefficient')
        pivot_hr = cox_df.pivot(index='Model', columns='Covariate', values='Hazard_Ratio')
        pivot_p = cox_df.pivot(index='Model', columns='Covariate', values='P_Value')
        
        # Save individual model summary tables
        pivot_coef.to_csv('generated/outputs/individual_coefficients_matrix.csv')
        pivot_hr.to_csv('generated/outputs/individual_hazard_ratios_matrix.csv')
        pivot_p.to_csv('generated/outputs/individual_pvalues_matrix.csv')
        
        print("✅ Individual model summary tables saved:")
        print("   • individual_coefficients_matrix.csv")
        print("   • individual_hazard_ratios_matrix.csv") 
        print("   • individual_pvalues_matrix.csv")
        
        # Print sample results
        print(f"\n📊 Sample Results ({len(cox_df)} total coefficients):")
        if not cox_df.empty:
            for _, row in cox_df.head(10).iterrows():
                print(f"   {row['Model']}: {row['Covariate']} - "
                      f"HR={row['Hazard_Ratio']:.3f} (p={row['P_Value']:.3f}{row['Significance']})")

def main():
    """Run individual model Cox coefficient extraction."""
    print("🔬 Individual Model Cox Coefficient Extraction")
    print("=" * 48)
    print("This tool extracts detailed Cox regression results for each individual model:")
    print("• Individual model coefficients")
    print("• Hazard ratios per model") 
    print("• P-values per model")
    print("• Confidence intervals")
    print("• Significance indicators")
    print()
    
    os.makedirs('generated/outputs', exist_ok=True)
    
    extractor = CoxCoefficientExtractor()
    cox_df = extractor.extract_cox_coefficients()
    
    if cox_df is not None:
        extractor.create_cox_summary_table(cox_df)
        print(f"\n🎉 Individual model coefficient extraction complete!")
        print(f"📊 Extracted coefficients for {cox_df['Model'].nunique()} individual models")
        print(f"📈 Analyzed {cox_df['Covariate'].nunique()} covariates per model")
        print(f"\n📁 Files saved in generated/outputs/:")
        print("   • individual_cox_coefficients.csv")
        print("   • individual_coefficients_matrix.csv")
        print("   • individual_hazard_ratios_matrix.csv")
        print("   • individual_pvalues_matrix.csv")
    else:
        print("❌ No individual model coefficients could be extracted")
        print("💡 Make sure you have processed_data/ with model subdirectories")
        print("   containing *_long.csv files with survival data")

if __name__ == "__main__":
    main() 