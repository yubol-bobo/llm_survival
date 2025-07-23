#!/usr/bin/env python3
"""
Extract Individual Model Coefficients (Baseline Analysis)
=========================================================
Extracts detailed coefficients for each individual model using baseline Cox regression,
providing the individual model equivalent of the combined analysis.

Usage:
    python extract_individual_model_coefficients.py

Outputs:
    - Individual model baseline coefficients with hazard ratios and p-values
    - Model-by-model coefficient tables with interpretations
    - Comparison with combined model results
"""

import pandas as pd
import numpy as np
import os
from lifelines import CoxPHFitter
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

class IndividualModelCoefficientExtractor:
    """Extract and analyze coefficients for each individual model."""
    
    def __init__(self):
        self.individual_coefficients = []
        self.model_summaries = {}
        
    def load_individual_model_data(self):
        """Load processed data for each individual model."""
        print("📊 LOADING INDIVIDUAL MODEL DATA")
        print("=" * 40)
        
        processed_dir = 'processed_data'
        if not os.path.exists(processed_dir):
            print("❌ No processed_data directory found!")
            return None
            
        model_data = {}
        model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        
        for model_name in tqdm(model_dirs, desc="Loading model data"):
            long_file = os.path.join(processed_dir, model_name, f'{model_name}_long.csv')
            
            if os.path.exists(long_file):
                try:
                    df = pd.read_csv(long_file)
                    model_data[model_name] = df
                    print(f"   ✅ {model_name}: {len(df):,} observations, {df['failure'].sum():,} events")
                except Exception as e:
                    print(f"   ❌ Error loading {model_name}: {e}")
        
        if not model_data:
            print("❌ No model data loaded!")
            return None
            
        print(f"\n📊 Loaded {len(model_data)} individual models")
        return model_data
    
    def fit_individual_model_coefficients(self, model_data):
        """Fit Cox models for each individual model and extract coefficients."""
        print("\n🔍 FITTING INDIVIDUAL MODEL COEFFICIENTS")
        print("=" * 45)
        
        # Define survival analysis columns
        covariates = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'prompt_complexity']
        duration_col = 'round'
        event_col = 'failure'
        
        for model_name, df in model_data.items():
            print(f"\n🤖 Analyzing {model_name}:")
            print(f"   📊 Total observations: {len(df):,}")
            print(f"   🎯 Total events: {df[event_col].sum():,}")
            
            # Prepare clean data
            survival_cols = covariates + [duration_col, event_col]
            clean_data = df[survival_cols].dropna()
            
            print(f"   📋 Clean observations: {len(clean_data):,}")
            print(f"   🎯 Clean events: {clean_data[event_col].sum():,}")
            
            if len(clean_data) < 50 or clean_data[event_col].sum() < 5:
                print(f"   ⚠️ Insufficient data for {model_name}")
                continue
            
            # Fit Cox model
            try:
                cph = CoxPHFitter()
                cph.fit(clean_data, duration_col=duration_col, event_col=event_col, show_progress=False)
                
                print(f"   ✅ Model fitted successfully:")
                print(f"      • C-Index: {cph.concordance_index_:.4f}")
                print(f"      • AIC: {getattr(cph, 'AIC_partial_', 'N/A')}")
                
                # Extract coefficients
                for covariate in covariates:
                    if covariate in cph.params_.index:
                        coef = cph.params_[covariate]
                        hr = np.exp(coef)
                        p_val = cph.summary.loc[covariate, 'p']
                        ci_lower = cph.summary.loc[covariate, 'coef lower 95%']
                        ci_upper = cph.summary.loc[covariate, 'coef upper 95%']
                        hr_ci_lower = np.exp(ci_lower)
                        hr_ci_upper = np.exp(ci_upper)
                        
                        # Significance
                        if p_val < 0.001:
                            significance = '***'
                        elif p_val < 0.01:
                            significance = '**'
                        elif p_val < 0.05:
                            significance = '*'
                        else:
                            significance = 'ns'
                        
                        self.individual_coefficients.append({
                            'Model': model_name,
                            'Analysis_Type': 'Individual_Baseline',
                            'Covariate': covariate,
                            'Coefficient': coef,
                            'Hazard_Ratio': hr,
                            'P_Value': p_val,
                            'CI_Lower': ci_lower,
                            'CI_Upper': ci_upper,
                            'HR_CI_Lower': hr_ci_lower,
                            'HR_CI_Upper': hr_ci_upper,
                            'Significance': significance,
                            'C_Index': cph.concordance_index_,
                            'AIC': getattr(cph, 'AIC_partial_', np.nan),
                            'N_Observations': len(clean_data),
                            'N_Events': clean_data[event_col].sum()
                        })
                
                print(f"      • Extracted {len(covariates)} coefficients")
                
            except Exception as e:
                print(f"   ❌ Model fitting failed: {e}")
    
    def interpret_coefficient(self, covariate, coefficient, hazard_ratio, p_value):
        """Generate interpretation for a coefficient."""
        # Significance
        if p_value < 0.001:
            sig_text = "Highly significant"
        elif p_value < 0.01:
            sig_text = "Significant" 
        elif p_value < 0.05:
            sig_text = "Marginally significant"
        else:
            sig_text = "Not significant"
        
        # Effect interpretation
        if covariate == 'prompt_to_prompt_drift':
            if hazard_ratio > 1e6:
                effect = "Extreme failure risk increase"
            elif hazard_ratio > 1000:
                effect = "Massive failure risk increase"
            elif hazard_ratio > 100:
                effect = "Large failure risk increase"
            elif hazard_ratio > 10:
                effect = "Moderate failure risk increase"
            else:
                effect = "Small failure risk increase"
        elif covariate == 'context_to_prompt_drift':
            if hazard_ratio > 100:
                effect = "Large context vulnerability"
            elif hazard_ratio > 10:
                effect = "Moderate context vulnerability"
            elif hazard_ratio > 1.1:
                effect = "Slight context vulnerability"
            else:
                effect = "Minimal context effect"
        elif covariate == 'cumulative_drift':
            if hazard_ratio < 1e-5:
                effect = "Strong protective effect (adaptation)"
            elif hazard_ratio < 1e-3:
                effect = "Moderate protective effect"
            elif hazard_ratio < 0.9:
                effect = "Slight protective effect"
            else:
                effect = "Minimal cumulative effect"
        elif covariate == 'prompt_complexity':
            if abs(coefficient) < 0.001:
                effect = "Minimal complexity effect"
            elif hazard_ratio > 1.01:
                effect = "Slight complexity risk increase"
            elif hazard_ratio < 0.99:
                effect = "Slight complexity protective effect"
            else:
                effect = "No complexity effect"
        else:
            effect = "Unknown effect pattern"
        
        return f"{effect} ({sig_text})"
    
    def format_number(self, num, is_pvalue=False, is_hazard_ratio=False):
        """Format numbers for display."""
        if pd.isna(num):
            return "N/A"
        
        if is_pvalue:
            if num < 0.001:
                return "<0.001"
            else:
                return f"{num:.3f}"
        
        if is_hazard_ratio:
            if num >= 1e6:
                return f"{num:.2e}"
            elif num >= 1000:
                return f"{num:,.0f}"
            elif num >= 1:
                return f"{num:.2f}"
            else:
                return f"{num:.2e}"
        
        # Regular coefficient
        return f"{num:.3f}"
    
    def create_model_summaries(self):
        """Create detailed summaries for each model."""
        print("\n📊 CREATING INDIVIDUAL MODEL SUMMARIES")
        print("=" * 45)
        
        df = pd.DataFrame(self.individual_coefficients)
        
        for model_name in df['Model'].unique():
            model_data = df[df['Model'] == model_name]
            
            if model_data.empty:
                continue
            
            # Create summary data for this model
            summary_data = []
            
            predictor_order = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'prompt_complexity']
            
            for predictor in predictor_order:
                row_data = model_data[model_data['Covariate'] == predictor]
                if not row_data.empty:
                    row = row_data.iloc[0]
                    interpretation = self.interpret_coefficient(
                        predictor, row['Coefficient'], row['Hazard_Ratio'], row['P_Value']
                    )
                    
                    summary_data.append({
                        'Predictor': predictor.replace('_', '-').title(),
                        'Coefficient': self.format_number(row['Coefficient']),
                        'Hazard_Ratio': self.format_number(row['Hazard_Ratio'], is_hazard_ratio=True),
                        'P_Value': self.format_number(row['P_Value'], is_pvalue=True),
                        'Significance': row['Significance'],
                        'Interpretation': interpretation
                    })
            
            # Create model summary
            model_summary = {
                'model_name': model_name,
                'c_index': model_data['C_Index'].iloc[0],
                'aic': model_data['AIC'].iloc[0],
                'n_observations': model_data['N_Observations'].iloc[0],
                'n_events': model_data['N_Events'].iloc[0],
                'coefficients': summary_data
            }
            
            self.model_summaries[model_name] = model_summary
        
        print(f"✅ Created summaries for {len(self.model_summaries)} models")
    
    def create_markdown_report(self):
        """Create markdown report with individual model coefficients."""
        markdown_content = []
        
        markdown_content.append("# 📊 Individual Model Baseline Coefficient Analysis\n")
        markdown_content.append("Detailed coefficient analysis for each individual LLM using baseline Cox regression.\n")
        markdown_content.append("**Note:** This is the individual model equivalent of the previous combined analysis.\n\n")
        
        # Sort models by C-Index (descending)
        sorted_models = sorted(
            self.model_summaries.items(),
            key=lambda x: x[1]['c_index'],
            reverse=True
        )
        
        # Overall summary
        markdown_content.append("## 🎯 Individual Model Performance Summary\n\n")
        markdown_content.append("| Model | C-Index | AIC | Observations | Events |\n")
        markdown_content.append("|-------|---------|-----|--------------|--------|\n")
        
        for model_name, summary in sorted_models:
            markdown_content.append(
                f"| **{model_name}** | {summary['c_index']:.3f} | {summary['aic']:.1f} | "
                f"{summary['n_observations']:,} | {summary['n_events']:,} |\n"
            )
        
        markdown_content.append("\n---\n\n")
        
        # Individual model details
        for model_name, summary in sorted_models:
            markdown_content.append(f"## 🤖 {model_name.upper()}\n")
            
            # Model performance
            markdown_content.append("### Model Performance\n")
            discriminative_ability = "excellent" if summary['c_index'] > 0.85 else "good"
            markdown_content.append(f"- **Concordance Index:** {summary['c_index']:.3f} ({discriminative_ability} discriminative ability)\n")
            markdown_content.append(f"- **AIC:** {summary['aic']:.1f}\n")
            markdown_content.append(f"- **Valid predictions:** {summary['n_observations']:,} turn-level observations\n")
            markdown_content.append(f"- **Failure events:** {summary['n_events']:,} conversation breakdowns\n\n")
            
            # Coefficient table
            markdown_content.append("### Key Findings\n\n")
            markdown_content.append("| Predictor | Coefficient | Hazard Ratio | p-value | Significance | Interpretation |\n")
            markdown_content.append("|-----------|-------------|--------------|---------|--------------|----------------|\n")
            
            for coef_data in summary['coefficients']:
                markdown_content.append(
                    f"| {coef_data['Predictor']} | {coef_data['Coefficient']} | "
                    f"{coef_data['Hazard_Ratio']} | {coef_data['P_Value']} | "
                    f"{coef_data['Significance']} | {coef_data['Interpretation']} |\n"
                )
            
            markdown_content.append("\n---\n\n")
        
        return "".join(markdown_content)
    
    def save_results(self):
        """Save individual model coefficient results."""
        print("\n💾 SAVING INDIVIDUAL MODEL COEFFICIENT RESULTS")
        print("=" * 50)
        
        os.makedirs('generated/outputs', exist_ok=True)
        
        # Save detailed coefficients CSV
        if self.individual_coefficients:
            df = pd.DataFrame(self.individual_coefficients)
            df.to_csv('generated/outputs/individual_model_baseline_coefficients.csv', index=False)
            print(f"✅ Detailed coefficients: {len(df)} records")
        
        # Save markdown report
        markdown_content = self.create_markdown_report()
        with open('generated/outputs/individual_model_coefficient_report.md', 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print("✅ Markdown report saved")
        
        # Save model summaries JSON
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        summaries_serializable = convert_numpy_types(self.model_summaries)
        
        with open('generated/outputs/individual_model_summaries.json', 'w') as f:
            json.dump(summaries_serializable, f, indent=2)
        print("✅ Model summaries JSON saved")
        
        print("\n📁 Generated files:")
        print("   • individual_model_baseline_coefficients.csv")
        print("   • individual_model_coefficient_report.md") 
        print("   • individual_model_summaries.json")
    
    def print_sample_results(self):
        """Print sample results for the best performing model."""
        if not self.model_summaries:
            return
        
        # Get best model by C-Index
        best_model = max(self.model_summaries.items(), key=lambda x: x[1]['c_index'])
        model_name, summary = best_model
        
        print(f"\n🏆 SAMPLE: {model_name.upper()} (Best C-Index)")
        print("=" * 50)
        print(f"C-Index: {summary['c_index']:.3f} (excellent discriminative ability)")
        print(f"AIC: {summary['aic']:.1f}")
        print(f"Valid predictions: {summary['n_observations']:,} observations")
        print(f"Failure events: {summary['n_events']:,} breakdowns")
        print("\nKey Findings:")
        print("-" * 80)
        print(f"{'Predictor':<25} {'Coefficient':<12} {'Hazard Ratio':<15} {'p-value':<10} {'Interpretation'}")
        print("-" * 80)
        
        for coef_data in summary['coefficients']:
            print(f"{coef_data['Predictor']:<25} {coef_data['Coefficient']:<12} "
                  f"{coef_data['Hazard_Ratio']:<15} {coef_data['P_Value']:<10} {coef_data['Interpretation']}")
    
    def compare_with_combined_analysis(self):
        """Compare individual model results with combined analysis."""
        print("\n🔍 COMPARISON WITH COMBINED ANALYSIS")
        print("=" * 45)
        
        # Load existing combined analysis if available
        combined_file = 'generated/outputs/cox_model_coefficients.csv'
        if os.path.exists(combined_file):
            combined_df = pd.read_csv(combined_file)
            print(f"✅ Found combined analysis: {len(combined_df)} records")
            
            print("\n📊 Analysis Type Comparison:")
            print(f"   • Combined analysis: {combined_df['Model'].nunique()} models")
            print(f"   • Individual analysis: {len(self.model_summaries)} models")
            print(f"   • Data source: Same processed survival data")
            print(f"   • Key difference: Individual vs combined modeling approach")
        else:
            print("❌ No combined analysis found for comparison")
    
    def run_complete_analysis(self):
        """Run complete individual model coefficient analysis."""
        print("🔬 INDIVIDUAL MODEL BASELINE COEFFICIENT ANALYSIS")
        print("=" * 55)
        print("This analysis provides individual model coefficients to complement")
        print("the combined model analysis you showed earlier.\n")
        
        # Load data
        model_data = self.load_individual_model_data()
        if model_data is None:
            return
        
        # Fit individual models
        self.fit_individual_model_coefficients(model_data)
        
        # Create summaries
        self.create_model_summaries()
        
        # Save results
        self.save_results()
        
        # Print sample
        self.print_sample_results()
        
        # Compare with combined analysis
        self.compare_with_combined_analysis()
        
        print(f"\n🎉 INDIVIDUAL MODEL ANALYSIS COMPLETE!")
        print("=" * 45)
        print(f"📊 Analysis Summary:")
        print(f"   • Models analyzed: {len(self.model_summaries)}")
        print(f"   • Total coefficients: {len(self.individual_coefficients)}")
        print(f"   • Analysis type: Individual model baseline Cox regression")
        print(f"   • Comparison: Individual vs your original combined approach")

def main():
    """Run individual model coefficient analysis."""
    extractor = IndividualModelCoefficientExtractor()
    extractor.run_complete_analysis()

if __name__ == "__main__":
    main() 