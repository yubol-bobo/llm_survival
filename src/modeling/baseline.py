#!/usr/bin/env python3
"""
Combined Baseline Survival Modeling for LLM Robustness Analysis
==============================================================

Combined modeling implementation that:
1. Loads data from data/raw/ and data/processed/ directories
2. Combines all LLM data into a single dataset
3. Uses 'model' as a covariate with dummy encoding
4. Performs Cox PH survival analysis with conversation-level frailty
5. Provides direct statistical comparisons between LLMs
6. Exports all results to results/outputs/baseline/

Frailty Approaches (in order of preference):
- Conversation-level frailty: Accounts for conversation-specific effects
- Subject-stratified: Stratifies by subject cluster when conversation frailty fails
- Standard Cox PH: Fallback when frailty approaches fail

Features:
- Hazard ratios comparing all LLMs to claude_35 baseline
- Statistical significance testing for model differences
- Unified confidence intervals for all comparisons

Usage:
    python src/modeling/baseline.py

Outputs:
    - results/outputs/baseline/model_coefficients.csv
    - results/outputs/baseline/model_performance.csv  
    - results/outputs/baseline/hazard_ratios.csv
    - results/outputs/baseline/p_values.csv
    - results/outputs/baseline/complete_results.csv
"""

import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BaselineModeling:
    """Combined baseline survival modeling for LLM robustness analysis."""
    
    def __init__(self):
        self.models_data = {}
        self.subject_clusters = self._create_subject_clusters()
        
    def _create_subject_clusters(self):
        """Create 7 meaningful subject clusters from 39 individual subjects."""
        return {
            'Business_Economics': [
                'accounting', 'econometrics', 'management', 
                'marketing', 'microeconomics'
            ],
            'General_Knowledge': [
                'common sense', 'truthful'
            ],
            'Humanities': [
                'formal_logic', 'philosophy', 'prehistory', 
                'us_history', 'world_history', 'world_religions'
            ],
            'Law_Legal': [
                'international_law', 'jurisprudence', 'law'
            ],
            'Medical_Health': [
                'anatomy', 'biology', 'clinical_knowledge', 
                'human_sexuality', 'medical_genetics', 'medicine', 
                'nutrition', 'virology'
            ],
            'STEM': [
                'abstract_algebra', 'astronomy', 'chemistry', 
                'computer_science', 'computer_security', 'conceptual_physics', 
                'electrical_engineering', 'machine_learning', 'mathematics', 
                'physics', 'statistics'
            ],
            'Social_Sciences': [
                'global_facts', 'moral_scenarios', 'psychology', 'sociology'
            ]
        }
        
    def _map_subject_to_cluster(self, subject):
        """Map individual subject to cluster."""
        for cluster_name, subjects in self.subject_clusters.items():
            if subject in subjects:
                return cluster_name
        return 'Other'
        
    def load_data(self):
        """Load and process data from data/ directories."""
        print("Loading training data from data/processed/train/")
        print("=" * 65)

        # Load cleaned reference data from TRAIN split ONLY
        cleaned_data_path = 'data/processed/cleaned_data_with_clusters.csv'
        if not os.path.exists(cleaned_data_path):
            raise FileNotFoundError(f"Processed metadata not found: {cleaned_data_path}. Run the data_split stage first.")

        cleaned_data = pd.read_csv(cleaned_data_path)

        # Use existing subject_cluster from the data, or map if missing
        if 'subject_cluster' not in cleaned_data.columns:
            cleaned_data['subject_cluster'] = cleaned_data['subject'].apply(self._map_subject_to_cluster)

        # Ensure difficulty level metadata is available
        if 'difficulty_level' not in cleaned_data.columns:
            cleaned_data['difficulty_level'] = cleaned_data['level']

        # Load processed model data from TRAIN split
        processed_dir = 'data/processed/train'
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Train data not found: {processed_dir}. Run train/test split first using --stage data_split!")
        
        model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        
        for model_name in tqdm(model_dirs, desc="Loading models"):
            model_path = os.path.join(processed_dir, model_name)
            
            # Load static and long format data
            static_path = os.path.join(model_path, f'{model_name}_static.csv')
            long_path = os.path.join(model_path, f'{model_name}_long.csv')
            
            if os.path.exists(static_path) and os.path.exists(long_path):
                static_df = pd.read_csv(static_path)
                long_df = pd.read_csv(long_path)
                
                # Use the original level as difficulty_level
                static_df['difficulty_level'] = static_df['level']
                long_df['difficulty_level'] = long_df['level']
                
                self.models_data[model_name] = {
                    'static': static_df,
                    'long': long_df
                }
                
        print(f"✅ Loaded {len(self.models_data)} models")
        return self.models_data

    def fit_combined_survival_model(self):
        """Fit a single Cox PH model combining all LLMs with 'model' as a covariate."""
        print("\n🔍 FITTING COMBINED SURVIVAL MODEL")
        print("=" * 40)
        print("Treating all LLMs as single dataset with 'model' as covariate")
        
        # Combine all model data
        combined_data = []
        
        for model_name in tqdm(self.models_data.keys(), desc="Combining data"):
            try:
                long_df = self.models_data[model_name]['long'].copy()
                
                # Add model name as feature
                long_df['model'] = model_name
                
                # Prepare the data with drift covariates and new covariates
                drift_covariates = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                                  'cumulative_drift', 'prompt_complexity']
                
                # Select required columns including new covariates
                required_cols = ['round', 'failure', 'conversation_id', 'subject_cluster', 'difficulty_level', 'model'] + drift_covariates
                available_cols = [col for col in required_cols if col in long_df.columns]
                model_data = long_df[available_cols].copy()
                
                # Drop rows with NaN in critical columns
                model_data = model_data.dropna(subset=['round', 'failure'] + drift_covariates)
                
                if len(model_data) > 0:
                    combined_data.append(model_data)
                    
            except Exception as e:
                print(f"❌ Failed to combine {model_name}: {e}")
                continue
        
        if not combined_data:
            print("❌ No data available for combined modeling")
            return None
        
        # Concatenate all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"📊 Combined dataset: {len(combined_df)} observations from {len(self.models_data)} models")
        
        # Encode model names for Cox regression
        le = LabelEncoder()
        combined_df['model_encoded'] = le.fit_transform(combined_df['model'])
        
        # Try different fitting approaches with frailty
        fitted_model = None
        frailty_used = 'failed'
        
        drift_covariates = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                          'cumulative_drift', 'prompt_complexity']
        
        # Add model covariates (using dummy encoding for better interpretation)
        model_dummies = pd.get_dummies(combined_df['model'], prefix='model', drop_first=True)
        combined_df = pd.concat([combined_df, model_dummies], axis=1)
        model_covariate_cols = model_dummies.columns.tolist()
        
        # Add subject cluster covariates (using dummy encoding)
        if 'subject_cluster' in combined_df.columns:
            subject_dummies = pd.get_dummies(combined_df['subject_cluster'], prefix='subject', drop_first=True)
            combined_df = pd.concat([combined_df, subject_dummies], axis=1)
            subject_covariate_cols = subject_dummies.columns.tolist()
        else:
            subject_covariate_cols = []
        
        # Add difficulty level covariates (using dummy encoding)
        if 'difficulty_level' in combined_df.columns:
            difficulty_dummies = pd.get_dummies(combined_df['difficulty_level'], prefix='difficulty', drop_first=True)
            combined_df = pd.concat([combined_df, difficulty_dummies], axis=1)
            difficulty_covariate_cols = difficulty_dummies.columns.tolist()
        else:
            difficulty_covariate_cols = []
        
        # Combine all covariate columns
        all_covariates = drift_covariates + model_covariate_cols + subject_covariate_cols + difficulty_covariate_cols
        
        # 1. Try conversation-level frailty with all covariates
        try:
            if 'conversation_id' in combined_df.columns and combined_df['conversation_id'].nunique() > 10:
                cph = CoxPHFitter()
                frailty_data = combined_df[all_covariates + ['round', 'failure', 'conversation_id']]
                frailty_data = frailty_data.dropna()
                cph.fit(frailty_data, duration_col='round', event_col='failure', 
                       cluster_col='conversation_id', show_progress=False)
                fitted_model = cph
                frailty_used = 'conversation_frailty'
                print(f"✅ Conversation frailty successful with {len(all_covariates)} covariates")
        except Exception as e:
            print(f"⚠️  Conversation frailty failed: {e}")
        
        # 2. Try subject-stratified approach (but don't stratify by subject since it's now a covariate)
        if fitted_model is None:
            try:
                cph = CoxPHFitter()
                basic_data = combined_df[all_covariates + ['round', 'failure']]
                basic_data = basic_data.dropna()
                cph.fit(basic_data, duration_col='round', event_col='failure', show_progress=False)
                fitted_model = cph
                frailty_used = 'no_frailty_all_covariates'
                print(f"✅ Standard Cox PH successful with {len(all_covariates)} covariates")
            except Exception as e:
                print(f"⚠️  Standard Cox PH with all covariates failed: {e}")
        
        # 3. Fallback to basic model without subject/difficulty covariates
        if fitted_model is None:
            try:
                cph = CoxPHFitter()
                basic_covariates = drift_covariates + model_covariate_cols
                basic_data = combined_df[basic_covariates + ['round', 'failure']]
                basic_data = basic_data.dropna()
                cph.fit(basic_data, duration_col='round', event_col='failure', show_progress=False)
                fitted_model = cph
                frailty_used = 'no_frailty_basic'
                print(f"✅ Fallback basic model successful with {len(basic_covariates)} covariates")
            except Exception as e:
                print(f"❌ All fitting approaches failed: {e}")
                return None
        
        # Extract results
        combined_result = {
            'model': 'COMBINED_ALL_LLMS',
            'frailty_type': frailty_used,
            'n_observations': len(combined_df),
            'n_events': combined_df['failure'].sum(),
            'n_conversations': combined_df['conversation_id'].nunique() if 'conversation_id' in combined_df.columns else np.nan,
            'n_subjects': combined_df['subject_cluster'].nunique() if 'subject_cluster' in combined_df.columns else np.nan,
            'n_models': len(self.models_data),
            'c_index': fitted_model.concordance_index_,
            'aic': getattr(fitted_model, 'AIC_partial_', np.nan),
            'log_likelihood': fitted_model.log_likelihood_,
            'model_names': ', '.join(list(self.models_data.keys()))
        }
        
        # Add coefficients and p-values for all covariates (drift, model, subject, difficulty)
        export_covariates = drift_covariates + model_covariate_cols + subject_covariate_cols + difficulty_covariate_cols
        for covariate in export_covariates:
            if covariate in fitted_model.params_.index:
                combined_result[f'{covariate}_coef'] = fitted_model.params_[covariate]
                combined_result[f'{covariate}_pval'] = fitted_model.summary.loc[covariate, 'p']
                combined_result[f'{covariate}_hr'] = np.exp(fitted_model.params_[covariate])
                
                # Add confidence intervals if available
                try:
                    ci_lower = fitted_model.confidence_intervals_.loc[covariate, '95% CI (lower)']
                    ci_upper = fitted_model.confidence_intervals_.loc[covariate, '95% CI (upper)']
                    combined_result[f'{covariate}_hr_ci_lower'] = np.exp(ci_lower)
                    combined_result[f'{covariate}_hr_ci_upper'] = np.exp(ci_upper)
                except (KeyError, AttributeError):
                    combined_result[f'{covariate}_hr_ci_lower'] = np.nan
                    combined_result[f'{covariate}_hr_ci_upper'] = np.nan
        
        self.combined_results = pd.DataFrame([combined_result])
        self.combined_model = fitted_model
        
        print(f"✅ Combined model fitted successfully")
        print(f"   📊 {len(combined_df)} observations, {combined_df['failure'].sum()} events")
        print(f"   🎯 C-index: {fitted_model.concordance_index_:.4f}")
        print(f"   🔧 Frailty approach: {frailty_used}")
        
        return self.combined_results

    def export_results(self, output_dir='results/outputs/baseline'):
        """Export combined modeling results to CSV files."""
        print(f"\n💾 EXPORTING COMBINED MODEL RESULTS TO {output_dir}/")
        print("=" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export combined model results
        if hasattr(self, 'combined_results'):
            # 1. Model performance metrics
            performance_cols = ['model', 'frailty_type', 'n_observations', 'n_events', 'n_conversations', 'n_subjects', 'n_models', 'c_index', 'aic', 'log_likelihood', 'model_names']
            available_cols = [col for col in performance_cols if col in self.combined_results.columns]
            performance_df = self.combined_results[available_cols].copy()
            performance_df.to_csv(f'{output_dir}/model_performance.csv', index=False)
            print(f"✅ Exported model_performance.csv")
            
            # 2. Model coefficients  
            coef_cols = [col for col in self.combined_results.columns if '_coef' in col or col == 'model']
            if len(coef_cols) > 1:
                coefficients_df = self.combined_results[coef_cols].copy()
                coefficients_df.to_csv(f'{output_dir}/model_coefficients.csv', index=False)
                print(f"✅ Exported model_coefficients.csv")
            
            # 3. Hazard ratios with confidence intervals
            hr_cols = [col for col in self.combined_results.columns if '_hr' in col or col == 'model']
            if len(hr_cols) > 1:
                hazard_ratios_df = self.combined_results[hr_cols].copy()
                hazard_ratios_df.to_csv(f'{output_dir}/hazard_ratios.csv', index=False)
                print(f"✅ Exported hazard_ratios.csv")
            
            # 4. P-values for significance testing
            pval_cols = [col for col in self.combined_results.columns if '_pval' in col or col == 'model']
            if len(pval_cols) > 1:
                pvalues_df = self.combined_results[pval_cols].copy()
                pvalues_df.to_csv(f'{output_dir}/p_values.csv', index=False)
                print(f"✅ Exported p_values.csv")
            
            # 5. Complete results
            self.combined_results.to_csv(f'{output_dir}/complete_results.csv', index=False)
            print(f"✅ Exported complete_results.csv")
        else:
            print("❌ No combined results found to export")
        
        print(f"📁 All results saved to: {output_dir}/")

    def run_complete_analysis(self):
        """Run complete combined baseline modeling pipeline."""
        print("🚀 STARTING COMBINED BASELINE MODELING PIPELINE")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Fit combined model
        self.fit_combined_survival_model()
        
        # Export results
        self.export_results()
        
        print("\n✅ COMBINED BASELINE MODELING COMPLETED!")
        print("=" * 40)
        if hasattr(self, 'combined_results'):
            print(f"🔗 Successfully fitted combined model with all {len(self.models_data)} LLMs")
            print(f"🎯 C-index: {self.combined_model.concordance_index_:.4f}")
        print(f"📁 Results saved to results/outputs/baseline/")
        
        return self.combined_results


def main():
    """Main execution function."""
    baseline = BaselineModeling()
    results = baseline.run_complete_analysis()
    return results


if __name__ == "__main__":
    main()