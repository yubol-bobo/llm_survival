#!/usr/bin/env python3
"""
Advanced Survival Modeling for LLM Robustness Analysis
===========        # Load cleaned reference data (use the file with existing clusters)
        cleaned_data_path = 'data/raw/cleaned_data_with_clusters.csv'
        if not os.path.exists(cleaned_data_path):
            raise FileNotFoundError(f"Required file not found: {cleaned_data_path}")
            
        cleaned_data = pd.read_csv(cleaned_data_path)
        
        # Use existing subject_cluster from the data, or map if missing
        if 'subject_cluster' not in cleaned_data.columns:
            cleaned_data['subject_cluster'] = cleaned_data['subject'].apply(self._map_subject_to_cluster)====================================

Advanced modeling implementation that:
1. Loads data from data/raw/ and data/processed/ directories
2. Fits combined interaction model with drift×model effects
3. Compares combined interaction vs combined baseline models
4. Exports all results to results/outputs/advanced/

Features:
- Combined interaction modeling (all LLMs together)
- Drift×model interaction effects (40 interaction terms)  
- Proper model comparison (combined baseline vs combined interaction)
- Statistical significance testing via likelihood ratio tests

Usage:
    python src/modeling/advanced.py

Outputs:
    - results/outputs/advanced/interaction_models.csv
    - results/outputs/advanced/interaction_effects.csv
    - results/outputs/advanced/model_comparisons.csv
"""

import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


class AdvancedModeling:
    """Advanced stratified survival modeling for LLM robustness analysis."""
    
    def __init__(self):
        self.models_data = {}
        self.interaction_results = {}
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
        print("📊 LOADING DATA FROM data/raw/ AND data/processed/")
        print("=" * 50)
        
        # Load cleaned reference data
        cleaned_data_path = 'data/raw/cleaned_data - cleaned_data.csv'
        if not os.path.exists(cleaned_data_path):
            raise FileNotFoundError(f"Required file not found: {cleaned_data_path}")
            
        cleaned_data = pd.read_csv(cleaned_data_path)
        cleaned_data['subject_cluster'] = cleaned_data['subject'].apply(self._map_subject_to_cluster)
        
        # Use the original level as difficulty_level
        cleaned_data['difficulty_level'] = cleaned_data['level']
        
        # Build lookup for questions
        question_lookup = {}
        for _, row in cleaned_data.iterrows():
            question_lookup[str(row['question']).strip()] = {
                'subject_cluster': row['subject_cluster'],
                'difficulty_level': row['difficulty_level']
            }
        
        # Load processed model data
        processed_dir = 'data/processed'
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
        
        model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        
        for model_name in tqdm(model_dirs, desc="Loading models"):
            model_path = os.path.join(processed_dir, model_name)
            
            # Load long format data for survival analysis
            long_path = os.path.join(model_path, f'{model_name}_long.csv')
            
            if os.path.exists(long_path):
                long_df = pd.read_csv(long_path)
                
                # Data already has subject_cluster, just add difficulty_level from level
                long_df['difficulty_level'] = long_df['level']
                
                self.models_data[model_name] = long_df
                
        print(f"✅ Loaded {len(self.models_data)} models")
        return self.models_data



    def fit_interaction_models(self):
        """Fit Cox PH model with drift × model interactions (combined all models)."""
        print("\n🔗 FITTING COMBINED INTERACTION MODEL")
        print("=" * 40)
        
        # Combine all model data (similar to baseline approach)
        combined_data = []
        
        for model_name in self.models_data.keys():
            try:
                model_data = self.models_data[model_name].copy()
                
                # Add model name as feature
                model_data['model'] = model_name
                
                # Prepare the data with required covariates
                required_cols = ['round', 'failure', 'model', 'subject_cluster', 'difficulty_level',
                               'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                               'cumulative_drift', 'prompt_complexity']
                
                available_cols = [col for col in required_cols if col in model_data.columns]
                filtered_data = model_data[available_cols].copy()
                
                # Drop rows with NaN in critical columns
                filtered_data = filtered_data.dropna(subset=['round', 'failure', 'subject_cluster', 'difficulty_level'])
                
                if len(filtered_data) > 0:
                    combined_data.append(filtered_data)
                    
            except Exception as e:
                print(f"❌ Failed to combine {model_name}: {e}")
                continue
        
        if not combined_data:
            print("❌ No data available for interaction modeling")
            return pd.DataFrame()
        
        # Concatenate all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"📊 Combined dataset: {len(combined_df)} observations from {len(self.models_data)} models")
        
        # Create dummy variables (same structure as baseline)
        model_dummies = pd.get_dummies(combined_df['model'], prefix='model', drop_first=True)
        subject_dummies = pd.get_dummies(combined_df['subject_cluster'], prefix='subject', drop_first=True)
        difficulty_dummies = pd.get_dummies(combined_df['difficulty_level'], prefix='difficulty', drop_first=True)
        
        # Combine base data with dummies
        combined_df = pd.concat([combined_df, model_dummies, subject_dummies, difficulty_dummies], axis=1)
        
        # Base covariates (same as baseline model)
        drift_covariates = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                          'cumulative_drift', 'prompt_complexity']
        model_covariate_cols = model_dummies.columns.tolist()
        subject_covariate_cols = subject_dummies.columns.tolist()
        difficulty_covariate_cols = difficulty_dummies.columns.tolist()
        
        base_covariates = drift_covariates + model_covariate_cols + subject_covariate_cols + difficulty_covariate_cols
        
        # Create interaction terms: drift × model
        interaction_terms = []
        for drift_var in drift_covariates:
            for model_col in model_covariate_cols:
                interaction_name = f'{drift_var}_x_{model_col}'
                combined_df[interaction_name] = combined_df[drift_var] * combined_df[model_col]
                interaction_terms.append(interaction_name)
        
        # Prepare data for modeling
        all_covariates = base_covariates + interaction_terms
        modeling_data = combined_df[all_covariates + ['round', 'failure']].dropna()
        
        if len(modeling_data) < 50 or modeling_data['failure'].sum() < 10:
            print("❌ Insufficient data for interaction modeling")
            return pd.DataFrame()
        
        try:
            # Fit full interaction model
            cph_full = CoxPHFitter()
            cph_full.fit(modeling_data, duration_col='round', event_col='failure', show_progress=False)
            
            # Fit reduced model (baseline structure without interactions)
            reduced_data = modeling_data[base_covariates + ['round', 'failure']]
            cph_reduced = CoxPHFitter()
            cph_reduced.fit(reduced_data, duration_col='round', event_col='failure', show_progress=False)
            
            # Calculate model comparison statistics
            lr_statistic = 2 * (cph_full.log_likelihood_ - cph_reduced.log_likelihood_)
            df = len(interaction_terms)
            lr_pvalue = 1 - stats.chi2.cdf(lr_statistic, df) if df > 0 else np.nan
            
            # Count significant interactions
            significant_interactions = []
            for term in interaction_terms:
                if term in cph_full.params_.index:
                    pval = cph_full.summary.loc[term, 'p']
                    if pval < 0.05:
                        significant_interactions.append(term)
            
            # Store combined results
            result = {
                'model': 'COMBINED_WITH_INTERACTIONS',
                'analysis_type': 'drift_model_interactions',
                'n_observations': len(modeling_data),
                'n_events': modeling_data['failure'].sum(),
                'n_models': len(self.models_data),
                'full_c_index': cph_full.concordance_index_,
                'reduced_c_index': cph_reduced.concordance_index_,
                'full_aic': getattr(cph_full, 'AIC_partial_', np.nan),
                'reduced_aic': getattr(cph_reduced, 'AIC_partial_', np.nan),
                'full_log_likelihood': cph_full.log_likelihood_,
                'reduced_log_likelihood': cph_reduced.log_likelihood_,
                'lr_statistic': lr_statistic,
                'lr_df': df,
                'lr_pvalue': lr_pvalue,
                'n_interaction_terms': len(interaction_terms),
                'n_significant_interactions': len(significant_interactions),
                'significant_interactions': '; '.join(significant_interactions)
            }
            
            # Add interaction coefficients
            for term in interaction_terms:
                if term in cph_full.params_.index:
                    coef = cph_full.params_[term]
                    pval = cph_full.summary.loc[term, 'p']
                    hr = np.exp(coef)
                    
                    result[f'{term}_coef'] = coef
                    result[f'{term}_pval'] = pval
                    result[f'{term}_hr'] = hr
            
            self.interaction_results = pd.DataFrame([result])
            self.interaction_model = cph_full
            self.baseline_model = cph_reduced
            
            print(f"✅ Combined interaction model fitted successfully")
            print(f"   Full model C-index: {cph_full.concordance_index_:.4f}")
            print(f"   Reduced model C-index: {cph_reduced.concordance_index_:.4f}")
            print(f"   LR test p-value: {lr_pvalue:.6f}")
            print(f"   Significant interactions: {len(significant_interactions)}/{len(interaction_terms)}")
            
            return self.interaction_results
            
        except Exception as e:
            print(f"❌ Interaction modeling failed: {e}")
            return pd.DataFrame()

    def fit_subject_stratified_models(self):
        """Fit subject-stratified Cox models with frailty approximation."""
        print("\n📚 FITTING SUBJECT-STRATIFIED MODELS")
        print("=" * 40)
        
        results = []
        frailty_effects = []
        
        for model_name in tqdm(self.models_data.keys(), desc="Subject stratified"):
            try:
                data = self.models_data[model_name]
                
                # Define covariates
                drift_covariates = [col for col in ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                                                  'cumulative_drift', 'prompt_complexity'] 
                                  if col in data.columns]
                
                if len(drift_covariates) < 2:
                    continue
                
                # Prepare data with subject encoding
                data_encoded = data.copy()
                le_subject = LabelEncoder()
                data_encoded['subject_encoded'] = le_subject.fit_transform(data_encoded['subject_cluster'].astype(str))
                
                survival_cols = drift_covariates + ['round', 'failure', 'subject_encoded']
                model_data = data_encoded[survival_cols].dropna()
                
                if len(model_data) < 20 or model_data['failure'].sum() < 5:
                    continue
                
                # Fit stratified Cox model
                cph = CoxPHFitter()
                cph.fit(model_data, duration_col='round', event_col='failure',
                       strata=['subject_encoded'], show_progress=False)
                
                # Store stratified results
                result = {
                    'model': model_name,
                    'analysis_type': 'subject_stratified',
                    'n_observations': len(model_data),
                    'n_events': model_data['failure'].sum(),
                    'c_index': cph.concordance_index_,
                    'aic': getattr(cph, 'AIC_partial_', np.nan),
                    'n_strata': len(data['subject_cluster'].unique())
                }
                
                # Add coefficients
                for covariate in drift_covariates:
                    if covariate in cph.params_.index:
                        result[f'{covariate}_coef'] = cph.params_[covariate]
                        result[f'{covariate}_pval'] = cph.summary.loc[covariate, 'p']
                        result[f'{covariate}_hr'] = np.exp(cph.params_[covariate])
                
                results.append(result)
                
                # Calculate subject frailty effects
                for subject in data['subject_cluster'].unique():
                    subj_data = data[data['subject_cluster'] == subject]
                    if len(subj_data) > 5 and subj_data['failure'].sum() > 1:
                        try:
                            subj_cph = CoxPHFitter()
                            subj_survival_data = subj_data[drift_covariates + ['round', 'failure']].dropna()
                            subj_cph.fit(subj_survival_data, duration_col='round', event_col='failure', show_progress=False)
                            
                            baseline_hazard = subj_cph.baseline_hazard_
                            hazard_mean = baseline_hazard.mean().iloc[0] if len(baseline_hazard.shape) > 1 else baseline_hazard.mean()
                            
                            frailty_effects.append({
                                'model': model_name,
                                'subject_cluster': subject,
                                'frailty_type': 'subject',
                                'baseline_hazard': float(hazard_mean),
                                'n_observations': len(subj_data),
                                'n_events': subj_data['failure'].sum()
                            })
                        except:
                            continue
                
            except Exception as e:
                print(f"❌ Subject stratified failed {model_name}: {e}")
                continue
        
        self.subject_stratified = pd.DataFrame(results)
        self.subject_frailty = pd.DataFrame(frailty_effects)
        return self.subject_stratified, self.subject_frailty

    def fit_difficulty_stratified_models(self):
        """Fit difficulty-stratified Cox models with frailty approximation."""
        print("\n📈 FITTING DIFFICULTY-STRATIFIED MODELS")
        print("=" * 40)
        
        results = []
        frailty_effects = []
        
        for model_name in tqdm(self.models_data.keys(), desc="Difficulty stratified"):
            try:
                data = self.models_data[model_name]
                
                # Define covariates
                drift_covariates = [col for col in ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                                                  'cumulative_drift', 'prompt_complexity'] 
                                  if col in data.columns]
                
                if len(drift_covariates) < 2:
                    continue
                
                # Prepare data with difficulty encoding
                data_encoded = data.copy()
                le_difficulty = LabelEncoder()
                data_encoded['difficulty_encoded'] = le_difficulty.fit_transform(data_encoded['difficulty_level'].astype(str))
                
                survival_cols = drift_covariates + ['round', 'failure', 'difficulty_encoded']
                model_data = data_encoded[survival_cols].dropna()
                
                if len(model_data) < 20 or model_data['failure'].sum() < 5:
                    continue
                
                # Fit stratified Cox model
                cph = CoxPHFitter()
                cph.fit(model_data, duration_col='round', event_col='failure',
                       strata=['difficulty_encoded'], show_progress=False)
                
                # Store stratified results
                result = {
                    'model': model_name,
                    'analysis_type': 'difficulty_stratified',
                    'n_observations': len(model_data),
                    'n_events': model_data['failure'].sum(),
                    'c_index': cph.concordance_index_,
                    'aic': getattr(cph, 'AIC_partial_', np.nan),
                    'n_strata': len(data['difficulty_level'].unique())
                }
                
                # Add coefficients
                for covariate in drift_covariates:
                    if covariate in cph.params_.index:
                        result[f'{covariate}_coef'] = cph.params_[covariate]
                        result[f'{covariate}_pval'] = cph.summary.loc[covariate, 'p']
                        result[f'{covariate}_hr'] = np.exp(cph.params_[covariate])
                
                results.append(result)
                
                # Calculate difficulty frailty effects
                for difficulty in data['difficulty_level'].unique():
                    diff_data = data[data['difficulty_level'] == difficulty]
                    if len(diff_data) > 5 and diff_data['failure'].sum() > 1:
                        try:
                            diff_cph = CoxPHFitter()
                            diff_survival_data = diff_data[drift_covariates + ['round', 'failure']].dropna()
                            diff_cph.fit(diff_survival_data, duration_col='round', event_col='failure', show_progress=False)
                            
                            baseline_hazard = diff_cph.baseline_hazard_
                            hazard_mean = baseline_hazard.mean().iloc[0] if len(baseline_hazard.shape) > 1 else baseline_hazard.mean()
                            
                            frailty_effects.append({
                                'model': model_name,
                                'difficulty_level': difficulty,
                                'frailty_type': 'difficulty',
                                'baseline_hazard': float(hazard_mean),
                                'n_observations': len(diff_data),
                                'n_events': diff_data['failure'].sum()
                            })
                        except:
                            continue
                
            except Exception as e:
                print(f"❌ Difficulty stratified failed {model_name}: {e}")
                continue
        
        self.difficulty_stratified = pd.DataFrame(results)
        self.difficulty_frailty = pd.DataFrame(frailty_effects)
        return self.difficulty_stratified, self.difficulty_frailty

    def export_results(self, output_dir='results/outputs/advanced'):
        """Export all advanced modeling results to CSV files."""
        print(f"\n💾 EXPORTING RESULTS TO {output_dir}/")
        print("=" * 40)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Interaction modeling results
        if hasattr(self, 'interaction_results'):
            self.interaction_results.to_csv(f'{output_dir}/interaction_models.csv', index=False)
            print(f"✅ Exported interaction_models.csv")
            
            # Extract detailed interaction effects
            interaction_details = []
            for _, row in self.interaction_results.iterrows():
                for col in self.interaction_results.columns:
                    if '_x_' in col and col.endswith('_coef'):
                        interaction_term = col.replace('_coef', '')
                        pval_col = f'{interaction_term}_pval'
                        hr_col = f'{interaction_term}_hr'
                        
                        if pval_col in row and hr_col in row:
                            interaction_details.append({
                                'interaction_term': interaction_term,
                                'coefficient': row[col],
                                'p_value': row[pval_col],
                                'hazard_ratio': row[hr_col],
                                'significant': row[pval_col] < 0.05
                            })
            
            if interaction_details:
                interaction_df = pd.DataFrame(interaction_details)
                interaction_df.to_csv(f'{output_dir}/interaction_effects.csv', index=False)
                print(f"✅ Exported interaction_effects.csv")
        
        # 2. Model comparison (combined baseline vs combined interaction)
        if hasattr(self, 'interaction_results'):
            # The interaction_results already contains the comparison:
            # - full_c_index: Combined interaction model C-index
            # - reduced_c_index: Combined baseline model C-index (equivalent to Stage 1)
            # - lr_pvalue: Statistical significance test
            comparison = self.interaction_results[['analysis_type', 'n_observations', 'n_events', 'n_models',
                                                  'full_c_index', 'reduced_c_index', 'lr_pvalue']].copy()
            comparison['c_index_improvement'] = comparison['full_c_index'] - comparison['reduced_c_index']
            comparison['improvement_percentage'] = (comparison['c_index_improvement'] / comparison['reduced_c_index']) * 100
            
            comparison.to_csv(f'{output_dir}/model_comparisons.csv', index=False)
            print(f"✅ Exported model_comparisons.csv")
        
        print(f"📁 All results saved to: {output_dir}/")

    def run_complete_analysis(self):
        """Run complete advanced modeling pipeline."""
        print("🚀 STARTING ADVANCED MODELING PIPELINE")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Fit interaction models
        self.fit_interaction_models()
        
        # Export results
        self.export_results()
        
        print("\n✅ ADVANCED MODELING COMPLETED!")
        print("=" * 30)
        print(f"📊 Analyzed {len(self.models_data)} models")
        print(f"📁 Results saved to results/outputs/advanced/")
        
        return {
            'interactions': self.interaction_results if hasattr(self, 'interaction_results') else None,
            'comparison': 'combined_baseline_vs_combined_interaction'
        }


def main():
    """Main execution function."""
    advanced = AdvancedModeling()
    results = advanced.run_complete_analysis()
    return results


if __name__ == "__main__":
    main()