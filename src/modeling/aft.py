#!/usr/bin/env python3
"""
Accelerated Failure Time (AFT) Models for LLM Conversation Analysis
Implements various AFT models (Log-Normal, Log-Logistic, Weibull) as alternatives 
to Cox regression when proportional hazards assumption is violated
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import (WeibullAFTFitter, LogNormalAFTFitter, 
                      LogLogisticAFTFitter)
from lifelines.utils import concordance_index
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class AFTModeling:
    """Accelerated Failure Time Models for LLM Analysis"""
    
    def __init__(self):
        self.models_data = {}
        self.fitted_models = {}
        self.model_datasets = {}  # Store datasets used for each model
        self.results = {}
        self.combined_data = None
        self.subject_clusters = self._create_subject_clusters()
        
    def _create_subject_clusters(self):
        """Create 7 meaningful subject clusters from 39 individual subjects."""
        return {
            'STEM': [
                'mathematics', 'statistics', 'abstract_algebra',
                'physics', 'conceptual_physics', 'astronomy', 
                'chemistry', 'computer_science', 'computer_security', 
                'machine_learning', 'electrical_engineering'
            ],
            'Medical_Health': [
                'medicine', 'clinical_knowledge', 'medical_genetics',
                'biology', 'anatomy', 'virology',
                'nutrition', 'human_sexuality'
            ],
            'Social_Sciences': [
                'psychology', 'sociology',
                'moral_scenarios', 'global_facts'
            ],
            'Humanities': [
                'philosophy', 'jurisprudence', 'logical_fallacies',
                'moral_disputes', 'world_religions'
            ],
            'Business_Economics': [
                'business_ethics', 'econometrics', 'microeconomics',
                'macroeconomics', 'management', 'marketing'
            ],
            'Arts_Culture': [
                'prehistory', 'us_foreign_policy', 'high_school_geography',
                'miscellaneous', 'high_school_government_and_politics'
            ],
            'Education': [
                'elementary_mathematics', 'high_school_biology',
                'high_school_chemistry', 'high_school_mathematics',
                'high_school_physics', 'high_school_world_history'
            ]
        }
        
    def _map_subject_to_cluster(self, subject):
        """Map individual subject to cluster."""
        for cluster_name, subjects in self.subject_clusters.items():
            if subject in subjects:
                return cluster_name
        return 'Other'
    
    def _prepare_aft_data(self):
        """Prepare data for AFT models (numeric features only)"""
        # Base numeric columns
        numeric_cols = ['round', 'failure', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                       'cumulative_drift', 'prompt_complexity']
        
        # Get dummy variable columns (these are already numeric)
        model_cols = [col for col in self.combined_data.columns if col.startswith('model_')]
        subject_cols = [col for col in self.combined_data.columns if col.startswith('subject_') and col != 'subject_cluster']
        difficulty_cols = [col for col in self.combined_data.columns if col.startswith('difficulty_') and col != 'difficulty_level']
        
        # Combine all numeric columns
        final_cols = numeric_cols + model_cols + subject_cols + difficulty_cols
        aft_data = self.combined_data[final_cols].copy()
        
        # Convert boolean columns to int for AFT models
        bool_cols = aft_data.select_dtypes(include=['bool']).columns
        aft_data[bool_cols] = aft_data[bool_cols].astype(int)
        
        return aft_data
    
    def _prepare_interaction_data(self):
        """Prepare data for AFT models with interactions"""
        # Start with base data
        interaction_data = self.combined_data.copy()
        
        # Add significant interaction terms
        significant_interactions = [
            ('prompt_to_prompt_drift', 'model_mistral_large'),
            ('context_to_prompt_drift', 'model_gpt_oss'),
            ('cumulative_drift', 'model_llama_33'),
            ('prompt_complexity', 'model_qwen_3'),
            ('context_to_prompt_drift', 'model_deepseek_r1')
        ]
        
        interaction_count = 0
        for drift_var, model_var in significant_interactions:
            if drift_var in interaction_data.columns and model_var in interaction_data.columns:
                interaction_name = f"{drift_var}_x_{model_var}"
                interaction_data[interaction_name] = interaction_data[drift_var] * interaction_data[model_var]
                interaction_count += 1
        
        print(f"Created {interaction_count} interaction terms")
        
        # Prepare final dataset (numeric only)
        numeric_cols = ['round', 'failure', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                       'cumulative_drift', 'prompt_complexity']
        model_cols = [col for col in interaction_data.columns if col.startswith('model_')]
        subject_cols = [col for col in interaction_data.columns if col.startswith('subject_') and col != 'subject_cluster']
        difficulty_cols = [col for col in interaction_data.columns if col.startswith('difficulty_') and col != 'difficulty_level']
        interaction_cols = [col for col in interaction_data.columns if '_x_' in col]
        
        final_cols = numeric_cols + model_cols + subject_cols + difficulty_cols + interaction_cols
        aft_data = interaction_data[final_cols].copy()
        
        # Convert boolean columns to int for AFT models
        bool_cols = aft_data.select_dtypes(include=['bool']).columns
        aft_data[bool_cols] = aft_data[bool_cols].astype(int)
        
        return aft_data
        
    def load_data(self):
        """Load and prepare data for AFT modeling"""
        print("📊 LOADING TRAINING DATA FOR AFT MODELING")
        print("=" * 50)

        try:
            # Load processed model data from TRAIN split ONLY
            processed_dir = 'data/processed/train'
            if not os.path.exists(processed_dir):
                raise FileNotFoundError(f"Train data not found: {processed_dir}. Run train/test split first using --stage data_split!")
            
            model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
            
            self.models_data = {}
            combined_data = []
            
            for model_name in tqdm(model_dirs, desc="Loading models"):
                model_path = os.path.join(processed_dir, model_name)
                
                # Load long format data
                long_path = os.path.join(model_path, f'{model_name}_long.csv')
                
                if os.path.exists(long_path):
                    long_df = pd.read_csv(long_path)
                    long_df['model'] = model_name
                    self.models_data[model_name] = {'long': long_df}
                    
                    required_cols = ['round', 'failure', 'conversation_id', 'model', 
                                   'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                                   'cumulative_drift', 'prompt_complexity', 'level', 
                                   'subject', 'subject_cluster']
                    
                    available_cols = [col for col in required_cols if col in long_df.columns]
                    model_subset = long_df[available_cols].copy()
                    
                    # Add difficulty_level and ensure subject_cluster is available
                    if 'level' in model_subset.columns:
                        model_subset['difficulty_level'] = model_subset['level']
                    if 'subject_cluster' not in model_subset.columns and 'subject' in model_subset.columns:
                        model_subset['subject_cluster'] = model_subset['subject'].apply(self._map_subject_to_cluster)
                    
                    model_subset = model_subset.dropna(subset=['round', 'failure', 'prompt_to_prompt_drift', 
                                                             'context_to_prompt_drift', 'cumulative_drift', 
                                                             'prompt_complexity'])
                    
                    if len(model_subset) > 0:
                        combined_data.append(model_subset)
            
            if combined_data:
                self.combined_data = pd.concat(combined_data, ignore_index=True)
                
                # Create model dummy variables
                model_dummies = pd.get_dummies(self.combined_data['model'], prefix='model', drop_first=True)
                self.combined_data = pd.concat([self.combined_data, model_dummies], axis=1)
                
                # Create subject cluster dummy variables
                if 'subject_cluster' in self.combined_data.columns:
                    subject_dummies = pd.get_dummies(self.combined_data['subject_cluster'], 
                                                   prefix='subject', drop_first=True)
                    self.combined_data = pd.concat([self.combined_data, subject_dummies], axis=1)
                    print(f"✅ Added {subject_dummies.shape[1]} subject cluster features")
                
                # Create difficulty level dummy variables
                if 'difficulty_level' in self.combined_data.columns:
                    difficulty_dummies = pd.get_dummies(self.combined_data['difficulty_level'], 
                                                       prefix='difficulty', drop_first=True)
                    self.combined_data = pd.concat([self.combined_data, difficulty_dummies], axis=1)
                    print(f"✅ Added {difficulty_dummies.shape[1]} difficulty level features")
                
                print(f"✅ Loaded {len(self.models_data)} models")
                print(f"✅ Combined dataset: {self.combined_data.shape[0]} observations")
                print(f"✅ Total features: {self.combined_data.shape[1]} columns")
                print(f"✅ Event rate: {self.combined_data['failure'].mean()*100:.1f}%")
                return True
            else:
                print("❌ No valid data found")
                return False
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def fit_weibull_aft(self):
        """Fit Weibull AFT model"""
        print("\n🔧 FITTING WEIBULL AFT MODEL")
        print("=" * 40)
        
        try:
            # Prepare numeric data for AFT
            aft_data = self._prepare_aft_data()
            
            waft = WeibullAFTFitter(penalizer=0.01)
            waft.fit(aft_data, duration_col='round', event_col='failure')
            
            self.fitted_models['weibull_aft'] = waft
            self.model_datasets['weibull_aft'] = aft_data
            
            result = {
                'model_type': 'Weibull AFT',
                'n_observations': len(aft_data),
                'n_events': aft_data['failure'].sum(),
                'c_index': waft.concordance_index_,
                'aic': waft.AIC_,
                'bic': waft.BIC_,
                'log_likelihood': waft.log_likelihood_,
                'rho_param': getattr(waft, 'rho_', np.nan),
                'lambda_param': getattr(waft, 'lambda_', np.nan)
            }
            
            print(f"✅ Weibull AFT fitted successfully")
            print(f"   C-index: {waft.concordance_index_:.4f}")
            print(f"   AIC: {waft.AIC_:.2f}")
            print(f"   BIC: {waft.BIC_:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Weibull AFT fitting failed: {e}")
            return None
    
    def fit_lognormal_aft(self):
        """Fit Log-Normal AFT model"""
        print("\n🔧 FITTING LOG-NORMAL AFT MODEL")
        print("=" * 40)
        
        try:
            # Prepare numeric data for AFT
            aft_data = self._prepare_aft_data()
            
            lnaft = LogNormalAFTFitter(penalizer=0.01)
            lnaft.fit(aft_data, duration_col='round', event_col='failure')
            
            self.fitted_models['lognormal_aft'] = lnaft
            self.model_datasets['lognormal_aft'] = aft_data
            
            result = {
                'model_type': 'Log-Normal AFT',
                'n_observations': len(aft_data),
                'n_events': aft_data['failure'].sum(),
                'c_index': lnaft.concordance_index_,
                'aic': lnaft.AIC_,
                'bic': lnaft.BIC_,
                'log_likelihood': lnaft.log_likelihood_,
                'mu_param': getattr(lnaft, 'mu_', np.nan),
                'sigma_param': getattr(lnaft, 'sigma_', np.nan)
            }
            
            print(f"✅ Log-Normal AFT fitted successfully")
            print(f"   C-index: {lnaft.concordance_index_:.4f}")
            print(f"   AIC: {lnaft.AIC_:.2f}")
            print(f"   BIC: {lnaft.BIC_:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Log-Normal AFT fitting failed: {e}")
            return None
    
    def fit_loglogistic_aft(self):
        """Fit Log-Logistic AFT model"""
        print("\n🔧 FITTING LOG-LOGISTIC AFT MODEL")
        print("=" * 40)
        
        try:
            # Prepare numeric data for AFT
            aft_data = self._prepare_aft_data()
            
            llaft = LogLogisticAFTFitter(penalizer=0.01)
            llaft.fit(aft_data, duration_col='round', event_col='failure')
            
            self.fitted_models['loglogistic_aft'] = llaft
            self.model_datasets['loglogistic_aft'] = aft_data
            
            result = {
                'model_type': 'Log-Logistic AFT',
                'n_observations': len(aft_data),
                'n_events': aft_data['failure'].sum(),
                'c_index': llaft.concordance_index_,
                'aic': llaft.AIC_,
                'bic': llaft.BIC_,
                'log_likelihood': llaft.log_likelihood_,
                'alpha_param': getattr(llaft, 'alpha_', np.nan),
                'beta_param': getattr(llaft, 'beta_', np.nan)
            }
            
            print(f"✅ Log-Logistic AFT fitted successfully")
            print(f"   C-index: {llaft.concordance_index_:.4f}")
            print(f"   AIC: {llaft.AIC_:.2f}")
            print(f"   BIC: {llaft.BIC_:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Log-Logistic AFT fitting failed: {e}")
            return None
    

    
    def fit_aft_with_interactions(self, model_type='weibull'):
        """Fit AFT model with interaction terms"""
        print(f"\n⚡ FITTING {model_type.upper()} AFT WITH INTERACTIONS")
        print("=" * 50)
        
        try:
            # Select AFT model
            if model_type.lower() == 'weibull':
                aft_model = WeibullAFTFitter(penalizer=0.02)
            elif model_type.lower() == 'lognormal':
                aft_model = LogNormalAFTFitter(penalizer=0.02)
            elif model_type.lower() == 'loglogistic':
                aft_model = LogLogisticAFTFitter(penalizer=0.02)
            else:
                aft_model = WeibullAFTFitter(penalizer=0.02)
            
            # Prepare interaction data for AFT
            aft_data = self._prepare_interaction_data()
            
            # Count interaction terms
            interaction_count = len([col for col in aft_data.columns if '_x_' in col])
            
            # Fit model
            aft_model.fit(aft_data, duration_col='round', event_col='failure')
            
            model_key = f"{model_type}_aft_interactions"
            self.fitted_models[model_key] = aft_model
            self.model_datasets[model_key] = aft_data  # Store the interaction dataset
            
            result = {
                'model_type': f'{model_type.title()} AFT + Interactions',
                'n_observations': len(aft_data),
                'n_events': aft_data['failure'].sum(),
                'n_interactions': interaction_count,
                'c_index': aft_model.concordance_index_,
                'aic': aft_model.AIC_,
                'bic': aft_model.BIC_,
                'log_likelihood': aft_model.log_likelihood_
            }
            
            print(f"✅ {model_type.title()} AFT + Interactions fitted successfully")
            print(f"   C-index: {aft_model.concordance_index_:.4f}")
            print(f"   AIC: {aft_model.AIC_:.2f}")
            print(f"   Interactions: {interaction_count}")
            
            # Store coefficients
            coef_df = aft_model.summary.copy()
            coef_df['model_type'] = f'{model_type}_aft_interactions'
            coef_df['analysis_type'] = 'aft_interactions'
            
            if 'interaction_coefficients' not in self.results:
                self.results['interaction_coefficients'] = []
            self.results['interaction_coefficients'].append(coef_df)
            
            return result
            
        except Exception as e:
            print(f"❌ {model_type.title()} AFT + Interactions fitting failed: {e}")
            return None
    
    def compare_aft_models(self):
        """Compare all fitted AFT models"""
        print("\n📊 AFT MODEL COMPARISON")
        print("=" * 50)
        
        # Collect all results
        all_results = []
        
        # Basic AFT models
        models_to_fit = [
            ('weibull', self.fit_weibull_aft),
            ('lognormal', self.fit_lognormal_aft),
            ('loglogistic', self.fit_loglogistic_aft)
        ]
        
        for model_name, fit_func in models_to_fit:
            result = fit_func()
            if result:
                all_results.append(result)
        
        # AFT models with interactions
        for model_type in ['weibull', 'lognormal', 'loglogistic']:
            result = self.fit_aft_with_interactions(model_type)
            if result:
                all_results.append(result)
        
        if not all_results:
            print("❌ No models fitted successfully")
            return None
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        print("AFT Model Performance Comparison:")
        print(comparison_df[['model_type', 'c_index', 'aic', 'bic']].to_string(index=False))
        
        # Find best models
        best_aic = comparison_df.loc[comparison_df['aic'].idxmin()]
        best_bic = comparison_df.loc[comparison_df['bic'].idxmin()]
        best_cindex = comparison_df.loc[comparison_df['c_index'].idxmax()]
        
        print(f"\n🏆 BEST MODELS:")
        print(f"   Best AIC: {best_aic['model_type']} (AIC: {best_aic['aic']:.2f})")
        print(f"   Best BIC: {best_bic['model_type']} (BIC: {best_bic['bic']:.2f})")
        print(f"   Best C-index: {best_cindex['model_type']} (C-index: {best_cindex['c_index']:.4f})")
        
        self.results['model_comparison'] = comparison_df
        return comparison_df
    
    def perform_residual_analysis(self):
        """Perform alternative model diagnostics for AFT models"""
        print("\n🔍 AFT MODEL DIAGNOSTIC ANALYSIS")
        print("=" * 50)
        
        diagnostic_results = []
        
        for model_name, model in self.fitted_models.items():
            try:
                print(f"\nAnalyzing model diagnostics for {model_name}...")
                
                # Get the correct dataset for this model
                model_data = self.model_datasets.get(model_name, self.combined_data)
                
                # AFT models don't have traditional residuals like Cox models
                # Instead, we'll do alternative diagnostics
                
                # 1. Model predictions vs actual survival times
                if hasattr(model, 'predict_expectation'):
                    expectations = model.predict_expectation(model_data)
                    actual_times = model_data['round'].values
                    
                    # Calculate prediction accuracy metrics
                    mae = np.mean(np.abs(expectations - actual_times))
                    rmse = np.sqrt(np.mean((expectations - actual_times)**2))
                    correlation = np.corrcoef(expectations, actual_times)[0, 1]
                    
                    print(f"  ✅ Prediction Analysis:")
                    print(f"     Mean Absolute Error: {mae:.2f}")
                    print(f"     Root Mean Square Error: {rmse:.2f}")
                    print(f"     Correlation (pred vs actual): {correlation:.4f}")
                    
                    diagnostic_results.append({
                        'model': model_name,
                        'diagnostic_type': 'prediction_accuracy',
                        'mae': mae,
                        'rmse': rmse,
                        'correlation': correlation,
                        'c_index': model.concordance_index_,
                        'aic': model.AIC_,
                        'bic': model.BIC_
                    })
                
                # 2. Check for outliers in predictions
                elif hasattr(model, 'predict_survival_function'):
                    # Use survival function approach
                    survival_func = model.predict_survival_function(model_data)
                    if survival_func is not None and len(survival_func.columns) > 0:
                        print(f"  ✅ Survival Function Analysis:")
                        print(f"     Generated survival curves for {len(survival_func.columns)} subjects")
                        print(f"     Time points analyzed: {len(survival_func.index)}")
                        
                        diagnostic_results.append({
                            'model': model_name,
                            'diagnostic_type': 'survival_function',
                            'n_subjects': len(survival_func.columns),
                            'n_timepoints': len(survival_func.index),
                            'c_index': model.concordance_index_,
                            'aic': model.AIC_,
                            'bic': model.BIC_
                        })
                
                # 3. Model fit statistics
                else:
                    print(f"  ✅ Model Fit Statistics:")
                    print(f"     C-index: {model.concordance_index_:.4f}")
                    print(f"     AIC: {model.AIC_:.2f}")
                    print(f"     BIC: {model.BIC_:.2f}")
                    print(f"     Log-likelihood: {model.log_likelihood_:.2f}")
                    
                    diagnostic_results.append({
                        'model': model_name,
                        'diagnostic_type': 'fit_statistics',
                        'c_index': model.concordance_index_,
                        'aic': model.AIC_,
                        'bic': model.BIC_,
                        'log_likelihood': model.log_likelihood_
                    })
                
            except Exception as e:
                print(f"  ⚠️  Diagnostic analysis failed for {model_name}: {e}")
                # Still add basic model info
                diagnostic_results.append({
                    'model': model_name,
                    'diagnostic_type': 'basic_info',
                    'c_index': getattr(model, 'concordance_index_', np.nan),
                    'aic': getattr(model, 'AIC_', np.nan),
                    'bic': getattr(model, 'BIC_', np.nan),
                    'error': str(e)
                })
                continue
        
        if diagnostic_results:
            self.results['diagnostic_analysis'] = pd.DataFrame(diagnostic_results)
            print(f"\n✅ AFT diagnostic analysis completed for {len(diagnostic_results)} models")
            print("📝 Note: AFT models use different diagnostics than Cox models")
            print("   - Prediction accuracy metrics instead of residuals")
            print("   - Survival function analysis for model validation")
        else:
            print("\n⚠️  No diagnostic results available")
        
        return diagnostic_results
    
    def plot_aft_diagnostics(self):
        """Generate diagnostic plots for AFT models"""
        print("\n📈 GENERATING AFT DIAGNOSTIC PLOTS")
        print("=" * 40)
        
        try:
            n_models = len(self.fitted_models)
            if n_models == 0:
                print("❌ No fitted models to plot")
                return
            
            print(f"🔍 Found {n_models} fitted models for diagnostics")
            
            # Calculate grid size
            n_cols = min(3, n_models)
            n_rows = (n_models + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            # Plot diagnostics for each model
            for i, (model_name, model) in enumerate(self.fitted_models.items()):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                print(f"📊 Creating diagnostic for {model_name}")
                
                try:
                    # Get the correct dataset for this model
                    model_data = self.model_datasets.get(model_name, self.combined_data)
                    
                    # Try to get model predictions
                    if hasattr(model, 'predict_survival_function'):
                        # Use survival function predictions for diagnostic
                        survival_func = model.predict_survival_function(model_data)
                        
                        if survival_func is not None and len(survival_func.columns) > 0:
                            # Plot survival function for first few subjects
                            for j, col in enumerate(survival_func.columns[:5]):  # First 5 subjects
                                ax.plot(survival_func.index, survival_func[col], alpha=0.3)
                            
                            ax.set_xlabel('Time')
                            ax.set_ylabel('Survival Probability')
                            ax.set_title(f'{model_name}\nSample Survival Curves')
                            ax.grid(True, alpha=0.3)
                            ax.set_ylim(0, 1)
                        else:
                            ax.text(0.5, 0.5, 'No survival predictions available', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{model_name}\n(No predictions)')
                    
                    elif hasattr(model, 'predict_expectation'):
                        # Use expected lifetime predictions
                        expectations = model.predict_expectation(model_data)
                        
                        if expectations is not None and len(expectations) > 0:
                            # Histogram of predicted lifetimes
                            ax.hist(expectations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                            ax.set_xlabel('Predicted Lifetime')
                            ax.set_ylabel('Frequency')
                            ax.set_title(f'{model_name}\nPredicted Lifetimes Distribution')
                            ax.grid(True, alpha=0.3)
                        else:
                            ax.text(0.5, 0.5, 'No expectations available', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{model_name}\n(No expectations)')
                    
                    else:
                        # Simple diagnostic - show model AIC/BIC if available
                        model_info = []
                        if hasattr(model, 'AIC_'):
                            model_info.append(f'AIC: {model.AIC_:.1f}')
                        if hasattr(model, 'BIC_'):
                            model_info.append(f'BIC: {model.BIC_:.1f}')
                        if hasattr(model, 'concordance_index_'):
                            model_info.append(f'C-index: {model.concordance_index_:.4f}')
                        
                        info_text = '\n'.join(model_info) if model_info else 'Model fitted successfully'
                        ax.text(0.5, 0.5, info_text, ha='center', va='center', 
                               transform=ax.transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                        ax.set_title(f'{model_name}\nModel Summary')
                        ax.axis('off')
                
                except Exception as e:
                    print(f"⚠️  Error with {model_name}: {e}")
                    ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                           ha='center', va='center', transform=ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                    ax.set_title(f'{model_name}\n(Error)')
                    ax.axis('off')
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Create AFT directory if it doesn't exist
            os.makedirs('results/figures/aft', exist_ok=True)
            save_path = 'results/figures/aft/aft_diagnostics.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Close to avoid display issues
            
            print(f"✅ AFT diagnostic plots saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_model_comparison(self):
        """Plot model comparison metrics"""
        print("\n📊 PLOTTING MODEL COMPARISON")
        print("=" * 40)
        
        try:
            if 'model_comparison' not in self.results:
                print("❌ No model comparison data available")
                return
            
            comp_df = self.results['model_comparison']
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # AIC comparison
            ax1 = axes[0]
            bars1 = ax1.bar(range(len(comp_df)), comp_df['aic'])
            ax1.set_xticks(range(len(comp_df)))
            ax1.set_xticklabels(comp_df['model_type'], rotation=45, ha='right')
            ax1.set_ylabel('AIC')
            ax1.set_title('Model Comparison by AIC\n(Lower is Better)')
            ax1.grid(True, alpha=0.3)
            
            # Highlight best AIC
            best_aic_idx = comp_df['aic'].idxmin()
            bars1[best_aic_idx].set_color('green')
            bars1[best_aic_idx].set_alpha(0.8)
            
            # BIC comparison
            ax2 = axes[1]
            bars2 = ax2.bar(range(len(comp_df)), comp_df['bic'])
            ax2.set_xticks(range(len(comp_df)))
            ax2.set_xticklabels(comp_df['model_type'], rotation=45, ha='right')
            ax2.set_ylabel('BIC')
            ax2.set_title('Model Comparison by BIC\n(Lower is Better)')
            ax2.grid(True, alpha=0.3)
            
            # Highlight best BIC
            best_bic_idx = comp_df['bic'].idxmin()
            bars2[best_bic_idx].set_color('green')
            bars2[best_bic_idx].set_alpha(0.8)
            
            # C-index comparison
            ax3 = axes[2]
            bars3 = ax3.bar(range(len(comp_df)), comp_df['c_index'])
            ax3.set_xticks(range(len(comp_df)))
            ax3.set_xticklabels(comp_df['model_type'], rotation=45, ha='right')
            ax3.set_ylabel('C-index')
            ax3.set_title('Model Comparison by C-index\n(Higher is Better)')
            ax3.grid(True, alpha=0.3)
            
            # Highlight best C-index
            best_cindex_idx = comp_df['c_index'].idxmax()
            bars3[best_cindex_idx].set_color('green')
            bars3[best_cindex_idx].set_alpha(0.8)
            
            plt.tight_layout()
            
            # Create AFT directory if it doesn't exist
            os.makedirs('results/figures/aft', exist_ok=True)
            save_path = 'results/figures/aft/aft_model_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Close to avoid display issues
            
            print(f"✅ Model comparison plots saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Model comparison plotting failed: {e}")
    
    def save_results(self):
        """Save all AFT results to CSV files"""
        print("\n💾 SAVING AFT RESULTS")
        print("=" * 30)
        
        try:
            os.makedirs('results/outputs/aft', exist_ok=True)
            
            # 1. Save model comparison (main results)
            if 'model_comparison' in self.results:
                self.results['model_comparison'].to_csv(
                    'results/outputs/aft/model_comparison.csv', index=False)
                print("✅ Model comparison saved")
            
            # 2. Save residual analysis
            if 'residual_analysis' in self.results:
                self.results['residual_analysis'].to_csv(
                    'results/outputs/aft/residual_analysis.csv', index=False)
                print("✅ Residual analysis saved")
            
            # 3. Save all model coefficients and summaries
            all_coefficients = []
            model_performance = []
                        
            for model_name, model in self.fitted_models.items():
                try:
                    # Model coefficients
                    summary_df = model.summary.copy()
                    summary_df['model_name'] = model_name
                    summary_df['feature'] = summary_df.index
                    summary_df = summary_df.reset_index(drop=True)
                    all_coefficients.append(summary_df)
                    
                    # Model performance metrics
                    performance = {
                        'model_name': model_name,
                        'model_type': model_name.replace('_aft', '').replace('_interactions', ''),
                        'c_index': getattr(model, 'concordance_index_', np.nan),
                        'aic': getattr(model, 'AIC_', np.nan),
                        'bic': getattr(model, 'BIC_', np.nan),
                        'log_likelihood': getattr(model, 'log_likelihood_', np.nan),
                        'n_observations': len(self.combined_data) if hasattr(self, 'combined_data') else np.nan,
                        'n_events': self.combined_data['failure'].sum() if hasattr(self, 'combined_data') else np.nan,
                        'event_rate': self.combined_data['failure'].mean() if hasattr(self, 'combined_data') else np.nan
                    }
                    model_performance.append(performance)
                    
                except Exception as e:
                    print(f"⚠️  Warning: Could not save {model_name}: {e}")
                    continue
            
            # Save combined coefficients
            if all_coefficients:
                combined_coef_df = pd.concat(all_coefficients, ignore_index=True)
                combined_coef_df.to_csv('results/outputs/aft/all_coefficients.csv', index=False)
                print("✅ All model coefficients saved")
            
            # Save model performance summary
            if model_performance:
                performance_df = pd.DataFrame(model_performance)
                performance_df.to_csv('results/outputs/aft/model_performance.csv', index=False)
                print("✅ Model performance summary saved")
            
            # 4. Save interaction coefficients (if available)
            if 'interaction_coefficients' in self.results:
                interaction_combined = []
                for coef_df in self.results['interaction_coefficients']:
                    if len(coef_df) > 0:
                        interaction_combined.append(coef_df)
                
                if interaction_combined:
                    interaction_df = pd.concat(interaction_combined, ignore_index=True)
                    interaction_df.to_csv('results/outputs/aft/interaction_coefficients.csv', index=False)
                    print("✅ Interaction coefficients saved")
            
            # 5. Save feature importance ranking
            if hasattr(self, 'fitted_models') and len(self.fitted_models) > 0:
                # Use the best model for feature importance
                best_model_name = None
                best_c_index = 0
                
                for name, model in self.fitted_models.items():
                    c_idx = getattr(model, 'concordance_index_', 0)
                    if c_idx > best_c_index:
                        best_c_index = c_idx
                        best_model_name = name
                
                if best_model_name:
                    best_model = self.fitted_models[best_model_name]
                    if hasattr(best_model, 'summary'):
                        feature_importance = best_model.summary.copy()
                        feature_importance['abs_coef'] = abs(feature_importance['coef'])
                        feature_importance['acceleration_factor'] = np.exp(feature_importance['coef'])
                        feature_importance['effect_direction'] = feature_importance['coef'].apply(
                            lambda x: 'Accelerates Failure' if x < 0 else 'Delays Failure')
                        feature_importance['significance'] = feature_importance['p'].apply(
                            lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
                        feature_importance['feature'] = feature_importance.index
                        feature_importance['best_model'] = best_model_name
                        feature_importance = feature_importance.sort_values('abs_coef', ascending=False)
                        
                        feature_importance.to_csv('results/outputs/aft/feature_importance.csv', index=False)
                        print("✅ Feature importance ranking saved")
            
            # 6. Save model rankings summary
            if 'model_comparison' in self.results:
                comp_df = self.results['model_comparison']
                
                # Create model ranking summary
                ranking_summary = []
                
                # Best by each metric
                if 'c_index' in comp_df.columns:
                    best_c_index = comp_df.loc[comp_df['c_index'].idxmax()]
                    ranking_summary.append({
                        'metric': 'c_index',
                        'best_model': best_c_index['model_type'],
                        'best_value': best_c_index['c_index'],
                        'interpretation': 'Higher is better - Predictive accuracy'
                    })
                
                if 'aic' in comp_df.columns:
                    best_aic = comp_df.loc[comp_df['aic'].idxmin()]
                    ranking_summary.append({
                        'metric': 'aic',
                        'best_model': best_aic['model_type'],
                        'best_value': best_aic['aic'],
                        'interpretation': 'Lower is better - Model parsimony'
                    })
                
                if 'bic' in comp_df.columns:
                    best_bic = comp_df.loc[comp_df['bic'].idxmin()]
                    ranking_summary.append({
                        'metric': 'bic',
                        'best_model': best_bic['model_type'],
                        'best_value': best_bic['bic'],
                        'interpretation': 'Lower is better - Model parsimony with sample size penalty'
                    })
                
                if ranking_summary:
                    ranking_df = pd.DataFrame(ranking_summary)
                    ranking_df.to_csv('results/outputs/aft/model_rankings.csv', index=False)
                    print("✅ Model rankings summary saved")
            
            print("📁 All AFT results saved to results/outputs/aft/")
            print(f"📊 Files saved: model_comparison.csv, model_performance.csv, all_coefficients.csv")
            print(f"📈 Additional: feature_importance.csv, model_rankings.csv")
            
        except Exception as e:
            print(f"❌ Saving failed: {e}")
    
    def run_complete_analysis(self):
        """Run complete AFT modeling pipeline"""
        print("🔬 ACCELERATED FAILURE TIME (AFT) MODELING FOR LLM ANALYSIS")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        print("=" * 80)
        
        # Compare all AFT models
        self.compare_aft_models()
        
        print("=" * 80)
        
        # Residual analysis
        self.perform_residual_analysis()
        
        print("=" * 80)
        
        # Generate diagnostic plots
        self.plot_aft_diagnostics()
        
        print("=" * 80)
        
        # Plot model comparison
        self.plot_model_comparison()
        
        print("=" * 80)
        
        # Save results
        self.save_results()
        
        print("=" * 80)
        print("🎉 AFT MODELING COMPLETED!")

def main():
    """Main execution function"""
    aft_analysis = AFTModeling()
    aft_analysis.run_complete_analysis()

if __name__ == "__main__":
    main()