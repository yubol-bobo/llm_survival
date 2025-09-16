#!/usr/bin/env python3
"""
Weibull Survival Modeling for LLM Conversation Analysis
Implements Weibull regression as an alternative to Cox regression
when proportional hazards assumption is violated
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import WeibullFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class WeibullModeling:
    """Weibull Survival Modeling for LLM Analysis"""
    
    def __init__(self):
        self.models_data = {}
        self.fitted_models = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare data for Weibull modeling"""
        print("📊 LOADING DATA FOR WEIBULL MODELING")
        print("=" * 50)
        
        try:
            # Load processed model data
            processed_dir = 'data/processed'
            if not os.path.exists(processed_dir):
                raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
            
            model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
            
            self.models_data = {}
            
            for model_name in tqdm(model_dirs, desc="Loading models"):
                model_path = os.path.join(processed_dir, model_name)
                
                # Load long format data
                long_path = os.path.join(model_path, f'{model_name}_long.csv')
                
                if os.path.exists(long_path):
                    long_df = pd.read_csv(long_path)
                    self.models_data[model_name] = {'long': long_df}
            
            print(f"✅ Loaded {len(self.models_data)} models")
            
            # Also create combined dataset for AFT regression
            self._create_combined_data()
            return True
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def _create_combined_data(self):
        """Create combined dataset for AFT regression"""
        combined_data = []
        
        for model_name, model_data in self.models_data.items():
            long_df = model_data['long'].copy()
            long_df['model'] = model_name
            
            required_cols = ['round', 'failure', 'conversation_id', 'model', 
                           'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                           'cumulative_drift', 'prompt_complexity']
            
            available_cols = [col for col in required_cols if col in long_df.columns]
            model_subset = long_df[available_cols].copy()
            model_subset = model_subset.dropna()
            
            if len(model_subset) > 0:
                combined_data.append(model_subset)
        
        if combined_data:
            self.combined_data = pd.concat(combined_data, ignore_index=True)
            
            # Create model dummy variables
            model_dummies = pd.get_dummies(self.combined_data['model'], prefix='model', drop_first=True)
            self.combined_data = pd.concat([self.combined_data, model_dummies], axis=1)
            
            print(f"✅ Combined dataset: {self.combined_data.shape[0]} observations")
            print(f"✅ Event rate: {self.combined_data['failure'].mean()*100:.1f}%")
    
    def fit_weibull_individual_models(self):
        """Fit individual Weibull models for each LLM"""
        print("\n🔧 FITTING INDIVIDUAL WEIBULL MODELS")
        print("=" * 50)
        
        individual_results = []
        
        for model_name, model_data in self.models_data.items():
            try:
                print(f"\nFitting Weibull model for {model_name}...")
                
                # Use long format data
                long_df = model_data['long'].copy()
                
                # Remove missing values
                long_df = long_df.dropna(subset=['round', 'failure'])
                
                if len(long_df) < 10:
                    print(f"⚠️  Insufficient data for {model_name}")
                    continue
                
                # Fit Weibull model
                wf = WeibullFitter()
                wf.fit(long_df['round'], long_df['failure'])
                
                # Store fitted model
                self.fitted_models[model_name] = wf
                
                # Extract parameters
                lambda_param = wf.lambda_
                rho_param = wf.rho_
                
                # Calculate metrics
                try:
                    median_survival = wf.median_survival_time_
                except:
                    median_survival = wf.lambda_ * (np.log(2) ** (1/wf.rho_))
                
                try:
                    mean_survival = wf.lambda_ * np.math.gamma(1 + 1/wf.rho_)
                except:
                    mean_survival = np.nan
                
                # AIC and log-likelihood
                try:
                    aic = wf.AIC_
                except:
                    aic = np.nan
                    
                try:
                    log_likelihood = wf._log_likelihood
                except:
                    try:
                        log_likelihood = wf.log_likelihood_
                    except:
                        log_likelihood = np.nan
                
                individual_results.append({
                    'model': model_name,
                    'n_observations': len(long_df),
                    'n_events': long_df['failure'].sum(),
                    'lambda_param': lambda_param,
                    'rho_param': rho_param,
                    'median_survival': median_survival,
                    'mean_survival': mean_survival,
                    'aic': aic,
                    'log_likelihood': log_likelihood
                })
                
                print(f"  ✅ {model_name}: λ={lambda_param:.4f}, ρ={rho_param:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed for {model_name}: {e}")
                continue
        
        # Save individual results
        self.results['individual_weibull'] = pd.DataFrame(individual_results)
        return self.results['individual_weibull']
    
    def fit_weibull_aft_regression(self):
        """Fit Weibull AFT regression with covariates"""
        print("\n🔧 FITTING WEIBULL AFT REGRESSION")
        print("=" * 50)
        
        try:
            if not hasattr(self, 'combined_data') or self.combined_data is None:
                print("❌ No combined data available for AFT regression")
                return None
            
            print(f"Combined data shape: {self.combined_data.shape}")
            print(f"Events: {self.combined_data['failure'].sum()}/{len(self.combined_data)} ({self.combined_data['failure'].mean()*100:.1f}%)")
            
            # Prepare covariates
            drift_vars = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
            model_vars = [col for col in self.combined_data.columns if col.startswith('model_')]
            
            # Prepare data for AFT regression (exclude non-numeric columns)
            aft_data = self.combined_data.copy()
            numeric_cols = ['round', 'failure', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                           'cumulative_drift', 'prompt_complexity']
            model_cols = [col for col in aft_data.columns if col.startswith('model_')]
            
            # Select only numeric columns
            final_cols = numeric_cols + model_cols
            aft_data = aft_data[final_cols]
            
            print(f"AFT regression using {len(final_cols)} variables")
            
            # AFT regression
            waft = WeibullAFTFitter(penalizer=0.01)
            waft.fit(aft_data, duration_col='round', event_col='failure')
            
            # Store results
            self.fitted_models['combined_aft'] = waft
            
            # Extract results
            aft_results = {
                'model': 'COMBINED_WEIBULL_AFT',
                'n_observations': len(self.combined_data),
                'n_events': self.combined_data['failure'].sum(),
                'concordance_index': waft.concordance_index_,
                'aic': waft.AIC_,
                'log_likelihood': waft.log_likelihood_,
                'rho_param': getattr(waft, 'rho_', getattr(waft, 'params_', {}).get('rho_', np.nan)),
                'lambda_param': getattr(waft, 'lambda_', getattr(waft, 'params_', {}).get('lambda_', np.nan))
            }
            
            print(f"✅ AFT Model fitted successfully")
            print(f"   C-index: {waft.concordance_index_:.4f}")
            print(f"   AIC: {waft.AIC_:.2f}")
            print(f"   ρ (shape): {waft.rho_:.4f}")
            print(f"   λ (scale): {waft.lambda_:.4f}")
            
            # Coefficients analysis
            coef_df = waft.summary.copy()
            coef_df['model'] = 'COMBINED_WEIBULL_AFT'
            coef_df['analysis_type'] = 'weibull_aft_regression'
            
            self.results['aft_regression'] = aft_results
            self.results['aft_coefficients'] = coef_df
            
            return aft_results
            
        except Exception as e:
            print(f"❌ AFT regression failed: {e}")
            return None
    
    def fit_weibull_with_interactions(self):
        """Fit Weibull AFT with interaction terms"""
        print("\n⚡ FITTING WEIBULL AFT WITH INTERACTIONS")
        print("=" * 50)
        
        try:
            if not hasattr(self, 'combined_data') or self.combined_data is None:
                print("❌ No combined data available for interaction modeling")
                return None
            
            # Create interaction data from pre-existing combined data
            interaction_data = self.combined_data.copy()
            
            # Create selected interaction terms (only the most significant ones)
            significant_interactions = [
                ('prompt_to_prompt_drift', 'model_mistral_large'),
                ('prompt_to_prompt_drift', 'model_qwen_max'),
                ('prompt_to_prompt_drift', 'model_deepseek_r1'),
                ('cumulative_drift', 'model_mistral_large'),
                ('cumulative_drift', 'model_qwen_max')
            ]
            
            interaction_count = 0
            for drift_var, model_var in significant_interactions:
                if drift_var in interaction_data.columns and model_var in interaction_data.columns:
                    interaction_name = f"{drift_var}_x_{model_var}"
                    interaction_data[interaction_name] = interaction_data[drift_var] * interaction_data[model_var]
                    interaction_count += 1
            
            print(f"Created {interaction_count} interaction terms")
            
            # Prepare data for AFT regression (exclude non-numeric columns)
            numeric_cols = ['round', 'failure', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                           'cumulative_drift', 'prompt_complexity']
            model_cols = [col for col in interaction_data.columns if col.startswith('model_')]
            interaction_cols = [col for col in interaction_data.columns if '_x_' in col]
            
            final_cols = numeric_cols + model_cols + interaction_cols
            aft_data = interaction_data[final_cols]
            
            print(f"Interaction AFT using {len(final_cols)} variables")
            
            # Fit AFT with interactions
            waft_int = WeibullAFTFitter(penalizer=0.02)  # Add regularization
            waft_int.fit(aft_data, duration_col='round', event_col='failure')
            
            # Store results
            self.fitted_models['interaction_aft'] = waft_int
            
            interaction_results = {
                'model': 'WEIBULL_AFT_INTERACTIONS',
                'n_observations': len(aft_data),
                'n_events': aft_data['failure'].sum(),
                'n_interactions': interaction_count,
                'concordance_index': waft_int.concordance_index_,
                'aic': waft_int.AIC_,
                'log_likelihood': waft_int.log_likelihood_,
                'rho_param': getattr(waft_int, 'rho_', getattr(waft_int, 'params_', {}).get('rho_', np.nan)),
                'lambda_param': getattr(waft_int, 'lambda_', getattr(waft_int, 'params_', {}).get('lambda_', np.nan))
            }
            
            print(f"✅ Interaction AFT Model fitted successfully")
            print(f"   C-index: {waft_int.concordance_index_:.4f}")
            print(f"   AIC: {waft_int.AIC_:.2f}")
            print(f"   Interaction terms: {interaction_count}")
            
            # Coefficients analysis
            int_coef_df = waft_int.summary.copy()
            int_coef_df['model'] = 'WEIBULL_AFT_INTERACTIONS'
            int_coef_df['analysis_type'] = 'weibull_aft_interactions'
            
            self.results['interaction_aft'] = interaction_results
            self.results['interaction_coefficients'] = int_coef_df
            
            return interaction_results
            
        except Exception as e:
            print(f"❌ Interaction AFT failed: {e}")
            return None
    
    def compare_models(self):
        """Compare different Weibull models"""
        print("\n📊 WEIBULL MODEL COMPARISON")
        print("=" * 50)
        
        comparisons = []
        
        # Individual models comparison
        if 'individual_weibull' in self.results:
            ind_df = self.results['individual_weibull']
            if len(ind_df) > 0 and 'aic' in ind_df.columns:
                # Find valid (non-NaN) AIC values
                valid_aic = ind_df.dropna(subset=['aic'])
                if len(valid_aic) > 0:
                    best_individual = valid_aic.loc[valid_aic['aic'].idxmin()]
                    
                    comparisons.append({
                        'model_type': 'Best Individual Weibull',
                        'model_name': best_individual['model'],
                        'c_index': None,  # Individual models don't have C-index
                        'aic': best_individual['aic'],
                        'log_likelihood': best_individual['log_likelihood'],
                        'parameters': f"λ={best_individual['lambda_param']:.4f}, ρ={best_individual['rho_param']:.4f}"
            })
        
        # AFT regression
        if 'aft_regression' in self.results:
            aft_res = self.results['aft_regression']
            comparisons.append({
                'model_type': 'Weibull AFT Regression',
                'model_name': 'Combined',
                'c_index': aft_res['concordance_index'],
                'aic': aft_res['aic'],
                'log_likelihood': aft_res['log_likelihood'],
                'parameters': f"λ={aft_res['lambda_param']:.4f}, ρ={aft_res['rho_param']:.4f}"
            })
        
        # Interaction AFT
        if 'interaction_aft' in self.results:
            int_res = self.results['interaction_aft']
            comparisons.append({
                'model_type': 'Weibull AFT + Interactions',
                'model_name': 'Combined',
                'c_index': int_res['concordance_index'],
                'aic': int_res['aic'],
                'log_likelihood': int_res['log_likelihood'],
                'parameters': f"λ={int_res['lambda_param']:.4f}, ρ={int_res['rho_param']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparisons)
        
        print("Model Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best model by AIC
        if len(comparison_df) > 0:
            best_model = comparison_df.loc[comparison_df['aic'].idxmin()]
            print(f"\n🏆 Best Model by AIC: {best_model['model_type']}")
            print(f"   AIC: {best_model['aic']:.2f}")
            if best_model['c_index'] is not None:
                print(f"   C-index: {best_model['c_index']:.4f}")
        
        self.results['model_comparison'] = comparison_df
        return comparison_df
    
    def plot_survival_curves(self):
        """Plot survival curves for different models"""
        print("\n📈 GENERATING SURVIVAL PLOTS")
        print("=" * 30)
        
        try:
            # Plot individual Weibull models
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Individual survival curves
            ax1 = axes[0, 0]
            for model_name, wf in self.fitted_models.items():
                if isinstance(wf, WeibullFitter):
                    wf.survival_function_.plot(ax=ax1, label=model_name)
            
            ax1.set_title('Individual Weibull Survival Curves')
            ax1.set_xlabel('Conversation Round')
            ax1.set_ylabel('Survival Probability')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Hazard functions
            ax2 = axes[0, 1]
            for model_name, wf in self.fitted_models.items():
                if isinstance(wf, WeibullFitter):
                    wf.hazard_.plot(ax=ax2, label=model_name)
            
            ax2.set_title('Individual Weibull Hazard Functions')
            ax2.set_xlabel('Conversation Round')
            ax2.set_ylabel('Hazard Rate')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Parameter comparison
            ax3 = axes[1, 0]
            if 'individual_weibull' in self.results:
                ind_df = self.results['individual_weibull']
                ax3.scatter(ind_df['lambda_param'], ind_df['rho_param'], s=100, alpha=0.7)
                
                for i, row in ind_df.iterrows():
                    ax3.annotate(row['model'], 
                               (row['lambda_param'], row['rho_param']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
                
                ax3.set_xlabel('λ (Scale Parameter)')
                ax3.set_ylabel('ρ (Shape Parameter)')
                ax3.set_title('Weibull Parameters by Model')
                ax3.grid(True, alpha=0.3)
            
            # AIC comparison
            ax4 = axes[1, 1]
            if 'model_comparison' in self.results:
                comp_df = self.results['model_comparison']
                bars = ax4.bar(range(len(comp_df)), comp_df['aic'])
                ax4.set_xticks(range(len(comp_df)))
                ax4.set_xticklabels(comp_df['model_type'], rotation=45, ha='right')
                ax4.set_ylabel('AIC')
                ax4.set_title('Model Comparison by AIC')
                ax4.grid(True, alpha=0.3)
                
                # Highlight best model
                best_idx = comp_df['aic'].idxmin()
                bars[best_idx].set_color('red')
                bars[best_idx].set_alpha(0.8)
            
            plt.tight_layout()
            plt.savefig('results/figures/weibull_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Survival plots saved to results/figures/weibull_analysis.png")
            
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
    
    def save_results(self):
        """Save all results to CSV files"""
        print("\n💾 SAVING WEIBULL RESULTS")
        print("=" * 30)
        
        try:
            os.makedirs('results/outputs/weibull', exist_ok=True)
            
            # Save individual results
            if 'individual_weibull' in self.results:
                self.results['individual_weibull'].to_csv(
                    'results/outputs/weibull/individual_weibull_models.csv', index=False)
                print("✅ Individual Weibull results saved")
            
            # Save AFT regression results
            if 'aft_coefficients' in self.results:
                self.results['aft_coefficients'].to_csv(
                    'results/outputs/weibull/aft_regression_coefficients.csv', index=False)
                print("✅ AFT regression coefficients saved")
            
            # Save interaction results
            if 'interaction_coefficients' in self.results:
                self.results['interaction_coefficients'].to_csv(
                    'results/outputs/weibull/aft_interaction_coefficients.csv', index=False)
                print("✅ AFT interaction coefficients saved")
            
            # Save model comparison
            if 'model_comparison' in self.results:
                self.results['model_comparison'].to_csv(
                    'results/outputs/weibull/model_comparison.csv', index=False)
                print("✅ Model comparison saved")
            
            print("📁 All results saved to results/outputs/weibull/")
            
        except Exception as e:
            print(f"❌ Saving failed: {e}")
    
    def run_complete_analysis(self):
        """Run complete Weibull modeling pipeline"""
        print("🔬 WEIBULL SURVIVAL MODELING FOR LLM ANALYSIS")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        print("=" * 80)
        
        # Individual models
        self.fit_weibull_individual_models()
        
        print("=" * 80)
        
        # AFT regression
        self.fit_weibull_aft_regression()
        
        print("=" * 80)
        
        # AFT with interactions
        self.fit_weibull_with_interactions()
        
        print("=" * 80)
        
        # Model comparison
        self.compare_models()
        
        print("=" * 80)
        
        # Generate plots
        self.plot_survival_curves()
        
        print("=" * 80)
        
        # Save results
        self.save_results()
        
        print("=" * 80)
        print("🎉 WEIBULL MODELING COMPLETED!")

def main():
    """Main execution function"""
    weibull_analysis = WeibullModeling()
    weibull_analysis.run_complete_analysis()

if __name__ == "__main__":
    main()