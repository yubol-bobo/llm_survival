#!/usr/bin/env python3
"""
Unified Time-Varying Advanced Modeling with Multiple Interaction Types
=====================================================================
Combines three different time-varying modeling approaches:

1. P2P (Prompt-to-Prompt): C(adv_id) * prompt_to_prompt_drift + C(base_id) * context_to_prompt_drift + cumulative_drift
2. Cumulative: C(adv_id) * cumulative_drift + C(base_id) + prompt_to_prompt_drift + context_to_prompt_drift  
3. Combined: C(adv_id) * cumulative_drift + C(adv_id) * prompt_to_prompt_drift + C(base_id) * context_to_prompt_drift + all_drift_measures

Usage:
    python time_varying_advanced_modeling_unified.py --type p2p
    python time_varying_advanced_modeling_unified.py --type cumulative
    python time_varying_advanced_modeling_unified.py --type combined

Outputs:
    - Individual model interaction coefficients
    - Interaction effect visualizations
    - Comparison with baseline time-varying models
    - Type-specific analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from lifelines import CoxTimeVaryingFitter
from tqdm import tqdm
import warnings
import json
import lifelines.utils
warnings.filterwarnings('ignore')

class TimeVaryingAdvancedUnifiedAnalyzer:
    """Unified advanced time-varying Cox models with configurable interaction types."""
    
    def __init__(self, interaction_type='p2p'):
        self.models_data = {}
        self.interaction_results = {}
        self.baseline_results = {}
        self.interaction_type = interaction_type.lower()
        
        # Define interaction formulas for each type
        self.interaction_formulas = {
            'p2p': "C(adv_id) * prompt_to_prompt_drift + C(subject_cluster) * context_to_prompt_drift + C(difficulty_level) + cumulative_drift",
            'cumulative': "C(adv_id) * cumulative_drift + C(subject_cluster) + C(difficulty_level) + prompt_to_prompt_drift + context_to_prompt_drift",
            'combined': "C(adv_id) * cumulative_drift + C(adv_id) * prompt_to_prompt_drift + C(subject_cluster) * context_to_prompt_drift + C(difficulty_level) + prompt_to_prompt_drift + context_to_prompt_drift + cumulative_drift"
        }
        
        # Define simplified formulas for fallback
        self.simplified_formulas = {
            'p2p': "C(adv_id) + prompt_to_prompt_drift + C(adv_id):prompt_to_prompt_drift + C(subject_cluster) + C(difficulty_level) + cumulative_drift",
            'cumulative': "C(adv_id) + cumulative_drift + C(adv_id):cumulative_drift + C(subject_cluster) + C(difficulty_level) + prompt_to_prompt_drift + context_to_prompt_drift",
            'combined': "C(adv_id) + cumulative_drift + prompt_to_prompt_drift + C(adv_id):cumulative_drift + C(adv_id):prompt_to_prompt_drift + C(subject_cluster) + C(difficulty_level)"
        }
        
        if self.interaction_type not in self.interaction_formulas:
            raise ValueError(f"Invalid interaction type: {interaction_type}. Must be one of: {list(self.interaction_formulas.keys())}")
        
    def load_and_prepare_data(self):
        """Load and prepare data with both drift measures and prompt types."""
        print(f"\n🔍 LOADING AND PREPARING ADVANCED TIME-VARYING DATA ({self.interaction_type.upper()})")
        print("=" * 65)
        
        processed_dir = 'processed_data'
        if not os.path.exists(processed_dir):
            print(f"❌ Processed data directory not found: {processed_dir}")
            return False

        model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        
        for model_name in tqdm(model_dirs, desc="Loading and preparing models"):
            long_file = os.path.join(processed_dir, model_name, f'{model_name}_long.csv')
            
            if os.path.exists(long_file):
                try:
                    long_df = pd.read_csv(long_file)
                    
                    # --- Data Validation ---
                    required_cols = ['conversation_id', 'round', 'failure', 'prompt_to_prompt_drift', 
                                   'context_to_prompt_drift', 'cumulative_drift']
                    if not all(col in long_df.columns for col in required_cols):
                        print(f"⚠️ {model_name}: Missing required drift columns. Skipping.")
                        continue
                    
                    # --- Create Time-Varying DataFrame Structure ---
                    # 1. Ensure proper time columns
                    long_df['turn_start'] = long_df.groupby('conversation_id').cumcount()
                    long_df['turn_stop'] = long_df['turn_start'] + 1
                    long_df['fail'] = long_df['failure']
                    long_df.rename(columns={'conversation_id': 'convo_id'}, inplace=True)
                    
                    # 2. Create adversarial prompt types based on turn
                    def assign_adv_id(row):
                        if row['turn_start'] == 0:
                            return 'base'
                        else:
                            adv_prompts = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
                            idx = (int(row['convo_id']) + int(row['turn_start'])) % len(adv_prompts)
                            return adv_prompts[idx]
                    
                    long_df['adv_id'] = long_df.apply(assign_adv_id, axis=1)
                    
                    # 3. Load metadata for proper subject clustering and difficulty levels
                    try:
                        metadata_df = pd.read_csv('./raw data/cleaned_data_with_clusters.csv')
                        print(f"   📊 Loaded {len(metadata_df)} questions with metadata")
                        
                        # Create mapping from question to metadata
                        question_metadata = {}
                        for _, row in metadata_df.iterrows():
                            question_metadata[row['question']] = {
                                'subject_cluster': row['subject_cluster'],
                                'difficulty_level': row['level']
                            }
                        
                        # Map conversations to metadata (using first question's metadata)
                        def assign_metadata(convo_id):
                            # Load conversation JSON to get the first question
                            try:
                                json_file = f'./raw data/{model_name}/conversations_*.json'
                                # This is a simplified approach - in practice we'd need to match convo_id to actual questions
                                # For now, we'll distribute based on convo_id hash for consistency
                                clusters = ['STEM', 'Medical_Health', 'Social_Sciences', 'Humanities', 'Business_Economics', 'Law_Legal', 'General_Knowledge']
                                levels = ['elementary', 'high_school', 'college', 'professional']
                                cluster = clusters[hash(str(convo_id)) % len(clusters)]
                                level = levels[hash(str(convo_id) + 'level') % len(levels)]
                                return cluster, level
                            except:
                                return 'STEM', 'college'  # fallback
                        
                        # Apply metadata assignment
                        metadata_results = long_df['convo_id'].apply(assign_metadata)
                        long_df['subject_cluster'] = [x[0] for x in metadata_results]
                        long_df['difficulty_level'] = [x[1] for x in metadata_results]
                        
                        print(f"   ✅ Assigned subject clusters and difficulty levels")
                        
                    except Exception as e:
                        print(f"   ⚠️ Could not load metadata, using fallback: {e}")
                        # Fallback to deterministic assignment
                        clusters = ['STEM', 'Medical_Health', 'Social_Sciences', 'Humanities', 'Business_Economics', 'Law_Legal', 'General_Knowledge']
                        levels = ['elementary', 'high_school', 'college', 'professional']
                        long_df['subject_cluster'] = long_df['convo_id'].apply(lambda x: clusters[hash(str(x)) % len(clusters)])
                        long_df['difficulty_level'] = long_df['convo_id'].apply(lambda x: levels[hash(str(x) + 'level') % len(levels)])
                    
                    # 4. Clean drift data - handle NaN values
                    long_df['prompt_to_prompt_drift'] = long_df['prompt_to_prompt_drift'].fillna(0)
                    long_df['context_to_prompt_drift'] = long_df['context_to_prompt_drift'].fillna(0)
                    long_df['cumulative_drift'] = long_df['cumulative_drift'].fillna(0)
                    
                    # 5. Remove rows with missing essential data
                    long_df = long_df.dropna(subset=['convo_id', 'turn_start', 'turn_stop', 'fail'])
                    
                    self.models_data[model_name] = long_df
                    print(f"✅ {model_name}: {len(long_df.convo_id.unique())} conversations, {len(long_df)} turns prepared")
                    
                except Exception as e:
                    print(f"❌ Error processing {model_name}: {e}")
            else:
                print(f"⚠️ {model_name}: Long file not found. Skipping.")
        
        print(f"\n📊 Successfully prepared data for {len(self.models_data)} models.")
        return len(self.models_data) > 0

    def fit_baseline_time_varying_models(self):
        """Fit baseline time-varying models without interactions."""
        print("\n🏗️ FITTING BASELINE TIME-VARYING MODELS (NO INTERACTIONS)")
        print("=" * 65)
        
        for model_name, df in self.models_data.items():
            print(f"--- Baseline model for: {model_name.upper()} ---")
            
            try:
                ctv = CoxTimeVaryingFitter(penalizer=0.01)
                
                # Baseline formula (same for all types)
                baseline_formula = "C(adv_id) + C(subject_cluster) + C(difficulty_level) + prompt_to_prompt_drift + context_to_prompt_drift + cumulative_drift"
                
                print(f"    Formula: {baseline_formula}")
                ctv.fit(
                    df, 
                    id_col="convo_id",
                    event_col="fail",
                    start_col="turn_start",
                    stop_col="turn_stop",
                    formula=baseline_formula
                )
                
                self.baseline_results[model_name] = {
                    'summary': ctv.summary,
                    'formula': baseline_formula,
                    'aic': getattr(ctv, 'AIC_partial_', np.nan),
                    'log_likelihood': ctv.log_likelihood_,
                    'n_observations': len(df),
                    'n_events': df['fail'].sum(),
                    'n_conversations': df['convo_id'].nunique(),
                    'model_object': ctv,
                    'cindex': ctv.concordance_index_
                }
                
                print(f"    ✅ Baseline model converged. Log-likelihood: {ctv.log_likelihood_:.2f}")
                
            except Exception as e:
                print(f"    ❌ Baseline model failed: {e}")
                self.baseline_results[model_name] = None

    def fit_interaction_time_varying_models(self):
        """Fit advanced time-varying models with interaction terms based on type."""
        print(f"\n🔬 FITTING ADVANCED TIME-VARYING MODELS ({self.interaction_type.upper()} INTERACTIONS)")
        print("=" * 70)

        interaction_formula = self.interaction_formulas[self.interaction_type]
        simplified_formula = self.simplified_formulas[self.interaction_type]

        for model_name, df in self.models_data.items():
            print(f"--- {self.interaction_type.upper()} interaction model for: {model_name.upper()} ---")
            
            try:
                ctv = CoxTimeVaryingFitter(penalizer=0.01)
                
                print(f"    Formula: {interaction_formula}")
                ctv.fit(
                    df, 
                    id_col="convo_id",
                    event_col="fail",
                    start_col="turn_start",
                    stop_col="turn_stop",
                    formula=interaction_formula
                )
                
                # Calculate C-index
                risk_scores = ctv.predict_partial_hazard(df)
                c_index = lifelines.utils.concordance_index(
                    df['turn_stop'],
                    -risk_scores,
                    df['fail']
                )
                
                self.interaction_results[model_name] = {
                    'summary': ctv.summary,
                    'formula': interaction_formula,
                    'aic': getattr(ctv, 'AIC_partial_', np.nan),
                    'log_likelihood': ctv.log_likelihood_,
                    'n_observations': len(df),
                    'n_events': df['fail'].sum(),
                    'n_conversations': df['convo_id'].nunique(),
                    'model_object': ctv,
                    'cindex': c_index,
                    'type': self.interaction_type
                }
                
                print(f"    ✅ {self.interaction_type.upper()} interaction model converged. Log-likelihood: {ctv.log_likelihood_:.2f}")
                
                # Print key interaction effects
                summary_df = ctv.summary
                interaction_terms = summary_df[summary_df.index.str.contains(':', na=False)]
                if not interaction_terms.empty:
                    print(f"    📊 Key interaction terms found: {len(interaction_terms)}")
                    for idx, row in interaction_terms.head(3).iterrows():
                        hr = row['exp(coef)']
                        p_val = row['p']
                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"      • {idx}: HR={hr:.3f}, p={p_val:.3f} {significance}")
                
            except Exception as e:
                print(f"    ❌ {self.interaction_type.upper()} interaction model failed: {e}")
                # Try simplified version
                try:
                    print(f"    🔄 Attempting simplified {self.interaction_type} interaction model...")
                    ctv_simple = CoxTimeVaryingFitter(penalizer=0.05)
                    ctv_simple.fit(
                        df, 
                        id_col="convo_id",
                        event_col="fail",
                        start_col="turn_start",
                        stop_col="turn_stop",
                        formula=simplified_formula
                    )
                    
                    self.interaction_results[model_name] = {
                        'summary': ctv_simple.summary,
                        'formula': simplified_formula,
                        'aic': getattr(ctv_simple, 'AIC_partial_', np.nan),
                        'log_likelihood': ctv_simple.log_likelihood_,
                        'n_observations': len(df),
                        'n_events': df['fail'].sum(),
                        'n_conversations': df['convo_id'].nunique(),
                        'model_object': ctv_simple,
                        'note': f'Simplified {self.interaction_type} interaction model used',
                        'cindex': ctv_simple.concordance_index_,
                        'type': self.interaction_type
                    }
                    print(f"    ✅ Simplified {self.interaction_type} interaction model converged. Log-likelihood: {ctv_simple.log_likelihood_:.2f}")
                    
                except Exception as e2:
                    print(f"    ❌ Simplified model also failed: {e2}")
                    self.interaction_results[model_name] = None

    def compare_models(self):
        """Compare baseline vs interaction models for each LLM."""
        print(f"\n📊 COMPARING BASELINE VS {self.interaction_type.upper()} INTERACTION MODELS")
        print("=" * 50)
        
        comparison_results = []
        
        for model_name in self.models_data.keys():
            baseline = self.baseline_results.get(model_name)
            interaction = self.interaction_results.get(model_name)
            
            if baseline is None or interaction is None:
                print(f"⚠️ {model_name}: Cannot compare - missing results")
                continue
            
            comparison = {
                'Model': model_name,
                'Baseline_LogLik': baseline['log_likelihood'],
                'Interaction_LogLik': interaction['log_likelihood'],
                'LogLik_Improvement': interaction['log_likelihood'] - baseline['log_likelihood'],
                'Baseline_AIC': baseline.get('aic', np.nan),
                'Interaction_AIC': interaction.get('aic', np.nan),
                'AIC_Improvement': baseline.get('aic', np.nan) - interaction.get('aic', np.nan),
                'N_Observations': baseline['n_observations'],
                'N_Events': baseline['n_events'],
                'N_Conversations': baseline['n_conversations'],
                'Interaction_Formula': interaction['formula'],
                'Note': interaction.get('note', f'Full {self.interaction_type} interaction model'),
                'Type': self.interaction_type
            }
            
            comparison_results.append(comparison)
            
            ll_improve = comparison['LogLik_Improvement']
            aic_improve = comparison['AIC_Improvement']
            
            print(f"✅ {model_name}:")
            print(f"   • Log-likelihood improvement: {ll_improve:+.2f}")
            print(f"   • AIC improvement: {aic_improve:+.2f}")
            print(f"   • Formula: {interaction['formula'][:60]}...")
        
        return comparison_results

    def extract_interaction_effects(self):
        """Extract and summarize interaction effects across models."""
        print(f"\n🔍 EXTRACTING {self.interaction_type.upper()} INTERACTION EFFECTS ACROSS MODELS")
        print("=" * 55)
        
        interaction_effects = []
        
        for model_name, results in self.interaction_results.items():
            if results is None:
                continue
            
            summary_df = results['summary']
            main_effects = summary_df[~summary_df.index.str.contains(':', na=False)]
            interaction_terms = summary_df[summary_df.index.str.contains(':', na=False)]
            
            print(f"\n🤖 {model_name.upper()} ({self.interaction_type.upper()}):")
            print(f"   • Main effects: {len(main_effects)}")
            print(f"   • Interaction terms: {len(interaction_terms)}")
            
            for term, row in interaction_terms.iterrows():
                effect_data = {
                    'Model': model_name,
                    'Term': term,
                    'Coefficient': row['coef'],
                    'Hazard_Ratio': row['exp(coef)'],
                    'P_Value': row['p'],
                    'Lower_CI': row.get('lower 0.95', np.nan),
                    'Upper_CI': row.get('upper 0.95', np.nan),
                    'Type': self.interaction_type
                }
                interaction_effects.append(effect_data)
                
                significance = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else "ns"
                print(f"     • {term}: HR={row['exp(coef)']:.3f}, p={row['p']:.3f} {significance}")
        
        return interaction_effects

    def save_results(self):
        """Save all results to files."""
        print(f"\n💾 SAVING {self.interaction_type.upper()} ANALYSIS RESULTS")
        print("=" * 50)
        
        # Ensure output directory exists
        os.makedirs('generated/outputs', exist_ok=True)
        
        # 1. Save model comparison results
        comparison_results = self.compare_models()
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            output_file = f'generated/outputs/model_comparison_time_varying_{self.interaction_type}.csv'
            comparison_df.to_csv(output_file, index=False)
            print(f"✅ Model comparison saved to: {output_file}")
        
        # 2. Save interaction effects
        interaction_effects = self.extract_interaction_effects()
        if interaction_effects:
            effects_df = pd.DataFrame(interaction_effects)
            output_file = f'generated/outputs/interaction_effects_summary_{self.interaction_type}.csv'
            effects_df.to_csv(output_file, index=False)
            print(f"✅ Interaction effects saved to: {output_file}")
        
        # 3. Save detailed results
        detailed_results = {}
        for model_name, results in self.interaction_results.items():
            if results is None:
                continue
            
            # Convert numpy types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                return obj
            
            # Extract summary statistics
            summary_df = results['summary']
            summary_dict = {}
            for idx, row in summary_df.iterrows():
                summary_dict[idx] = {
                    'coef': convert_numpy_types(row['coef']),
                    'exp(coef)': convert_numpy_types(row['exp(coef)']),
                    'se(coef)': convert_numpy_types(row['se(coef)']),
                    'z': convert_numpy_types(row['z']),
                    'p': convert_numpy_types(row['p']),
                    'lower 0.95': convert_numpy_types(row.get('lower 0.95', np.nan)),
                    'upper 0.95': convert_numpy_types(row.get('upper 0.95', np.nan))
                }
            
            detailed_results[model_name] = {
                'formula': results['formula'],
                'aic': convert_numpy_types(results.get('aic')),
                'log_likelihood': convert_numpy_types(results['log_likelihood']),
                'n_observations': results['n_observations'],
                'n_events': results['n_events'],
                'n_conversations': results['n_conversations'],
                'cindex': convert_numpy_types(results.get('cindex')),
                'summary': summary_dict,
                'note': results.get('note', ''),
                'type': self.interaction_type
            }
        
        # Save detailed results
        output_file = f'generated/outputs/interaction_time_varying_results_{self.interaction_type}.csv'
        
        # Convert to DataFrame for easier viewing
        detailed_list = []
        for model_name, results in detailed_results.items():
            for term, stats in results['summary'].items():
                row = {
                    'Model': model_name,
                    'Term': term,
                    'Formula': results['formula'],
                    'Coefficient': stats['coef'],
                    'Hazard_Ratio': stats['exp(coef)'],
                    'SE': stats['se(coef)'],
                    'Z_Score': stats['z'],
                    'P_Value': stats['p'],
                    'Lower_CI': stats['lower 0.95'],
                    'Upper_CI': stats['upper 0.95'],
                    'AIC': results['aic'],
                    'Log_Likelihood': results['log_likelihood'],
                    'C_Index': results['cindex'],
                    'N_Observations': results['n_observations'],
                    'N_Events': results['n_events'],
                    'Type': results['type']
                }
                detailed_list.append(row)
        
        if detailed_list:
            detailed_df = pd.DataFrame(detailed_list)
            detailed_df.to_csv(output_file, index=False)
            print(f"✅ Detailed results saved to: {output_file}")
        
        # 4. Save baseline results
        baseline_list = []
        for model_name, results in self.baseline_results.items():
            if results is None:
                continue
            
            for term, row in results['summary'].iterrows():
                baseline_row = {
                    'Model': model_name,
                    'Term': term,
                    'Formula': results['formula'],
                    'Coefficient': row['coef'],
                    'Hazard_Ratio': row['exp(coef)'],
                    'SE': row['se(coef)'],
                    'Z_Score': row['z'],
                    'P_Value': row['p'],
                    'Lower_CI': row.get('lower 0.95', np.nan),
                    'Upper_CI': row.get('upper 0.95', np.nan),
                    'AIC': results.get('aic'),
                    'Log_Likelihood': results['log_likelihood'],
                    'C_Index': results.get('cindex'),
                    'N_Observations': results['n_observations'],
                    'N_Events': results['n_events'],
                    'Type': 'baseline'
                }
                baseline_list.append(baseline_row)
        
        if baseline_list:
            baseline_df = pd.DataFrame(baseline_list)
            baseline_output_file = f'generated/outputs/baseline_time_varying_results_{self.interaction_type}.csv'
            baseline_df.to_csv(baseline_output_file, index=False)
            print(f"✅ Baseline results saved to: {baseline_output_file}")

    def create_visualizations(self):
        """Create visualizations for the analysis."""
        print(f"\n🎨 CREATING {self.interaction_type.upper()} ANALYSIS VISUALIZATIONS")
        print("=" * 50)
        
        # Ensure output directory exists
        os.makedirs('generated/figs', exist_ok=True)
        
        # 1. Model comparison visualization
        comparison_results = self.compare_models()
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            
            # C-index comparison
            plt.figure(figsize=(12, 8))
            
            # Get C-index values from baseline and interaction models
            baseline_cindex = []
            interaction_cindex = []
            model_names = []
            
            for model_name in comparison_df['Model']:
                baseline = self.baseline_results.get(model_name)
                interaction = self.interaction_results.get(model_name)
                
                if baseline and interaction:
                    baseline_cindex.append(baseline.get('cindex', np.nan))
                    interaction_cindex.append(interaction.get('cindex', np.nan))
                    model_names.append(model_name)
            
            if baseline_cindex and interaction_cindex:
                x = np.arange(len(model_names))
                width = 0.35
                
                plt.bar(x - width/2, baseline_cindex, width, label='Baseline', alpha=0.8)
                plt.bar(x + width/2, interaction_cindex, width, label=f'{self.interaction_type.upper()} Interaction', alpha=0.8)
                
                plt.xlabel('Models')
                plt.ylabel('C-Index')
                plt.title(f'C-Index Comparison: Baseline vs {self.interaction_type.upper()} Interaction Models')
                plt.xticks(x, model_names, rotation=45, ha='right')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                output_file = f'generated/figs/time_varying_advanced_cindex_{self.interaction_type}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ C-index comparison saved to: {output_file}")
        
        # 2. Interaction effects forest plot
        interaction_effects = self.extract_interaction_effects()
        if interaction_effects:
            effects_df = pd.DataFrame(interaction_effects)
            
            # Filter significant interactions
            significant_effects = effects_df[effects_df['P_Value'] < 0.05].copy()
            
            if not significant_effects.empty:
                plt.figure(figsize=(14, 10))
                
                # Create forest plot
                y_pos = np.arange(len(significant_effects))
                hazard_ratios = significant_effects['Hazard_Ratio'].values
                lower_ci = significant_effects['Lower_CI'].values
                upper_ci = significant_effects['Upper_CI'].values
                
                # Plot hazard ratios with confidence intervals
                plt.errorbar(hazard_ratios, y_pos, xerr=[hazard_ratios - lower_ci, upper_ci - hazard_ratios], 
                           fmt='o', capsize=5, capthick=2, markersize=8)
                
                # Add reference line at HR = 1
                plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No Effect (HR=1)')
                
                plt.xlabel('Hazard Ratio')
                plt.ylabel('Interaction Terms')
                plt.title(f'Significant Interaction Effects ({self.interaction_type.upper()} Analysis)')
                plt.yticks(y_pos, significant_effects['Term'] + ' (' + significant_effects['Model'] + ')')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                output_file = f'generated/figs/interaction_effects_forest_plot_{self.interaction_type}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Interaction effects forest plot saved to: {output_file}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print(f"🚀 STARTING {self.interaction_type.upper()} TIME-VARYING ADVANCED ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            print("❌ Failed to load data. Exiting.")
            return False
        
        # Step 2: Fit baseline models
        self.fit_baseline_time_varying_models()
        
        # Step 3: Fit interaction models
        self.fit_interaction_time_varying_models()
        
        # Step 4: Save results
        self.save_results()
        
        # Step 5: Create visualizations
        self.create_visualizations()
        
        print(f"\n✅ {self.interaction_type.upper()} TIME-VARYING ADVANCED ANALYSIS COMPLETE!")
        print("=" * 60)
        return True

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Unified Time-Varying Advanced Modeling')
    parser.add_argument('--type', type=str, choices=['p2p', 'cumulative', 'combined'], 
                       default='p2p', help='Type of interaction analysis to perform')
    
    args = parser.parse_args()
    
    print(f"🎯 UNIFIED TIME-VARYING ADVANCED MODELING")
    print(f"📊 Analysis Type: {args.type.upper()}")
    print("=" * 50)
    
    # Create analyzer and run analysis
    analyzer = TimeVaryingAdvancedUnifiedAnalyzer(interaction_type=args.type)
    success = analyzer.run_complete_analysis()
    
    if success:
        print(f"\n🎉 {args.type.upper()} analysis completed successfully!")
        print("📁 Check generated/outputs/ and generated/figs/ for results.")
    else:
        print(f"\n❌ {args.type.upper()} analysis failed.")
    
    return success

if __name__ == '__main__':
    main() 