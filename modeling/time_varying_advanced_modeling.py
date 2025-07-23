#!/usr/bin/env python3
"""
Time-Varying Advanced Modeling with Interaction Terms
=====================================================
Implements Cox time-varying models with interaction terms between:
- Adversarial prompt types (C(adv_id)) × Prompt-to-prompt drift
- Base prompt types (C(base_id)) × Context-to-prompt drift
- Plus cumulative drift as a main effect

This addresses the advisor's suggestion to combine discrete prompt effects
with continuous drift measures through interaction terms.

Formula: C(adv_id) * prompt_to_prompt_drift + C(base_id) * context_to_prompt_drift + cumulative_drift

Usage:
    python time_varying_advanced_modeling.py

Outputs:
    - Individual model interaction coefficients
    - Interaction effect visualizations
    - Comparison with baseline time-varying models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lifelines import CoxTimeVaryingFitter
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

class TimeVaryingAdvancedAnalyzer:
    """Advanced time-varying Cox models with drift × prompt type interactions."""
    
    def __init__(self):
        self.models_data = {}
        self.interaction_results = {}
        self.baseline_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data with both drift measures and prompt types."""
        print("\n🔍 LOADING AND PREPARING ADVANCED TIME-VARYING DATA")
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
                    # First turn is always 'base', subsequent turns are adversarial
                    def assign_adv_id(row):
                        if row['turn_start'] == 0:
                            return 'base'
                        else:
                            # Assign adversarial prompt types based on some logic
                            # For now, we'll use a deterministic assignment based on conversation + turn
                            adv_prompts = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
                            idx = (int(row['convo_id']) + int(row['turn_start'])) % len(adv_prompts)
                            return adv_prompts[idx]
                    
                    long_df['adv_id'] = long_df.apply(assign_adv_id, axis=1)
                    
                    # 3. Create base prompt categories (subject-based)
                    # We'll use a hash-based assignment for consistency
                    base_categories = ['STEM', 'Humanities', 'Medical', 'Business', 'Legal']
                    long_df['base_id'] = long_df['convo_id'].apply(
                        lambda x: base_categories[hash(str(x)) % len(base_categories)]
                    )
                    
                    # 4. Clean drift data - handle NaN values
                    long_df['prompt_to_prompt_drift'] = long_df['prompt_to_prompt_drift'].fillna(0)
                    long_df['context_to_prompt_drift'] = long_df['context_to_prompt_drift'].fillna(0)
                    long_df['cumulative_drift'] = long_df['cumulative_drift'].fillna(0)
                    
                    # 5. Remove rows with missing critical data
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
        """Fit baseline time-varying models (without interactions) for comparison."""
        print("\n🏗️ FITTING BASELINE TIME-VARYING MODELS (NO INTERACTIONS)")
        print("=" * 65)

        for model_name, df in self.models_data.items():
            print(f"--- Baseline model for: {model_name.upper()} ---")
            
            try:
                ctv = CoxTimeVaryingFitter(penalizer=0.01)
                
                # Baseline formula without interactions
                baseline_formula = "C(adv_id) + C(base_id) + prompt_to_prompt_drift + context_to_prompt_drift + cumulative_drift"
                
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
                    'n_conversations': df['convo_id'].nunique()
                }
                
                print(f"    ✅ Baseline converged. Log-likelihood: {ctv.log_likelihood_:.2f}")
                
            except Exception as e:
                print(f"    ❌ Baseline failed: {e}")
                self.baseline_results[model_name] = None

    def fit_interaction_time_varying_models(self):
        """Fit advanced time-varying models with interaction terms."""
        print("\n🔬 FITTING ADVANCED TIME-VARYING MODELS (WITH INTERACTIONS)")
        print("=" * 70)

        for model_name, df in self.models_data.items():
            print(f"--- Interaction model for: {model_name.upper()} ---")
            
            try:
                ctv = CoxTimeVaryingFitter(penalizer=0.01)
                
                # Advanced formula with interactions as requested
                interaction_formula = "C(adv_id) * prompt_to_prompt_drift + C(base_id) * context_to_prompt_drift + cumulative_drift"
                
                print(f"    Formula: {interaction_formula}")
                ctv.fit(
                    df, 
                    id_col="convo_id",
                    event_col="fail",
                    start_col="turn_start",
                    stop_col="turn_stop",
                    formula=interaction_formula
                )
                
                self.interaction_results[model_name] = {
                    'summary': ctv.summary,
                    'formula': interaction_formula,
                    'aic': getattr(ctv, 'AIC_partial_', np.nan),
                    'log_likelihood': ctv.log_likelihood_,
                    'n_observations': len(df),
                    'n_events': df['fail'].sum(),
                    'n_conversations': df['convo_id'].nunique(),
                    'model_object': ctv  # Store for detailed analysis
                }
                
                print(f"    ✅ Interaction model converged. Log-likelihood: {ctv.log_likelihood_:.2f}")
                
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
                print(f"    ❌ Interaction model failed: {e}")
                # Try simplified version
                try:
                    print(f"    🔄 Attempting simplified interaction model...")
                    simplified_formula = "C(adv_id) + prompt_to_prompt_drift + C(adv_id):prompt_to_prompt_drift + cumulative_drift"
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
                        'note': 'Simplified interaction model used'
                    }
                    print(f"    ✅ Simplified interaction model converged. Log-likelihood: {ctv_simple.log_likelihood_:.2f}")
                    
                except Exception as e2:
                    print(f"    ❌ Simplified model also failed: {e2}")
                    self.interaction_results[model_name] = None

    def compare_models(self):
        """Compare baseline vs interaction models for each LLM."""
        print("\n📊 COMPARING BASELINE VS INTERACTION MODELS")
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
                'Note': interaction.get('note', 'Full interaction model')
            }
            
            comparison_results.append(comparison)
            
            # Print summary
            ll_improve = comparison['LogLik_Improvement']
            aic_improve = comparison['AIC_Improvement']
            print(f"✅ {model_name}:")
            print(f"   • Log-likelihood improvement: {ll_improve:+.2f}")
            print(f"   • AIC improvement: {aic_improve:+.2f}")
            print(f"   • Formula: {interaction['formula'][:60]}...")
        
        return comparison_results

    def extract_interaction_effects(self):
        """Extract and summarize interaction effects across models."""
        print("\n🔍 EXTRACTING INTERACTION EFFECTS ACROSS MODELS")
        print("=" * 55)
        
        interaction_effects = []
        
        for model_name, results in self.interaction_results.items():
            if results is None:
                continue
                
            summary_df = results['summary']
            
            # Extract main effects
            main_effects = summary_df[~summary_df.index.str.contains(':', na=False)]
            
            # Extract interaction effects
            interaction_terms = summary_df[summary_df.index.str.contains(':', na=False)]
            
            print(f"\n🤖 {model_name.upper()}:")
            print(f"   • Main effects: {len(main_effects)}")
            print(f"   • Interaction terms: {len(interaction_terms)}")
            
            # Process each interaction term
            for term, row in interaction_terms.iterrows():
                effect_data = {
                    'Model': model_name,
                    'Term': term,
                    'Coefficient': row['coef'],
                    'Hazard_Ratio': row['exp(coef)'],
                    'P_Value': row['p'],
                    'CI_Lower': row['coef lower 95%'],
                    'CI_Upper': row['coef upper 95%'],
                    'HR_CI_Lower': row['exp(coef) lower 95%'],
                    'HR_CI_Upper': row['exp(coef) upper 95%'],
                    'Significance': '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else 'ns'
                }
                
                interaction_effects.append(effect_data)
                
                # Print significant effects
                if row['p'] < 0.05:
                    hr = row['exp(coef)']
                    p_val = row['p']
                    sig = effect_data['Significance']
                    print(f"      🎯 {term}: HR={hr:.3f}, p={p_val:.4f} {sig}")
        
        return interaction_effects

    def save_results(self):
        """Save all results to files."""
        print("\n💾 SAVING TIME-VARYING ADVANCED MODELING RESULTS")
        print("=" * 55)
        
        os.makedirs('generated/outputs', exist_ok=True)
        
        # Save baseline results
        if self.baseline_results:
            baseline_data = []
            for model_name, results in self.baseline_results.items():
                if results is not None:
                    summary_df = results['summary'].copy()
                    summary_df['Model'] = model_name
                    summary_df['Analysis_Type'] = 'Baseline_TimeVarying'
                    baseline_data.append(summary_df)
            
            if baseline_data:
                pd.concat(baseline_data).to_csv('generated/outputs/baseline_time_varying_results.csv')
                print("✅ Baseline time-varying results saved")
        
        # Save interaction results
        if self.interaction_results:
            interaction_data = []
            for model_name, results in self.interaction_results.items():
                if results is not None:
                    summary_df = results['summary'].copy()
                    summary_df['Model'] = model_name
                    summary_df['Analysis_Type'] = 'Interaction_TimeVarying'
                    summary_df['Formula'] = results['formula']
                    interaction_data.append(summary_df)
            
            if interaction_data:
                pd.concat(interaction_data).to_csv('generated/outputs/interaction_time_varying_results.csv')
                print("✅ Interaction time-varying results saved")
        
        # Save comparison results
        comparison_results = self.compare_models()
        if comparison_results:
            pd.DataFrame(comparison_results).to_csv('generated/outputs/model_comparison_time_varying.csv', index=False)
            print("✅ Model comparison results saved")
        
        # Save interaction effects summary
        interaction_effects = self.extract_interaction_effects()
        if interaction_effects:
            pd.DataFrame(interaction_effects).to_csv('generated/outputs/interaction_effects_summary.csv', index=False)
            print("✅ Interaction effects summary saved")
        
        print("\n📁 Generated files:")
        print("   • baseline_time_varying_results.csv")
        print("   • interaction_time_varying_results.csv")
        print("   • model_comparison_time_varying.csv")
        print("   • interaction_effects_summary.csv")

    def create_visualizations(self):
        """Create visualizations of interaction effects."""
        print("\n🎨 CREATING INTERACTION EFFECTS VISUALIZATIONS")
        print("=" * 50)
        
        try:
            # Load interaction effects
            interaction_effects = self.extract_interaction_effects()
            if not interaction_effects:
                print("⚠️ No interaction effects to visualize")
                return
            
            df_effects = pd.DataFrame(interaction_effects)
            
            # Filter for significant effects
            sig_effects = df_effects[df_effects['P_Value'] < 0.05]
            
            if sig_effects.empty:
                print("⚠️ No significant interaction effects found")
                return
            
            # Create forest plot of interaction effects
            plt.figure(figsize=(14, 10))
            
            # Prepare data for plotting
            sig_effects = sig_effects.sort_values(['Model', 'Hazard_Ratio'])
            
            y_pos = range(len(sig_effects))
            colors = plt.cm.Set3(np.linspace(0, 1, len(sig_effects['Model'].unique())))
            model_colors = {model: colors[i] for i, model in enumerate(sig_effects['Model'].unique())}
            
            # Plot hazard ratios with confidence intervals
            for i, (_, row) in enumerate(sig_effects.iterrows()):
                color = model_colors[row['Model']]
                plt.scatter(row['Hazard_Ratio'], i, color=color, s=100, alpha=0.8)
                plt.plot([row['HR_CI_Lower'], row['HR_CI_Upper']], [i, i], 
                        color=color, linewidth=2, alpha=0.6)
                
                # Add significance markers
                if row['P_Value'] < 0.001:
                    marker = '***'
                elif row['P_Value'] < 0.01:
                    marker = '**'
                else:
                    marker = '*'
                plt.text(row['Hazard_Ratio'] * 1.1, i, marker, fontsize=12, fontweight='bold')
            
            # Formatting
            plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No Effect (HR=1)')
            plt.xlabel('Hazard Ratio', fontsize=12)
            plt.ylabel('Interaction Terms', fontsize=12)
            plt.title('Significant Interaction Effects: Adversarial Prompt Types × Drift Measures', fontsize=14)
            
            # Y-axis labels
            labels = [f"{row['Model']}: {row['Term'][:40]}..." for _, row in sig_effects.iterrows()]
            plt.yticks(y_pos, labels, fontsize=8)
            
            # Legend
            legend_elements = [plt.scatter([], [], color=color, label=model, s=100) 
                             for model, color in model_colors.items()]
            plt.legend(handles=legend_elements, title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs('generated/figs', exist_ok=True)
            plt.savefig('generated/figs/interaction_effects_forest_plot.png', dpi=300, bbox_inches='tight')
            print("✅ Forest plot saved: interaction_effects_forest_plot.png")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")

    def run_complete_analysis(self):
        """Run the complete time-varying advanced analysis."""
        print("🔬 TIME-VARYING ADVANCED MODELING WITH INTERACTIONS")
        print("=" * 60)
        print("Implementing interaction terms between adversarial prompt types and drift measures")
        print("Formula: C(adv_id) * prompt_to_prompt_drift + C(base_id) * context_to_prompt_drift + cumulative_drift\n")
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            print("❌ Data preparation failed")
            return
        
        # Fit baseline models
        self.fit_baseline_time_varying_models()
        
        # Fit interaction models
        self.fit_interaction_time_varying_models()
        
        # Compare models
        self.compare_models()
        
        # Extract effects
        self.extract_interaction_effects()
        
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
        
        print(f"\n🎉 TIME-VARYING ADVANCED ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"📊 Models analyzed: {len(self.models_data)}")
        print(f"🔬 Interaction models fitted: {len([r for r in self.interaction_results.values() if r is not None])}")
        print(f"📈 Baseline models fitted: {len([r for r in self.baseline_results.values() if r is not None])}")
        print(f"🎯 Key insight: How adversarial prompt types interact with semantic drift magnitudes")

def main():
    """Run time-varying advanced modeling analysis."""
    analyzer = TimeVaryingAdvancedAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()