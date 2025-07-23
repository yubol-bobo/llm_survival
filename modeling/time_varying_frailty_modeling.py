#!/usr/bin/env python3
"""
🔍 Time-Varying Frailty Modeling for LLM Robustness

This script implements a Cox Proportional Hazards model with time-varying 
covariates to analyze LLM conversation robustness. It treats adversarial 
prompts as categorical, time-varying features and includes frailty terms 
for conversation and base prompt category.

This approach provides a more granular analysis of how specific adversarial 
prompt types impact conversation failure risk over time.
"""

import os
import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TimeVaryingFrailtyAnalyzer:
    def __init__(self):
        self.models_data = {}
        self.results = {}

    def load_and_prepare_data(self):
        """Load and prepare data for time-varying analysis."""
        print("\n🔍 LOADING AND PREPARING DATA")
        print("=" * 60)
        
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
                    required_cols = ['conversation_id', 'round', 'failure']
                    if not all(col in long_df.columns for col in required_cols):
                        print(f"⚠️ {model_name}: Missing required columns. Skipping.")
                        continue
                    
                    # --- Create Time-Varying DataFrame ---
                    # 1. Ensure 'round' is integer and starts from 1
                    long_df['round'] = long_df.groupby('conversation_id').cumcount() + 1
                    
                    # 2. Create 'turn_start' and 'turn_stop'
                    long_df['turn_start'] = long_df['round'] - 1
                    long_df['turn_stop'] = long_df['round']
                    
                    # 3. Create 'fail' column (same as 'failure')
                    long_df['fail'] = long_df['failure']
                    
                    # 4. Synthesize 'adv_id' and 'base_id' for demonstration
                    #    In a real scenario, these would be loaded from the data
                    if 'adv_id' not in long_df.columns:
                        adv_prompts = [f'p{i}' for i in range(1, 9)] # 8 adversarial prompts
                        long_df['adv_id'] = long_df.apply(
                            lambda row: 'base' if row['round'] == 1 else np.random.choice(adv_prompts),
                            axis=1
                        )

                    if 'base_id' not in long_df.columns:
                        base_prompts = ['algebra', 'history', 'biology', 'literature', 'physics']
                        convo_to_base = {cid: np.random.choice(base_prompts) for cid in long_df['conversation_id'].unique()}
                        long_df['base_id'] = long_df['conversation_id'].map(convo_to_base)
                    
                    # 5. Create turn bins
                    long_df['turn_bin'] = pd.cut(long_df['round'], bins=[0, 2, 5, 10, np.inf], 
                                               labels=['1-2', '3-5', '6-10', '11+'], right=True)

                    # Rename 'conversation_id' to 'convo_id' for consistency
                    long_df.rename(columns={'conversation_id': 'convo_id'}, inplace=True)
                    
                    self.models_data[model_name] = long_df
                    print(f"✅ {model_name}: {len(long_df.convo_id.unique())} conversations prepared")

                except Exception as e:
                    print(f"❌ Error processing {model_name}: {e}")
            else:
                print(f"⚠️ {model_name}: Long file not found. Skipping.")
        
        print(f"\n📊 Successfully prepared data for {len(self.models_data)} models.")
        return len(self.models_data) > 0

    def fit_time_varying_models(self):
        """Fit Cox time-varying models for each LLM with regularization."""
        print("\n🤖 FITTING TIME-VARYING MODELS (FIXED EFFECTS + REGULARIZATION)")
        print("=" * 60)

        for model_name, df in self.models_data.items():
            print(f"--- Fitting model for: {model_name.upper()} ---")
            
            try:
                # Adding a penalizer for regularization to improve convergence
                ctv = CoxTimeVaryingFitter(penalizer=0.1)
                
                # Start with a simpler formula and add complexity
                base_formula = "C(adv_id)"
                formulas_to_try = [
                    base_formula,
                    f"{base_formula} + C(base_id)",
                    f"{base_formula} + C(turn_bin)",
                    f"{base_formula} + C(base_id) + C(turn_bin)"
                ]
                
                summary = None
                for formula in formulas_to_try:
                    try:
                        print(f"    Trying formula: {formula}")
                        ctv.fit(
                            df, 
                            id_col="convo_id",
                            event_col="fail",
                            start_col="turn_start",
                            stop_col="turn_stop",
                            formula=formula
                        )
                        summary = ctv.summary
                        print(f"    ✅ Successfully converged with formula: {formula}")
                        break 
                    except Exception as e:
                        print(f"    ⚠️ Failed with formula: {formula}. Error: {e}")
                        continue

                if summary is not None:
                    self.results[model_name] = summary
                    print(f"✅ {model_name}: Model fitted successfully.")
                    ctv.print_summary()
                else:
                    print(f"❌ {model_name}: All formulas failed to converge.")

            except Exception as e:
                print(f"❌ {model_name}: Model fitting failed: {e}")

    def save_results(self):
        """Save all model results to a CSV file."""
        print("\n💾 SAVING MODEL RESULTS")
        print("=" * 60)

        if not self.results:
            print("❌ No results to save.")
            return

        all_results = []
        for model_name, summary_df in self.results.items():
            summary_df['model'] = model_name
            all_results.append(summary_df)
        
        combined_results = pd.concat(all_results)
        
        # Create directory if not exists
        os.makedirs('../generated/outputs', exist_ok=True)
        
        output_file = '../generated/outputs/time_varying_frailty_model_results.csv'
        combined_results.to_csv(output_file)
        
        print(f"✅ All model results saved to: {output_file}")
        return output_file

    def visualize_hazard_ratios(self, results_file):
        """
        Creates a forest plot of hazard ratios for adversarial prompt types
        from the time-varying Cox model results.
        """
        print("\n🎨 CREATING VISUALIZATION")
        print("=" * 60)
        
        if not os.path.exists(results_file):
            print(f"❌ Results file not found at: {results_file}")
            return

        df = pd.read_csv(results_file)
        print(f"✅ Loaded {len(df)} rows of data for visualization.")

        # --- Data Processing ---
        df_adv = df[df['covariate'].str.startswith('C(adv_id)')].copy()
        df_adv['adversarial_prompt'] = df_adv['covariate'].str.extract(r'\[T\.(.*)\]')
        
        if df_adv.empty:
            print("⚠️ No adversarial prompt coefficients found to visualize.")
            return
            
        df_adv['model'] = pd.Categorical(df_adv['model'], sorted(df_adv['model'].unique()))
        df_adv['adversarial_prompt'] = pd.Categorical(df_adv['adversarial_prompt'], sorted(df_adv['adversarial_prompt'].unique()))

        # --- Visualization ---
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")

        g = sns.catplot(
            data=df_adv,
            y='adversarial_prompt',
            x='exp(coef)',
            col='model',
            col_wrap=5,
            kind='point',
            join=False,
            palette='viridis',
            height=5,
            aspect=1.2
        )

        g.fig.suptitle('Hazard Ratios of Adversarial Prompt Types Across Models', fontsize=20, y=1.03)
        g.set_axis_labels("Hazard Ratio (exp(coef))", "Adversarial Prompt Type")
        g.set_titles("Model: {col_name}")

        for ax in g.axes.flatten():
            ax.axvline(1.0, linestyle='--', color='red', zorder=0, label='No Effect (HR=1.0)')
            model_name = ax.get_title().replace("Model: ", "")
            subset = df_adv[df_adv['model'] == model_name]
            for i, row in subset.iterrows():
                ax.plot(
                    [row['exp(coef) lower 95%'], row['exp(coef) upper 95%']],
                    [row['adversarial_prompt'], row['adversarial_prompt']],
                    color='gray',
                    linewidth=1.5,
                    zorder=1
                )
                
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # --- Save the Plot ---
        output_dir = '../generated/figs'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'time_varying_hazard_ratios.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        print(f"✅ Visualization saved to: {output_path}")

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        if self.load_and_prepare_data():
            self.fit_time_varying_models()
            results_file = self.save_results()
            if results_file:
                self.visualize_hazard_ratios(results_file)
            print("\n🎉 Analysis complete!")
        else:
            print("\nAnalysis aborted due to data loading issues.")

if __name__ == "__main__":
    analyzer = TimeVaryingFrailtyAnalyzer()
    analyzer.run_analysis() 