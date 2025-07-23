#!/usr/bin/env python3
"""
🔍 Prompt-to-Prompt Drift Analysis by Turns

This script analyzes how prompt-to-prompt drift changes over conversation turns
for each LLM model and creates comprehensive visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DriftByTurnsAnalyzer:
    def __init__(self):
        self.models_data = {}
        self.drift_by_turns = {}
        
    def load_long_data(self):
        """Load long-format data for all models."""
        print("\n🔍 LOADING LONG-FORMAT DATA FOR DRIFT ANALYSIS")
        print("=" * 60)
        
        processed_dir = 'processed_data'
        if not os.path.exists(processed_dir):
            print("❌ No processed_data directory found!")
            return
        
        model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        
        for model_name in tqdm(model_dirs, desc="Loading models"):
            long_file = os.path.join(processed_dir, model_name, f'{model_name}_long.csv')
            
            if os.path.exists(long_file):
                try:
                    long_df = pd.read_csv(long_file)
                    
                    # Check for required columns
                    required_cols = ['prompt_to_prompt_drift']
                    turn_col = None
                    
                    # Find turn/round column
                    for col in ['round', 'turn', 'time_step']:
                        if col in long_df.columns:
                            turn_col = col
                            break
                    
                    if turn_col and all(col in long_df.columns for col in required_cols):
                        self.models_data[model_name] = {
                            'data': long_df,
                            'turn_col': turn_col
                        }
                        print(f"✅ {model_name}: {len(long_df):,} turns, turn column: '{turn_col}'")
                    else:
                        print(f"⚠️ {model_name}: Missing required columns")
                        
                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")
        
        print(f"\n📊 Successfully loaded {len(self.models_data)} models for drift analysis")
    
    def calculate_drift_by_turns(self):
        """Calculate average drift by turn/round for each model."""
        print("\n📈 CALCULATING DRIFT BY TURNS FOR EACH MODEL")
        print("=" * 60)
        
        for model_name, model_info in tqdm(self.models_data.items(), desc="Analyzing drift"):
            long_df = model_info['data']
            turn_col = model_info['turn_col']
            
            try:
                # Group by turn and calculate statistics
                drift_stats = long_df.groupby(turn_col).agg({
                    'prompt_to_prompt_drift': ['mean', 'std', 'count', 'median', 'min', 'max']
                }).round(4)
                
                # Flatten column names
                drift_stats.columns = ['_'.join(col).strip() for col in drift_stats.columns]
                drift_stats = drift_stats.reset_index()
                drift_stats['model'] = model_name
                
                # Store results
                self.drift_by_turns[model_name] = drift_stats
                
                print(f"✅ {model_name}: {len(drift_stats)} turn levels analyzed")
                
            except Exception as e:
                print(f"❌ Error analyzing {model_name}: {e}")
        
        print(f"\n📊 Drift analysis completed for {len(self.drift_by_turns)} models")
    
    def create_comprehensive_visualization(self):
        """Create comprehensive drift by turns visualization as separate PNG files."""
        print("\n🎨 CREATING SEPARATE DRIFT BY TURNS VISUALIZATIONS")
        print("=" * 60)
        
        if not self.drift_by_turns:
            print("❌ No drift data available for visualization")
            return
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        os.makedirs('generated/figs', exist_ok=True)
        
        # Plot colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.drift_by_turns)))
        
        # ==================================================
        # PLOT 1: Main drift trends by turns
        # ==================================================
        plt.figure(figsize=(16, 10))
        
        # Plot each model's drift trend
        for i, (model_name, drift_data) in enumerate(self.drift_by_turns.items()):
            turn_col = [col for col in drift_data.columns if col.startswith('round') or col.startswith('turn') or col.startswith('time_step')][0]
            
            # Plot mean drift with error bars (std)
            plt.errorbar(
                drift_data[turn_col], 
                drift_data['prompt_to_prompt_drift_mean'],
                yerr=drift_data['prompt_to_prompt_drift_std'],
                label=model_name.replace('_', ' ').title(),
                marker='o',
                linewidth=3,
                markersize=8,
                alpha=0.8,
                color=colors[i],
                capsize=4,
                capthick=2
            )
        
        plt.xlabel('Conversation Turn/Round', fontsize=16, fontweight='bold')
        plt.ylabel('Average Prompt-to-Prompt Drift', fontsize=16, fontweight='bold')
        plt.title('📈 Prompt-to-Prompt Drift Evolution by Conversation Turns\nAll Models Comparison', fontsize=18, fontweight='bold', pad=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.grid(True, alpha=0.3, linewidth=1)
        plt.xlim(1, None)
        
        plt.tight_layout()
        
        # Add annotation with proper spacing
        plt.figtext(0.5, 0.01, '🔍 Shows how semantic drift changes as conversations progress across different LLM models', 
                   ha='center', fontsize=12, style='italic')
        
        plt.savefig('generated/figs/drift_evolution_trends.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        plt.close()
        
        print("✅ Plot 1 saved: 'generated/figs/drift_evolution_trends.png'")
        
        # ==================================================
        # PLOT 2: Heatmap of drift by model and turn
        # ==================================================
        plt.figure(figsize=(14, 10))
        
        # Prepare data for heatmap
        heatmap_data = []
        for model_name, drift_data in self.drift_by_turns.items():
            turn_col = [col for col in drift_data.columns if col.startswith('round') or col.startswith('turn') or col.startswith('time_step')][0]
            
            for _, row in drift_data.iterrows():
                heatmap_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Turn': row[turn_col],
                    'Drift': row['prompt_to_prompt_drift_mean']
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create pivot table for heatmap
        pivot_data = heatmap_df.pivot(index='Model', columns='Turn', values='Drift')
        
        # Limit to first 15 turns for readability
        max_turns = min(15, pivot_data.columns.max())
        pivot_data_subset = pivot_data.iloc[:, :max_turns]
        
        sns.heatmap(pivot_data_subset, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   center=pivot_data_subset.mean().mean(),
                   cbar_kws={'label': 'Average Drift'},
                   annot_kws={'fontsize': 10},
                   linewidths=0.5)
        
        plt.title('🔥 Drift Intensity Heatmap by Model and Turn\n(First 15 Conversation Turns)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Conversation Turn', fontsize=14, fontweight='bold')
        plt.ylabel('LLM Model', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Add annotation with proper spacing
        plt.figtext(0.5, 0.01, '🌡️ Red = High Drift (Semantic Inconsistency), Blue = Low Drift (Semantic Consistency)', 
                   ha='center', fontsize=11, style='italic')
        
        plt.savefig('generated/figs/drift_intensity_heatmap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        plt.close()
        
        print("✅ Plot 2 saved: 'generated/figs/drift_intensity_heatmap.png'")
        
        # ==================================================
        # PLOT 3: Model ranking by average drift
        # ==================================================
        plt.figure(figsize=(12, 10))
        
        # Calculate model-level statistics
        model_stats = []
        for model_name, drift_data in self.drift_by_turns.items():
            stats = {
                'Model': model_name.replace('_', ' ').title(),
                'Avg_Drift': drift_data['prompt_to_prompt_drift_mean'].mean(),
                'Max_Drift': drift_data['prompt_to_prompt_drift_mean'].max(),
                'Min_Drift': drift_data['prompt_to_prompt_drift_mean'].min(),
                'Drift_Range': drift_data['prompt_to_prompt_drift_mean'].max() - drift_data['prompt_to_prompt_drift_mean'].min(),
                'Turn_Count': len(drift_data)
            }
            model_stats.append(stats)
        
        model_stats_df = pd.DataFrame(model_stats).sort_values('Avg_Drift', ascending=True)
        
        # Bar plot of average drift by model
        bars = plt.barh(range(len(model_stats_df)), model_stats_df['Avg_Drift'], 
                       color=colors[:len(model_stats_df)], alpha=0.8, edgecolor='black', linewidth=1)
        
        plt.yticks(range(len(model_stats_df)), model_stats_df['Model'], fontsize=12)
        plt.xlabel('Average Prompt-to-Prompt Drift', fontsize=14, fontweight='bold')
        plt.title('📊 LLM Model Ranking by Average Drift\nLower Drift = More Semantically Consistent', fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3, linewidth=1)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, model_stats_df['Avg_Drift'])):
            plt.text(value + 0.001, i, f'{value:.3f}', 
                    va='center', fontweight='bold', fontsize=11)
        
        # Add ranking annotations
        for i, rank in enumerate(range(1, len(model_stats_df) + 1)):
            plt.text(-0.002, i, f'#{rank}', va='center', ha='right', 
                    fontweight='bold', fontsize=12, color='darkblue')
        
        plt.tight_layout()
        
        # Add annotation with proper spacing
        plt.figtext(0.5, 0.01, '🏆 Lower values indicate more consistent semantic flow across conversation turns', 
                   ha='center', fontsize=11, style='italic')
        
        plt.savefig('generated/figs/model_drift_rankings.png', 
                   dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
        plt.close()
        
        print("✅ Plot 3 saved: 'generated/figs/model_drift_rankings.png'")
        
        # ==================================================
        # Save data files
        # ==================================================
        os.makedirs('generated/outputs', exist_ok=True)
        
        # Combine all model data
        all_drift_data = []
        for model_name, drift_data in self.drift_by_turns.items():
            model_data = drift_data.copy()
            model_data['model'] = model_name
            all_drift_data.append(model_data)
        
        combined_drift_df = pd.concat(all_drift_data, ignore_index=True)
        combined_drift_df.to_csv('generated/outputs/drift_by_turns_analysis.csv', index=False)
        
        # Save model summary stats
        model_stats_df.to_csv('generated/outputs/drift_by_turns_model_summary.csv', index=False)
        
        print("✅ Data saved:")
        print(f"   📊 'generated/outputs/drift_by_turns_analysis.csv' ({len(combined_drift_df)} rows)")
        print(f"   📋 'generated/outputs/drift_by_turns_model_summary.csv' ({len(model_stats_df)} models)")
        
        print("\n🎉 THREE SEPARATE VISUALIZATIONS CREATED:")
        print("   📈 'generated/figs/drift_evolution_trends.png' - Trend lines for all models")
        print("   🔥 'generated/figs/drift_intensity_heatmap.png' - Turn-by-turn intensity map")
        print("   📊 'generated/figs/model_drift_rankings.png' - Model performance ranking")
    
    def create_detailed_turn_analysis(self):
        """Create detailed analysis of specific turns."""
        print("\n📋 CREATING DETAILED TURN ANALYSIS")
        print("=" * 50)
        
        # Focus on first 10 turns for detailed analysis
        detailed_analysis = []
        
        for model_name, drift_data in self.drift_by_turns.items():
            turn_col = [col for col in drift_data.columns if col.startswith('round') or col.startswith('turn') or col.startswith('time_step')][0]
            
            # Get first 10 turns
            first_10_turns = drift_data[drift_data[turn_col] <= 10].copy()
            
            for _, row in first_10_turns.iterrows():
                detailed_analysis.append({
                    'Model': model_name,
                    'Turn': row[turn_col],
                    'Mean_Drift': row['prompt_to_prompt_drift_mean'],
                    'Std_Drift': row['prompt_to_prompt_drift_std'],
                    'Median_Drift': row['prompt_to_prompt_drift_median'],
                    'Min_Drift': row['prompt_to_prompt_drift_min'],
                    'Max_Drift': row['prompt_to_prompt_drift_max'],
                    'N_Observations': row['prompt_to_prompt_drift_count']
                })
        
        detailed_df = pd.DataFrame(detailed_analysis)
        detailed_df.to_csv('generated/outputs/detailed_drift_by_turns_first_10.csv', index=False)
        
        print(f"✅ Detailed turn analysis saved: {len(detailed_df)} turn-model combinations")
        
        # Print summary insights
        print("\n🎯 KEY INSIGHTS:")
        
        # Find models with increasing vs decreasing drift trends
        trend_analysis = {}
        for model_name, drift_data in self.drift_by_turns.items():
            turn_col = [col for col in drift_data.columns if col.startswith('round') or col.startswith('turn') or col.startswith('time_step')][0]
            
            first_5_turns = drift_data[drift_data[turn_col] <= 5]
            if len(first_5_turns) >= 3:
                first_turn_drift = first_5_turns['prompt_to_prompt_drift_mean'].iloc[0]
                last_turn_drift = first_5_turns['prompt_to_prompt_drift_mean'].iloc[-1]
                trend = "Increasing" if last_turn_drift > first_turn_drift else "Decreasing"
                trend_analysis[model_name] = {
                    'trend': trend,
                    'change': last_turn_drift - first_turn_drift
                }
        
        increasing_models = [m for m, t in trend_analysis.items() if t['trend'] == 'Increasing']
        decreasing_models = [m for m, t in trend_analysis.items() if t['trend'] == 'Decreasing']
        
        print(f"   📈 Models with INCREASING drift over first 5 turns: {len(increasing_models)}")
        for model in increasing_models[:3]:  # Show top 3
            change = trend_analysis[model]['change']
            print(f"      • {model}: +{change:.3f} drift increase")
        
        print(f"   📉 Models with DECREASING drift over first 5 turns: {len(decreasing_models)}")
        for model in decreasing_models[:3]:  # Show top 3
            change = abs(trend_analysis[model]['change'])
            print(f"      • {model}: -{change:.3f} drift decrease")
        
    def run_complete_analysis(self):
        """Run the complete drift by turns analysis."""
        print("\n🚀 STARTING COMPLETE DRIFT BY TURNS ANALYSIS")
        print("=" * 70)
        
        # Load data
        self.load_long_data()
        
        if not self.models_data:
            print("❌ No model data loaded. Exiting.")
            return
        
        # Calculate drift statistics
        self.calculate_drift_by_turns()
        
        # Create visualizations
        self.create_comprehensive_visualization()
        
        # Detailed analysis
        self.create_detailed_turn_analysis()
        
        print("\n🎉 COMPLETE DRIFT BY TURNS ANALYSIS FINISHED!")
        print("=" * 50)
        print("📁 Generated Visualization Files:")
        print("   📈 'generated/figs/drift_evolution_trends.png' - Trend lines for all models")
        print("   🔥 'generated/figs/drift_intensity_heatmap.png' - Turn-by-turn intensity map")
        print("   📊 'generated/figs/model_drift_rankings.png' - Model performance ranking")
        print("\n📁 Generated Data Files:")
        print("   📊 'generated/outputs/drift_by_turns_analysis.csv'")
        print("   📋 'generated/outputs/drift_by_turns_model_summary.csv'")
        print("   🔍 'generated/outputs/detailed_drift_by_turns_first_10.csv'")

if __name__ == "__main__":
    analyzer = DriftByTurnsAnalyzer()
    analyzer.run_complete_analysis() 