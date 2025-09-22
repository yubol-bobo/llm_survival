#!/usr/bin/env python3
"""
AFT Model Visualization Suite
Comprehensive visualizations for Accelerated Failure Time models
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import ast
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

class AFTVisualizer:
    """Create comprehensive visualizations for AFT model results"""
    
    def __init__(self):
        self.results_dir = 'results/outputs/aft'
        self.figures_dir = 'results/figures/aft'
        self.data = {}
        self._round_level_data = None

        # Create figures directory if it doesn't exist
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def load_results(self):
        """Load all AFT results"""
        print("📊 LOADING AFT RESULTS FOR VISUALIZATION")
        print("=" * 50)
        
        try:
            # Load main results files
            files_to_load = {
                'model_comparison': 'model_comparison.csv',
                'model_performance': 'model_performance.csv',
                'all_coefficients': 'all_coefficients.csv',
                'feature_importance': 'feature_importance.csv',
                'model_rankings': 'model_rankings.csv'
            }
            
            for key, filename in files_to_load.items():
                filepath = os.path.join(self.results_dir, filename)
                if os.path.exists(filepath):
                    self.data[key] = pd.read_csv(filepath)
                    print(f"✅ Loaded {filename}: {self.data[key].shape}")
                else:
                    print(f"⚠️  Missing {filename}")
            
            # Load interaction coefficients if available
            interaction_file = os.path.join(self.results_dir, 'interaction_coefficients.csv')
            if os.path.exists(interaction_file):
                self.data['interaction_coefficients'] = pd.read_csv(interaction_file)
                print(f"✅ Loaded interaction_coefficients.csv: {self.data['interaction_coefficients'].shape}")
            
            print(f"📈 Ready to create visualizations with {len(self.data)} datasets")
            return True
            
        except Exception as e:
            print(f"❌ Loading failed: {e}")
            return False
    
    def plot_model_performance_comparison(self):
        """Create comprehensive model performance comparison"""
        print("\n📊 CREATING MODEL PERFORMANCE COMPARISON")
        print("=" * 50)
        
        if 'model_comparison' not in self.data:
            print("❌ No model comparison data available")
            return
            
        try:
            comp_df = self.data['model_comparison'].copy()
            
            # Debug: Print data info
            print(f"🔍 Data shape: {comp_df.shape}")
            print(f"🔍 C-index range: {comp_df['c_index'].min():.4f} - {comp_df['c_index'].max():.4f}")
            print(f"🔍 AIC range: {comp_df['aic'].min():.1f} - {comp_df['aic'].max():.1f}")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AFT Model Performance Comparison', fontsize=16, fontweight='bold')
            
            # 1. C-index comparison (top-left)
            ax1 = axes[0, 0]
            models = comp_df['model_type'].values
            c_indices = comp_df['c_index'].values
            
            # Ensure we have valid data
            if len(models) == 0 or len(c_indices) == 0:
                ax1.text(0.5, 0.5, 'No C-index data available', ha='center', va='center', transform=ax1.transAxes)
            else:
                # Color coding: best model in gold, others in skyblue  
                best_idx = np.argmax(c_indices)
                colors = ['gold' if i == best_idx else 'skyblue' for i in range(len(models))]
                
                bars1 = ax1.barh(range(len(models)), c_indices, color=colors, alpha=0.8)
                ax1.set_yticks(range(len(models)))
                ax1.set_yticklabels([self._truncate_label(model, 18) for model in models])
                ax1.set_xlabel('C-index (Concordance Index)')
                ax1.set_title('Predictive Performance\n(Higher is Better)', fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')
                
                # Set reasonable x-axis limits
                c_min, c_max = c_indices.min(), c_indices.max()
                ax1.set_xlim(c_min - 0.005, c_max + 0.005)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars1, c_indices)):
                    ax1.text(val + (c_max - c_min) * 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{val:.4f}', va='center', ha='left',
                            fontweight='bold' if i == best_idx else 'normal')
                
                # Add performance threshold line
                ax1.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, label='Excellent (0.8)')
                ax1.legend()
            
            # 2. AIC comparison (top-right)
            ax2 = axes[0, 1]
            aic_values = comp_df['aic'].values
            
            if len(aic_values) == 0:
                ax2.text(0.5, 0.5, 'No AIC data available', ha='center', va='center', transform=ax2.transAxes)
            else:
                best_aic_idx = np.argmin(aic_values)
                colors2 = ['gold' if i == best_aic_idx else 'lightcoral' for i in range(len(models))]
                
                bars2 = ax2.barh(range(len(models)), aic_values, color=colors2, alpha=0.8)
                ax2.set_yticks(range(len(models)))
                ax2.set_yticklabels([self._truncate_label(model, 18) for model in models])
                ax2.set_xlabel('AIC (Akaike Information Criterion)')
                ax2.set_title('Model Fit Quality\n(Lower is Better)', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                aic_range = aic_values.max() - aic_values.min()
                for i, (bar, val) in enumerate(zip(bars2, aic_values)):
                    ax2.text(val + aic_range * 0.02, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
                            va='center', ha='left', fontweight='bold' if i == best_aic_idx else 'normal')
            
            # 3. BIC comparison (bottom-left)
            ax3 = axes[1, 0]
            bic_values = comp_df['bic'].values
            
            if len(bic_values) == 0:
                ax3.text(0.5, 0.5, 'No BIC data available', ha='center', va='center', transform=ax3.transAxes)
            else:
                best_bic_idx = np.argmin(bic_values)
                colors3 = ['gold' if i == best_bic_idx else 'lightgreen' for i in range(len(models))]
                
                bars3 = ax3.barh(range(len(models)), bic_values, color=colors3, alpha=0.8)
                ax3.set_yticks(range(len(models)))
                ax3.set_yticklabels([self._truncate_label(model, 18) for model in models])
                ax3.set_xlabel('BIC (Bayesian Information Criterion)')
                ax3.set_title('Model Complexity vs Fit\n(Lower is Better)', fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                bic_range = bic_values.max() - bic_values.min()
                for i, (bar, val) in enumerate(zip(bars3, bic_values)):
                    ax3.text(val + bic_range * 0.02, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
                            va='center', ha='left', fontweight='bold' if i == best_bic_idx else 'normal')
            
            # 4. Model summary table (bottom-right)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            if len(comp_df) > 0:
                # Create summary table
                summary_data = []
                best_c_index = comp_df.loc[comp_df['c_index'].idxmax()]
                best_aic = comp_df.loc[comp_df['aic'].idxmin()]
                best_bic = comp_df.loc[comp_df['bic'].idxmin()]
                
                summary_data.append(['Best C-index', self._truncate_label(best_c_index['model_type'], 15), f"{best_c_index['c_index']:.4f}"])
                summary_data.append(['Best AIC', self._truncate_label(best_aic['model_type'], 15), f"{best_aic['aic']:.1f}"])
                summary_data.append(['Best BIC', self._truncate_label(best_bic['model_type'], 15), f"{best_bic['bic']:.1f}"])
                
                # Add performance insights
                avg_c_index = comp_df['c_index'].mean()
                best_improvement = ((best_c_index['c_index'] - avg_c_index) / avg_c_index) * 100
                
                summary_data.append(['Avg C-index', 'All Models', f"{avg_c_index:.4f}"])
                summary_data.append(['Best Improvement', f"+{best_improvement:.1f}%", 'vs Average'])
                
                # Create table
                table = ax4.table(cellText=summary_data, 
                                colLabels=['Metric', 'Best Model', 'Value'],
                                cellLoc='center', loc='center',
                                colColours=['lightgray']*3)
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.8)
                ax4.set_title('Performance Summary', fontweight='bold', pad=20)
            else:
                ax4.text(0.5, 0.5, 'No summary data available', ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_performance_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()  # Close figure to free memory
            
            print(f"✅ Model performance comparison saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Performance comparison failed: {e}")
    
    def plot_feature_importance_analysis(self):
        """Create feature importance analysis visualization"""
        print("\n🎯 CREATING FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        if 'feature_importance' not in self.data:
            print("❌ No feature importance data available")
            return
            
        try:
            importance_df = self.data['feature_importance'].copy()
            
            # Clean feature names
            importance_df['clean_feature'] = importance_df['feature'].apply(self._clean_feature_name)
            
            # Get top features
            top_features = importance_df.head(12)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AFT Model Feature Importance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Top feature coefficients (top-left)
            ax1 = axes[0, 0]
            
            # Color by effect direction
            colors = ['red' if coef < 0 else 'green' for coef in top_features['coef']]
            
            bars1 = ax1.barh(range(len(top_features)), abs(top_features['coef']), color=colors, alpha=0.7)
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels([self._truncate_label(name, 20) for name in top_features['clean_feature']])
            ax1.set_xlabel('Absolute Coefficient Value')
            ax1.set_title('Feature Impact Magnitude\n(Red=Accelerates Failure, Green=Delays Failure)', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, val, significance in zip(bars1, abs(top_features['coef']), top_features['significance']):
                ax1.text(val + max(abs(top_features['coef']))*0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}{significance}', va='center', fontsize=9)
            
            # 2. Acceleration factors (top-right)
            ax2 = axes[0, 1]
            
            # Filter out extreme values for better visualization
            acc_factors = top_features['acceleration_factor'].copy()
            acc_factors = np.clip(acc_factors, 0.01, 10)  # Clip extreme values
            
            colors2 = ['red' if af < 1 else 'green' if af > 1 else 'gray' for af in acc_factors]
            
            bars2 = ax2.barh(range(len(top_features)), acc_factors, color=colors2, alpha=0.7)
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels([self._truncate_label(name, 20) for name in top_features['clean_feature']])
            ax2.set_xlabel('Acceleration Factor (clipped 0.01-10)')
            ax2.set_title('Survival Time Acceleration\n(>1 = Protective, <1 = Harmful)', fontweight='bold')
            ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5, label='Neutral (1.0)')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.legend()
            
            # Add value labels
            for bar, val_orig, val_clip in zip(bars2, top_features['acceleration_factor'], acc_factors):
                label = f'{val_clip:.2f}' if val_orig == val_clip else f'{val_clip:.2f}*'
                ax2.text(val_clip + 0.1, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=9)
            
            # 3. P-value significance (bottom-left)
            ax3 = axes[1, 0]
            
            # Create significance categories
            p_values = top_features['p']
            sig_categories = []
            colors3 = []
            
            for p in p_values:
                if p < 0.001:
                    sig_categories.append('p < 0.001\n(***)')
                    colors3.append('darkgreen')
                elif p < 0.01:
                    sig_categories.append('p < 0.01\n(**)')
                    colors3.append('green')
                elif p < 0.05:
                    sig_categories.append('p < 0.05\n(*)')
                    colors3.append('orange')
                else:
                    sig_categories.append('p ≥ 0.05\n(ns)')
                    colors3.append('red')
            
            # Plot negative log p-values for better visualization
            neg_log_p = -np.log10(p_values + 1e-16)  # Add small constant to avoid log(0)
            
            bars3 = ax3.barh(range(len(top_features)), neg_log_p, color=colors3, alpha=0.7)
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels([self._truncate_label(name, 20) for name in top_features['clean_feature']])
            ax3.set_xlabel('-log₁₀(p-value)')
            ax3.set_title('Statistical Significance\n(Higher bars = More significant)', fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add significance thresholds
            ax3.axvline(x=-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p=0.05')
            ax3.axvline(x=-np.log10(0.01), color='green', linestyle='--', alpha=0.7, label='p=0.01')
            ax3.axvline(x=-np.log10(0.001), color='darkgreen', linestyle='--', alpha=0.7, label='p=0.001')
            ax3.legend()
            
            # 4. Effect direction summary (bottom-right)
            ax4 = axes[1, 1]
            
            # Count effect directions
            effect_counts = top_features['effect_direction'].value_counts()
            colors4 = ['red' if 'Accelerates' in effect else 'green' for effect in effect_counts.index]
            
            wedges, texts, autotexts = ax4.pie(effect_counts.values, labels=effect_counts.index, 
                                             colors=colors4, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Feature Effect Distribution\n(Top Features)', fontweight='bold')
            
            # Make text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_feature_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Feature importance analysis saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Feature importance analysis failed: {e}")
    
    def plot_model_coefficients_heatmap(self):
        """Create coefficient heatmap across all models"""
        print("\n🔥 CREATING MODEL COEFFICIENTS HEATMAP")
        print("=" * 50)
        
        if 'all_coefficients' not in self.data:
            print("❌ No coefficients data available")
            return
            
        try:
            coef_df = self.data['all_coefficients'].copy()
            
            # Clean up feature names and model names
            coef_df['clean_feature'] = coef_df['feature'].apply(self._clean_feature_name)
            coef_df['clean_model'] = coef_df['model_name'].str.replace('_aft', '').str.replace('_interactions', ' + Int')
            
            # Pivot to create heatmap data
            heatmap_data = coef_df.pivot_table(index='clean_feature', columns='clean_model', 
                                             values='coef', aggfunc='mean')
            
            # Filter to most important features (by average absolute coefficient)
            feature_importance = heatmap_data.abs().mean(axis=1).sort_values(ascending=False)
            top_features = feature_importance.head(15).index
            heatmap_subset = heatmap_data.loc[top_features]
            
            # Create figure
            plt.figure(figsize=(14, 10))
            
            # Create heatmap
            mask = heatmap_subset.isnull()
            sns.heatmap(heatmap_subset, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       mask=mask, cbar_kws={'label': 'Coefficient Value'})
            
            plt.title('AFT Model Coefficients Heatmap\n(Top 15 Features)', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Model Type', fontweight='bold')
            plt.ylabel('Features', fontweight='bold')
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add interpretation guide
            plt.figtext(0.02, 0.02, 'Red = Accelerates Failure (Harmful) | Blue = Delays Failure (Protective)', 
                       fontsize=10, style='italic')
            
            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_coefficients_heatmap.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Coefficients heatmap saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Coefficients heatmap failed: {e}")
    
    def plot_model_rankings_dashboard(self):
        """Create comprehensive model rankings dashboard"""
        print("\n🏆 CREATING MODEL RANKINGS DASHBOARD")
        print("=" * 50)
        
        if 'model_rankings' not in self.data or 'model_performance' not in self.data:
            print("❌ Missing required data for rankings dashboard")
            return
            
        try:
            rankings_df = self.data['model_rankings'].copy()
            performance_df = self.data['model_performance'].copy()
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AFT Model Rankings Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Best models by metric (top-left)
            ax1 = axes[0, 0]
            
            metrics = rankings_df['metric'].values
            models = [self._truncate_label(model, 15) for model in rankings_df['best_model'].values]
            values = rankings_df['best_value'].values
            
            colors_metrics = ['gold', 'silver', 'orange']
            bars1 = ax1.bar(metrics, values, color=colors_metrics, alpha=0.8)
            
            ax1.set_ylabel('Best Value')
            ax1.set_title('Best Models by Metric', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add model names on bars
            for bar, model, value in zip(bars1, models, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{model}\n{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 2. Model performance distribution (top-right)
            ax2 = axes[0, 1]
            
            # Create performance score (normalized combination of metrics)
            perf_subset = performance_df[performance_df['c_index'].notna()].copy()
            
            if len(perf_subset) > 0:
                # Normalize C-index (higher is better)
                c_index_norm = (perf_subset['c_index'] - perf_subset['c_index'].min()) / \
                              (perf_subset['c_index'].max() - perf_subset['c_index'].min())
                
                # Normalize AIC (lower is better, so invert)
                aic_norm = (perf_subset['aic'].max() - perf_subset['aic']) / \
                          (perf_subset['aic'].max() - perf_subset['aic'].min())
                
                # Combined performance score
                perf_subset['performance_score'] = (c_index_norm + aic_norm) / 2
                perf_subset = perf_subset.sort_values('performance_score', ascending=False)
                
                model_names = [self._truncate_label(name.replace('_aft', '').replace('_interactions', ' + Int'), 12) 
                              for name in perf_subset['model_name']]
                
                colors2 = plt.cm.RdYlGn(perf_subset['performance_score'])
                bars2 = ax2.barh(model_names, perf_subset['performance_score'], color=colors2)
                
                ax2.set_xlabel('Performance Score (0-1)')
                ax2.set_title('Overall Model Performance\n(C-index + AIC combined)', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')
                
                # Add score labels
                for bar, score in zip(bars2, perf_subset['performance_score']):
                    ax2.text(score + 0.02, bar.get_y() + bar.get_height()/2, 
                            f'{score:.3f}', va='center', fontsize=9)
            
            # 3. Model complexity analysis (bottom-left)
            ax3 = axes[1, 0]
            
            # Compare basic vs interaction models
            basic_models = performance_df[~performance_df['model_name'].str.contains('interactions', na=False)]
            interaction_models = performance_df[performance_df['model_name'].str.contains('interactions', na=False)]
            
            categories = ['Basic Models', 'With Interactions']
            avg_c_index = [basic_models['c_index'].mean(), interaction_models['c_index'].mean()]
            avg_aic = [basic_models['aic'].mean(), interaction_models['aic'].mean()]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars3a = ax3.bar(x - width/2, avg_c_index, width, label='Avg C-index', color='skyblue', alpha=0.8)
            
            # Create secondary y-axis for AIC
            ax3_twin = ax3.twinx()
            bars3b = ax3_twin.bar(x + width/2, avg_aic, width, label='Avg AIC', color='lightcoral', alpha=0.8)
            
            ax3.set_xlabel('Model Type')
            ax3.set_ylabel('Average C-index', color='blue')
            ax3_twin.set_ylabel('Average AIC', color='red')
            ax3.set_title('Basic vs Interaction Models', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(categories)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars3a, avg_c_index):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9, color='blue')
            
            for bar, val in zip(bars3b, avg_aic):
                ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                             f'{val:.0f}', ha='center', va='bottom', fontsize=9, color='red')
            
            # 4. Recommendation summary (bottom-right)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create recommendations text
            recommendations = []
            
            if len(rankings_df) > 0:
                best_overall = rankings_df[rankings_df['metric'] == 'c_index'].iloc[0] if len(rankings_df[rankings_df['metric'] == 'c_index']) > 0 else None
                best_parsimony = rankings_df[rankings_df['metric'] == 'aic'].iloc[0] if len(rankings_df[rankings_df['metric'] == 'aic']) > 0 else None
                
                if best_overall is not None:
                    recommendations.append(f"🏆 BEST OVERALL MODEL:")
                    recommendations.append(f"   {best_overall['best_model']}")
                    recommendations.append(f"   C-index: {best_overall['best_value']:.4f}")
                    recommendations.append("")
                
                if best_parsimony is not None:
                    recommendations.append(f"⚖️ MOST PARSIMONIOUS:")
                    recommendations.append(f"   {best_parsimony['best_model']}")
                    recommendations.append(f"   AIC: {best_parsimony['best_value']:.1f}")
                    recommendations.append("")
                
                recommendations.extend([
                    "📊 KEY INSIGHTS:",
                    "• All AFT models show excellent performance (C > 0.82)",
                    "• Log-Normal AFT offers best parsimony",
                    "• Log-Logistic AFT provides best predictions",
                    "• Interactions don't always improve performance",
                    "",
                    "🎯 RECOMMENDATION:",
                    "Use Log-Logistic AFT for production deployment"
                ])
            
            # Display recommendations
            recommendation_text = '\n'.join(recommendations)
            ax4.text(0.05, 0.95, recommendation_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_rankings_dashboard.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Model rankings dashboard saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Rankings dashboard failed: {e}")
    
    def plot_survival_insights_analysis(self):
        """Create survival insights analysis"""
        print("\n📈 CREATING SURVIVAL INSIGHTS ANALYSIS")
        print("=" * 50)
        
        try:
            # Load the best performing model results for survival analysis
            if 'feature_importance' not in self.data:
                print("❌ No feature importance data for survival analysis")
                return
            
            importance_df = self.data['feature_importance'].copy()
            
            # Create survival scenarios based on feature importance
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AFT Model Survival Insights Analysis', fontsize=16, fontweight='bold')
            
            # 1. Risk factors ranking (top-left)
            ax1 = axes[0, 0]
            
            # Get risk factors (negative coefficients = accelerate failure)
            risk_factors = importance_df[importance_df['coef'] < 0].copy()
            risk_factors['risk_magnitude'] = abs(risk_factors['coef'])
            risk_factors = risk_factors.sort_values('risk_magnitude', ascending=True).tail(8)
            
            if len(risk_factors) > 0:
                clean_names = [self._clean_feature_name(name) for name in risk_factors['feature']]
                clean_names = [self._truncate_label(name, 25) for name in clean_names]
                
                bars1 = ax1.barh(clean_names, risk_factors['risk_magnitude'], color='red', alpha=0.7)
                ax1.set_xlabel('Risk Magnitude (|Coefficient|)')
                ax1.set_title('Top Risk Factors\n(Features that Accelerate Failure)', fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='x')
                
                # Add significance indicators
                for bar, significance in zip(bars1, risk_factors['significance']):
                    ax1.text(bar.get_width() + max(risk_factors['risk_magnitude'])*0.02, 
                            bar.get_y() + bar.get_height()/2, significance, va='center', fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'No significant risk factors found', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Top Risk Factors', fontweight='bold')
            
            # 2. Protective factors ranking (top-right)
            ax2 = axes[0, 1]
            
            # Get protective factors (positive coefficients = delay failure)
            protective_factors = importance_df[importance_df['coef'] > 0].copy()
            protective_factors['protection_magnitude'] = protective_factors['coef']
            protective_factors = protective_factors.sort_values('protection_magnitude', ascending=True).tail(8)
            
            if len(protective_factors) > 0:
                clean_names2 = [self._clean_feature_name(name) for name in protective_factors['feature']]
                clean_names2 = [self._truncate_label(name, 25) for name in clean_names2]
                
                bars2 = ax2.barh(clean_names2, protective_factors['protection_magnitude'], color='green', alpha=0.7)
                ax2.set_xlabel('Protection Magnitude (Coefficient)')
                ax2.set_title('Top Protective Factors\n(Features that Delay Failure)', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')
                
                # Add significance indicators
                for bar, significance in zip(bars2, protective_factors['significance']):
                    ax2.text(bar.get_width() + max(protective_factors['protection_magnitude'])*0.02, 
                            bar.get_y() + bar.get_height()/2, significance, va='center', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No significant protective factors found', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Top Protective Factors', fontweight='bold')
            
            # 3. Effect sizes distribution (bottom-left)
            ax3 = axes[1, 0]
            
            # Create histogram of effect sizes
            all_effects = importance_df['coef'].values
            
            ax3.hist(all_effects, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral Effect')
            ax3.set_xlabel('Coefficient Value')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Feature Effects', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Add annotations
            risk_count = len(importance_df[importance_df['coef'] < 0])
            protective_count = len(importance_df[importance_df['coef'] > 0])
            
            annotation_text = f'Risk Factors: {risk_count}\nProtective Factors: {protective_count}'
            ax3.text(0.02, 0.98, annotation_text, transform=ax3.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 4. Model insights summary (bottom-right)
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Generate insights summary
            insights = []
            
            if len(risk_factors) > 0:
                top_risk = risk_factors.iloc[-1]
                insights.append("🚨 HIGHEST RISK FACTOR:")
                insights.append(f"   {self._clean_feature_name(top_risk['feature'])}")
                insights.append(f"   Impact: {top_risk['risk_magnitude']:.3f}")
                insights.append("")
            
            if len(protective_factors) > 0:
                top_protection = protective_factors.iloc[-1]
                insights.append("🛡️ STRONGEST PROTECTION:")
                insights.append(f"   {self._clean_feature_name(top_protection['feature'])}")
                insights.append(f"   Impact: +{top_protection['protection_magnitude']:.3f}")
                insights.append("")
            
            # Overall model insights
            if 'model_comparison' in self.data:
                best_model = self.data['model_comparison'].loc[self.data['model_comparison']['c_index'].idxmax()]
                insights.extend([
                    "📊 MODEL PERFORMANCE:",
                    f"   Best Model: {best_model['model_type']}",
                    f"   C-index: {best_model['c_index']:.4f}",
                    f"   Excellent Performance: {'Yes' if best_model['c_index'] > 0.8 else 'No'}",
                    "",
                    "🎯 DEPLOYMENT INSIGHTS:",
                    "• Monitor high-risk features closely",
                    "• Leverage protective factors in design",
                    "• Implement early warning systems",
                    "• Consider intervention thresholds"
                ])
            
            insights_text = '\n'.join(insights)
            ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_survival_insights.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Survival insights analysis saved to {save_path}")
            
        except Exception as e:
            print(f"❌ Survival insights analysis failed: {e}")
    
    def _load_round_level_data(self):
        """Load combined long-format conversation data for round tracking."""
        if self._round_level_data is not None:
            return self._round_level_data

        processed_dir = Path('data/processed')
        if not processed_dir.exists():
            print("�?O Processed data directory missing for driver dynamics")
            return None

        frames = []
        for model_dir in processed_dir.iterdir():
            if not model_dir.is_dir():
                continue

            long_path = model_dir / f"{model_dir.name}_long.csv"
            if not long_path.exists():
                continue

            try:
                df = pd.read_csv(long_path)
            except Exception as exc:
                print(f"�?O Failed to read {long_path}: {exc}")
                continue

            if 'round' not in df.columns:
                continue

            df = df[df['round'].between(1, 8)].copy()
            if df.empty:
                continue

            if 'model' in df.columns:
                model_dummies = pd.get_dummies(df['model'], prefix='model', drop_first=True)
                df = pd.concat([df, model_dummies], axis=1)

            if 'difficulty_level' not in df.columns and 'level' in df.columns:
                df['difficulty_level'] = df['level']

            if 'difficulty_level' in df.columns:
                diff_dummies = pd.get_dummies(df['difficulty_level'], prefix='difficulty', drop_first=True)
                df = pd.concat([df, diff_dummies], axis=1)

            if 'subject_cluster' in df.columns:
                subject_dummies = pd.get_dummies(df['subject_cluster'], prefix='subject', drop_first=True)
                df = pd.concat([df, subject_dummies], axis=1)

            frames.append(df)

        if not frames:
            print("�?O No round-level datasets available")
            return None

        self._round_level_data = pd.concat(frames, ignore_index=True)
        return self._round_level_data

    def _get_best_model_coefficients(self):
        """Fetch coefficients for the top-performing AFT model."""
        if 'model_performance' not in self.data or 'all_coefficients' not in self.data:
            return None, None

        perf_df = self.data['model_performance']
        perf_df = perf_df[perf_df['c_index'].notna()]
        if perf_df.empty:
            return None, None

        best_row = perf_df.sort_values('c_index', ascending=False).iloc[0]
        best_model = best_row['model_name']

        coeff_df = self.data['all_coefficients']
        coeff_df = coeff_df[coeff_df['model_name'] == best_model].copy()
        if coeff_df.empty:
            return best_model, None

        def _extract_tokens(raw_value):
            try:
                parsed = ast.literal_eval(raw_value)
            except Exception:
                return None, None

            if isinstance(parsed, tuple) and len(parsed) >= 2:
                return parsed[0], parsed[1]
            return None, None

        coeff_df[['param', 'feature_name']] = coeff_df['feature'].apply(
            lambda item: pd.Series(_extract_tokens(item))
        )
        coeff_df = coeff_df[coeff_df['feature_name'].notna()]
        if coeff_df.empty:
            return best_model, None

        primary_params = {'lambda_', 'mu_', 'alpha_'}
        filtered = coeff_df[coeff_df['param'].isin(primary_params)].copy()
        if filtered.empty:
            filtered = coeff_df.copy()

        coeff_map = dict(zip(filtered['feature_name'], filtered['coef'].astype(float)))
        return best_model, coeff_map

    def plot_driver_dynamics_over_time(self):
        """Plot how driver effects evolve over the eight follow-up rounds."""
        print("\n🎬 CREATING DRIVER DYNAMICS OVER TIME")
        print("=" * 60)

        # Load round-level data
        combined_df = self._load_round_level_data()
        if combined_df is None:
            print("❌ Driver dynamics skipped (data unavailable)")
            return

        # Get best model coefficients 
        if 'feature_importance' not in self.data:
            print("❌ Driver dynamics skipped (feature importance unavailable)")
            return
            
        # Use feature importance data which has clean coefficient mapping
        importance_df = self.data['feature_importance'].copy()
        
        # Create coefficient map from feature importance data
        coeff_map = {}
        for _, row in importance_df.iterrows():
            feature_name = str(row['feature'])
            # Extract clean feature name from tuple format
            if 'prompt_to_prompt_drift' in feature_name:
                coeff_map['prompt_to_prompt_drift'] = row['coef']
            elif 'context_to_prompt_drift' in feature_name:
                coeff_map['context_to_prompt_drift'] = row['coef']
            elif 'cumulative_drift' in feature_name:
                coeff_map['cumulative_drift'] = row['coef']
            elif 'prompt_complexity' in feature_name:
                coeff_map['prompt_complexity'] = row['coef']
        
        print(f"📊 Found coefficients for {len(coeff_map)} drift features")
        
        # Focus on drift features that we have coefficients for
        drift_features = [feat for feat in ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                                          'cumulative_drift', 'prompt_complexity'] if feat in coeff_map]
        
        if not drift_features:
            print("❌ No drift features with coefficients found")
            return
            
        # Keep all data - we now have drift data for all 8 rounds!  
        print(f"📈 Data loaded: {len(combined_df)} observations across {combined_df['round'].nunique()} rounds")
        
        # Check data availability by round
        for i, feature in enumerate(drift_features):
            non_null_by_round = combined_df.groupby('round')[feature].count()
            total_by_round = combined_df.groupby('round').size()
            print(f"   {feature}: available for rounds {list(non_null_by_round[non_null_by_round > 0].index)}")
            for round_num in sorted(combined_df['round'].unique()):
                avail = non_null_by_round.get(round_num, 0)
                total = total_by_round.get(round_num, 0)
                pct = (avail/total*100) if total > 0 else 0
                print(f"     Round {round_num}: {avail}/{total} ({pct:.1f}%)")
        
        clean_df = combined_df.copy()  # Keep all data
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AFT Driver Dynamics: How Risk Factors Evolve Across 8 Rounds', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Individual drift features over time (top-left)
        ax1 = axes[0, 0]
        colors = ['red', 'orange', 'green', 'blue']
        
        # Force all 8 rounds to be shown
        all_rounds = list(range(1, 9))
        
        for i, feature in enumerate(drift_features):
            # Calculate mean feature value by round (now we have all 8 rounds!)
            round_means = clean_df.groupby('round')[feature].mean()
            
            # Get data for all available rounds (should be 1-8)
            available_rounds = [r for r in all_rounds if r in round_means.index and not pd.isna(round_means[r])]
            available_means = [round_means[r] for r in available_rounds]
            
            if len(available_rounds) > 0:
                # Calculate acceleration factor: exp(coeff * mean_value)
                coeff = coeff_map[feature]
                acceleration_factors = np.exp(coeff * np.array(available_means))
                
                # Plot all available data with solid line
                ax1.plot(available_rounds, acceleration_factors, 
                        marker='o', linewidth=2, color=colors[i % len(colors)], linestyle='-',
                        label=f'{self._clean_feature_name(feature)} (β={coeff:.3f})')
                
                print(f"   Plotted {feature}: {len(available_rounds)} rounds ({min(available_rounds)}-{max(available_rounds)})")
        
        ax1.axhline(1.0, color='black', linestyle='--', alpha=0.7, label='Neutral (AF=1.0)')
        ax1.set_xlabel('Follow-up Round')
        ax1.set_ylabel('Acceleration Factor')
        ax1.set_title('Individual Drift Feature Evolution')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visibility
        ax1.set_xlim(0.5, 8.5)  # Show all 8 rounds
        ax1.set_xticks(range(1, 9))  # Show all round numbers
        
        # Plot 2: Combined risk score over time (top-right)
        ax2 = axes[0, 1]
        
        # Calculate combined risk score for all 8 rounds
        combined_risk = []
        risk_rounds = []
        
        for round_num in range(1, 9):
            round_data = clean_df[clean_df['round'] == round_num]
            if len(round_data) == 0:
                continue
                
            # Calculate weighted risk score
            risk_score = 0
            valid_features = 0
            
            for feature in drift_features:
                if feature in round_data.columns:
                    mean_val = round_data[feature].mean()
                    if not np.isnan(mean_val):
                        coeff = coeff_map[feature]
                        # Convert to risk: negative coeff = risk, positive = protective
                        risk_contribution = -coeff * mean_val if coeff < 0 else coeff * mean_val
                        risk_score += risk_contribution
                        valid_features += 1
            
            # Only include rounds with at least some valid data
            if valid_features > 0:
                combined_risk.append(risk_score)
                risk_rounds.append(round_num)
        
        if len(combined_risk) > 0:
            # Plot actual data
            actual_risk_rounds = [r for r in risk_rounds if r <= 4]
            actual_risk_scores = [combined_risk[risk_rounds.index(r)] for r in actual_risk_rounds]
            
            ax2.plot(actual_risk_rounds, actual_risk_scores, 
                    marker='s', linewidth=3, color='purple', label='Combined Risk Score')
            
            # Add note about data coverage
            ax2.text(0.95, 0.05, f'Data coverage:\nRounds 1-{max(risk_rounds)}', 
                    transform=ax2.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax2.set_xlabel('Follow-up Round')
        ax2.set_ylabel('Combined Risk Score')
        ax2.set_title('Overall Risk Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0.5, 8.5)  # Show all 8 rounds
        ax2.set_xticks(range(1, 9))  # Show all round numbers
        
        # Plot 3: Feature value distributions by round (bottom-left)
        ax3 = axes[1, 0]
        
        # Create boxplot for most important feature
        most_important_feature = max(drift_features, key=lambda x: abs(coeff_map[x]))
        
        round_data_for_box = []
        round_labels = []
        
        # Check all 8 rounds but only include those with data
        for round_num in range(1, 9):
            round_vals = clean_df[clean_df['round'] == round_num][most_important_feature].dropna()
            if len(round_vals) > 0:
                round_data_for_box.append(round_vals)
                round_labels.append(f'R{round_num}')
        
        if round_data_for_box:
            bp = ax3.boxplot(round_data_for_box, labels=round_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
        
        ax3.set_xlabel('Follow-up Round')
        ax3.set_ylabel(f'{self._clean_feature_name(most_important_feature)} Value')
        ax3.set_title(f'Distribution of {self._clean_feature_name(most_important_feature)}\n(Most Important Feature)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics (bottom-right)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = []
        summary_text.append("🎯 KEY INSIGHTS:")
        summary_text.append("")
        
        for feature in drift_features:
            coeff = coeff_map[feature]
            effect = "🚨 RISK" if coeff < 0 else "🛡️ PROTECTIVE"
            summary_text.append(f"{effect}: {self._clean_feature_name(feature)}")
            summary_text.append(f"   Coefficient: {coeff:.3f}")
            
            # Calculate trend
            round_means = clean_df.groupby('round')[feature].mean()
            if len(round_means) > 1:
                trend = "↗️ Increasing" if round_means.iloc[-1] > round_means.iloc[0] else "↘️ Decreasing"
                summary_text.append(f"   Trend: {trend}")
            summary_text.append("")
        
        # Add model info
        if 'model_performance' in self.data:
            best_model_row = self.data['model_performance'].loc[self.data['model_performance']['c_index'].idxmax()]
            summary_text.append(f"🏆 Best Model: {best_model_row['model_type']}")
            summary_text.append(f"   C-index: {best_model_row['c_index']:.4f}")
        
        summary_str = '\n'.join(summary_text)
        ax4.text(0.05, 0.95, summary_str, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = f'{self.figures_dir}/driver_dynamics_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✅ Driver dynamics plot saved to {output_path}")

    def _clean_feature_name(self, feature_name):
        """Clean up feature names for better display"""
        if pd.isna(feature_name):
            return "Unknown"
        
        # Handle tuple representations
        feature_str = str(feature_name)
        
        # Remove tuple formatting
        feature_str = feature_str.replace('(', '').replace(')', '').replace("'", '').replace(',', '')
        
        # Clean common patterns
        cleanups = {
            'alpha_': '',
            'beta_': '',
            'model_': '',
            '_': ' ',
            'prompt to prompt drift': 'Prompt-to-Prompt Drift',
            'context to prompt drift': 'Context-to-Prompt Drift',
            'cumulative drift': 'Cumulative Drift',
            'prompt complexity': 'Prompt Complexity',
            'Intercept': 'Baseline (Intercept)'
        }
        
        for old, new in cleanups.items():
            feature_str = feature_str.replace(old, new)
        
        return feature_str.strip().title()
    
    def _truncate_label(self, label, max_length):
        """Truncate labels for better display"""
        if len(label) <= max_length:
            return label
        return label[:max_length-3] + '...'
    
    def plot_subject_cluster_analysis(self):
        """Create subject cluster analysis visualization"""
        print("\n📚 CREATING SUBJECT CLUSTER ANALYSIS")
        print("=" * 50)
        
        if 'all_coefficients' not in self.data:
            print("❌ No coefficient data available for subject cluster analysis")
            return
            
        try:
            coef_df = self.data['all_coefficients'].copy()
            
            # Filter for subject cluster coefficients (handle tuple string format)
            subject_coefs = coef_df[coef_df['feature'].str.contains('subject_')].copy()
            
            if subject_coefs.empty:
                print("⚠️  No subject cluster coefficients found")
                return
                
            # Extract clean feature names from tuple strings
            subject_coefs['cluster'] = subject_coefs['feature'].str.extract(r"'(subject_[^']*)'")[0]
            subject_coefs = subject_coefs.dropna(subset=['cluster'])
            subject_coefs['cluster'] = subject_coefs['cluster'].str.replace('subject_', '')
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AFT Models: Subject Cluster Analysis', fontsize=16, fontweight='bold')
            
            # 1. Coefficient magnitude by cluster (top-left)
            ax1 = axes[0, 0]
            
            # Group by cluster and get mean absolute coefficient
            cluster_impact = subject_coefs.groupby('cluster')['coef'].apply(
                lambda x: np.mean(np.abs(x))
            ).sort_values(ascending=False)
            
            bars = ax1.bar(range(len(cluster_impact)), cluster_impact.values)
            ax1.set_xticks(range(len(cluster_impact)))
            ax1.set_xticklabels(cluster_impact.index, rotation=45, ha='right')
            ax1.set_title('Subject Cluster Impact (Mean |Coefficient|)')
            ax1.set_ylabel('Mean Absolute Coefficient')
            
            # Color bars by impact level
            for i, bar in enumerate(bars):
                if cluster_impact.values[i] > cluster_impact.median():
                    bar.set_color('crimson')
                else:
                    bar.set_color('steelblue')
            
            # 2. Coefficient heatmap by model and cluster (top-right)
            ax2 = axes[0, 1]
            
            # Create pivot table for heatmap
            heatmap_data = subject_coefs.pivot_table(
                index='cluster', columns='model_name', values='coef', 
                aggfunc='mean', fill_value=0
            )
            
            if not heatmap_data.empty:
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', 
                           center=0, ax=ax2)
                ax2.set_title('Subject Cluster Coefficients by Model')
                ax2.set_ylabel('Subject Cluster')
            else:
                ax2.text(0.5, 0.5, 'No heatmap data available', ha='center', va='center')
            
            # 3. Risk vs Protective factors (bottom-left)
            ax3 = axes[1, 0]
            
            positive_coefs = subject_coefs[subject_coefs['coef'] > 0]
            negative_coefs = subject_coefs[subject_coefs['coef'] < 0]
            
            risk_clusters = positive_coefs.groupby('cluster')['coef'].mean().sort_values()
            protective_clusters = negative_coefs.groupby('cluster')['coef'].mean().sort_values()
            
            y_pos = np.arange(len(risk_clusters))
            ax3.barh(y_pos, risk_clusters.values, color='crimson', alpha=0.7, label='Risk Factors')
            
            if not protective_clusters.empty:
                y_neg = np.arange(len(protective_clusters)) - len(protective_clusters) - 1
                ax3.barh(y_neg, protective_clusters.values, color='green', alpha=0.7, label='Protective Factors')
                all_labels = list(protective_clusters.index) + list(risk_clusters.index)
                all_positions = list(y_neg) + list(y_pos)
            else:
                all_labels = list(risk_clusters.index)
                all_positions = list(y_pos)
            
            ax3.set_yticks(all_positions)
            ax3.set_yticklabels(all_labels)
            ax3.set_title('Risk vs Protective Subject Clusters')
            ax3.set_xlabel('Mean Coefficient')
            ax3.legend()
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # 4. Model agreement on clusters (bottom-right)
            ax4 = axes[1, 1]
            
            # Calculate agreement (standard deviation of coefficients across models)
            cluster_agreement = subject_coefs.groupby('cluster')['coef'].std().sort_values()
            
            bars = ax4.bar(range(len(cluster_agreement)), cluster_agreement.values)
            ax4.set_xticks(range(len(cluster_agreement)))
            ax4.set_xticklabels(cluster_agreement.index, rotation=45, ha='right')
            ax4.set_title('Model Agreement on Subject Clusters\n(Lower = More Agreement)')
            ax4.set_ylabel('Coefficient Standard Deviation')
            
            # Color by agreement level
            for i, bar in enumerate(bars):
                if cluster_agreement.values[i] > cluster_agreement.median():
                    bar.set_color('orange')  # High disagreement
                else:
                    bar.set_color('lightgreen')  # High agreement
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f'{self.figures_dir}/aft_subject_cluster_analysis.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print("✅ Subject cluster analysis saved successfully")
            
        except Exception as e:
            print(f"❌ Subject cluster analysis failed: {e}")
            plt.close()
    
    def plot_difficulty_level_analysis(self):
        """Create difficulty level analysis visualization"""
        print("\n📈 CREATING DIFFICULTY LEVEL ANALYSIS")
        print("=" * 50)
        
        if 'all_coefficients' not in self.data:
            print("❌ No coefficient data available for difficulty level analysis")
            return
            
        try:
            coef_df = self.data['all_coefficients'].copy()
            
            # Filter for difficulty level coefficients (handle tuple string format)
            difficulty_coefs = coef_df[coef_df['feature'].str.contains('difficulty_')].copy()
            
            if difficulty_coefs.empty:
                print("⚠️  No difficulty level coefficients found")
                return
                
            # Extract clean feature names from tuple strings
            difficulty_coefs['level'] = difficulty_coefs['feature'].str.extract(r"'(difficulty_[^']*)'")[0]
            difficulty_coefs = difficulty_coefs.dropna(subset=['level'])
            difficulty_coefs['level'] = difficulty_coefs['level'].str.replace('difficulty_', '')
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AFT Models: Difficulty Level Analysis', fontsize=16, fontweight='bold')
            
            # 1. Coefficient progression by difficulty (top-left)
            ax1 = axes[0, 0]
            
            # Define proper ordering for difficulty levels
            level_order = ['high_school', 'college', 'professional']  # elementary is reference
            available_levels = [level for level in level_order if level in difficulty_coefs['level'].values]
            
            if available_levels:
                level_means = []
                level_stds = []
                
                for level in available_levels:
                    level_data = difficulty_coefs[difficulty_coefs['level'] == level]['coef']
                    level_means.append(level_data.mean())
                    level_stds.append(level_data.std())
                
                bars = ax1.bar(range(len(available_levels)), level_means, yerr=level_stds, 
                              capsize=5, alpha=0.7)
                ax1.set_xticks(range(len(available_levels)))
                ax1.set_xticklabels(available_levels, rotation=45, ha='right')
                ax1.set_title('Difficulty Level Impact (vs Elementary)')
                ax1.set_ylabel('Mean Coefficient ± SD')
                ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Color by effect direction
                for i, (bar, mean_val) in enumerate(zip(bars, level_means)):
                    if mean_val > 0:
                        bar.set_color('crimson')  # Increases failure risk
                    else:
                        bar.set_color('green')    # Decreases failure risk
            
            # 2. Model consistency across difficulty levels (top-right)
            ax2 = axes[0, 1]
            
            heatmap_data = difficulty_coefs.pivot_table(
                index='level', columns='model_name', values='coef', 
                aggfunc='mean', fill_value=0
            )
            
            if not heatmap_data.empty:
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdBu_r', 
                           center=0, ax=ax2)
                ax2.set_title('Difficulty Level Coefficients by Model')
                ax2.set_ylabel('Difficulty Level')
            else:
                ax2.text(0.5, 0.5, 'No heatmap data available', ha='center', va='center')
            
            # 3. Effect size distribution (bottom-left)
            ax3 = axes[1, 0]
            
            # Distribution of coefficients by difficulty level
            for level in available_levels:
                level_data = difficulty_coefs[difficulty_coefs['level'] == level]['coef']
                ax3.hist(level_data, alpha=0.6, label=level, bins=10)
            
            ax3.set_title('Distribution of Difficulty Level Effects')
            ax3.set_xlabel('Coefficient Value')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # 4. Cross-model agreement (bottom-right)
            ax4 = axes[1, 1]
            
            # Calculate agreement (inverse of standard deviation)
            level_agreement = difficulty_coefs.groupby('level')['coef'].std().sort_values()
            
            bars = ax4.bar(range(len(level_agreement)), level_agreement.values)
            ax4.set_xticks(range(len(level_agreement)))
            ax4.set_xticklabels(level_agreement.index, rotation=45, ha='right')
            ax4.set_title('Model Agreement on Difficulty Levels\n(Lower = More Agreement)')
            ax4.set_ylabel('Coefficient Standard Deviation')
            
            # Color by agreement level
            for i, bar in enumerate(bars):
                if level_agreement.values[i] > level_agreement.median():
                    bar.set_color('orange')   # High disagreement
                else:
                    bar.set_color('lightgreen')   # High agreement
            
            plt.tight_layout()
            
            # Save plot
            plt.savefig(f'{self.figures_dir}/aft_difficulty_level_analysis.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print("✅ Difficulty level analysis saved successfully")
            
        except Exception as e:
            print(f"❌ Difficulty level analysis failed: {e}")
            plt.close()
    
    def plot_hazard_ratio_by_difficulty(self):
        """Create instantaneous hazard ratio plots by difficulty level for AFT models"""
        print("\n📊 CREATING HAZARD RATIO BY DIFFICULTY")
        print("=" * 50)

        # Load round-level data for hazard calculations
        combined_df = self._load_round_level_data()
        if combined_df is None:
            print("❌ No round-level data available for hazard ratio plots")
            return

        # Get best model coefficients
        best_model, coeff_map = self._get_best_model_coefficients()
        if coeff_map is None:
            print("❌ No model coefficients available for hazard calculation")
            return

        print(f"📈 Using best model: {best_model}")
        print(f"📊 Coefficients available: {len(coeff_map)}")

        try:
            # Define difficulty levels and models
            difficulty_levels = ['elementary', 'high_school', 'college', 'professional']
            available_levels = [level for level in difficulty_levels if f'difficulty_{level}' in combined_df.columns or level == 'elementary']

            # Get unique models from the data
            unique_models = sorted(combined_df['model'].unique()) if 'model' in combined_df.columns else ['Combined']

            # Create subplot grid (2x2 for 4 difficulty levels)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AFT Hazard Ratio by Difficulty Level (Instantaneous)', fontsize=16, fontweight='bold')
            axes = axes.flatten()

            # Color palette for models
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))

            for level_idx, difficulty in enumerate(available_levels):
                if level_idx >= 4:  # Only plot first 4 levels
                    break

                ax = axes[level_idx]
                ax.set_title(f'{difficulty.replace("_", " ").title()} Level (AFT)')

                # Filter data for this difficulty level
                if difficulty == 'elementary':
                    # Elementary is reference level (all difficulty dummies = 0)
                    level_data = combined_df.copy()
                    for other_level in ['high_school', 'college', 'professional']:
                        if f'difficulty_{other_level}' in level_data.columns:
                            level_data = level_data[level_data[f'difficulty_{other_level}'] == 0]
                else:
                    # Other levels where the specific dummy = 1
                    if f'difficulty_{difficulty}' in combined_df.columns:
                        level_data = combined_df[combined_df[f'difficulty_{difficulty}'] == 1].copy()
                    else:
                        continue

                if len(level_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {difficulty}', ha='center', va='center', transform=ax.transAxes)
                    continue

                # Calculate hazard ratio for each model in this difficulty level
                for model_idx, model in enumerate(unique_models):
                    if 'model' in level_data.columns:
                        model_data = level_data[level_data['model'] == model].copy()
                    else:
                        model_data = level_data.copy()

                    if len(model_data) == 0:
                        continue

                    # Calculate instantaneous hazard ratio for each round
                    rounds = sorted(model_data['round'].unique())
                    hazard_ratios = []

                    for round_num in rounds:
                        round_data = model_data[model_data['round'] == round_num]

                        # Calculate linear predictor (risk score) using available coefficients
                        risk_score = 0

                        # Add coefficients for available features
                        for feature, coeff in coeff_map.items():
                            if feature in round_data.columns:
                                feature_mean = round_data[feature].mean()
                                if not np.isnan(feature_mean):
                                    risk_score += coeff * feature_mean

                        # Add difficulty level effect (elementary is reference)
                        if difficulty != 'elementary' and f'difficulty_{difficulty}' in coeff_map:
                            risk_score += coeff_map[f'difficulty_{difficulty}']

                        # Add model effects if available
                        if 'model' in round_data.columns and f'model_{model}' in coeff_map:
                            risk_score += coeff_map[f'model_{model}']

                        # Convert to hazard ratio using AFT assumption
                        # AFT: hazard ratio = exp(-risk_score) (negative because AFT accelerates/decelerates time)
                        hazard_ratio = np.exp(-risk_score)
                        hazard_ratios.append(hazard_ratio)

                    # Plot hazard ratio curve
                    if len(rounds) > 0 and len(hazard_ratios) > 0:
                        ax.plot(rounds, hazard_ratios,
                               color=colors[model_idx], linewidth=2,
                               label=model.replace('_', ' ').title(), marker='o')

                ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Baseline (HR=1.0)')
                ax.set_xlabel('Conversation Round')
                ax.set_ylabel('Hazard Ratio')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(1, 8)
                ax.set_yscale('log')  # Log scale for hazard ratios

                # Add legend only to first subplot
                if level_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Hide empty subplots
            for idx in range(len(available_levels), 4):
                axes[idx].set_visible(False)

            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_hazard_ratio_by_difficulty.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ AFT hazard ratio by difficulty saved to {save_path}")

        except Exception as e:
            print(f"❌ AFT hazard ratio by difficulty failed: {e}")

    def plot_cumulative_hazard_by_difficulty(self):
        """Create cumulative hazard plots by difficulty level for AFT models"""
        print("\n📊 CREATING CUMULATIVE HAZARD BY DIFFICULTY")
        print("=" * 50)

        # Load round-level data for hazard calculations
        combined_df = self._load_round_level_data()
        if combined_df is None:
            print("❌ No round-level data available for cumulative hazard plots")
            return

        # Get best model coefficients
        best_model, coeff_map = self._get_best_model_coefficients()
        if coeff_map is None:
            print("❌ No model coefficients available for hazard calculation")
            return

        print(f"📈 Using best model: {best_model}")
        print(f"📊 Coefficients available: {len(coeff_map)}")

        try:
            # Define difficulty levels and models
            difficulty_levels = ['elementary', 'high_school', 'college', 'professional']
            available_levels = [level for level in difficulty_levels if f'difficulty_{level}' in combined_df.columns or level == 'elementary']

            # Get unique models from the data
            unique_models = sorted(combined_df['model'].unique()) if 'model' in combined_df.columns else ['Combined']

            # Create subplot grid (2x2 for 4 difficulty levels)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('AFT Cumulative Hazard by Difficulty Level', fontsize=16, fontweight='bold')
            axes = axes.flatten()

            # Color palette for models
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))

            for level_idx, difficulty in enumerate(available_levels):
                if level_idx >= 4:  # Only plot first 4 levels
                    break

                ax = axes[level_idx]
                ax.set_title(f'{difficulty.replace("_", " ").title()} Level (AFT)')

                # Filter data for this difficulty level
                if difficulty == 'elementary':
                    # Elementary is reference level (all difficulty dummies = 0)
                    level_data = combined_df.copy()
                    for other_level in ['high_school', 'college', 'professional']:
                        if f'difficulty_{other_level}' in level_data.columns:
                            level_data = level_data[level_data[f'difficulty_{other_level}'] == 0]
                else:
                    # Other levels where the specific dummy = 1
                    if f'difficulty_{difficulty}' in combined_df.columns:
                        level_data = combined_df[combined_df[f'difficulty_{difficulty}'] == 1].copy()
                    else:
                        continue

                if len(level_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {difficulty}', ha='center', va='center', transform=ax.transAxes)
                    continue

                # Calculate cumulative hazard for each model in this difficulty level
                for model_idx, model in enumerate(unique_models):
                    if 'model' in level_data.columns:
                        model_data = level_data[level_data['model'] == model].copy()
                    else:
                        model_data = level_data.copy()

                    if len(model_data) == 0:
                        continue

                    # Calculate cumulative hazard for each round
                    rounds = sorted(model_data['round'].unique())
                    cumulative_hazard = []
                    running_cumulative = 0.0

                    for round_num in rounds:
                        round_data = model_data[model_data['round'] == round_num]

                        # Calculate linear predictor (risk score) using available coefficients
                        risk_score = 0

                        # Add coefficients for available features
                        for feature, coeff in coeff_map.items():
                            if feature in round_data.columns:
                                feature_mean = round_data[feature].mean()
                                if not np.isnan(feature_mean):
                                    risk_score += coeff * feature_mean

                        # Add difficulty level effect (elementary is reference)
                        if difficulty != 'elementary' and f'difficulty_{difficulty}' in coeff_map:
                            risk_score += coeff_map[f'difficulty_{difficulty}']

                        # Add model effects if available
                        if 'model' in round_data.columns and f'model_{model}' in coeff_map:
                            risk_score += coeff_map[f'model_{model}']

                        # Convert to instantaneous hazard using AFT assumption
                        baseline_hazard_rate = 0.05  # Base hazard rate per round
                        hazard_ratio = np.exp(-risk_score)  # AFT uses negative of Cox model
                        instantaneous_hazard = baseline_hazard_rate * hazard_ratio

                        # Accumulate hazard (this ensures monotonic increase)
                        running_cumulative += instantaneous_hazard
                        cumulative_hazard.append(running_cumulative)

                    # Plot cumulative hazard curve
                    if len(rounds) > 0 and len(cumulative_hazard) > 0:
                        ax.plot(rounds, cumulative_hazard,
                               color=colors[model_idx], linewidth=2,
                               label=model.replace('_', ' ').title(), marker='o')

                ax.set_xlabel('Conversation Round')
                ax.set_ylabel('Cumulative Hazard')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(1, 8)

                # Add legend only to first subplot
                if level_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Hide empty subplots
            for idx in range(len(available_levels), 4):
                axes[idx].set_visible(False)

            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_cumulative_hazard_by_difficulty.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ AFT cumulative hazard by difficulty saved to {save_path}")

        except Exception as e:
            print(f"❌ AFT cumulative hazard by difficulty failed: {e}")

    def plot_hazard_ratio_by_subject_cluster(self):
        """Create instantaneous hazard ratio plots by subject cluster for AFT models"""
        print("\n📚 CREATING HAZARD RATIO BY SUBJECT CLUSTER")
        print("=" * 50)

        # Load round-level data
        combined_df = self._load_round_level_data()
        if combined_df is None:
            print("❌ No round-level data available for hazard ratio plots")
            return

        # Get best model coefficients
        best_model, coeff_map = self._get_best_model_coefficients()
        if coeff_map is None:
            print("❌ No model coefficients available for hazard calculation")
            return

        try:
            # Get available subject clusters
            subject_cols = [col for col in combined_df.columns if col.startswith('subject_')]
            if 'subject_cluster' in combined_df.columns:
                subject_clusters = sorted(combined_df['subject_cluster'].unique())
            else:
                # Infer clusters from dummy variables
                subject_clusters = [col.replace('subject_', '') for col in subject_cols]

            # Limit to 8 clusters for better visualization
            subject_clusters = subject_clusters[:8]

            # Get unique models
            unique_models = sorted(combined_df['model'].unique()) if 'model' in combined_df.columns else ['Combined']

            # Create subplot grid (2x4 for 8 clusters)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('AFT Hazard Ratio by Subject Cluster (Instantaneous)', fontsize=16, fontweight='bold')
            axes = axes.flatten()

            # Color palette for models
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))

            for cluster_idx, cluster in enumerate(subject_clusters):
                if cluster_idx >= 8:  # Only plot first 8 clusters
                    break

                ax = axes[cluster_idx]
                ax.set_title(f'{cluster.replace("_", " ").title()} (AFT)')

                # Filter data for this subject cluster
                if 'subject_cluster' in combined_df.columns:
                    cluster_data = combined_df[combined_df['subject_cluster'] == cluster].copy()
                elif f'subject_{cluster}' in combined_df.columns:
                    cluster_data = combined_df[combined_df[f'subject_{cluster}'] == 1].copy()
                else:
                    continue

                if len(cluster_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {cluster}', ha='center', va='center', transform=ax.transAxes)
                    continue

                # Calculate hazard ratio for each model in this cluster
                for model_idx, model in enumerate(unique_models):
                    if 'model' in cluster_data.columns:
                        model_data = cluster_data[cluster_data['model'] == model].copy()
                    else:
                        model_data = cluster_data.copy()

                    if len(model_data) == 0:
                        continue

                    # Calculate instantaneous hazard ratio for each round
                    rounds = sorted(model_data['round'].unique())
                    hazard_ratios = []

                    for round_num in rounds:
                        round_data = model_data[model_data['round'] == round_num]

                        # Calculate linear predictor using available coefficients
                        risk_score = 0

                        # Add coefficients for available features
                        for feature, coeff in coeff_map.items():
                            if feature in round_data.columns:
                                feature_mean = round_data[feature].mean()
                                if not np.isnan(feature_mean):
                                    risk_score += coeff * feature_mean

                        # Add subject cluster effect
                        if f'subject_{cluster}' in coeff_map:
                            risk_score += coeff_map[f'subject_{cluster}']

                        # Add model effects if available
                        if 'model' in round_data.columns and f'model_{model}' in coeff_map:
                            risk_score += coeff_map[f'model_{model}']

                        # Convert to hazard ratio using AFT assumption
                        hazard_ratio = np.exp(-risk_score)
                        hazard_ratios.append(hazard_ratio)

                    # Plot hazard ratio curve
                    if len(rounds) > 0 and len(hazard_ratios) > 0:
                        ax.plot(rounds, hazard_ratios,
                               color=colors[model_idx], linewidth=2,
                               label=model.replace('_', ' ').title(), marker='o')

                ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Baseline (HR=1.0)')
                ax.set_xlabel('Conversation Round')
                ax.set_ylabel('Hazard Ratio')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(1, 8)
                ax.set_yscale('log')  # Log scale for hazard ratios

                # Add legend only to first subplot
                if cluster_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Hide empty subplots
            for idx in range(len(subject_clusters), 8):
                axes[idx].set_visible(False)

            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_hazard_ratio_by_subject_cluster.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ AFT hazard ratio by subject cluster saved to {save_path}")

        except Exception as e:
            print(f"❌ AFT hazard ratio by subject cluster failed: {e}")

    def plot_cumulative_hazard_by_subject_cluster(self):
        """Create cumulative hazard plots by subject cluster for AFT models"""
        print("\n📚 CREATING CUMULATIVE HAZARD BY SUBJECT CLUSTER")
        print("=" * 50)

        # Load round-level data
        combined_df = self._load_round_level_data()
        if combined_df is None:
            print("❌ No round-level data available for cumulative hazard plots")
            return

        # Get best model coefficients
        best_model, coeff_map = self._get_best_model_coefficients()
        if coeff_map is None:
            print("❌ No model coefficients available for hazard calculation")
            return

        try:
            # Get available subject clusters
            subject_cols = [col for col in combined_df.columns if col.startswith('subject_')]
            if 'subject_cluster' in combined_df.columns:
                subject_clusters = sorted(combined_df['subject_cluster'].unique())
            else:
                # Infer clusters from dummy variables
                subject_clusters = [col.replace('subject_', '') for col in subject_cols]

            # Limit to 8 clusters for better visualization
            subject_clusters = subject_clusters[:8]

            # Get unique models
            unique_models = sorted(combined_df['model'].unique()) if 'model' in combined_df.columns else ['Combined']

            # Create subplot grid (2x4 for 8 clusters)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle('AFT Cumulative Hazard by Subject Cluster', fontsize=16, fontweight='bold')
            axes = axes.flatten()

            # Color palette for models
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))

            for cluster_idx, cluster in enumerate(subject_clusters):
                if cluster_idx >= 8:  # Only plot first 8 clusters
                    break

                ax = axes[cluster_idx]
                ax.set_title(f'{cluster.replace("_", " ").title()} (AFT)')

                # Filter data for this subject cluster
                if 'subject_cluster' in combined_df.columns:
                    cluster_data = combined_df[combined_df['subject_cluster'] == cluster].copy()
                elif f'subject_{cluster}' in combined_df.columns:
                    cluster_data = combined_df[combined_df[f'subject_{cluster}'] == 1].copy()
                else:
                    continue

                if len(cluster_data) == 0:
                    ax.text(0.5, 0.5, f'No data for {cluster}', ha='center', va='center', transform=ax.transAxes)
                    continue

                # Calculate cumulative hazard for each model in this cluster
                for model_idx, model in enumerate(unique_models):
                    if 'model' in cluster_data.columns:
                        model_data = cluster_data[cluster_data['model'] == model].copy()
                    else:
                        model_data = cluster_data.copy()

                    if len(model_data) == 0:
                        continue

                    # Calculate cumulative hazard for each round
                    rounds = sorted(model_data['round'].unique())
                    cumulative_hazard = []
                    running_cumulative = 0.0

                    for round_num in rounds:
                        round_data = model_data[model_data['round'] == round_num]

                        # Calculate linear predictor using available coefficients
                        risk_score = 0

                        # Add coefficients for available features
                        for feature, coeff in coeff_map.items():
                            if feature in round_data.columns:
                                feature_mean = round_data[feature].mean()
                                if not np.isnan(feature_mean):
                                    risk_score += coeff * feature_mean

                        # Add subject cluster effect
                        if f'subject_{cluster}' in coeff_map:
                            risk_score += coeff_map[f'subject_{cluster}']

                        # Add model effects if available
                        if 'model' in round_data.columns and f'model_{model}' in coeff_map:
                            risk_score += coeff_map[f'model_{model}']

                        # Convert to instantaneous hazard using AFT assumption
                        baseline_hazard_rate = 0.05  # Base hazard rate per round
                        hazard_ratio = np.exp(-risk_score)  # AFT uses negative of Cox model
                        instantaneous_hazard = baseline_hazard_rate * hazard_ratio

                        # Accumulate hazard (this ensures monotonic increase)
                        running_cumulative += instantaneous_hazard
                        cumulative_hazard.append(running_cumulative)

                    # Plot cumulative hazard curve
                    if len(rounds) > 0 and len(cumulative_hazard) > 0:
                        ax.plot(rounds, cumulative_hazard,
                               color=colors[model_idx], linewidth=2,
                               label=model.replace('_', ' ').title(), marker='o')

                ax.set_xlabel('Conversation Round')
                ax.set_ylabel('Cumulative Hazard')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(1, 8)

                # Add legend only to first subplot
                if cluster_idx == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Hide empty subplots
            for idx in range(len(subject_clusters), 8):
                axes[idx].set_visible(False)

            plt.tight_layout()
            save_path = f'{self.figures_dir}/aft_cumulative_hazard_by_subject_cluster.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ AFT cumulative hazard by subject cluster saved to {save_path}")

        except Exception as e:
            print(f"❌ AFT cumulative hazard by subject cluster failed: {e}")

    def create_all_visualizations(self):
        """Create all AFT visualizations"""
        print("🎨 CREATING COMPREHENSIVE AFT VISUALIZATIONS")
        print("=" * 80)

        if not self.load_results():
            return

        print("=" * 80)

        # Create all visualizations
        self.plot_model_performance_comparison()
        print("=" * 80)

        self.plot_feature_importance_analysis()
        print("=" * 80)

        self.plot_model_coefficients_heatmap()
        print("=" * 80)

        self.plot_model_rankings_dashboard()
        print("=" * 80)

        self.plot_survival_insights_analysis()
        print("=" * 80)

        self.plot_driver_dynamics_over_time()
        print("=" * 80)

        self.plot_subject_cluster_analysis()
        print("=" * 80)

        self.plot_difficulty_level_analysis()
        print("=" * 80)

        # Add hazard ratio plots (instantaneous)
        self.plot_hazard_ratio_by_difficulty()
        print("=" * 80)

        self.plot_hazard_ratio_by_subject_cluster()
        print("=" * 80)

        # Add cumulative hazard plots
        self.plot_cumulative_hazard_by_difficulty()
        print("=" * 80)

        self.plot_cumulative_hazard_by_subject_cluster()
        print("=" * 80)
        
        print("🎉 ALL AFT VISUALIZATIONS COMPLETED!")
        print(f"📁 Saved to: {self.figures_dir}/")
        print("🖼️  Files created:")
        print("   • aft_performance_comparison.png - Model performance dashboard")
        print("   • aft_feature_importance.png - Feature importance analysis")
        print("   • aft_coefficients_heatmap.png - Cross-model coefficient comparison")
        print("   • aft_rankings_dashboard.png - Comprehensive model rankings")
        print("   • aft_survival_insights.png - Risk/protective factors analysis")
        print("   • driver_dynamics_evolution.png - Driver dynamics across rounds")
        print("   • aft_subject_cluster_analysis.png - Subject cluster impact analysis")
        print("   • aft_difficulty_level_analysis.png - Difficulty level progression analysis")
        print("   • aft_hazard_ratio_by_difficulty.png - Instantaneous hazard ratio by difficulty level")
        print("   • aft_hazard_ratio_by_subject_cluster.png - Instantaneous hazard ratio by subject cluster")
        print("   • aft_cumulative_hazard_by_difficulty.png - Cumulative hazard by difficulty level")
        print("   • aft_cumulative_hazard_by_subject_cluster.png - Cumulative hazard by subject cluster")

def main():
    """Main execution function"""
    visualizer = AFTVisualizer()
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main()
