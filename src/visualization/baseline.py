#!/usr/bin/env python3
"""
Baseline Combined Model Visualizations - AI Conference Ready
==========================================================

Professional visualizations for combined baseline survival modeling results.
Creates separate, publication-ready plots optimized for top AI conferences:
- No titles (clean appearance)
- No subplots (individual files)
- Professional color schemes
- High-quality styling

All plots are saved individually to results/figures/baseline/

Usage:
    python src/visualization/baseline.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Import baseline modeling
from modeling.baseline import BaselineModeling

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Professional color palette for AI conferences
COLORS = {
    'primary': '#2E86AB',     # Professional blue
    'secondary': '#A23B72',   # Deep magenta  
    'success': '#F18F01',     # Warm orange
    'danger': '#C73E1D',      # Strong red
    'safe': '#4CAF50',        # Material green
    'risk': '#FF5722',        # Material deep orange
    'warning': '#FF9800',     # Orange warning
    'neutral': '#607D8B',     # Blue grey
    'accent1': '#9C27B0',     # Purple
    'accent2': '#00BCD4',     # Cyan
    'accent3': '#FFC107',     # Amber
    'gradient': ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#9C27B0', '#00BCD4', '#FF5722', '#FFC107', '#607D8B']
}

class BaselineCombinedVisualizer:
    """Visualizations for combined baseline survival modeling results."""
    
    def __init__(self, results_dir='results/outputs/baseline', output_dir='results/figures/baseline'):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all result files
        self.load_results()
        
        # Color schemes
        self.model_colors = {
            'claude_35': '#1f77b4',      # Blue (reference)
            'deepseek_r1': '#ff7f0e',    # Orange
            'gemini_25': '#2ca02c',      # Green
            'gpt_default': '#d62728',    # Red
            'gpt_oss': '#9467bd',        # Purple
            'llama_33': '#8c564b',       # Brown
            'llama_4_maverick': '#e377c2', # Pink
            'llama_4_scout': '#7f7f7f',  # Gray
            'mistral_large': '#bcbd22',  # Olive
            'qwen_3': '#17becf',         # Cyan
            'qwen_max': '#ff9896'        # Light Red
        }
    
    def load_results(self):
        """Load all result CSV files."""
        self.hazard_ratios = pd.read_csv(f'{self.results_dir}/hazard_ratios.csv')
        self.coefficients = pd.read_csv(f'{self.results_dir}/model_coefficients.csv')
        self.p_values = pd.read_csv(f'{self.results_dir}/p_values.csv')
        self.performance = pd.read_csv(f'{self.results_dir}/model_performance.csv')
        
        print(f"✅ Loaded baseline results from {self.results_dir}/")
    
    def plot_hazard_ratio_comparison(self):
        """Create separate hazard ratio comparison plots."""
        
        # Extract model hazard ratios
        model_hrs = []
        hr_cols = [col for col in self.hazard_ratios.columns 
                  if 'model_' in col and '_hr' in col and '_ci' not in col]
        
        for col in hr_cols:
            model_name = col.replace('model_', '').replace('_hr', '')
            hr_val = self.hazard_ratios[col].iloc[0]
            
            # Get p-value for significance
            pval_col = col.replace('_hr', '_pval')
            pval = self.p_values[pval_col].iloc[0] if pval_col in self.p_values.columns else 1.0
            
            model_hrs.append({
                'model': model_name,
                'hazard_ratio': hr_val,
                'p_value': pval,
                'significant': pval < 0.05
            })
        
        # Add claude_35 as reference (HR = 1.0, p = reference)
        model_hrs.append({
            'model': 'claude_35',
            'hazard_ratio': 1.0,
            'p_value': np.nan,
            'significant': False
        })
        
        hr_df = pd.DataFrame(model_hrs).sort_values('hazard_ratio')
        
        # Plot 1: Hazard Ratios
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Professional color gradient based on risk level
        colors = []
        for hr in hr_df['hazard_ratio']:
            if hr < 0.3:
                colors.append(COLORS['safe'])
            elif hr < 0.7:
                colors.append(COLORS['success'])
            elif hr < 1.0:
                colors.append(COLORS['primary'])
            elif hr < 1.3:
                colors.append(COLORS['danger'])
            else:
                colors.append(COLORS['risk'])
        
        bars = ax.barh(range(len(hr_df)), hr_df['hazard_ratio'], color=colors, alpha=0.8)
        
        # Add significance markers with professional styling
        for i, (_, row) in enumerate(hr_df.iterrows()):
            if row['significant']:
                ax.text(row['hazard_ratio'] + 0.05, i, '***', 
                       ha='left', va='center', fontweight='bold', fontsize=14, color='black')
        
        # Reference line
        ax.axvline(x=1.0, color=COLORS['neutral'], linestyle='--', alpha=0.8, linewidth=2)
        
        # Clean styling
        ax.set_yticks(range(len(hr_df)))
        ax.set_yticklabels(hr_df['model'], fontsize=13)
        ax.set_xlabel('Hazard Ratio (vs claude_35)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(hr_df['hazard_ratio']) * 1.15)
        
        # Add value labels on bars
        for i, (bar, hr) in enumerate(zip(bars, hr_df['hazard_ratio'])):
            ax.text(hr + 0.02, bar.get_y() + bar.get_height()/2, f'{hr:.3f}', 
                   va='center', ha='left', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hazard_ratios.png')
        plt.savefig(f'{self.output_dir}/hazard_ratios.pdf')
        plt.close()
        
        # Plot 2: Risk Reduction Percentages
        fig, ax = plt.subplots(figsize=(10, 8))
        
        risk_reduction = [(1 - hr) * 100 if hr < 1 else -(hr - 1) * 100 
                         for hr in hr_df['hazard_ratio']]
        
        colors_risk = [COLORS['safe'] if r > 50 else COLORS['success'] if r > 0 else COLORS['danger'] 
                      for r in risk_reduction]
        
        bars = ax.barh(range(len(hr_df)), risk_reduction, color=colors_risk, alpha=0.8)
        ax.axvline(x=0, color=COLORS['neutral'], linestyle='-', alpha=0.8, linewidth=2)
        
        ax.set_yticks(range(len(hr_df)))
        ax.set_yticklabels(hr_df['model'], fontsize=13)
        ax.set_xlabel('Risk Change (% vs claude_35)', fontsize=14, fontweight='bold')
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, risk_reduction)):
            label_x = pct + (3 if pct > 0 else -3)
            ax.text(label_x, i, f'{pct:.1f}%', 
                   ha='left' if pct > 0 else 'right', va='center', 
                   fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/risk_reduction.png')
        plt.savefig(f'{self.output_dir}/risk_reduction.pdf')
        plt.close()
        
        print("✅ Created hazard ratio and risk reduction plots")
    
    def plot_coefficient_effects(self):
        """Plot separate coefficient effects for drift measures and model comparisons."""
        
        # Plot 1: Drift Effects
        drift_effects = []
        drift_cols = ['prompt_to_prompt_drift_coef', 'context_to_prompt_drift_coef', 
                     'cumulative_drift_coef', 'prompt_complexity_coef']
        drift_names = ['Prompt-to-Prompt\nDrift', 'Context-to-Prompt\nDrift', 
                      'Cumulative\nDrift', 'Prompt\nComplexity']
        
        for col, name in zip(drift_cols, drift_names):
            if col in self.coefficients.columns:
                coef_val = self.coefficients[col].iloc[0]
                
                # Get p-value
                pval_col = col.replace('_coef', '_pval')
                pval = self.p_values[pval_col].iloc[0] if pval_col in self.p_values.columns else 1.0
                
                drift_effects.append({
                    'effect': name,
                    'coefficient': coef_val,
                    'p_value': pval,
                    'significant': pval < 0.05
                })
        
        drift_df = pd.DataFrame(drift_effects)
        
        # Drift effects plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [COLORS['danger'] if coef > 0 else COLORS['safe'] for coef in drift_df['coefficient']]
        
        bars = ax.bar(range(len(drift_df)), drift_df['coefficient'], color=colors, alpha=0.8)
        
        # Add significance markers
        for i, (_, row) in enumerate(drift_df.iterrows()):
            if row['significant']:
                y_pos = row['coefficient'] + (0.5 if row['coefficient'] > 0 else -0.5)
                ax.text(i, y_pos, '***', ha='center', 
                       va='bottom' if row['coefficient'] > 0 else 'top',
                       fontweight='bold', fontsize=16, color='black')
        
        # Add coefficient values on bars
        for i, (bar, coef) in enumerate(zip(bars, drift_df['coefficient'])):
            if abs(coef) > 1:  # Only show values for significant coefficients
                ax.text(i, coef/2, f'{coef:.1f}', ha='center', va='center', 
                       fontweight='bold', fontsize=11, color='white')
        
        ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.8, linewidth=2)
        ax.set_xticks(range(len(drift_df)))
        ax.set_xticklabels(drift_df['effect'], fontsize=12)
        ax.set_ylabel('Log Hazard Ratio', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/drift_effects.png')
        plt.savefig(f'{self.output_dir}/drift_effects.pdf')
        plt.close()
        
        # Plot 2: Model Effects
        model_effects = []
        model_cols = [col for col in self.coefficients.columns 
                     if 'model_' in col and '_coef' in col]
        
        for col in model_cols:
            model_name = col.replace('model_', '').replace('_coef', '')
            coef_val = self.coefficients[col].iloc[0]
            
            # Get p-value
            pval_col = col.replace('_coef', '_pval')
            pval = self.p_values[pval_col].iloc[0] if pval_col in self.p_values.columns else 1.0
            
            model_effects.append({
                'model': model_name,
                'coefficient': coef_val,
                'p_value': pval,
                'significant': pval < 0.05
            })
        
        # Add claude_35 as reference (coefficient = 0)
        model_effects.append({
            'model': 'claude_35',
            'coefficient': 0.0,
            'p_value': np.nan,
            'significant': False
        })
        
        model_df = pd.DataFrame(model_effects).sort_values('coefficient')
        
        # Model effects plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = []
        for coef in model_df['coefficient']:
            if coef < -1.5:
                colors.append(COLORS['safe'])
            elif coef < -0.5:
                colors.append(COLORS['success'])
            elif coef < 0.5:
                colors.append(COLORS['primary'])
            else:
                colors.append(COLORS['danger'])
        
        bars = ax.barh(range(len(model_df)), model_df['coefficient'], color=colors, alpha=0.8)
        
        # Add significance markers
        for i, (_, row) in enumerate(model_df.iterrows()):
            if row['significant']:
                x_pos = row['coefficient'] + (0.1 if row['coefficient'] > 0 else -0.1)
                ax.text(x_pos, i, '***', ha='left' if row['coefficient'] > 0 else 'right', 
                       va='center', fontweight='bold', fontsize=14, color='black')
        
        # Add coefficient values
        for i, (bar, coef) in enumerate(zip(bars, model_df['coefficient'])):
            if abs(coef) > 0.1:  # Show values for non-zero coefficients
                label_x = coef/2 if abs(coef) > 0.3 else (coef + 0.15 if coef > 0 else coef - 0.15)
                color = 'white' if abs(coef) > 0.3 else 'black'
                ax.text(label_x, i, f'{coef:.2f}', ha='center', va='center', 
                       fontweight='bold', fontsize=10, color=color)
        
        ax.axvline(x=0, color=COLORS['neutral'], linestyle='-', alpha=0.8, linewidth=2)
        ax.set_yticks(range(len(model_df)))
        ax.set_yticklabels(model_df['model'], fontsize=12)
        ax.set_xlabel('Log Hazard Ratio (vs claude_35)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_coefficients.png')
        plt.savefig(f'{self.output_dir}/model_coefficients.pdf')
        plt.close()
        
        print("✅ Created drift effects and model coefficients plots")
    
    def plot_significance_heatmap(self):
        """Create statistical significance heatmap."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for heatmap
        effects_data = []
        
        # Drift effects
        drift_cols = ['prompt_to_prompt_drift_pval', 'context_to_prompt_drift_pval', 
                     'cumulative_drift_pval', 'prompt_complexity_pval']
        drift_names = ['Prompt-to-Prompt Drift', 'Context-to-Prompt Drift', 
                      'Cumulative Drift', 'Prompt Complexity']
        
        for col, name in zip(drift_cols, drift_names):
            if col in self.p_values.columns:
                pval = self.p_values[col].iloc[0]
                effects_data.append({'Effect': name, 'Type': 'Drift Effect', 'P_Value': pval})
        
        # Model effects
        model_cols = [col for col in self.p_values.columns 
                     if 'model_' in col and '_pval' in col]
        
        for col in model_cols:
            model_name = col.replace('model_', '').replace('_pval', '')
            pval = self.p_values[col].iloc[0]
            effects_data.append({'Effect': f'vs {model_name}', 'Type': 'Model Effect', 'P_Value': pval})
        
        effects_df = pd.DataFrame(effects_data)
        
        # Create significance categories
        def significance_category(pval):
            if pval < 0.001:
                return '***'
            elif pval < 0.01:
                return '**'
            elif pval < 0.05:
                return '*'
            else:
                return 'ns'
        
        def significance_color(pval):
            if pval < 0.001:
                return 0.9  # Very significant
            elif pval < 0.01:
                return 0.7  # Significant
            elif pval < 0.05:
                return 0.5  # Marginally significant
            else:
                return 0.1  # Not significant
        
        effects_df['Significance'] = effects_df['P_Value'].apply(significance_category)
        effects_df['Color_Value'] = effects_df['P_Value'].apply(significance_color)
        
        # Create pivot table for heatmap
        pivot_df = effects_df.pivot(index='Effect', columns='Type', values='Color_Value')
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=False, cmap='Reds', cbar_kws={'label': 'Statistical Significance'},
                   ax=ax, linewidths=0.5)
        
        # Add significance annotations
        for i, effect in enumerate(pivot_df.index):
            for j, effect_type in enumerate(pivot_df.columns):
                if not pd.isna(pivot_df.iloc[i, j]):
                    sig_text = effects_df[(effects_df['Effect'] == effect) & 
                                        (effects_df['Type'] == effect_type)]['Significance'].iloc[0]
                    ax.text(j + 0.5, i + 0.5, sig_text, ha='center', va='center',
                           fontsize=12, fontweight='bold', color='white')
        
        ax.set_title('Statistical Significance of Effects\n(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Effect Type', fontsize=12)
        ax.set_ylabel('Variables', fontsize=12)
        
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/3_significance_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/3_significance_heatmap.pdf', 
                   bbox_inches='tight')
        plt.close()
        
        print("✅ Created significance heatmap")
    
    def plot_model_performance_overview(self):
        """Create model performance overview plots."""
        # Performance metrics
        c_index = self.performance['c_index'].iloc[0]
        n_obs = self.performance['n_observations'].iloc[0]
        n_events = self.performance['n_events'].iloc[0]
        n_models = self.performance['n_models'].iloc[0]
        
        # Plot 1: Model Performance Gauge (C-index)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wedges, texts, autotexts = ax.pie([c_index, 1-c_index], 
                                         labels=['C-index', 'Remaining'], 
                                         colors=[COLORS['success'], '#E8E8E8'],
                                         startangle=90,
                                         autopct='%1.3f',
                                         textprops={'fontsize': 14, 'fontweight': 'bold'})
        
        ax.text(0, 0, f'{c_index:.4f}', ha='center', va='center', 
                fontsize=24, fontweight='bold', color=COLORS['primary'])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_gauge.png')
        plt.savefig(f'{self.output_dir}/performance_gauge.pdf')
        plt.close()
        
        # Plot 2: Dataset Overview
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Total\nObservations', 'Failure\nEvents', 'LLM\nModels', 'Event\nRate (%)']
        values = [n_obs, n_events, n_models, (n_events/n_obs)*100]
        colors = [COLORS['primary'], COLORS['warning'], COLORS['success'], COLORS['danger']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel('Count / Percentage', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{val:.0f}' if val > 1 else f'{val:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dataset_overview.png')
        plt.savefig(f'{self.output_dir}/dataset_overview.pdf')
        plt.close()
        
        # Plot 3: Risk Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_hrs = []
        hr_cols = [col for col in self.hazard_ratios.columns 
                  if 'model_' in col and '_hr' in col and '_ci' not in col]
        
        for col in hr_cols:
            hr_val = self.hazard_ratios[col].iloc[0]
            model_hrs.append(hr_val)
        
        model_hrs.append(1.0)  # Add claude_35 reference
        
        ax.hist(model_hrs, bins=8, color=COLORS['primary'], alpha=0.7, 
                edgecolor=COLORS['neutral'], linewidth=2)
        ax.axvline(x=1.0, color=COLORS['danger'], linestyle='--', linewidth=3, 
                   label='Reference (claude_35)')
        ax.set_xlabel('Hazard Ratio', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Models', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/risk_distribution.png')
        plt.savefig(f'{self.output_dir}/risk_distribution.pdf')
        plt.close()
        
        # Plot 4: Top Safest Models
        fig, ax = plt.subplots(figsize=(10, 8))
        
        model_names = []
        for col in hr_cols:
            model_name = col.replace('model_', '').replace('_hr', '')
            model_names.append(model_name)
        model_names.append('claude_35')
        
        ranking_data = list(zip(model_names, model_hrs))
        ranking_data.sort(key=lambda x: x[1])
        
        models, hrs = zip(*ranking_data[:8])  # Top 8 safest models
        
        colors = []
        for hr in hrs:
            if hr < 0.5:
                colors.append(COLORS['safe'])
            elif hr < 0.8:
                colors.append(COLORS['success'])
            elif hr < 1.2:
                colors.append(COLORS['primary'])
            else:
                colors.append(COLORS['danger'])
        
        bars = ax.barh(range(len(models)), hrs, color=colors, alpha=0.8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=12)
        ax.set_xlabel('Hazard Ratio', fontsize=14, fontweight='bold')
        ax.axvline(x=1.0, color=COLORS['neutral'], linestyle='--', alpha=0.8, linewidth=2)
        
        # Add hazard ratio values
        for i, (bar, hr) in enumerate(zip(bars, hrs)):
            x_pos = hr + 0.02 if hr < 2.0 else hr - 0.1
            ax.text(x_pos, i, f'{hr:.2f}', va='center', 
                   fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/safest_models.png')
        plt.savefig(f'{self.output_dir}/safest_models.pdf')
        plt.close()
        
        print("✅ Created performance gauge, dataset overview, risk distribution, and safest models plots")
    
    def plot_risk_ranking_detailed(self):
        """Create detailed risk ranking visualizations."""
        # Extract all model data
        model_data = []
        hr_cols = [col for col in self.hazard_ratios.columns 
                  if 'model_' in col and '_hr' in col and '_ci' not in col]
        
        for col in hr_cols:
            model_name = col.replace('model_', '').replace('_hr', '')
            hr_val = self.hazard_ratios[col].iloc[0]
            
            # Get p-value
            pval_col = col.replace('_hr', '_pval')
            pval = self.p_values[pval_col].iloc[0] if pval_col in self.p_values.columns else 1.0
            
            model_data.append({
                'model': model_name,
                'hazard_ratio': hr_val,
                'p_value': pval,
                'risk_level': 'Low' if hr_val < 0.5 else 'Medium' if hr_val < 1.0 else 'High'
            })
        
        # Add claude_35 as reference
        model_data.append({
            'model': 'claude_35',
            'hazard_ratio': 1.0,
            'p_value': np.nan,
            'risk_level': 'Reference'
        })
        
        df = pd.DataFrame(model_data).sort_values('hazard_ratio')
        
        # Plot 1: Complete Risk Ranking - Publication Ready
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Clean, academic color scheme - subtle but clear distinction
        def get_bar_color(hr, significant):
            if hr < 0.3:
                return '#2E7D32' if significant else '#4CAF50'  # Dark/light green for very safe
            elif hr < 0.7:
                return '#388E3C' if significant else '#66BB6A'  # Medium green for safe
            elif hr < 1.0:
                return '#1976D2' if significant else '#42A5F5'  # Blue for moderate
            else:
                return '#D32F2F' if significant else '#F44336'  # Red for risky
        
        # Create bars with clean styling
        colors = [get_bar_color(row['hazard_ratio'], row['p_value'] < 0.05) 
                 for _, row in df.iterrows()]
        
        bars = ax.barh(range(len(df)), df['hazard_ratio'], 
                      color=colors, alpha=0.85, height=0.7, 
                      edgecolor='white', linewidth=0.5)
        
        # Clean model labels - simple text on y-axis
        model_labels = []
        for _, row in df.iterrows():
            # Clean model name without significance markers
            label = row['model'].replace('_', ' ').title()
            model_labels.append(label)
        
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(model_labels, fontsize=11)
        
        # Clean hazard ratio values - minimal but clear
        for i, (_, row) in enumerate(df.iterrows()):
            hr_val = row['hazard_ratio']
            # Position text appropriately
            if hr_val < 0.8:
                x_pos = hr_val + 0.03
                ha = 'left'
            else:
                x_pos = hr_val - 0.03
                ha = 'right'
            
            ax.text(x_pos, i, f'{hr_val:.3f}', 
                   va='center', ha=ha, fontsize=9, 
                   fontweight='bold', color='black')
        
        # Minimal reference lines
        ax.axvline(x=1.0, color='#424242', linestyle='-', alpha=0.8, linewidth=1.5)
        ax.axvline(x=0.5, color='#757575', linestyle='--', alpha=0.6, linewidth=1)
        
        # Clean axes
        ax.set_xlabel('Hazard Ratio (relative to Claude 3.5)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(df['hazard_ratio']) * 1.15)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Subtle grid for better readability
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Clean legend - simple and informative
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E7D32', alpha=0.85, label='Low Risk (HR < 0.5)'),
            Patch(facecolor='#1976D2', alpha=0.85, label='Medium Risk (0.5 ≤ HR < 1.0)'),
            Patch(facecolor='#D32F2F', alpha=0.85, label='High Risk (HR ≥ 1.0)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10, 
                 frameon=True, fancybox=False, shadow=False, framealpha=0.95)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/complete_risk_ranking.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.output_dir}/complete_risk_ranking.pdf', bbox_inches='tight')
        plt.close()
        
        print("✅ Created publication-ready complete risk ranking plot")
    
    def plot_round_specific_hazard_dynamics(self):
        """Create line plot showing log hazard ratios across follow-up rounds for each model."""
        print("🎨 Generating round-specific hazard dynamics plot...")
        
        # Load the combined model to get round-specific predictions
        try:
            import sys
            sys.path.append('src')
            from modeling.baseline import BaselineModeling
            
            # Create baseline instance and load data
            baseline = BaselineModeling()
            baseline.load_data()
            
            # Calculate round-specific hazard ratios for each model
            models_hazard_data = []
            
            for model_name in baseline.models_data.keys():
                long_df = baseline.models_data[model_name]['long'].copy()
                
                # Calculate empirical hazard rate at each round
                round_hazards = []
                for round_num in range(1, 9):  # Rounds 1-8
                    round_data = long_df[long_df['round'] == round_num]
                    
                    if len(round_data) > 0:
                        # Calculate empirical hazard rate (failures / at-risk)
                        failures = round_data['failure'].sum()
                        at_risk = len(round_data)
                        
                        if at_risk > 0:
                            hazard_rate = failures / at_risk
                            # Add small constant to avoid log(0)
                            log_hazard = np.log(max(hazard_rate, 1e-6))
                        else:
                            log_hazard = np.log(1e-6)
                    else:
                        log_hazard = np.log(1e-6)
                    
                    round_hazards.append(log_hazard)
                
                models_hazard_data.append({
                    'model': model_name,
                    'rounds': list(range(1, 9)),
                    'log_hazards': round_hazards
                })
            
            # Create the line plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Use a color palette with enough distinct colors for all models
            colors = plt.cm.tab20(np.linspace(0, 1, len(models_hazard_data)))
            
            for i, model_data in enumerate(models_hazard_data):
                model_name = model_data['model'].replace('_', ' ').title()
                rounds = model_data['rounds']
                log_hazards = model_data['log_hazards']
                
                # Plot the line
                ax.plot(rounds, log_hazards, 
                       color=colors[i], 
                       linewidth=2.5, 
                       marker='o', 
                       markersize=6,
                       label=model_name,
                       alpha=0.8)
            
            # Customize the plot
            ax.set_xlabel('Follow-up Round', fontsize=14, fontweight='bold')
            ax.set_ylabel('Log Hazard Rate', fontsize=14, fontweight='bold')
            ax.set_xticks(range(1, 9))
            ax.set_xticklabels([f'Round {i}' for i in range(1, 9)])
            
            # Add reference line at y=0 (log(1) = 0, representing neutral hazard)
            ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', alpha=0.6, linewidth=1)
            
            # Customize grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Legend with two columns to fit all models
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=10, ncol=1)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/round_hazard_dynamics.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/round_hazard_dynamics.pdf', bbox_inches='tight')
            plt.close()
            
            print("✅ Created round-specific hazard dynamics line plot")
            
        except Exception as e:
            print(f"❌ Error creating round hazard dynamics plot: {e}")
    
    def plot_cumulative_hazard_dynamics(self):
        """Create line plot showing cumulative hazard rates across follow-up rounds for each model."""
        print("🎨 Generating cumulative hazard dynamics plot...")
        
        # Load the combined model to get round-specific predictions
        try:
            import sys
            sys.path.append('src')
            from modeling.baseline import BaselineModeling
            
            # Create baseline instance and load data
            baseline = BaselineModeling()
            baseline.load_data()
            
            # Calculate cumulative hazard rates for each model
            models_cumulative_data = []
            
            for model_name in baseline.models_data.keys():
                long_df = baseline.models_data[model_name]['long'].copy()
                
                # Calculate empirical hazard rate at each round and cumulate
                round_hazards = []
                cumulative_hazard = 0.0
                
                for round_num in range(1, 9):  # Rounds 1-8
                    round_data = long_df[long_df['round'] == round_num]
                    
                    if len(round_data) > 0:
                        # Calculate empirical hazard rate (failures / at-risk)
                        failures = round_data['failure'].sum()
                        at_risk = len(round_data)
                        
                        if at_risk > 0:
                            hazard_rate = failures / at_risk
                        else:
                            hazard_rate = 0.0
                    else:
                        hazard_rate = 0.0
                    
                    # Add to cumulative hazard
                    cumulative_hazard += hazard_rate
                    round_hazards.append(cumulative_hazard)
                
                models_cumulative_data.append({
                    'model': model_name,
                    'rounds': list(range(1, 9)),
                    'cumulative_hazards': round_hazards
                })
            
            # Create the line plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Use a color palette with enough distinct colors for all models
            colors = plt.cm.tab20(np.linspace(0, 1, len(models_cumulative_data)))
            
            for i, model_data in enumerate(models_cumulative_data):
                model_name = model_data['model'].replace('_', ' ').title()
                rounds = model_data['rounds']
                cumulative_hazards = model_data['cumulative_hazards']
                
                # Plot the line
                ax.plot(rounds, cumulative_hazards, 
                       color=colors[i], 
                       linewidth=2.5, 
                       marker='o', 
                       markersize=6,
                       label=model_name,
                       alpha=0.8)
            
            # Customize the plot
            ax.set_xlabel('Follow-up Round', fontsize=14, fontweight='bold')
            ax.set_ylabel('Cumulative Hazard Rate', fontsize=14, fontweight='bold')
            ax.set_xticks(range(1, 9))
            ax.set_xticklabels([f'Round {i}' for i in range(1, 9)])
            
            # Customize grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Legend with two columns to fit all models
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                     fontsize=10, ncol=1)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            
            # Start y-axis from 0 for cumulative interpretation
            ax.set_ylim(bottom=0)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/cumulative_hazard_dynamics.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/cumulative_hazard_dynamics.pdf', bbox_inches='tight')
            plt.close()
            
            print("✅ Created cumulative hazard dynamics line plot")
            
        except Exception as e:
            print(f"❌ Error creating cumulative hazard dynamics plot: {e}")
    
    def plot_cumulative_hazard_by_difficulty(self):
        """Create 2x2 subplot showing cumulative hazard dynamics by difficulty level."""
        print("🎨 Generating cumulative hazard by difficulty level plot...")
        
        try:
            import sys
            sys.path.append('src')
            from modeling.baseline import BaselineModeling
            
            # Create baseline instance and load data
            baseline = BaselineModeling()
            baseline.load_data()
            
            # Get difficulty levels
            difficulty_levels = ['elementary', 'high_school', 'college', 'professional']
            
            # Create 2x2 subplot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            # Use consistent colors for models across subplots
            model_names = list(baseline.models_data.keys())
            colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))
            model_color_map = {model: colors[i] for i, model in enumerate(model_names)}
            
            for diff_idx, difficulty in enumerate(difficulty_levels):
                ax = axes[diff_idx]
                
                # Calculate cumulative hazard rates for each model at this difficulty level
                for model_name in baseline.models_data.keys():
                    long_df = baseline.models_data[model_name]['long'].copy()
                    
                    # Filter by difficulty level
                    if 'level' in long_df.columns:
                        diff_data = long_df[long_df['level'] == difficulty]
                    else:
                        # Skip if no level data
                        continue
                    
                    if len(diff_data) == 0:
                        continue
                    
                    # Calculate cumulative hazard for this model and difficulty
                    round_hazards = []
                    cumulative_hazard = 0.0
                    
                    for round_num in range(1, 9):  # Rounds 1-8
                        round_data = diff_data[diff_data['round'] == round_num]
                        
                        if len(round_data) > 0:
                            failures = round_data['failure'].sum()
                            at_risk = len(round_data)
                            
                            if at_risk > 0:
                                hazard_rate = failures / at_risk
                            else:
                                hazard_rate = 0.0
                        else:
                            hazard_rate = 0.0
                        
                        cumulative_hazard += hazard_rate
                        round_hazards.append(cumulative_hazard)
                    
                    # Plot the line for this model
                    model_display = model_name.replace('_', ' ').title()
                    ax.plot(range(1, 9), round_hazards,
                           color=model_color_map[model_name],
                           linewidth=2,
                           marker='o',
                           markersize=4,
                           label=model_display,
                           alpha=0.8)
                
                # Customize each subplot
                ax.set_title(f'{difficulty.replace("_", " ").title()} Level', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Follow-up Round', fontsize=12)
                ax.set_ylabel('Cumulative Hazard Rate', fontsize=12)
                ax.set_xticks(range(1, 9))
                ax.set_xticklabels([f'R{i}' for i in range(1, 9)])
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_axisbelow(True)
                ax.set_ylim(bottom=0)
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#CCCCCC')
                ax.spines['bottom'].set_color('#CCCCCC')

            # Add a single legend for all subplots at the bottom, flattened in 2 lines
            handles, labels = axes[0].get_legend_handles_labels()

            # Only create legend if there are handles to display
            if len(handles) > 0:
                ncol = len(handles) // 2 if len(handles) % 2 == 0 else (len(handles) + 1) // 2
                ncol = max(1, ncol)  # Ensure ncol is at least 1
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                          fontsize=10, ncol=ncol, frameon=False)
                plt.subplots_adjust(bottom=0.15)  # Reduce gap for legend at bottom
            else:
                print("⚠️  No data available for difficulty level plots - skipping legend")
                plt.subplots_adjust(bottom=0.05)  # Less gap if no legend

            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/cumulative_hazard_by_difficulty.png',
                       dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/cumulative_hazard_by_difficulty.pdf', 
                       bbox_inches='tight')
            plt.close()
            
            print("✅ Created cumulative hazard by difficulty level plot")
            
        except Exception as e:
            print(f"❌ Error creating cumulative hazard by difficulty plot: {e}")
    
    def plot_cumulative_hazard_by_subject_cluster(self):
        """Create 2x4 subplot showing cumulative hazard dynamics by subject cluster (7 clusters + 1 overall)."""
        try:
            print("🎨 Generating cumulative hazard by subject cluster plot...")
            
            baseline = BaselineModeling()
            baseline.load_data()
            
            # Define subject clusters (7 clusters + overall)
            subject_clusters = ['Business_Economics', 'General_Knowledge', 'Humanities', 
                              'Law_Legal', 'Medical_Health', 'STEM', 'Social_Sciences']
            
            # Create 2x4 subplot
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            # Use consistent colors for models across subplots
            model_names = list(baseline.models_data.keys())
            colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))
            model_color_map = {model: colors[i] for i, model in enumerate(model_names)}
            
            # Plot for each subject cluster
            for cluster_idx, cluster in enumerate(subject_clusters):
                ax = axes[cluster_idx]
                
                # Calculate cumulative hazard rates for each model in this cluster
                for model_name in baseline.models_data.keys():
                    long_df = baseline.models_data[model_name]['long'].copy()
                    
                    # Filter by subject cluster
                    if 'subject_cluster' in long_df.columns:
                        cluster_data = long_df[long_df['subject_cluster'] == cluster]
                    else:
                        continue
                    
                    if len(cluster_data) == 0:
                        continue
                    
                    # Calculate cumulative hazard for this model and cluster
                    round_hazards = []
                    cumulative_hazard = 0.0
                    
                    for round_num in range(1, 9):  # Rounds 1-8
                        round_data = cluster_data[cluster_data['round'] == round_num]
                        
                        if len(round_data) > 0:
                            failures = round_data['failure'].sum()
                            at_risk = len(round_data)
                            
                            if at_risk > 0:
                                hazard_rate = failures / at_risk
                            else:
                                hazard_rate = 0.0
                        else:
                            hazard_rate = 0.0
                        
                        cumulative_hazard += hazard_rate
                        round_hazards.append(cumulative_hazard)
                    
                    # Plot the line for this model
                    model_display = model_name.replace('_', ' ').title()
                    ax.plot(range(1, 9), round_hazards,
                           color=model_color_map[model_name],
                           linewidth=2,
                           marker='o',
                           markersize=4,
                           label=model_display,
                           alpha=0.8)
                
                # Customize each subplot
                ax.set_title(f'{cluster.replace("_", " & ")}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Follow-up Round', fontsize=12)
                ax.set_ylabel('Cumulative Hazard Rate', fontsize=12)
                ax.set_xticks(range(1, 9))
                ax.set_xticklabels([f'R{i}' for i in range(1, 9)])
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_axisbelow(True)
                ax.set_ylim(bottom=0)
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#CCCCCC')
                ax.spines['bottom'].set_color('#CCCCCC')
            
            # Plot overall analysis in the last subplot (index 7)
            ax = axes[7]
            
            # Calculate overall cumulative hazard rates for each model
            for model_name in baseline.models_data.keys():
                long_df = baseline.models_data[model_name]['long'].copy()
                
                # Calculate cumulative hazard for this model (overall)
                round_hazards = []
                cumulative_hazard = 0.0
                
                for round_num in range(1, 9):  # Rounds 1-8
                    round_data = long_df[long_df['round'] == round_num]
                    
                    if len(round_data) > 0:
                        failures = round_data['failure'].sum()
                        at_risk = len(round_data)
                        
                        if at_risk > 0:
                            hazard_rate = failures / at_risk
                        else:
                            hazard_rate = 0.0
                    else:
                        hazard_rate = 0.0
                    
                    cumulative_hazard += hazard_rate
                    round_hazards.append(cumulative_hazard)
                
                # Plot the line for this model
                model_display = model_name.replace('_', ' ').title()
                ax.plot(range(1, 9), round_hazards,
                       color=model_color_map[model_name],
                       linewidth=2,
                       marker='o',
                       markersize=4,
                       label=model_display,
                       alpha=0.8)
            
            # Customize overall subplot
            ax.set_title('Overall', fontsize=14, fontweight='bold')
            ax.set_xlabel('Follow-up Round', fontsize=12)
            ax.set_ylabel('Cumulative Hazard Rate', fontsize=12)
            ax.set_xticks(range(1, 9))
            ax.set_xticklabels([f'R{i}' for i in range(1, 9)])
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')

            # Add a single legend for all subplots at the bottom, flattened in 2 lines
            handles, labels = axes[0].get_legend_handles_labels()

            # Only create legend if there are handles to display
            if len(handles) > 0:
                ncol = len(handles) // 2 if len(handles) % 2 == 0 else (len(handles) + 1) // 2
                ncol = max(1, ncol)  # Ensure ncol is at least 1
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                          fontsize=10, ncol=ncol, frameon=False)
                plt.subplots_adjust(bottom=0.15)  # Reduce gap for legend at bottom
            else:
                print("⚠️  No data available for subject cluster plots - skipping legend")
                plt.subplots_adjust(bottom=0.05)  # Less gap if no legend

            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/cumulative_hazard_by_subject_cluster.png',
                       dpi=300, bbox_inches='tight')
            plt.savefig(f'{self.output_dir}/cumulative_hazard_by_subject_cluster.pdf', 
                       bbox_inches='tight')
            plt.close()
            
            print("✅ Created cumulative hazard by subject cluster plot")
            
        except Exception as e:
            print(f"❌ Error creating cumulative hazard by subject cluster plot: {e}")
    
    def generate_all_visualizations(self):
        """Generate all baseline visualization plots."""
        print("🎨 GENERATING COMBINED BASELINE VISUALIZATIONS")
        print("=" * 50)
        
        self.plot_hazard_ratio_comparison()
        self.plot_coefficient_effects()
        self.plot_significance_heatmap()
        self.plot_model_performance_overview()
        self.plot_risk_ranking_detailed()
        self.plot_round_specific_hazard_dynamics()
        self.plot_cumulative_hazard_dynamics()
        self.plot_cumulative_hazard_by_difficulty()
        self.plot_cumulative_hazard_by_subject_cluster()
        
        print(f"\n✅ ALL VISUALIZATIONS COMPLETED!")
        print(f"📁 Plots saved to: {self.output_dir}/")
        print("📊 Generated 15 individual professional plots:")
        print("   • hazard_ratio_comparison (bar chart)")
        print("   • drift_effects (coefficient effects)")
        print("   • model_coefficients (model effects)")
        print("   • significance_heatmap (p-values)")
        print("   • performance_gauge (C-index)")
        print("   • dataset_overview (data statistics)")
        print("   • risk_distribution (histogram)")
        print("   • safest_models (rankings)")
        print("   • complete_risk_ranking (detailed)")
        print("   • risk_category_distribution (pie chart)")
        print("   • round_hazard_dynamics (line plot)")
        print("   • cumulative_hazard_dynamics (cumulative line plot)")
        print("   • cumulative_hazard_by_difficulty (2x2 subplot by difficulty)")
        print("   • cumulative_hazard_by_subject_cluster (2x4 subplot by subject cluster)")
        print("   • All plots saved in PNG and PDF formats")


def main():
    """Main execution function."""
    visualizer = BaselineCombinedVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()