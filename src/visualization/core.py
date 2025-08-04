#!/usr/bin/env python3
"""
Core Visualization Utilities
============================
Shared functions, styling, and utilities for LLM survival analysis visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Rectangle

# Publication-ready style configuration
def setup_publication_style():
    """Configure matplotlib and seaborn for publication-ready plots"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

def ensure_output_dir(path='results/figures'):
    """Ensure output directory exists"""
    os.makedirs(path, exist_ok=True)

def load_model_data():
    """Load consistent model performance data"""
    return {
        'Model': ['CARG', 'Gemini-2.5', 'GPT-4', 'Llama-4-Maverick', 'Qwen-Max', 
                 'Mistral-Large', 'DeepSeek-R1', 'Llama-3.3', 'Llama-4-Scout', 'Claude-3.5'],
        'N_failures': [68, 78, 134, 174, 252, 269, 344, 377, 385, 453],
        'C_index': [0.750, 0.773, 0.754, 0.778, 0.780, 0.794, 0.801, 0.797, 0.770, 0.760],
        'Mean_TTF': [7.43, 7.43, 6.84, 6.25, 6.02, 5.28, 5.21, 4.59, 4.61, 4.38],
        'Failure_Rate': [0.126, 0.132, 0.245, 0.404, 0.495, 0.591, 0.658, 0.825, 0.795, 0.764]
    }

def get_model_colors():
    """Get consistent color scheme for models"""
    return {
        'CARG': '#1f77b4',
        'Gemini-2.5': '#ff7f0e', 
        'GPT-4': '#2ca02c',
        'Llama-4-Maverick': '#d62728',
        'Qwen-Max': '#9467bd',
        'Mistral-Large': '#8c564b',
        'DeepSeek-R1': '#e377c2',
        'Llama-3.3': '#7f7f7f',
        'Llama-4-Scout': '#bcbd22',
        'Claude-3.5': '#17becf'
    }

def save_figure(fig, filename, output_dir='results/figures', dpi=600, format='pdf'):
    """Save figure with consistent settings"""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, f"{filename}.{format}")
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
    plt.close(fig)
    print(f"✅ Saved: {filepath}")

def create_model_performance_comparison():
    """Create model performance comparison plots"""
    setup_publication_style()
    data = load_model_data()
    df = pd.DataFrame(data)
    colors = get_model_colors()
    
    # Plot 1: Failure ranking
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['Model'], df['N_failures'], 
                  color=[colors[model] for model in df['Model']])
    
    # Highlight top performers
    for i, (model, failures) in enumerate(zip(df['Model'], df['N_failures'])):
        if failures < 200:  # Elite performers
            bars[i].set_edgecolor('gold')
            bars[i].set_linewidth(3)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Number of Failures', fontweight='bold')
    ax.set_title('LLM Robustness Ranking by Failure Count', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    save_figure(fig, 'model_performance_comparison_1')
    
    # Plot 2: Performance trade-off
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['N_failures'], df['C_index'], 
                        s=100, alpha=0.7,
                        c=[colors[model] for model in df['Model']])
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        ax.annotate(model, (df['N_failures'][i], df['C_index'][i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Number of Failures (lower is better)', fontweight='bold')
    ax.set_ylabel('C-index (higher is better)', fontweight='bold')
    ax.set_title('Robustness vs. Discrimination Trade-off', fontweight='bold')
    plt.grid(alpha=0.3)
    
    save_figure(fig, 'model_performance_comparison_2')

def create_semantic_drift_effects():
    """Create semantic drift effects visualization"""
    setup_publication_style()
    
    # Hazard ratios for different drift types
    models = ['CARG', 'Gemini-2.5', 'GPT-4', 'Llama-4-Maverick', 'Qwen-Max']
    p2p_hr = [2.1, 2.3, 4.7, 3.2, 2.8]
    c2p_hr = [1.8, 1.9, 3.1, 2.4, 2.1]
    cum_hr = [0.3, 0.4, 0.8, 0.6, 0.5]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, p2p_hr, width, label='Prompt-to-Prompt Drift', alpha=0.8)
    bars2 = ax.bar(x, c2p_hr, width, label='Context-to-Prompt Drift', alpha=0.8)
    bars3 = ax.bar(x + width, cum_hr, width, label='Cumulative Drift', alpha=0.8)
    
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Neutral Effect (HR=1)')
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Hazard Ratio', fontweight='bold')
    ax.set_title('Semantic Drift Effects on Failure Risk', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_figure(fig, 'semantic_drift_effects')

# Initialize style when module is imported
setup_publication_style() 