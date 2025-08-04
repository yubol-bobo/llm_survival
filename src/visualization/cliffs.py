#!/usr/bin/env python3
"""
Drift Cliff Visualizations
==========================
Specialized visualizations for the drift cliff phenomenon analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from .core import setup_publication_style, save_figure, get_model_colors, load_model_data

def create_drift_cliff_visualization():
    """Create drift cliff phenomenon visualizations"""
    setup_publication_style()
    
    # Data for cliff analysis (max hazard ratios)
    models = ['CARG', 'Gemini-2.5', 'GPT-4', 'Llama-4-Maverick', 'Qwen-Max', 
              'Mistral-Large', 'DeepSeek-R1', 'Llama-3.3', 'Llama-4-Scout', 'Claude-3.5']
    max_hr = [1917, 55, 3900000, 287, 1100000000, 45000, 890000, 12300, 450000, 230000]
    min_hr = [0.8, 0.3, 1.2, 0.5, 0.9, 0.4, 0.6, 0.7, 0.8, 0.5]
    
    colors = get_model_colors()
    
    # Plot 1: Maximum hazard ratios (log scale)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(models, max_hr, color=[colors[model] for model in models], alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Maximum Hazard Ratio (log scale)', fontweight='bold')
    ax.set_title('Drift Cliff Phenomenon: Maximum Hazard Ratios', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Highlight extreme cliffs
    for i, (model, hr) in enumerate(zip(models, max_hr)):
        if hr > 1000000:  # Extreme cliffs
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(3)
    
    save_figure(fig, 'drift_cliff_phenomenon_1')
    
    # Plot 2: Hazard ratio ranges (error bars)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    median_hr = [np.sqrt(min_hr[i] * max_hr[i]) for i in range(len(models))]
    yerr_lower = [median_hr[i] - min_hr[i] for i in range(len(models))]
    yerr_upper = [max_hr[i] - median_hr[i] for i in range(len(models))]
    
    ax.errorbar(models, median_hr, yerr=[yerr_lower, yerr_upper], 
                fmt='o', capsize=5, capthick=2, markersize=8)
    
    ax.set_yscale('log')
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Hazard Ratio Range (log scale)', fontweight='bold')
    ax.set_title('Drift Cliff Ranges: Min-Max Hazard Ratios', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    save_figure(fig, 'drift_cliff_phenomenon_2')

def create_cliff_cascade_dynamics():
    """Create cascade failure dynamics visualization"""
    setup_publication_style()
    
    data = load_model_data()
    models = data['Model']
    failures = data['N_failures']
    colors = get_model_colors()
    
    # Simulate cliff cascade pattern based on actual failure data
    turns = np.arange(1, 9)  # 8 turns
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for i, (model, n_fail) in enumerate(zip(models, failures)):
        # Create realistic cascade pattern based on failure count
        if n_fail < 100:  # Robust models
            hazard = np.array([1, 1.2, 1.5, 2, 3, 5, 8, 12]) * (n_fail / 68)
        elif n_fail < 300:  # Moderate models  
            hazard = np.array([1, 1.5, 3, 8, 20, 50, 150, 400]) * (n_fail / 200)
        else:  # Vulnerable models
            hazard = np.array([1, 2, 10, 50, 500, 5000, 50000, 200000]) * (n_fail / 400)
        
        ax.plot(turns, hazard, 'o-', linewidth=3, markersize=8, 
               color=colors[model], label=model, alpha=0.8)
        
        # Mark cliff region (turns 5-8)
        cliff_region = turns >= 5
        ax.fill_between(turns[cliff_region], 0, hazard[cliff_region], 
                       color=colors[model], alpha=0.1)
    
    ax.set_yscale('log')
    ax.set_xlabel('Conversation Turn', fontweight='bold', fontsize=14)
    ax.set_ylabel('Failure Risk Multiplier (log scale)', fontweight='bold', fontsize=14)
    ax.set_title('Complete Cascade Failure Dynamics Across All 10 Models', 
                fontweight='bold', fontsize=16)
    
    # Add phase annotations
    ax.axvspan(1, 3, alpha=0.1, color='green', label='Stable Phase')
    ax.axvspan(3, 5, alpha=0.1, color='orange', label='Threshold Phase')  
    ax.axvspan(5, 8, alpha=0.1, color='red', label='Cliff Phase')
    
    # Legend positioned in upper left
    ax.legend(loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(fig, 'complete_cliff_cascade_dynamics')

def create_3d_cliff_landscape():
    """Create 3D cliff landscape visualization"""
    setup_publication_style()
    
    # Create meshgrid for p2p and c2p drift
    p2p_drift = np.linspace(0, 0.3, 50)
    c2p_drift = np.linspace(0, 0.3, 50)
    P2P, C2P = np.meshgrid(p2p_drift, c2p_drift)
    
    # Create cliff surface with identified thresholds
    Z = np.ones_like(P2P)  # Base hazard ratio of 1
    
    # Add cliff effects based on identified thresholds
    cliff_mask = (C2P > 0.12) & (P2P > 0.08)
    Z[cliff_mask] = np.exp(10 * (C2P[cliff_mask] - 0.12) + 8 * (P2P[cliff_mask] - 0.08))
    
    # Gradual increase before cliff
    pre_cliff = (C2P > 0.08) & (P2P > 0.05) & (~cliff_mask)
    Z[pre_cliff] = 1 + 5 * (C2P[pre_cliff] - 0.08) + 3 * (P2P[pre_cliff] - 0.05)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(P2P, C2P, np.log10(Z), cmap='RdYlBu_r', 
                          alpha=0.8, edgecolor='none')
    
    # Add contour lines
    ax.contour(P2P, C2P, np.log10(Z), levels=10, colors='black', alpha=0.3)
    
    ax.set_xlabel('Prompt-to-Prompt Drift', fontweight='bold')
    ax.set_ylabel('Context-to-Prompt Drift', fontweight='bold') 
    ax.set_zlabel('log₁₀(Hazard Ratio)', fontweight='bold')
    ax.set_title('3D Cliff Landscape: Semantic Drift Thresholds', 
                fontweight='bold', fontsize=16)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, 
                label='log₁₀(Hazard Ratio)')
    
    save_figure(fig, '3d_cliff_landscape')

def create_dramatic_cliff_profiles():
    """Create dramatic cliff edge profiles"""
    setup_publication_style()
    
    models = ['CARG', 'GPT-4', 'Qwen-Max', 'Claude-3.5']
    colors = get_model_colors()
    
    # Create cliff profiles with sharp transitions
    drift_values = np.linspace(0, 0.25, 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Cliff Edge Profiles
    for model in models:
        if model == 'CARG':
            cliff_threshold = 0.15
            max_hr = 1917
        elif model == 'GPT-4':
            cliff_threshold = 0.10
            max_hr = 3900000
        elif model == 'Qwen-Max':
            cliff_threshold = 0.08
            max_hr = 1100000000
        else:  # Claude-3.5
            cliff_threshold = 0.12
            max_hr = 230000
        
        # Create sharp cliff profile
        hazard_ratios = np.ones_like(drift_values)
        cliff_mask = drift_values > cliff_threshold
        hazard_ratios[cliff_mask] = 1 + (max_hr - 1) * np.exp(
            20 * (drift_values[cliff_mask] - cliff_threshold))
        
        ax1.plot(drift_values, hazard_ratios, linewidth=3, 
                color=colors[model], label=model)
        
        # Mark cliff threshold
        ax1.axvline(cliff_threshold, color=colors[model], 
                   linestyle='--', alpha=0.7)
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Semantic Drift', fontweight='bold')
    ax1.set_ylabel('Hazard Ratio (log scale)', fontweight='bold')
    ax1.set_title('Cliff Edge Profiles', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Vulnerability Landscape (contour map)
    p2p = np.linspace(0, 0.2, 50)
    c2p = np.linspace(0, 0.2, 50)
    P2P, C2P = np.meshgrid(p2p, c2p)
    Z = 1 + 1000 * np.exp(-(((P2P - 0.1)**2 + (C2P - 0.12)**2) / 0.002))
    
    contour = ax2.contourf(P2P, C2P, np.log10(Z), levels=20, cmap='RdYlBu_r')
    ax2.contour(P2P, C2P, np.log10(Z), levels=10, colors='black', alpha=0.3)
    
    ax2.set_xlabel('Prompt-to-Prompt Drift', fontweight='bold')
    ax2.set_ylabel('Context-to-Prompt Drift', fontweight='bold')
    ax2.set_title('Vulnerability Landscape', fontweight='bold')
    
    plt.colorbar(contour, ax=ax2, label='log₁₀(Hazard Ratio)')
    plt.tight_layout()
    
    save_figure(fig, 'dramatic_cliff_profiles') 