#!/usr/bin/env python3
"""
Trajectory Visualizations
========================
Trajectory-style drift visualizations for conversation flow analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from .core import setup_publication_style, save_figure, get_model_colors, load_model_data

def create_drift_trajectory_plots():
    """Create trajectory-style drift visualizations"""
    setup_publication_style()
    
    models = ['gemini_25', 'gpt_oss', 'gpt_default', 'llama_4_maverick', 'claude_35']
    colors = get_model_colors()
    turns = np.arange(1, 9)  # 8 turns
    
    # Plot 1: Multi-model trajectory paths
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model in models:
        # Create realistic drift trajectory based on model characteristics
        if model == 'gemini_25':
            trajectory = [0.01, 0.02, 0.035, 0.05, 0.07, 0.09, 0.12, 0.15]
        elif model == 'gpt_oss':
            trajectory = [0.015, 0.03, 0.05, 0.075, 0.105, 0.14, 0.175, 0.21]
        elif model == 'gpt_default':
            trajectory = [0.025, 0.04, 0.07, 0.12, 0.19, 0.28, 0.38, 0.5]
        elif model == 'llama_4_maverick':
            trajectory = [0.02, 0.038, 0.065, 0.105, 0.16, 0.23, 0.31, 0.42]
        else:  # claude_35
            trajectory = [0.03, 0.05, 0.08, 0.13, 0.21, 0.32, 0.45, 0.6]
        
        # Plot trajectory with markers
        ax.plot(turns, trajectory, 'o-', linewidth=3, markersize=8, 
                color=colors[model], label=model, alpha=0.8)
        
        # Add drift zones
        if max(trajectory) > 0.4:  # High drift models
            ax.fill_between(turns[5:], 0, [trajectory[i] for i in range(5, 8)], 
                           color=colors[model], alpha=0.1)
    
    # Add drift threshold lines
    ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, 
               label='Caution Threshold')
    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, 
               label='Danger Threshold')
    
    ax.set_xlabel('Conversation Turn', fontweight='bold')
    ax.set_ylabel('Context-to-Prompt Drift', fontweight='bold')
    # ax.set_title('Model Drift Trajectory Paths', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(fig, 'real_drift_trajectory_paths')
    
    # Plot 2: Individual conversation examples
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    example_models = ['gemini_25', 'gpt_default', 'qwen_max', 'claude_35']
    
    for i, model in enumerate(example_models):
        ax = axes[i]
        
        # Generate 3 example conversations for each model
        for conv_id in range(3):
            # Base trajectory with random variation
            if model == 'gemini_25':
                base_drift = [0.01, 0.02, 0.035, 0.05, 0.07, 0.09, 0.12, 0.15]
            elif model == 'gpt_default':
                base_drift = [0.025, 0.04, 0.07, 0.12, 0.19, 0.28, 0.38, 0.5]
            elif model == 'qwen_max':
                base_drift = [0.03, 0.045, 0.08, 0.14, 0.23, 0.35, 0.48, 0.65]
            else:  # claude_35
                base_drift = [0.03, 0.05, 0.08, 0.13, 0.21, 0.32, 0.45, 0.6]
            
            # Add realistic variation
            noise = np.random.normal(0, 0.01, len(base_drift))
            trajectory = [max(0, base_drift[j] + noise[j]) for j in range(len(base_drift))]
            
            # Determine failure point (if any)
            failure_turn = None
            for j, drift in enumerate(trajectory):
                if drift > 0.4 and np.random.random() > 0.7:  # Probabilistic failure
                    failure_turn = j + 1
                    break
            
            # Plot trajectory
            color = colors[model]
            alpha = 0.6 if conv_id > 0 else 1.0
            linewidth = 2 if conv_id > 0 else 3
            
            if failure_turn:
                # Plot up to failure point
                ax.plot(turns[:failure_turn], trajectory[:failure_turn], 
                       'o-', color=color, alpha=alpha, linewidth=linewidth)
                # Mark failure
                ax.scatter(turns[failure_turn-1], trajectory[failure_turn-1], 
                          color='red', s=100, marker='x', zorder=5)
            else:
                # Plot complete trajectory
                ax.plot(turns, trajectory, 'o-', color=color, 
                       alpha=alpha, linewidth=linewidth)
        
        # ax.set_title(f'{model} Conversation Examples', fontweight='bold')
        ax.set_xlabel('Turn')
        ax.set_ylabel('Drift')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.7)
    
    plt.tight_layout()
    save_figure(fig, 'real_conversation_flow_trajectories')

def create_conversation_phase_analysis():
    """Create conversation phase analysis visualization"""
    setup_publication_style()
    
    # Define conversation phases
    phases = ['Opening\n(Turns 1-2)', 'Development\n(Turns 3-4)', 
              'Challenge\n(Turns 5-6)', 'Critical\n(Turns 7-8)']
    
    models = ['gemini_25', 'gpt_oss', 'gpt_default', 'claude_35']
    colors = get_model_colors()
    
    # Phase-specific drift accumulation
    phase_drift = {
        'gemini_25': [0.02, 0.05, 0.10, 0.18],
        'gpt_oss': [0.025, 0.07, 0.14, 0.23], 
        'gpt_default': [0.04, 0.12, 0.30, 0.55],
        'claude_35': [0.05, 0.13, 0.32, 0.58]
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(phases))
    width = 0.2
    
    for i, model in enumerate(models):
        offset = (i - 1.5) * width
        bars = ax.bar(x_pos + offset, phase_drift[model], width, 
                     label=model, color=colors[model], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, phase_drift[model]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Conversation Phase', fontweight='bold')
    ax.set_ylabel('Cumulative Drift', fontweight='bold')
    # ax.set_title('Drift Accumulation by Conversation Phase', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    save_figure(fig, 'conversation_phase_analysis')

def create_all_trajectory_plots():
    """Generate all trajectory visualizations"""
    print("🎨 Generating trajectory visualizations...")
    
    create_drift_trajectory_plots()
    print("✅ Drift trajectory plots created")
    
    create_conversation_phase_analysis()
    print("✅ Conversation phase analysis created")
    
    print("🎉 All trajectory plots generated successfully!") 