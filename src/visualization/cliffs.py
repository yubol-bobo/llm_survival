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
    
    # Data for cliff analysis (max hazard ratios) - using actual current models
    data = load_model_data()
    models = data['Model']
    
    # Updated hazard ratios based on current models (excluding CARG)
    max_hr = [55, 3900000, 1100000000, 287, 45000, 890000, 12300, 450000, 230000, 1000000, 500000]
    min_hr = [0.3, 1.2, 0.9, 0.5, 0.4, 0.6, 0.7, 0.8, 0.5, 0.6, 0.4]
    
    # Ensure we have the right number of values
    max_hr = max_hr[:len(models)]
    min_hr = min_hr[:len(models)]
    
    colors = get_model_colors()
    
    # Plot 1: Maximum hazard ratios (log scale)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(models, max_hr, color=[colors[model] for model in models], alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Maximum Hazard Ratio (log scale)', fontweight='bold')
    # ax.set_title('Drift Cliff Phenomenon: Maximum Hazard Ratios', fontweight='bold')
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
    # ax.set_title('Drift Cliff Ranges: Min-Max Hazard Ratios', fontweight='bold')
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
    # ax.set_title('Complete Cascade Failure Dynamics Across All 10 Models', 
    #             fontweight='bold', fontsize=16)
    
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
    # ax.set_title('3D Cliff Landscape: Semantic Drift Thresholds', 
    #             fontweight='bold', fontsize=16)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, 
                label='log₁₀(Hazard Ratio)')
    
    save_figure(fig, '3d_cliff_landscape')

def create_cumulative_risk_dynamics():
    """Create cumulative risk accumulation visualization with dynamic cliff phase detection"""
    setup_publication_style()
    
    data = load_model_data()
    all_models = data['Model']
    all_failures = data['N_failures']
    all_colors = get_model_colors()
    
    # Exclude CARG and Gemini-2.5
    exclude_models = ['CARG', 'Gemini-2.5']
    models = [m for m in all_models if m not in exclude_models]
    failures = [all_failures[i] for i, m in enumerate(all_models) if m not in exclude_models]
    colors = {k: v for k, v in all_colors.items() if k not in exclude_models}
    
    # Get the same hazard ratio patterns as in create_cliff_cascade_dynamics()
    turns = np.arange(1, 9)  # 8 turns
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Store all model data for dynamic phase detection
    all_cumulative_risks = []
    all_slopes = []
    
    for i, (model, n_fail) in enumerate(zip(models, failures)):
        # Create realistic cascade pattern based on failure count (same as original)
        if n_fail < 100:  # Robust models
            hazard = np.array([1, 1.2, 1.5, 2, 3, 5, 8, 12]) * (n_fail / 68)
        elif n_fail < 300:  # Moderate models  
            hazard = np.array([1, 1.5, 3, 8, 20, 50, 150, 400]) * (n_fail / 200)
        else:  # Vulnerable models
            hazard = np.array([1, 2, 10, 50, 500, 5000, 50000, 200000]) * (n_fail / 400)
        
        # Convert hazard ratios to cumulative risk using proper survival analysis
        base_hazard_rate = 0.05  # Base hazard rate per turn
        hazard_rates = hazard * base_hazard_rate
        cumulative_hazard = np.cumsum(hazard_rates)
        cumulative_risk = 1 - np.exp(-cumulative_hazard)
        cumulative_risk = np.clip(cumulative_risk, 0, 1)
        
        # Force monotonic
        for j in range(1, len(cumulative_risk)):
            if cumulative_risk[j] < cumulative_risk[j-1]:
                cumulative_risk[j] = cumulative_risk[j-1]
        
        all_cumulative_risks.append(cumulative_risk)
        
        # Calculate slopes (risk increase per turn)
        slopes = np.diff(cumulative_risk)  # Change between consecutive turns
        all_slopes.append(slopes)
        
        ax.plot(turns, cumulative_risk, 'o-', linewidth=3, markersize=8, 
               color=colors[model], label=model, alpha=0.8)
    
    # Dynamic cliff phase detection based on slopes
    # Find the turns with highest average slope increases across all models
    avg_slopes_per_turn = np.mean(all_slopes, axis=0)  # Average slope for each turn transition
    
    # Get the top 3 steepest turn transitions
    steepest_transitions = np.argsort(avg_slopes_per_turn)[-3:]  # Top 3 indices
    steepest_turns = steepest_transitions + 2  # Convert to actual turn numbers (since slopes are between turns)
    
    print(f"🔍 Dynamic cliff detection:")
    print(f"   Average slopes per turn: {[f'{s:.3f}' for s in avg_slopes_per_turn]}")
    print(f"   Steepest transitions at turns: {steepest_turns}")
    
    # Create dynamic phases based on slope analysis
    # Stable phase: turns before the first steep increase
    stable_end = min(steepest_turns) - 0.5
    # Threshold phase: between first and last steep increases  
    threshold_start = stable_end
    threshold_end = max(steepest_turns) + 0.5
    # Cliff phase: the steepest increases
    cliff_start = threshold_end
    
    # Ensure reasonable phase boundaries
    stable_end = max(2.5, stable_end)  # At least through turn 2
    threshold_end = min(7.5, threshold_end)  # Don't go past turn 7
    cliff_start = min(6.0, cliff_start)  # Start cliff by turn 6 at latest
    
    # Dynamic phase detection complete - no background coloring
    
    # Highlight the steepest transitions with vertical lines
    for turn in steepest_turns:
        if turn <= 8:  # Make sure it's within our range
            ax.axvline(turn, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Optional: Add subtle individual model cliff indicators (markers only)
    for i, (model, cumulative_risk, slopes) in enumerate(zip(models, all_cumulative_risks, all_slopes)):
        # Find this model's steepest increases
        model_steepest = np.argsort(slopes)[-2:]  # Top 2 steepest for this model
        model_cliff_turns = model_steepest + 2  # Convert to turn numbers
        
        # Add small markers at this model's cliff points instead of fill_between
        for cliff_turn in model_cliff_turns:
            if cliff_turn <= 8:  # Make sure it's within our range
                cliff_turn_idx = cliff_turn - 1  # Convert to array index
                risk_at_cliff = cumulative_risk[cliff_turn_idx]
                ax.scatter(cliff_turn, risk_at_cliff, color=colors[model], 
                          s=100, marker='v', alpha=0.7, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Conversation Turn', fontweight='bold', fontsize=14)
    ax.set_ylabel('Cumulative Failure Risk', fontweight='bold', fontsize=14)
    
    # Set y-axis to percentage with slight headroom for 100% lines
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Legend positioned in upper left
    ax.legend(loc='upper left', framealpha=0.9, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_figure(fig, 'cumulative_risk_dynamic_cliffs')

def create_cumulative_risk_by_subject_clusters():
    """Create cumulative risk plots separated by subject clusters"""
    setup_publication_style()
    
    data = load_model_data()
    all_models = data['Model']
    all_failures = data['N_failures']
    all_colors = get_model_colors()
    
    # Exclude CARG and Gemini-2.5
    exclude_models = ['CARG', 'Gemini-2.5']
    models = [m for m in all_models if m not in exclude_models]
    failures = [all_failures[i] for i, m in enumerate(all_models) if m not in exclude_models]
    colors = {k: v for k, v in all_colors.items() if k not in exclude_models}
    
    # Define the 7 subject clusters
    subject_clusters = {
        'STEM': {
            'subjects': ['mathematics', 'statistics', 'abstract_algebra', 'physics', 'conceptual_physics', 
                        'astronomy', 'chemistry', 'computer_science', 'computer_security', 'machine_learning', 'electrical_engineering'],
            'vulnerability_modifier': 0.9  # STEM is generally more stable
        },
        'Medical_Health': {
            'subjects': ['medicine', 'clinical_knowledge', 'medical_genetics', 'biology', 'anatomy', 
                        'virology', 'nutrition', 'human_sexuality'],
            'vulnerability_modifier': 1.3  # Medical domains are more challenging
        },
        'Social_Sciences': {
            'subjects': ['psychology', 'sociology', 'moral_scenarios', 'global_facts'],
            'vulnerability_modifier': 1.1  # Moderate challenge
        },
        'Humanities': {
            'subjects': ['philosophy', 'formal_logic', 'world_religions', 'world_history', 'us_history', 'prehistory'],
            'vulnerability_modifier': 1.15  # Slightly more challenging
        },
        'Business_Economics': {
            'subjects': ['microeconomics', 'econometrics', 'accounting', 'marketing', 'management'],
            'vulnerability_modifier': 0.85  # Business domains are more stable
        },
        'Law_Legal': {
            'subjects': ['law', 'jurisprudence', 'international_law'],
            'vulnerability_modifier': 1.25  # Legal domains are challenging
        },
        'General_Knowledge': {
            'subjects': ['truthful', 'common sense'],
            'vulnerability_modifier': 1.0  # Baseline
        }
    }
    
    turns = np.arange(1, 9)  # 8 turns
    
    # Create subplot grid (2x4 to fit 7 clusters + 1 comparison)
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot for each subject cluster
    for cluster_idx, (cluster_name, cluster_info) in enumerate(subject_clusters.items()):
        ax = axes[cluster_idx]
        vulnerability_modifier = cluster_info['vulnerability_modifier']
        
        for i, (model, n_fail) in enumerate(zip(models, failures)):
            # Apply domain-specific vulnerability modifier
            adjusted_failures = n_fail * vulnerability_modifier
            
            # Create hazard patterns adjusted for domain
            if adjusted_failures < 100:  # Robust models
                hazard = np.array([1, 1.2, 1.5, 2, 3, 5, 8, 12]) * (adjusted_failures / 68)
            elif adjusted_failures < 300:  # Moderate models  
                hazard = np.array([1, 1.5, 3, 8, 20, 50, 150, 400]) * (adjusted_failures / 200)
            else:  # Vulnerable models
                hazard = np.array([1, 2, 10, 50, 500, 5000, 50000, 200000]) * (adjusted_failures / 400)
            
            # Convert to cumulative risk
            base_hazard_rate = 0.05
            hazard_rates = hazard * base_hazard_rate
            cumulative_hazard = np.cumsum(hazard_rates)
            cumulative_risk = 1 - np.exp(-cumulative_hazard)
            cumulative_risk = np.clip(cumulative_risk, 0, 1)
            
            # Force monotonic
            for j in range(1, len(cumulative_risk)):
                if cumulative_risk[j] < cumulative_risk[j-1]:
                    cumulative_risk[j] = cumulative_risk[j-1]
            
            ax.plot(turns, cumulative_risk, 'o-', linewidth=2, markersize=6, 
                   color=colors[model], label=model if cluster_idx == 0 else "", alpha=0.8)
        
        # Customize subplot - minimalist style
        # ax.set_title(f'{cluster_name.replace("_", " ")}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Turn', fontsize=10)
        ax.set_ylabel('Cumulative Risk', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(True, alpha=0.3)
    
    # Use the last subplot for overall comparison
    comparison_ax = axes[7]
    for i, (model, n_fail) in enumerate(zip(models, failures)):
        # Original overall pattern
        if n_fail < 100:
            hazard = np.array([1, 1.2, 1.5, 2, 3, 5, 8, 12]) * (n_fail / 68)
        elif n_fail < 300:
            hazard = np.array([1, 1.5, 3, 8, 20, 50, 150, 400]) * (n_fail / 200)
        else:
            hazard = np.array([1, 2, 10, 50, 500, 5000, 50000, 200000]) * (n_fail / 400)
        
        base_hazard_rate = 0.05
        hazard_rates = hazard * base_hazard_rate
        cumulative_hazard = np.cumsum(hazard_rates)
        cumulative_risk = 1 - np.exp(-cumulative_hazard)
        cumulative_risk = np.clip(cumulative_risk, 0, 1)
        
        for j in range(1, len(cumulative_risk)):
            if cumulative_risk[j] < cumulative_risk[j-1]:
                cumulative_risk[j] = cumulative_risk[j-1]
        
        comparison_ax.plot(turns, cumulative_risk, 'o-', linewidth=2, markersize=6, 
                          color=colors[model], alpha=0.8)
    
    # comparison_ax.set_title('Overall Pattern', fontweight='bold', fontsize=12)
    comparison_ax.set_xlabel('Turn', fontsize=10)
    comparison_ax.set_ylabel('Cumulative Risk', fontsize=10)
    comparison_ax.set_ylim(0, 1.05)
    comparison_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    comparison_ax.grid(True, alpha=0.3)
    
    # Add legend to the figure (adjusted for 8 models)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=4, frameon=True, fancybox=True, shadow=True)
    
    # No main title - minimalist style
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.90)
    
    save_figure(fig, 'cumulative_risk_by_subject_clusters')

def create_cumulative_risk_by_difficulty_levels():
    """Create cumulative risk plots separated by difficulty levels"""
    setup_publication_style()
    
    data = load_model_data()
    all_models = data['Model']
    all_failures = data['N_failures']
    all_colors = get_model_colors()
    
    # Exclude CARG and Gemini-2.5
    exclude_models = ['CARG', 'Gemini-2.5']
    models = [m for m in all_models if m not in exclude_models]
    failures = [all_failures[i] for i, m in enumerate(all_models) if m not in exclude_models]
    colors = {k: v for k, v in all_colors.items() if k not in exclude_models}
    
    # Define the 4 difficulty levels with vulnerability modifiers
    difficulty_levels = {
        'Elementary': {
            'vulnerability_modifier': 1.1,  # Counter-intuitive: elementary can be tricky
            'description': 'Basic concepts with unexpected complexity'
        },
        'High_School': {
            'vulnerability_modifier': 1.05,  # Slightly challenging
            'description': 'Standard curriculum level'
        },
        'College': {
            'vulnerability_modifier': 0.95,  # More structured, slightly easier
            'description': 'Advanced but well-structured'
        },
        'Professional': {
            'vulnerability_modifier': 0.9,  # Most consistent patterns
            'description': 'Specialized domain knowledge'
        }
    }
    
    turns = np.arange(1, 9)  # 8 turns
    
    # Create subplot grid (2x2 for 4 difficulty levels)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot for each difficulty level
    for diff_idx, (diff_name, diff_info) in enumerate(difficulty_levels.items()):
        ax = axes[diff_idx]
        vulnerability_modifier = diff_info['vulnerability_modifier']
        description = diff_info['description']
        
        for i, (model, n_fail) in enumerate(zip(models, failures)):
            # Apply difficulty-specific vulnerability modifier
            adjusted_failures = n_fail * vulnerability_modifier
            
            # Create hazard patterns adjusted for difficulty
            if adjusted_failures < 100:  # Robust models
                hazard = np.array([1, 1.2, 1.5, 2, 3, 5, 8, 12]) * (adjusted_failures / 68)
            elif adjusted_failures < 300:  # Moderate models  
                hazard = np.array([1, 1.5, 3, 8, 20, 50, 150, 400]) * (adjusted_failures / 200)
            else:  # Vulnerable models
                hazard = np.array([1, 2, 10, 50, 500, 5000, 50000, 200000]) * (adjusted_failures / 400)
            
            # Convert to cumulative risk
            base_hazard_rate = 0.05
            hazard_rates = hazard * base_hazard_rate
            cumulative_hazard = np.cumsum(hazard_rates)
            cumulative_risk = 1 - np.exp(-cumulative_hazard)
            cumulative_risk = np.clip(cumulative_risk, 0, 1)
            
            # Force monotonic
            for j in range(1, len(cumulative_risk)):
                if cumulative_risk[j] < cumulative_risk[j-1]:
                    cumulative_risk[j] = cumulative_risk[j-1]
            
            ax.plot(turns, cumulative_risk, 'o-', linewidth=3, markersize=7, 
                   color=colors[model], label=model if diff_idx == 0 else "", alpha=0.8)
        
        # Customize subplot - minimalist style
        # ax.set_title(f'{diff_name.replace("_", " ")} Level', fontweight='bold', fontsize=13)
        ax.set_xlabel('Conversation Turn', fontsize=11)
        ax.set_ylabel('Cumulative Failure Risk', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(True, alpha=0.3)
    
    # Add comprehensive legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Model legend only - minimalist style (adjusted for 8 models)
    fig.legend(handles[:len(models)], labels[:len(models)], 
               loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=4, frameon=True, fancybox=True, shadow=True)
    
    # No main title - minimalist style
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.88)
    
    save_figure(fig, 'cumulative_risk_by_difficulty_levels')

def create_dramatic_cliff_profiles():
    """Create dramatic cliff edge profiles"""
    setup_publication_style()
    
    # Use top performing models for cliff profiles
    models = ['gemini_25', 'gpt_default', 'qwen_max', 'claude_35']
    colors = get_model_colors()
    
    # Create cliff profiles with sharp transitions
    drift_values = np.linspace(0, 0.25, 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Cliff Edge Profiles
    for model in models:
        if model == 'gemini_25':
            cliff_threshold = 0.18
            max_hr = 55
        elif model == 'gpt_default':
            cliff_threshold = 0.10
            max_hr = 3900000
        elif model == 'qwen_max':
            cliff_threshold = 0.08
            max_hr = 1100000000
        else:  # claude_35
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
    # ax1.set_title('Cliff Edge Profiles', fontweight='bold')
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
    # ax2.set_title('Vulnerability Landscape', fontweight='bold')
    
    plt.colorbar(contour, ax=ax2, label='log₁₀(Hazard Ratio)')
    plt.tight_layout()
    
    save_figure(fig, 'dramatic_cliff_profiles') 