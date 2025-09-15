#!/usr/bin/env python3
"""
Quick Visualization Summary for Combined Baseline Results
========================================================

Quick summary script that can be called from the main analysis pipeline
to generate key visualization insights.

Usage:
    from src.visualization.baseline_summary import create_baseline_summary
    create_baseline_summary()
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_baseline_summary(results_dir='results/outputs/baseline', 
                          output_dir='results/figures/baseline'):
    """Create a quick summary visualization of baseline results."""
    
    # Load results
    hazard_ratios = pd.read_csv(f'{results_dir}/hazard_ratios.csv')
    performance = pd.read_csv(f'{results_dir}/model_performance.csv')
    p_values = pd.read_csv(f'{results_dir}/p_values.csv')
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top/Bottom 5 Models by Risk
    model_hrs = []
    hr_cols = [col for col in hazard_ratios.columns 
              if 'model_' in col and '_hr' in col and '_ci' not in col]
    
    for col in hr_cols:
        model_name = col.replace('model_', '').replace('_hr', '')
        hr_val = hazard_ratios[col].iloc[0]
        model_hrs.append((model_name, hr_val))
    
    model_hrs.append(('claude_35', 1.0))  # Reference
    model_hrs.sort(key=lambda x: x[1])
    
    # Plot safest and riskiest
    safest_5 = model_hrs[:5]
    riskiest_3 = model_hrs[-3:]
    
    models_plot = [m[0] for m in safest_5] + [m[0] for m in riskiest_3]
    hrs_plot = [m[1] for m in safest_5] + [m[1] for m in riskiest_3]
    colors = ['green'] * 5 + ['red'] * 3
    
    ax1.barh(range(len(models_plot)), hrs_plot, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(models_plot)))
    ax1.set_yticklabels(models_plot)
    ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Hazard Ratio')
    ax1.set_title('Safest (Green) vs Riskiest (Red) LLMs')
    
    # 2. Model Performance Metrics
    c_index = performance['c_index'].iloc[0]
    n_events = performance['n_events'].iloc[0]
    n_obs = performance['n_observations'].iloc[0]
    event_rate = (n_events / n_obs) * 100
    
    metrics = ['C-index', 'Event Rate (%)', 'Total Events', 'Observations']
    values = [c_index, event_rate, n_events/1000, n_obs/1000]  # Scale for visualization
    
    ax2.bar(metrics, values, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    ax2.set_title('Model Performance Overview')
    ax2.set_ylabel('Value (scaled)')
    
    # Add actual values as labels
    actual_vals = [f'{c_index:.3f}', f'{event_rate:.1f}%', f'{n_events}', f'{n_obs}']
    for i, (bar, val) in enumerate(zip(ax2.patches, actual_vals)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                val, ha='center', va='bottom', fontweight='bold')
    
    # 3. Risk Distribution
    all_hrs = [hr for _, hr in model_hrs]
    ax3.hist(all_hrs, bins=6, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Reference')
    ax3.set_xlabel('Hazard Ratio')
    ax3.set_ylabel('Number of Models')
    ax3.set_title('Risk Distribution Across LLMs')
    ax3.legend()
    
    # 4. Statistical Significance Count
    sig_counts = {'Highly Significant': 0, 'Significant': 0, 'Not Significant': 0}
    
    # Count significant model effects
    model_pval_cols = [col for col in p_values.columns 
                      if 'model_' in col and '_pval' in col]
    
    for col in model_pval_cols:
        pval = p_values[col].iloc[0]
        if pval < 0.001:
            sig_counts['Highly Significant'] += 1
        elif pval < 0.05:
            sig_counts['Significant'] += 1
        else:
            sig_counts['Not Significant'] += 1
    
    colors_sig = ['darkgreen', 'orange', 'red']
    ax4.pie(sig_counts.values(), labels=sig_counts.keys(), colors=colors_sig, 
           autopct='%1.1f%%', startangle=90)
    ax4.set_title('Statistical Significance\nof Model Effects')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/baseline_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/baseline_summary.pdf', bbox_inches='tight')
    plt.close()
    
    # Print key insights
    print("\n📊 BASELINE MODELING SUMMARY")
    print("=" * 40)
    print(f"🎯 Model Performance: C-index = {c_index:.4f}")
    print(f"📊 Dataset: {n_obs:,} observations, {n_events:,} events ({event_rate:.1f}% event rate)")
    print(f"🔧 Frailty: {performance['frailty_type'].iloc[0]}")
    
    print(f"\n🏆 TOP 3 SAFEST MODELS:")
    for i, (model, hr) in enumerate(safest_5[:3], 1):
        risk_reduction = (1 - hr) * 100 if hr < 1 else 0
        print(f"   {i}. {model}: HR = {hr:.3f} ({risk_reduction:.1f}% less risk)")
    
    print(f"\n⚠️  TOP 3 RISKIEST MODELS:")
    for i, (model, hr) in enumerate(riskiest_3, 1):
        risk_increase = (hr - 1) * 100 if hr > 1 else 0
        print(f"   {i}. {model}: HR = {hr:.3f} ({risk_increase:.1f}% more risk)")
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE:")
    for level, count in sig_counts.items():
        print(f"   {level}: {count} models")
    
    print(f"\n✅ Summary plot saved to: {output_dir}/baseline_summary.png")


if __name__ == "__main__":
    os.makedirs('results/figures/baseline', exist_ok=True)
    create_baseline_summary()