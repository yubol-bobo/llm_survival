#!/usr/bin/env python3
"""
CARG Vulnerability Analysis Visualization
========================================
Creates comprehensive visualizations of CARG's hazard patterns from time-varying advanced models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'

def create_carg_vulnerability_analysis():
    """Create comprehensive CARG vulnerability visualizations."""
    
    # Data from time-varying advanced model analysis
    prompt_drift_data = {
        'Prompt_Type': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'],
        'Cumulative_Drift_HR': [0.324, 1.432, 0.218, 0.106, 19.79, 0.547, 2.293, 2.975],
        'Prompt_to_Prompt_HR': [0.074, 40.69, 0.013, 0.0045, 10.00, 0.230, 0.032, 1917.02],
        'Risk_Level': ['LOW', 'HIGH', 'LOW', 'LOW', 'HIGH', 'LOW', 'MODERATE', 'EXTREME']
    }
    
    subject_data = {
        'Subject': ['STEM', 'Humanities', 'Legal', 'Medical', 'Business'],
        'Context_Drift_HR': [1.19, 1.52, 3.89, 38.15, np.nan],
        'Mean_Survival': [3.94, 3.11, 3.88, 2.53, 4.22],
        'N_Conversations': [16, 9, 17, 17, 9]
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Prompt Type Hazard Ratio Heatmap
    ax1 = plt.subplot(2, 3, 1)
    df_prompt = pd.DataFrame(prompt_drift_data)
    
    # Create heatmap data
    heatmap_data = df_prompt[['Cumulative_Drift_HR', 'Prompt_to_Prompt_HR']].T
    heatmap_data.columns = df_prompt['Prompt_Type']
    
    # Use log scale for better visualization
    heatmap_data_log = np.log10(heatmap_data + 0.001)  # Add small constant to avoid log(0)
    
    sns.heatmap(heatmap_data_log, annot=heatmap_data.values, fmt='.3f', 
                cmap='RdYlGn_r', center=0, ax=ax1,
                cbar_kws={'label': 'log10(Hazard Ratio)'})
    ax1.set_title('CARG: Prompt Type Vulnerability Matrix\n(Log Scale)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Prompt Type', fontweight='bold')
    ax1.set_ylabel('Drift Type', fontweight='bold')
    
    # 2. Subject Domain Risk Profile
    ax2 = plt.subplot(2, 3, 2)
    df_subject = pd.DataFrame(subject_data)
    df_subject_clean = df_subject.dropna(subset=['Context_Drift_HR'])
    
    bars = ax2.barh(df_subject_clean['Subject'], df_subject_clean['Context_Drift_HR'], 
                    color=['green', 'lightgreen', 'orange', 'red'])
    ax2.set_xlabel('Context-to-Prompt Drift Hazard Ratio', fontweight='bold')
    ax2.set_title('CARG: Subject Domain Risk Profile', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_subject_clean['Context_Drift_HR'])):
        ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontweight='bold')
    
    # 3. Subject Mean Survival Times
    ax3 = plt.subplot(2, 3, 3)
    colors = ['lightblue', 'lightcyan', 'lightyellow', 'lightcoral', 'lightgreen']
    bars = ax3.bar(df_subject['Subject'], df_subject['Mean_Survival'], color=colors)
    ax3.set_ylabel('Mean Survival (turns)', fontweight='bold')
    ax3.set_title('CARG: Mean Survival by Subject', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, df_subject['Mean_Survival']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', fontweight='bold')
    
    # 4. Risk Classification Pie Chart
    ax4 = plt.subplot(2, 3, 4)
    risk_counts = df_prompt['Risk_Level'].value_counts()
    colors_pie = {'LOW': 'green', 'MODERATE': 'orange', 'HIGH': 'red', 'EXTREME': 'darkred'}
    pie_colors = [colors_pie[risk] for risk in risk_counts.index]
    
    wedges, texts, autotexts = ax4.pie(risk_counts.values, labels=risk_counts.index, 
                                      autopct='%1.0f%%', colors=pie_colors, startangle=90)
    ax4.set_title('CARG: Risk Level Distribution\n(Prompt Types)', fontsize=14, fontweight='bold')
    
    # 5. Extreme Values Highlight
    ax5 = plt.subplot(2, 3, 5)
    extreme_data = df_prompt[df_prompt['Prompt_to_Prompt_HR'] > 1][['Prompt_Type', 'Prompt_to_Prompt_HR']]
    
    bars = ax5.bar(extreme_data['Prompt_Type'], extreme_data['Prompt_to_Prompt_HR'], 
                   color=['orange', 'red', 'red', 'darkred'])
    ax5.set_ylabel('Hazard Ratio (Log Scale)', fontweight='bold')
    ax5.set_title('CARG: High-Risk Prompt Types\n(Prompt-to-Prompt Drift)', fontsize=14, fontweight='bold')
    ax5.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, extreme_data['Prompt_to_Prompt_HR']):
        ax5.text(bar.get_x() + bar.get_width()/2, val * 1.2, 
                f'{val:.1f}', ha='center', fontweight='bold', color='darkred')
    
    # 6. Strategic Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    summary_text = """
    🏆 CARG STRATEGIC PROFILE
    
    ✅ SAFE ZONES (HR < 1.0):
    • p1, p3, p4, p6, p7 + Prompt-to-Prompt
    • STEM & Humanities domains
    • Business domain (best survival: 4.22)
    
    ⚠️ DANGER ZONES (HR > 10):
    • p8 + Prompt-to-Prompt (HR: 1917!)
    • p2 + Prompt-to-Prompt (HR: 40.7)
    • Medical + Context drift (HR: 38.2)
    • p5 + Cumulative drift (HR: 19.8)
    
    📊 KEY INSIGHTS:
    • Binary risk profile: extremely safe OR vulnerable
    • Medical domain = major weakness
    • 5/8 prompt types are highly protective
    • Ideal for controlled deployment
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('generated/figs/CARG_vulnerability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ CARG vulnerability analysis visualization saved!")
    
def create_comparison_heatmap():
    """Create a comparison heatmap showing CARG vs other top models."""
    
    # Sample data for top 5 models (based on N_failures)
    models_data = {
        'Model': ['CARG', 'Gemini-2.5', 'GPT-4', 'Qwen-Max', 'Llama-4-Mav'],
        'N_failures': [68, 78, 134, 252, 431],
        'Business_Survival': [4.22, 3.58, 3.65, 3.89, 3.67],
        'STEM_Survival': [3.94, 3.86, 3.75, 3.45, 3.89],
        'Medical_Survival': [2.53, 3.58, 3.42, 3.78, 3.44],
        'Legal_Survival': [3.88, 3.47, 3.55, 3.12, 3.78]
    }
    
    df_models = pd.DataFrame(models_data)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    survival_cols = ['Business_Survival', 'STEM_Survival', 'Medical_Survival', 'Legal_Survival']
    heatmap_data = df_models[survival_cols].T
    heatmap_data.columns = df_models['Model']
    heatmap_data.index = ['Business', 'STEM', 'Medical', 'Legal']
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=3.5,
                cbar_kws={'label': 'Mean Survival Time (turns)'})
    
    plt.title('Model Comparison: Subject Domain Performance\n(Mean Survival Times)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Models (Ranked by N_failures)', fontweight='bold')
    plt.ylabel('Subject Domains', fontweight='bold')
    
    # Add ranking information
    plt.figtext(0.02, 0.02, 'Models ranked by total failures (ascending): CARG=68, Gemini-2.5=78, GPT-4=134, Qwen-Max=252, Llama-4-Mav=431', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('generated/figs/model_comparison_domains.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Model comparison heatmap saved!")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs('generated/figs', exist_ok=True)
    
    print("🔬 Creating CARG vulnerability analysis...")
    create_carg_vulnerability_analysis()
    
    print("\n📊 Creating model comparison heatmap...")
    create_comparison_heatmap()
    
    print("\n🎉 All visualizations complete!") 