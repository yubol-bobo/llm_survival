#!/usr/bin/env python3
"""
Final Model Comparison Visualization - Separate Plots
====================================================
Creates publication-ready visualizations comparing all 10 models across multiple dimensions.
Each plot is saved as a separate high-quality figure.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def get_model_data():
    """Get the comprehensive model data."""
    model_data = {
        'Model': ['CARG', 'Gemini-2.5', 'Claude-3.5', 'GPT-4', 'Mistral-Large', 
                  'DeepSeek-R1', 'Llama-3.3', 'Llama-4-Scout', 'Llama-4-Mav', 'Qwen-Max'],
        'N_Failures': [68, 78, 134, 252, 284, 329, 359, 431, 496, 523],
        'Max_HR': [1917, 55, 151, 3917163, 98984, 16098, 51442, 42, 32, 1106942750],
        'Max_Protection_Pct': [99.5, 100.0, 99.7, 100.0, 100.0, 100.0, 100.0, 99.9, 100.0, 95.7],
        'Best_Domain': ['Business', 'Legal', 'STEM', 'Humanities', 'Business', 
                       'Business', 'Business', 'Balanced', 'Balanced', 'Medical'],
        'Best_Domain_Score': [4.22, 4.24, 4.25, 4.19, 3.71, 3.98, 4.22, 3.5, 3.5, 4.21],
        'Worst_Domain_Score': [2.53, 3.12, 2.11, 3.38, 2.85, 3.59, 3.67, 3.2, 3.2, 3.66],
        'Risk_Profile': ['Specialized', 'Protective', 'Specialized', 'High-Risk', 'High-Risk',
                        'Binary', 'Defensive', 'Balanced', 'Balanced', 'High-Risk'],
        'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    
    df = pd.DataFrame(model_data)
    df['Specialization_Gap'] = df['Best_Domain_Score'] - df['Worst_Domain_Score']
    df['Log_Max_HR'] = np.log10(df['Max_HR'])
    
    return df

def get_risk_colors():
    """Get the color scheme for risk profiles."""
    return {
        'Specialized': '#FF6B35',    # Orange
        'Protective': '#2ECC71',     # Green  
        'High-Risk': '#E74C3C',      # Red
        'Binary': '#9B59B6',         # Purple
        'Defensive': '#3498DB',      # Blue
        'Balanced': '#F39C12'        # Yellow
    }

def create_performance_overview():
    """Create the main performance overview scatter plot."""
    df = get_model_data()
    risk_colors = get_risk_colors()
    colors = [risk_colors[profile] for profile in df['Risk_Profile']]
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot: N_Failures vs Max_Protection
    scatter = plt.scatter(df['N_Failures'], df['Max_Protection_Pct'], 
                         s=250, c=colors, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add model labels
    for i, (x, y, model) in enumerate(zip(df['N_Failures'], df['Max_Protection_Pct'], df['Model'])):
        plt.annotate(model, (x, y), xytext=(8, 8), textcoords='offset points', 
                    fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.xlabel('Number of Failures', fontsize=16, fontweight='bold')
    plt.ylabel('Maximum Protection (%)', fontsize=16, fontweight='bold')
    plt.title('Model Performance Overview\n(Lower-Left = Better Performance)', fontsize=18, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.ylim(95, 100.5)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=12, label=profile)
                      for profile, color in risk_colors.items()]
    plt.legend(handles=legend_elements, loc='upper right', title='Risk Profiles', 
              title_fontsize=14, fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('generated/figs/1_performance_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Performance overview plot saved!")

def create_vulnerability_spectrum():
    """Create the vulnerability spectrum bar chart."""
    df = get_model_data()
    risk_colors = get_risk_colors()
    colors = [risk_colors[profile] for profile in df['Risk_Profile']]
    
    plt.figure(figsize=(14, 8))
    
    bars = plt.bar(range(len(df)), df['Log_Max_HR'], color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    plt.xlabel('Models', fontsize=16, fontweight='bold')
    plt.ylabel('Maximum Hazard Ratio (log₁₀ scale)', fontsize=16, fontweight='bold')
    plt.title('Vulnerability Spectrum Across All Models', fontsize=18, fontweight='bold', pad=20)
    plt.xticks(range(len(df)), df['Model'], rotation=45, ha='right', fontsize=12)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df['Max_HR'])):
        height = bar.get_height()
        if val >= 1000000:
            label = f'{val/1000000:.1f}M'
        elif val >= 1000:
            label = f'{val/1000:.0f}K'
        else:
            label = f'{val:.0f}'
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                label, ha='center', va='bottom', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('generated/figs/2_vulnerability_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Vulnerability spectrum plot saved!")

def create_specialization_analysis():
    """Create the domain specialization bar chart."""
    df = get_model_data()
    risk_colors = get_risk_colors()
    colors = [risk_colors[profile] for profile in df['Risk_Profile']]
    
    plt.figure(figsize=(14, 8))
    
    bars = plt.bar(range(len(df)), df['Specialization_Gap'], color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    plt.xlabel('Models', fontsize=16, fontweight='bold')
    plt.ylabel('Specialization Gap (turns)', fontsize=16, fontweight='bold')
    plt.title('Domain Specialization Analysis\n(Higher = More Specialized)', fontsize=18, fontweight='bold', pad=20)
    plt.xticks(range(len(df)), df['Model'], rotation=45, ha='right', fontsize=12)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['Specialization_Gap'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Add best/worst domain annotations
    for i, (model, best, worst) in enumerate(zip(df['Model'], df['Best_Domain'], df['Worst_Domain_Score'])):
        best_domain = df.iloc[i]['Best_Domain']
        plt.text(i, -0.2, f'Best: {best_domain}', ha='center', va='top', fontsize=9, 
                rotation=0, style='italic')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('generated/figs/3_specialization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Specialization analysis plot saved!")

def create_risk_profile_distribution():
    """Create the risk profile distribution pie chart."""
    df = get_model_data()
    risk_colors = get_risk_colors()
    
    plt.figure(figsize=(10, 8))
    
    risk_counts = df['Risk_Profile'].value_counts()
    colors_pie = [risk_colors[risk] for risk in risk_counts.index]
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(risk_counts.values, labels=risk_counts.index, 
                                      autopct='%1.0f%%', colors=colors_pie, startangle=90,
                                      textprops={'fontweight': 'bold', 'fontsize': 12},
                                      explode=[0.05] * len(risk_counts))  # Slight separation
    
    plt.title('Strategic Risk Profile Distribution', fontsize=18, fontweight='bold', pad=20)
    
    # Add count labels
    for i, (risk, count) in enumerate(risk_counts.items()):
        plt.text(0, -1.3 + i*0.15, f'{risk}: {count} models', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=risk_colors[risk], alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('generated/figs/4_risk_profile_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Risk profile distribution plot saved!")

def create_performance_rankings():
    """Create the performance rankings summary."""
    plt.figure(figsize=(12, 10))
    
    # Create text-based ranking visualization
    plt.axis('off')
    
    ranking_text = """🏆 COMPREHENSIVE MODEL RANKINGS
    
🥇 1. CARG (68 failures)
   • Specialized Champion Profile
   • 99.5% maximum protection capability
   • Business/STEM domain specialist
   • Predominantly protective approach
   
🥈 2. Gemini-2.5 (78 failures)  
   • Protective Specialist Profile
   • 100% maximum protection capability
   • Legal domain expert specialization
   • Excellent risk management
   
🥉 3. Claude-3.5 (134 failures)
   • STEM Specialist Profile
   • Most controlled risk profile (151x max HR)
   • Highest domain specialization gap
   • Reliable for technical applications
   
4. GPT-4 (252 failures)
   • High-Risk High-Reward Profile
   • 3.9M maximum vulnerability spikes
   • Humanities domain strength
   • Powerful but requires management
   
5. Mistral-Large (284 failures)
   • Balanced Performance Profile
   • Business domain focused
   • Moderate risk characteristics

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 BOTTOM 5 MODELS:
6. DeepSeek-R1 (329 failures) - Binary Performer
7. Llama-3.3 (359 failures) - Defensive Specialist  
8. Llama-4-Scout (431 failures) - Balanced Approach
9. Llama-4-Maverick (496 failures) - Controlled Risk
10. Qwen-Max (523 failures) - Highest Vulnerability

⚠️  VULNERABILITY CHAMPION: Qwen-Max
    1.1 BILLION x maximum failure risk

🛡️  PROTECTION CHAMPION: Llama-3.3
    100% maximum protection capability

🎯 SPECIALIZATION CHAMPION: Claude-3.5
    4.14 turns domain specialization gap"""
    
    plt.text(0.05, 0.95, ranking_text, transform=plt.gca().transAxes, fontsize=13,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", alpha=0.9, edgecolor="black"))
    
    plt.title('Final Model Rankings & Strategic Profiles', fontsize=20, fontweight='bold', 
              pad=30, y=0.98)
    
    plt.tight_layout()
    plt.savefig('generated/figs/5_performance_rankings.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Performance rankings plot saved!")

def create_deployment_heatmap():
    """Create deployment recommendation heatmap."""
    
    # Deployment recommendations matrix
    models = ['CARG', 'Gemini-2.5', 'Claude-3.5', 'GPT-4', 'Mistral-Large', 
              'DeepSeek-R1', 'Llama-3.3', 'Llama-4-Scout', 'Llama-4-Mav', 'Qwen-Max']
    
    domains = ['Business', 'STEM', 'Medical', 'Legal', 'Humanities']
    
    # Scores: 4=Excellent, 3=Strong, 2=Good, 1=Moderate, 0=Avoid
    deployment_matrix = np.array([
        [4, 3, 0, 2, 1],  # CARG
        [3, 0, 2, 4, 2],  # Gemini-2.5
        [1, 4, 2, 0, 1],  # Claude-3.5
        [3, 3, 1, 1, 4],  # GPT-4
        [2, 1, 1, 0, 2],  # Mistral-Large
        [4, 3, 2, 1, 1],  # DeepSeek-R1
        [4, 3, 2, 1, 3],  # Llama-3.3
        [2, 2, 2, 2, 2],  # Llama-4-Scout
        [2, 2, 2, 2, 2],  # Llama-4-Maverick
        [2, 1, 3, 0, 3],  # Qwen-Max
    ])
    
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(deployment_matrix, 
                xticklabels=domains, 
                yticklabels=models,
                annot=True, 
                fmt='d',
                cmap='RdYlGn',
                vmin=0, vmax=4,
                cbar_kws={'label': 'Deployment Suitability Score'},
                annot_kws={'fontweight': 'bold', 'fontsize': 12},
                linewidths=0.5)
    
    plt.title('Evidence-Based Model-Domain Deployment Matrix\n(4=Excellent, 3=Strong, 2=Good, 1=Moderate, 0=Avoid)', 
              fontsize=16, fontweight='bold', pad=25)
    plt.xlabel('Application Domains', fontweight='bold', fontsize=14)
    plt.ylabel('Models (Ranked by Overall Performance)', fontweight='bold', fontsize=14)
    
    # Add ranking numbers
    for i, model in enumerate(models):
        plt.text(-0.7, i + 0.5, f'{i+1}.', ha='center', va='center', 
                fontweight='bold', fontsize=14, color='darkblue')
    
    # Add legend explanation
    plt.figtext(0.02, 0.02, 
                'Based on comprehensive survival analysis of 10 LLMs across multi-turn adversarial interactions', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('generated/figs/6_deployment_recommendations.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Deployment recommendations heatmap saved!")

def main():
    """Main execution - creates all individual plots."""
    print("🎨 Creating individual model comparison visualizations...")
    print("=" * 70)
    
    # Create each plot separately
    print("\n1️⃣ Creating performance overview...")
    create_performance_overview()
    
    print("\n2️⃣ Creating vulnerability spectrum...")
    create_vulnerability_spectrum()
    
    print("\n3️⃣ Creating specialization analysis...")
    create_specialization_analysis()
    
    print("\n4️⃣ Creating risk profile distribution...")
    create_risk_profile_distribution()
    
    print("\n5️⃣ Creating performance rankings...")
    create_performance_rankings()
    
    print("\n6️⃣ Creating deployment recommendations...")
    create_deployment_heatmap()
    
    print("\n🎉 All individual visualizations complete!")
    print("📁 Individual files created:")
    print("  - generated/figs/1_performance_overview.png")
    print("  - generated/figs/2_vulnerability_spectrum.png") 
    print("  - generated/figs/3_specialization_analysis.png")
    print("  - generated/figs/4_risk_profile_distribution.png")
    print("  - generated/figs/5_performance_rankings.png")
    print("  - generated/figs/6_deployment_recommendations.png")
    print("\n🌟 Each plot is now a separate high-quality figure!")

if __name__ == "__main__":
    main() 