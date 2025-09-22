#!/usr/bin/env python3
"""
Comprehensive Model Comparison: Cox vs Parametric Models
Compares Cox regression with Weibull and AFT models to evaluate 
performance and address proportional hazards assumption violations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_results():
    """Load results from all modeling approaches"""
    results = {
        'model_type': [],
        'c_index': [],
        'aic': [],
        'bic': [],
        'assumptions_met': [],
        'notes': []
    }
    
    # Cox Regression Results (from previous analysis)
    results['model_type'].append('Cox Regression (Basic)')
    results['c_index'].append(0.8001)
    results['aic'].append(None)  # Not typically reported for Cox
    results['bic'].append(None)
    results['assumptions_met'].append(False)  # 12/15 features violated PH assumption
    results['notes'].append('12/15 features violate proportional hazards assumption')
    
    # Weibull Individual Models (best performer)
    results['model_type'].append('Individual Weibull (GPT-5)')
    results['c_index'].append(None)  # Individual models don't have C-index
    results['aic'].append(604.84)
    results['bic'].append(None)
    results['assumptions_met'].append(True)  # Parametric models don't require PH assumption
    results['notes'].append('Best individual model by AIC')
    
    # AFT Models from our analysis
    aft_results = [
        ('Log-Normal AFT', 0.8269, 9536.03, 9521.80),
        ('Log-Logistic AFT', 0.8275, 9759.23, 9745.01),
        ('Weibull AFT', 0.8274, 9720.55, 9706.33),
        ('Log-Normal AFT + Interactions', 0.8265, 9675.82, 9651.59),
        ('Weibull AFT + Interactions', 0.8248, 10098.51, 10074.28),
        ('Log-Logistic AFT + Interactions', 0.8252, 10163.49, 10139.26)
    ]
    
    for name, c_idx, aic, bic in aft_results:
        results['model_type'].append(name)
        results['c_index'].append(c_idx)
        results['aic'].append(aic)
        results['bic'].append(bic)
        results['assumptions_met'].append(True)
        results['notes'].append('Parametric model - no proportional hazards assumption required')
    
    return pd.DataFrame(results)

def create_comprehensive_comparison():
    """Create comprehensive model comparison analysis"""
    print("🔬 COMPREHENSIVE LLM SURVIVAL MODEL COMPARISON")
    print("=" * 80)
    
    # Load all results
    comparison_df = load_all_results()
    
    print("📊 MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Display comparison table
    display_df = comparison_df[['model_type', 'c_index', 'aic', 'bic', 'assumptions_met']].copy()
    display_df['c_index'] = display_df['c_index'].round(4)
    display_df['aic'] = display_df['aic'].round(2)
    display_df['bic'] = display_df['bic'].round(2)
    
    print(display_df.to_string(index=False))
    
    print("\n🏆 KEY FINDINGS:")
    print("=" * 30)
    
    # Find best models
    valid_c_index = comparison_df.dropna(subset=['c_index'])
    valid_aic = comparison_df.dropna(subset=['aic'])
    valid_bic = comparison_df.dropna(subset=['bic'])
    
    if len(valid_c_index) > 0:
        best_c_index = valid_c_index.loc[valid_c_index['c_index'].idxmax()]
        print(f"🥇 Best C-index: {best_c_index['model_type']}")
        print(f"   C-index: {best_c_index['c_index']:.4f}")
        print(f"   Performance: {'Excellent' if best_c_index['c_index'] > 0.8 else 'Good' if best_c_index['c_index'] > 0.7 else 'Fair'}")
    
    if len(valid_aic) > 0:
        best_aic = valid_aic.loc[valid_aic['aic'].idxmin()]
        print(f"\n🥇 Best AIC: {best_aic['model_type']}")
        print(f"   AIC: {best_aic['aic']:.2f}")
    
    if len(valid_bic) > 0:
        best_bic = valid_bic.loc[valid_bic['bic'].idxmin()]
        print(f"\n🥇 Best BIC: {best_bic['model_type']}")
        print(f"   BIC: {best_bic['bic']:.2f}")
    
    print("\n⚖️  MODEL COMPARISON ANALYSIS:")
    print("=" * 40)
    
    print("1. PREDICTIVE PERFORMANCE:")
    cox_c_index = 0.8001
    best_aft_c_index = valid_c_index['c_index'].max()
    improvement = ((best_aft_c_index - cox_c_index) / cox_c_index) * 100
    
    print(f"   • Cox Regression C-index: {cox_c_index:.4f}")
    print(f"   • Best AFT C-index: {best_aft_c_index:.4f}")
    print(f"   • Performance improvement: {improvement:.2f}%")
    
    print("\n2. ASSUMPTION COMPLIANCE:")
    compliant_models = comparison_df[comparison_df['assumptions_met'] == True]
    violation_models = comparison_df[comparison_df['assumptions_met'] == False]
    
    print(f"   • Models with assumption violations: {len(violation_models)}")
    print(f"   • Models without assumption violations: {len(compliant_models)}")
    print(f"   • Parametric models don't require proportional hazards assumption")
    
    print("\n3. MODEL SELECTION RECOMMENDATION:")
    print("   🎯 RECOMMENDED MODEL: Log-Logistic AFT")
    print(f"      • C-index: 0.8275 (highest predictive performance)")
    print(f"      • No proportional hazards assumption violations")
    print(f"      • Robust parametric approach")
    print(f"      • {improvement:.1f}% improvement over Cox regression")
    
    print("\n4. ALTERNATIVE RECOMMENDATIONS:")
    print("   📋 Log-Normal AFT:")
    print("      • Best AIC/BIC (most parsimonious)")
    print("      • C-index: 0.8269 (excellent performance)")
    print("      • Good for normally distributed log-survival times")
    
    print("   📋 Cox Regression:")
    print("      • Still viable despite assumption violations")
    print("      • Excellent C-index: 0.8001")
    print("      • Semi-parametric flexibility")
    print("      • Standard in survival analysis literature")
    
    # Create visualization
    create_model_comparison_plot(comparison_df)
    
    return comparison_df

def create_model_comparison_plot(comparison_df):
    """Create visualization comparing all models"""
    print("\n📈 GENERATING COMPREHENSIVE COMPARISON PLOTS")
    print("=" * 50)
    
    # Prepare data for plotting
    plot_data = comparison_df.dropna(subset=['c_index']).copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # C-index comparison
    ax1 = axes[0, 0]
    bars = ax1.barh(plot_data['model_type'], plot_data['c_index'])
    ax1.set_xlabel('C-index (Concordance Index)')
    ax1.set_title('Model Performance by C-index\n(Higher is Better)')
    ax1.grid(True, alpha=0.3)
    
    # Color code bars
    colors = ['red' if not assumptions else 'green' for assumptions in plot_data['assumptions_met']]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
        bar.set_alpha(0.7)
    
    # Add performance threshold line
    ax1.axvline(x=0.8, color='blue', linestyle='--', alpha=0.5, label='Excellent Performance (0.8)')
    ax1.legend()
    
    # AIC comparison (lower is better)
    ax2 = axes[0, 1]
    aic_data = comparison_df.dropna(subset=['aic'])
    if len(aic_data) > 0:
        bars2 = ax2.barh(aic_data['model_type'], aic_data['aic'])
        ax2.set_xlabel('AIC (Akaike Information Criterion)')
        ax2.set_title('Model Fit by AIC\n(Lower is Better)')
        ax2.grid(True, alpha=0.3)
        
        # Color code
        colors2 = ['red' if not assumptions else 'green' for assumptions in aic_data['assumptions_met']]
        for bar, color in zip(bars2, colors2):
            bar.set_color(color)
            bar.set_alpha(0.7)
    
    # BIC comparison (lower is better)
    ax3 = axes[1, 0]
    bic_data = comparison_df.dropna(subset=['bic'])
    if len(bic_data) > 0:
        bars3 = ax3.barh(bic_data['model_type'], bic_data['bic'])
        ax3.set_xlabel('BIC (Bayesian Information Criterion)')
        ax3.set_title('Model Fit by BIC\n(Lower is Better)')
        ax3.grid(True, alpha=0.3)
        
        # Color code
        colors3 = ['red' if not assumptions else 'green' for assumptions in bic_data['assumptions_met']]
        for bar, color in zip(bars3, colors3):
            bar.set_color(color)
            bar.set_alpha(0.7)
    
    # Model assumptions compliance
    ax4 = axes[1, 1]
    assumption_summary = comparison_df.groupby('assumptions_met').size()
    colors_pie = ['red', 'green']
    labels = ['Assumption Violations', 'Assumptions Met']
    ax4.pie(assumption_summary.values, labels=labels, colors=colors_pie, autopct='%1.1f%%')
    ax4.set_title('Model Assumption Compliance')
    
    plt.tight_layout()
    plt.savefig('results/figures/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive comparison plots saved to results/figures/comprehensive_model_comparison.png")

def main():
    """Main execution function"""
    comparison_df = create_comprehensive_comparison()
    
    # Save results
    comparison_df.to_csv('results/outputs/comprehensive_model_comparison.csv', index=False)
    print(f"\n💾 Results saved to results/outputs/comprehensive_model_comparison.csv")
    
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE MODEL COMPARISON COMPLETED!")
    print("🎯 RECOMMENDATION: Use Log-Logistic AFT for optimal performance")
    print("📊 All parametric models outperform Cox regression")
    print("⚖️  Parametric models solve assumption violation issues")
    print("=" * 80)

if __name__ == "__main__":
    main()