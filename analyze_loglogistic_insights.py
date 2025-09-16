#!/usr/bin/env python3
"""
Log-Logistic AFT Model Insights Analysis
Deep dive into the best performing model to understand:
1. Feature importance and hazard effects
2. Model-specific survival patterns
3. Risk stratification insights
4. Practical implications for LLM deployment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import LogLogisticAFTFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(__file__))

class LogLogisticInsights:
    """Analyze insights from the best performing Log-Logistic AFT model"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self.coefficients = None
        self.results = {}
        
    def load_and_fit_model(self):
        """Load data and fit the Log-Logistic AFT model"""
        print("🔬 LOADING DATA AND FITTING LOG-LOGISTIC AFT MODEL")
        print("=" * 60)
        
        try:
            # Load processed model data
            processed_dir = 'data/processed'
            if not os.path.exists(processed_dir):
                raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
            
            model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
            
            combined_data = []
            
            for model_name in tqdm(model_dirs, desc="Loading models"):
                model_path = os.path.join(processed_dir, model_name)
                long_path = os.path.join(model_path, f'{model_name}_long.csv')
                
                if os.path.exists(long_path):
                    long_df = pd.read_csv(long_path)
                    long_df['model'] = model_name
                    
                    required_cols = ['round', 'failure', 'conversation_id', 'model', 
                                   'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                                   'cumulative_drift', 'prompt_complexity']
                    
                    available_cols = [col for col in required_cols if col in long_df.columns]
                    model_subset = long_df[available_cols].copy()
                    model_subset = model_subset.dropna()
                    
                    if len(model_subset) > 0:
                        combined_data.append(model_subset)
            
            if not combined_data:
                raise ValueError("No valid data found")
            
            # Combine and prepare data
            self.data = pd.concat(combined_data, ignore_index=True)
            
            # Create model dummy variables
            model_dummies = pd.get_dummies(self.data['model'], prefix='model', drop_first=True)
            self.data = pd.concat([self.data, model_dummies], axis=1)
            
            # Prepare for AFT modeling (numeric columns only)
            numeric_cols = ['round', 'failure', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 
                           'cumulative_drift', 'prompt_complexity']
            model_cols = [col for col in self.data.columns if col.startswith('model_')]
            
            final_cols = numeric_cols + model_cols
            aft_data = self.data[final_cols].copy()
            
            print(f"✅ Loaded {len(model_dirs)} models")
            print(f"✅ Dataset: {aft_data.shape[0]} observations, {len(final_cols)} features")
            print(f"✅ Event rate: {aft_data['failure'].mean()*100:.1f}%")
            
            # Fit Log-Logistic AFT model
            print("\n🔧 FITTING LOG-LOGISTIC AFT MODEL")
            self.model = LogLogisticAFTFitter(penalizer=0.01)
            self.model.fit(aft_data, duration_col='round', event_col='failure')
            
            self.coefficients = self.model.summary.copy()
            
            print(f"✅ Model fitted successfully")
            print(f"   C-index: {self.model.concordance_index_:.4f}")
            print(f"   AIC: {self.model.AIC_:.2f}")
            print(f"   Log-likelihood: {self.model.log_likelihood_:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model fitting failed: {e}")
            return False
    
    def analyze_feature_importance(self):
        """Analyze feature importance and effects"""
        print("\n📊 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        if self.coefficients is None:
            print("❌ No coefficients available")
            return
        
        # Get coefficients and significance
        coef_analysis = self.coefficients.copy()
        coef_analysis['abs_coef'] = abs(coef_analysis['coef'])
        coef_analysis['significant'] = coef_analysis['p'] < 0.05
        coef_analysis['effect_direction'] = coef_analysis['coef'].apply(lambda x: 'Accelerates Failure' if x < 0 else 'Delays Failure')
        
        # Sort by absolute coefficient value (importance)
        coef_analysis = coef_analysis.sort_values('abs_coef', ascending=False)
        
        print("🔍 TOP FEATURE EFFECTS (Acceleration Factor interpretation):")
        print("   • Negative coefficients = INCREASE failure risk (accelerate failure)")
        print("   • Positive coefficients = DECREASE failure risk (delay failure)")
        print()
        
        # Display top features
        for idx, row in coef_analysis.head(10).iterrows():
            significance = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
            acceleration_factor = np.exp(row['coef'])  # AFT interpretation
            
            # Handle tuple indices (convert to string)
            idx_str = str(idx) if isinstance(idx, tuple) else idx
            feature_name = idx_str.replace('(', '').replace(')', '').replace("'", '').replace(',', ' ')
            
            print(f"🎯 {feature_name}:")
            print(f"   Coefficient: {row['coef']:.4f} {significance}")
            print(f"   Acceleration Factor: {acceleration_factor:.4f}")
            print(f"   Effect: {row['effect_direction']}")
            print(f"   P-value: {row['p']:.6f}")
            
            # Practical interpretation
            if 'drift' in feature_name.lower():
                if row['coef'] < 0:
                    print(f"   📈 Higher {feature_name} → {(1-acceleration_factor)*100:.1f}% faster failure")
                else:
                    print(f"   📉 Higher {feature_name} → {(acceleration_factor-1)*100:.1f}% slower failure")
            elif 'model_' in feature_name:
                model_name = feature_name.replace('model_', '').replace('_', ' ').title()
                if row['coef'] < 0:
                    print(f"   ⚠️  {model_name} fails {(1-acceleration_factor)*100:.1f}% faster than baseline")
                else:
                    print(f"   ✅ {model_name} fails {(acceleration_factor-1)*100:.1f}% slower than baseline")
            print()
        
        # Store results
        self.results['feature_importance'] = coef_analysis
        
        return coef_analysis
    
    def analyze_model_rankings(self):
        """Analyze LLM model performance rankings"""
        print("\n🏆 LLM MODEL PERFORMANCE RANKINGS")
        print("=" * 50)
        
        if self.coefficients is None:
            print("❌ No coefficients available")
            return
        
        # Extract model coefficients - handle tuple indices
        model_coefs = self.coefficients.copy()
        
        # Filter for model coefficients
        model_indices = []
        for idx in model_coefs.index:
            idx_str = str(idx) if isinstance(idx, tuple) else idx
            if 'model_' in idx_str.lower():
                model_indices.append(idx)
        
        model_coefs = model_coefs.loc[model_indices]
        # Extract model names from indices
        model_names = []
        for idx in model_coefs.index:
            idx_str = str(idx) if isinstance(idx, tuple) else str(idx)
            # Extract model name from string representation
            if 'model_' in idx_str:
                name_part = idx_str.split('model_')[1].split()[0].replace("'", '').replace(',', '')
                model_names.append(name_part.replace('_', ' ').title())
            else:
                model_names.append(idx_str.replace('_', ' ').title())
        
        model_coefs['model_name'] = model_names
        model_coefs['acceleration_factor'] = np.exp(model_coefs['coef'])
        model_coefs['failure_risk'] = model_coefs['coef'].apply(lambda x: 'Higher Risk' if x < 0 else 'Lower Risk')
        
        # Add baseline model (reference category)
        baseline_models = [m for m in self.data['model'].unique() if f"model_{m}" not in model_coefs.index]
        if baseline_models:
            baseline_row = pd.DataFrame({
                'coef': [0.0],
                'exp(coef)': [1.0],
                'se(coef)': [0.0],
                'coef lower 95%': [0.0],
                'coef upper 95%': [0.0],
                'exp(coef) lower 95%': [1.0],
                'exp(coef) upper 95%': [1.0],
                'cmp to': ['0.0'],
                'z': [0.0],
                'p': [1.0],
                'model_name': [baseline_models[0].replace('_', ' ').title()],
                'acceleration_factor': [1.0],
                'failure_risk': ['Baseline']
            }, index=[f'model_{baseline_models[0]}'])
            
            model_coefs = pd.concat([model_coefs, baseline_row])
        
        # Sort by acceleration factor (higher = more resilient)
        model_coefs = model_coefs.sort_values('acceleration_factor', ascending=False)
        
        print("📊 MODEL RESILIENCE RANKING (Acceleration Factor):")
        print("   • Higher Acceleration Factor = More Resilient (delays failure)")
        print("   • Lower Acceleration Factor = Less Resilient (accelerates failure)")
        print()
        
        for i, (idx, row) in enumerate(model_coefs.iterrows(), 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            risk_emoji = "✅" if row['failure_risk'] == 'Lower Risk' else "⚠️" if row['failure_risk'] == 'Higher Risk' else "🔘"
            
            print(f"{rank_emoji} {row['model_name']}")
            print(f"   Acceleration Factor: {row['acceleration_factor']:.4f}")
            print(f"   Risk Level: {row['failure_risk']} {risk_emoji}")
            
            if row['acceleration_factor'] != 1.0:
                if row['acceleration_factor'] > 1.0:
                    print(f"   Performance: {(row['acceleration_factor']-1)*100:.1f}% more resilient than baseline")
                else:
                    print(f"   Performance: {(1-row['acceleration_factor'])*100:.1f}% less resilient than baseline")
            
            if row['p'] < 0.05:
                print(f"   Significance: p = {row['p']:.4f} ⭐")
            else:
                print(f"   Significance: p = {row['p']:.4f} (not significant)")
            print()
        
        self.results['model_rankings'] = model_coefs
        return model_coefs
    
    def analyze_drift_patterns(self):
        """Analyze drift pattern effects"""
        print("\n🌊 CONVERSATION DRIFT PATTERN ANALYSIS")
        print("=" * 50)
        
        if self.coefficients is None:
            print("❌ No coefficients available")
            return
        
        # Extract drift-related coefficients
        drift_features = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 
                         'cumulative_drift', 'prompt_complexity']
        
        drift_analysis = []
        for feature in drift_features:
            # Look for feature in coefficients (handling tuple indices)
            found_idx = None
            for idx in self.coefficients.index:
                idx_str = str(idx) if isinstance(idx, tuple) else str(idx)
                if feature in idx_str:
                    found_idx = idx
                    break
            
            if found_idx is not None:
                coef_data = self.coefficients.loc[found_idx]
                acceleration_factor = np.exp(coef_data['coef'])
                
                drift_analysis.append({
                    'feature': feature,
                    'coefficient': coef_data['coef'],
                    'acceleration_factor': acceleration_factor,
                    'p_value': coef_data['p'],
                    'significant': coef_data['p'] < 0.05,
                    'effect_strength': 'Strong' if abs(coef_data['coef']) > 0.1 else 'Moderate' if abs(coef_data['coef']) > 0.05 else 'Weak'
                })
        
        drift_df = pd.DataFrame(drift_analysis)
        
        print("🔍 DRIFT FEATURE IMPACT:")
        for _, row in drift_df.iterrows():
            effect_emoji = "⚠️" if row['coefficient'] < 0 else "✅"
            significance_emoji = "⭐" if row['significant'] else "◯"
            
            print(f"{effect_emoji} {row['feature'].replace('_', ' ').title()}:")
            print(f"   Acceleration Factor: {row['acceleration_factor']:.4f}")
            print(f"   Effect Strength: {row['effect_strength']}")
            print(f"   Significance: {significance_emoji} (p = {row['p_value']:.4f})")
            
            if row['coefficient'] < 0:
                print(f"   Impact: {(1-row['acceleration_factor'])*100:.1f}% faster failure per unit increase")
            else:
                print(f"   Impact: {(row['acceleration_factor']-1)*100:.1f}% slower failure per unit increase")
            print()
        
        self.results['drift_patterns'] = drift_df
        return drift_df
    
    def generate_survival_insights(self):
        """Generate survival curve insights"""
        print("\n📈 SURVIVAL CURVE INSIGHTS")
        print("=" * 40)
        
        try:
            # Create representative scenarios
            scenarios = {
                'Low Risk Scenario': {
                    'prompt_to_prompt_drift': 0.1,
                    'context_to_prompt_drift': 0.05,
                    'cumulative_drift': 0.2,
                    'prompt_complexity': 0.3
                },
                'Medium Risk Scenario': {
                    'prompt_to_prompt_drift': 0.5,
                    'context_to_prompt_drift': 0.3,
                    'cumulative_drift': 0.8,
                    'prompt_complexity': 0.6
                },
                'High Risk Scenario': {
                    'prompt_to_prompt_drift': 1.0,
                    'context_to_prompt_drift': 0.8,
                    'cumulative_drift': 1.5,
                    'prompt_complexity': 0.9
                }
            }
            
            # Calculate survival probabilities
            time_points = [1, 5, 10, 15, 20, 25, 30]
            survival_results = []
            
            for scenario_name, scenario_values in scenarios.items():
                # Create scenario dataframe (need all model columns set to 0)
                scenario_df = pd.DataFrame([scenario_values])
                
                # Add all model dummy columns (set to 0 for baseline)
                model_cols = [col for col in self.model.params_.index if col.startswith('model_')]
                for col in model_cols:
                    scenario_df[col] = 0
                
                # Calculate survival probabilities
                for time_point in time_points:
                    try:
                        survival_prob = self.model.predict_survival_function(scenario_df).iloc[0, :].loc[time_point]
                        survival_results.append({
                            'scenario': scenario_name,
                            'time_point': time_point,
                            'survival_probability': survival_prob,
                            'failure_probability': 1 - survival_prob
                        })
                    except:
                        # If exact time point not available, interpolate
                        sf = self.model.predict_survival_function(scenario_df).iloc[0, :]
                        if time_point <= sf.index.max():
                            # Find closest time points
                            lower_idx = sf.index[sf.index <= time_point].max()
                            upper_idx = sf.index[sf.index >= time_point].min()
                            
                            if lower_idx == upper_idx:
                                survival_prob = sf.loc[lower_idx]
                            else:
                                # Linear interpolation
                                weight = (time_point - lower_idx) / (upper_idx - lower_idx)
                                survival_prob = sf.loc[lower_idx] * (1 - weight) + sf.loc[upper_idx] * weight
                            
                            survival_results.append({
                                'scenario': scenario_name,
                                'time_point': time_point,
                                'survival_probability': survival_prob,
                                'failure_probability': 1 - survival_prob
                            })
            
            survival_df = pd.DataFrame(survival_results)
            
            print("🎯 SURVIVAL PROBABILITY BY SCENARIO:")
            for scenario in scenarios.keys():
                scenario_data = survival_df[survival_df['scenario'] == scenario]
                print(f"\n{scenario}:")
                for _, row in scenario_data.iterrows():
                    print(f"   Round {row['time_point']:2d}: {row['survival_probability']*100:5.1f}% survival | {row['failure_probability']*100:5.1f}% failure")
            
            self.results['survival_insights'] = survival_df
            return survival_df
            
        except Exception as e:
            print(f"❌ Survival insights generation failed: {e}")
            return None
    
    def create_insights_visualization(self):
        """Create comprehensive insights visualization"""
        print("\n📊 GENERATING COMPREHENSIVE INSIGHTS VISUALIZATION")
        print("=" * 60)
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Feature Importance Plot
            ax1 = axes[0, 0]
            if 'feature_importance' in self.results:
                top_features = self.results['feature_importance'].head(8)
                colors = ['red' if coef < 0 else 'green' for coef in top_features['coef']]
                
                bars = ax1.barh(range(len(top_features)), abs(top_features['coef']), color=colors, alpha=0.7)
                ax1.set_yticks(range(len(top_features)))
                ax1.set_yticklabels([idx.replace('_', ' ').title() for idx in top_features.index])
                ax1.set_xlabel('Absolute Coefficient Value')
                ax1.set_title('Top Feature Importance\n(Red=Accelerates Failure, Green=Delays Failure)')
                ax1.grid(True, alpha=0.3)
            
            # 2. Model Rankings Plot
            ax2 = axes[0, 1]
            if 'model_rankings' in self.results:
                rankings = self.results['model_rankings']
                colors = ['red' if af < 1 else 'green' if af > 1 else 'gray' for af in rankings['acceleration_factor']]
                
                bars = ax2.barh(range(len(rankings)), rankings['acceleration_factor'], color=colors, alpha=0.7)
                ax2.set_yticks(range(len(rankings)))
                ax2.set_yticklabels(rankings['model_name'])
                ax2.set_xlabel('Acceleration Factor')
                ax2.set_title('LLM Model Resilience Rankings\n(Higher = More Resilient)')
                ax2.axvline(x=1, color='black', linestyle='--', alpha=0.5)
                ax2.grid(True, alpha=0.3)
            
            # 3. Drift Effects Plot
            ax3 = axes[1, 0]
            if 'drift_patterns' in self.results:
                drift_data = self.results['drift_patterns']
                colors = ['red' if coef < 0 else 'green' for coef in drift_data['coefficient']]
                
                bars = ax3.bar(range(len(drift_data)), drift_data['acceleration_factor'], color=colors, alpha=0.7)
                ax3.set_xticks(range(len(drift_data)))
                ax3.set_xticklabels([f.replace('_', '\n').title() for f in drift_data['feature']], rotation=45)
                ax3.set_ylabel('Acceleration Factor')
                ax3.set_title('Drift Pattern Effects\n(Red=Harmful, Green=Protective)')
                ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
                ax3.grid(True, alpha=0.3)
            
            # 4. Survival Scenarios Plot
            ax4 = axes[1, 1]
            if 'survival_insights' in self.results:
                survival_data = self.results['survival_insights']
                scenarios = survival_data['scenario'].unique()
                colors = ['green', 'orange', 'red']
                
                for i, scenario in enumerate(scenarios):
                    scenario_data = survival_data[survival_data['scenario'] == scenario]
                    ax4.plot(scenario_data['time_point'], scenario_data['survival_probability'], 
                            marker='o', label=scenario, color=colors[i], linewidth=2)
                
                ax4.set_xlabel('Conversation Round')
                ax4.set_ylabel('Survival Probability')
                ax4.set_title('Survival Curves by Risk Scenario')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('results/figures/loglogistic_insights.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Insights visualization saved to results/figures/loglogistic_insights.png")
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
    
    def generate_actionable_insights(self):
        """Generate actionable insights for LLM deployment"""
        print("\n🎯 ACTIONABLE INSIGHTS FOR LLM DEPLOYMENT")
        print("=" * 60)
        
        insights = []
        
        # Model-specific insights
        if 'model_rankings' in self.results:
            rankings = self.results['model_rankings'].sort_values('acceleration_factor', ascending=False)
            best_model = rankings.iloc[0]
            worst_model = rankings.iloc[-1]
            
            insights.append(f"🏆 BEST PERFORMING MODEL: {best_model['model_name']}")
            insights.append(f"   • {(best_model['acceleration_factor']-1)*100:.1f}% more resilient than baseline")
            insights.append(f"   • Recommended for production deployment")
            
            insights.append(f"\n⚠️  HIGHEST RISK MODEL: {worst_model['model_name']}")
            insights.append(f"   • {(1-worst_model['acceleration_factor'])*100:.1f}% higher failure rate")
            insights.append(f"   • Requires additional monitoring or safeguards")
        
        # Drift-based insights
        if 'drift_patterns' in self.results:
            drift_data = self.results['drift_patterns']
            most_harmful = drift_data.loc[drift_data['coefficient'].idxmin()]
            
            insights.append(f"\n🚨 HIGHEST RISK FACTOR: {most_harmful['feature'].replace('_', ' ').title()}")
            insights.append(f"   • {(1-most_harmful['acceleration_factor'])*100:.1f}% faster failure per unit increase")
            insights.append(f"   • Priority monitoring target")
        
        # Survival-based insights
        if 'survival_insights' in self.results:
            survival_data = self.results['survival_insights']
            
            # Find round where high-risk scenario drops below 50% survival
            high_risk = survival_data[survival_data['scenario'] == 'High Risk Scenario']
            critical_round = high_risk[high_risk['survival_probability'] < 0.5]['time_point'].iloc[0] if len(high_risk[high_risk['survival_probability'] < 0.5]) > 0 else None
            
            if critical_round:
                insights.append(f"\n⏰ CRITICAL CONVERSATION LENGTH: {critical_round} rounds")
                insights.append(f"   • High-risk conversations have <50% survival beyond round {critical_round}")
                insights.append(f"   • Implement intervention strategies before round {critical_round}")
        
        # Practical recommendations
        insights.extend([
            "\n📋 DEPLOYMENT RECOMMENDATIONS:",
            "1. 🎯 Model Selection: Deploy top-ranked models for critical applications",
            "2. 📊 Monitoring: Track drift metrics in real-time",
            "3. ⚡ Early Warning: Set alerts for high drift values",
            "4. 🔄 Intervention: Implement conversation reset mechanisms",
            "5. 📈 Optimization: Focus on reducing cumulative drift"
        ])
        
        # Print all insights
        for insight in insights:
            print(insight)
        
        self.results['actionable_insights'] = insights
        return insights
    
    def save_all_results(self):
        """Save all analysis results"""
        print("\n💾 SAVING LOG-LOGISTIC INSIGHTS ANALYSIS")
        print("=" * 50)
        
        try:
            os.makedirs('results/outputs/loglogistic_insights', exist_ok=True)
            
            # Save feature importance
            if 'feature_importance' in self.results:
                self.results['feature_importance'].to_csv(
                    'results/outputs/loglogistic_insights/feature_importance.csv')
                print("✅ Feature importance saved")
            
            # Save model rankings
            if 'model_rankings' in self.results:
                self.results['model_rankings'].to_csv(
                    'results/outputs/loglogistic_insights/model_rankings.csv')
                print("✅ Model rankings saved")
            
            # Save drift patterns
            if 'drift_patterns' in self.results:
                self.results['drift_patterns'].to_csv(
                    'results/outputs/loglogistic_insights/drift_patterns.csv', index=False)
                print("✅ Drift patterns saved")
            
            # Save survival insights
            if 'survival_insights' in self.results:
                self.results['survival_insights'].to_csv(
                    'results/outputs/loglogistic_insights/survival_scenarios.csv', index=False)
                print("✅ Survival scenarios saved")
            
            # Save actionable insights
            if 'actionable_insights' in self.results:
                with open('results/outputs/loglogistic_insights/actionable_insights.txt', 'w') as f:
                    f.write('\n'.join(self.results['actionable_insights']))
                print("✅ Actionable insights saved")
            
            # Save model coefficients
            if self.coefficients is not None:
                self.coefficients.to_csv('results/outputs/loglogistic_insights/model_coefficients.csv')
                print("✅ Model coefficients saved")
            
            print("📁 All insights saved to results/outputs/loglogistic_insights/")
            
        except Exception as e:
            print(f"❌ Saving failed: {e}")
    
    def run_complete_analysis(self):
        """Run complete Log-Logistic insights analysis"""
        print("🔬 LOG-LOGISTIC AFT MODEL INSIGHTS ANALYSIS")
        print("=" * 80)
        
        # Load and fit model
        if not self.load_and_fit_model():
            return
        
        print("=" * 80)
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        print("=" * 80)
        
        # Analyze model rankings
        self.analyze_model_rankings()
        
        print("=" * 80)
        
        # Analyze drift patterns
        self.analyze_drift_patterns()
        
        print("=" * 80)
        
        # Generate survival insights
        self.generate_survival_insights()
        
        print("=" * 80)
        
        # Create visualizations
        self.create_insights_visualization()
        
        print("=" * 80)
        
        # Generate actionable insights
        self.generate_actionable_insights()
        
        print("=" * 80)
        
        # Save all results
        self.save_all_results()
        
        print("=" * 80)
        print("🎉 LOG-LOGISTIC INSIGHTS ANALYSIS COMPLETED!")
        print("🔍 Key insights extracted for LLM deployment optimization")

def main():
    """Main execution function"""
    analyzer = LogLogisticInsights()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()