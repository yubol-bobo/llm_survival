#!/usr/bin/env python3
"""
Comprehensive Model Vulnerability Analysis
=========================================
Analyzes vulnerability patterns for all 10 LLMs from time-varying advanced models.
Extracts strategic insights, death traps, safe havens, and comparative advantages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

class LLMVulnerabilityAnalyzer:
    def __init__(self):
        self.models = ['CARG', 'claude_35', 'deepseek_r1', 'gemini_25', 'gpt_default', 
                       'llama_33', 'llama_4_maverick', 'llama_4_scout', 'mistral_large', 'qwen_max']
        self.model_profiles = {}
        self.comparative_insights = {}
        
    def load_data(self):
        """Load all interaction effects data."""
        print("📊 Loading interaction effects data...")
        
        # Load interaction effects
        self.interaction_df = pd.read_csv('generated/outputs/combined_interaction_effects_summary.csv')
        
        # Load subject analysis
        self.subject_df = pd.read_csv('generated/outputs/time_varying_subject_cluster_analysis_combined.csv')
        
        # Load difficulty analysis  
        self.difficulty_df = pd.read_csv('generated/outputs/time_varying_difficulty_level_analysis_combined.csv')
        
        print(f"✅ Loaded {len(self.interaction_df)} interaction effects")
        print(f"✅ Loaded subject analysis for {len(self.subject_df)} entries")
        print(f"✅ Loaded difficulty analysis for {len(self.difficulty_df)} entries")
        
    def analyze_single_model(self, model_name):
        """Analyze vulnerability patterns for a single model."""
        print(f"🔍 Analyzing {model_name}...")
        
        # Filter data for this model
        model_interactions = self.interaction_df[self.interaction_df['Model'] == model_name].copy()
        model_subjects = self.subject_df[self.subject_df['model'] == model_name].copy()
        model_difficulty = self.difficulty_df[self.difficulty_df['model'] == model_name].copy()
        
        profile = {
            'model_name': model_name,
            'death_traps': [],
            'safe_havens': [],
            'subject_strengths': [],
            'subject_weaknesses': [],
            'difficulty_profile': {},
            'strategic_insights': [],
            'unique_characteristics': []
        }
        
        if len(model_interactions) == 0:
            print(f"⚠️ No interaction data for {model_name}")
            return profile
            
        # Analyze interaction effects
        for _, row in model_interactions.iterrows():
            hr = row['Hazard_Ratio']
            term = row['Term']
            
            # Death traps (HR > 10)
            if hr > 10:
                profile['death_traps'].append({
                    'term': term,
                    'hazard_ratio': hr,
                    'risk_level': 'EXTREME' if hr > 100 else 'HIGH'
                })
                
            # Safe havens (HR < 0.1)  
            elif hr < 0.1:
                profile['safe_havens'].append({
                    'term': term,
                    'hazard_ratio': hr,
                    'protection_level': 'EXTREME' if hr < 0.01 else 'HIGH'
                })
        
        # Analyze subject performance
        if len(model_subjects) > 0:
            # Check actual column names
            subject_cols = model_subjects.columns.tolist()
            print(f"  Subject columns for {model_name}: {subject_cols}")
            
            if 'time_to_failure_mean' in subject_cols:
                model_subjects_sorted = model_subjects.sort_values('time_to_failure_mean', ascending=False)
                
                # Use the correct column name for subject
                subject_col = None
                for col in ['subject_cluster', 'subject', 'Subject']:
                    if col in subject_cols:
                        subject_col = col
                        break
                
                if subject_col:
                    profile['subject_strengths'] = model_subjects_sorted.head(2)[[subject_col, 'time_to_failure_mean']].to_dict('records')
                    profile['subject_weaknesses'] = model_subjects_sorted.tail(2)[[subject_col, 'time_to_failure_mean']].to_dict('records')
        
        # Analyze difficulty performance
        if len(model_difficulty) > 0:
            difficulty_clean = model_difficulty[model_difficulty['difficulty'].notna()].copy()
            if len(difficulty_clean) > 0:
                profile['difficulty_profile'] = difficulty_clean.set_index('difficulty')['time_to_failure_mean'].to_dict()
        
        # Generate strategic insights
        profile['strategic_insights'] = self._generate_strategic_insights(profile, model_name)
        profile['unique_characteristics'] = self._identify_unique_characteristics(profile, model_name)
        
        return profile
    
    def _generate_strategic_insights(self, profile, model_name):
        """Generate strategic insights based on vulnerability patterns."""
        insights = []
        
        # Death trap analysis
        if len(profile['death_traps']) > 0:
            extreme_traps = [trap for trap in profile['death_traps'] if trap['risk_level'] == 'EXTREME']
            if extreme_traps:
                max_hr = max([trap['hazard_ratio'] for trap in extreme_traps])
                insights.append(f"CATASTROPHIC VULNERABILITY: {max_hr:.1f}x failure risk in specific scenarios")
        
        # Safe haven analysis
        if len(profile['safe_havens']) > 0:
            extreme_safe = [haven for haven in profile['safe_havens'] if haven['protection_level'] == 'EXTREME']
            if extreme_safe:
                min_hr = min([haven['hazard_ratio'] for haven in extreme_safe])
                protection_pct = (1 - min_hr) * 100
                insights.append(f"EXTREME PROTECTION: {protection_pct:.1f}% risk reduction in favorable scenarios")
        
        # Balance analysis
        death_trap_count = len(profile['death_traps'])
        safe_haven_count = len(profile['safe_havens'])
        
        if safe_haven_count > death_trap_count * 2:
            insights.append("PREDOMINANTLY PROTECTIVE: More safe zones than danger zones")
        elif death_trap_count > safe_haven_count * 2:
            insights.append("HIGH RISK PROFILE: More danger zones than safe zones")
        else:
            insights.append("BALANCED RISK PROFILE: Similar numbers of safe and danger zones")
            
        return insights
    
    def _identify_unique_characteristics(self, profile, model_name):
        """Identify unique characteristics that differentiate this model."""
        characteristics = []
        
        # Extreme vulnerability patterns
        if profile['death_traps']:
            max_hr = max([trap['hazard_ratio'] for trap in profile['death_traps']])
            if max_hr > 1000:
                characteristics.append(f"Extreme vulnerability spikes (HR > 1000)")
            elif max_hr > 100:
                characteristics.append(f"High vulnerability concentration (HR > 100)")
        
        # Extreme protection patterns  
        if profile['safe_havens']:
            min_hr = min([haven['hazard_ratio'] for haven in profile['safe_havens']])
            if min_hr < 0.01:
                characteristics.append(f"Extreme protection zones (HR < 0.01)")
        
        # Subject specialization
        if profile['subject_strengths'] and profile['subject_weaknesses']:
            # Handle different possible column names
            subject_key = None
            for key in ['subject_cluster', 'subject', 'Subject']:
                if key in profile['subject_strengths'][0]:
                    subject_key = key
                    break
            
            if subject_key:
                best_subject = profile['subject_strengths'][0][subject_key]
                worst_subject = profile['subject_weaknesses'][0][subject_key]
                best_score = profile['subject_strengths'][0]['time_to_failure_mean']
                worst_score = profile['subject_weaknesses'][0]['time_to_failure_mean']
                
                if best_score - worst_score > 1.5:
                    characteristics.append(f"Strong domain specialization: {best_subject} >> {worst_subject}")
        
        return characteristics
    
    def analyze_all_models(self):
        """Analyze all models and generate comprehensive profiles."""
        print("🚀 Starting comprehensive analysis of all models...")
        
        for model in self.models:
            self.model_profiles[model] = self.analyze_single_model(model)
            
        print("✅ All models analyzed!")
        
    def generate_comparative_insights(self):
        """Generate insights comparing models against each other."""
        print("🔄 Generating comparative insights...")
        
        insights = {
            'most_vulnerable': None,
            'most_protected': None,
            'most_specialized': None,
            'most_balanced': None,
            'unique_patterns': {}
        }
        
        # Find most vulnerable (highest max HR)
        max_vulnerability = 0
        most_vulnerable_model = None
        for model, profile in self.model_profiles.items():
            if profile['death_traps']:
                max_hr = max([trap['hazard_ratio'] for trap in profile['death_traps']])
                if max_hr > max_vulnerability:
                    max_vulnerability = max_hr
                    most_vulnerable_model = model
        
        insights['most_vulnerable'] = {
            'model': most_vulnerable_model,
            'max_hazard_ratio': max_vulnerability
        }
        
        # Find most protected (lowest min HR)
        max_protection = 1.0
        most_protected_model = None
        for model, profile in self.model_profiles.items():
            if profile['safe_havens']:
                min_hr = min([haven['hazard_ratio'] for haven in profile['safe_havens']])
                if min_hr < max_protection:
                    max_protection = min_hr
                    most_protected_model = model
        
        insights['most_protected'] = {
            'model': most_protected_model,
            'min_hazard_ratio': max_protection,
            'protection_percentage': (1 - max_protection) * 100
        }
        
        # Find most specialized (biggest gap between best and worst subjects)
        max_specialization = 0
        most_specialized_model = None
        for model, profile in self.model_profiles.items():
            if profile['subject_strengths'] and profile['subject_weaknesses']:
                try:
                    best = profile['subject_strengths'][0]['time_to_failure_mean']
                    worst = profile['subject_weaknesses'][-1]['time_to_failure_mean']
                    gap = best - worst
                    if gap > max_specialization:
                        max_specialization = gap
                        most_specialized_model = model
                except (KeyError, IndexError):
                    continue
        
        insights['most_specialized'] = {
            'model': most_specialized_model,
            'specialization_gap': max_specialization
        }
        
        # Identify unique patterns
        for model, profile in self.model_profiles.items():
            unique_traits = []
            
            # Binary risk profile (CARG-like)
            death_count = len(profile['death_traps'])
            safe_count = len(profile['safe_havens'])
            moderate_count = 20 - death_count - safe_count  # Assuming ~20 total interactions
            
            if (death_count + safe_count) > moderate_count:
                unique_traits.append("Binary risk profile (extreme zones dominate)")
            
            # Extreme vulnerability spikes
            if profile['death_traps']:
                max_hr = max([trap['hazard_ratio'] for trap in profile['death_traps']])
                if max_hr > 1000:
                    unique_traits.append("Catastrophic vulnerability spikes")
            
            insights['unique_patterns'][model] = unique_traits
        
        self.comparative_insights = insights
        print("✅ Comparative insights generated!")
        
    def create_comprehensive_report(self):
        """Create comprehensive markdown report."""
        report = """# 🔬 Comprehensive LLM Vulnerability Analysis
## Time-Varying Advanced Models - All 10 Models

### 📊 Executive Summary

This analysis reveals the hidden vulnerability patterns of all 10 LLMs using time-varying advanced models with interaction effects. Each model exhibits unique strategic profiles, from CARG's binary protection patterns to other models' specialized strengths and weaknesses.

---

"""
        
        # Individual model profiles
        for model, profile in self.model_profiles.items():
            report += f"## 🤖 {model.upper()} - Detailed Profile\n\n"
            
            # Strategic insights
            if profile['strategic_insights']:
                report += "### 🎯 Strategic Profile\n"
                for insight in profile['strategic_insights']:
                    report += f"- **{insight}**\n"
                report += "\n"
            
            # Death traps
            if profile['death_traps']:
                report += "### ⚠️ Death Traps (HR > 10)\n"
                for trap in sorted(profile['death_traps'], key=lambda x: x['hazard_ratio'], reverse=True)[:5]:
                    risk_emoji = "💀" if trap['risk_level'] == 'EXTREME' else "🔴"
                    report += f"- {risk_emoji} **{trap['term']}**: HR = {trap['hazard_ratio']:.2f}\n"
                report += "\n"
            
            # Safe havens
            if profile['safe_havens']:
                report += "### 🛡️ Safe Havens (HR < 0.1)\n"
                for haven in sorted(profile['safe_havens'], key=lambda x: x['hazard_ratio'])[:5]:
                    protection_emoji = "🟢" if haven['protection_level'] == 'EXTREME' else "🟡"
                    protection_pct = (1 - haven['hazard_ratio']) * 100
                    report += f"- {protection_emoji} **{haven['term']}**: HR = {haven['hazard_ratio']:.4f} ({protection_pct:.1f}% protection)\n"
                report += "\n"
            
                         # Subject performance
             if profile['subject_strengths']:
                 report += "### 📚 Subject Domain Performance\n"
                 report += "**Strengths:**\n"
                 for strength in profile['subject_strengths']:
                     # Handle different column names
                     subject_key = None
                     for key in ['subject_cluster', 'subject', 'Subject']:
                         if key in strength:
                             subject_key = key
                             break
                     if subject_key:
                         report += f"- **{strength[subject_key]}**: {strength['time_to_failure_mean']:.2f} turns\n"
                 
                 if profile['subject_weaknesses']:
                     report += "\n**Weaknesses:**\n"
                     for weakness in profile['subject_weaknesses']:
                         # Handle different column names
                         subject_key = None
                         for key in ['subject_cluster', 'subject', 'Subject']:
                             if key in weakness:
                                 subject_key = key
                                 break
                         if subject_key:
                             report += f"- **{weakness[subject_key]}**: {weakness['time_to_failure_mean']:.2f} turns\n"
                 report += "\n"
            
            # Unique characteristics
            if profile['unique_characteristics']:
                report += "### ⭐ Unique Characteristics\n"
                for char in profile['unique_characteristics']:
                    report += f"- {char}\n"
                report += "\n"
            
            report += "---\n\n"
        
        # Comparative insights
        if self.comparative_insights:
            report += "## 🏆 Comparative Analysis\n\n"
            
            comp = self.comparative_insights
            
            if comp['most_vulnerable']:
                report += f"### ⚠️ Most Vulnerable Model\n"
                report += f"**{comp['most_vulnerable']['model']}** with maximum HR of {comp['most_vulnerable']['max_hazard_ratio']:.1f}\n\n"
            
            if comp['most_protected']:
                report += f"### 🛡️ Most Protected Model\n"
                report += f"**{comp['most_protected']['model']}** with minimum HR of {comp['most_protected']['min_hazard_ratio']:.4f} ({comp['most_protected']['protection_percentage']:.1f}% protection)\n\n"
            
            if comp['most_specialized']:
                report += f"### 🎯 Most Specialized Model\n"
                report += f"**{comp['most_specialized']['model']}** with {comp['most_specialized']['specialization_gap']:.2f} turns gap between best and worst subjects\n\n"
        
        return report
    
    def save_results(self):
        """Save all results to files."""
        print("💾 Saving comprehensive results...")
        
        # Save detailed report
        report = self.create_comprehensive_report()
        with open('comprehensive_model_analysis.md', 'w') as f:
            f.write(report)
        
        # Save model profiles as JSON
        with open('generated/outputs/model_vulnerability_profiles.json', 'w') as f:
            json.dump(self.model_profiles, f, indent=2)
        
        # Save comparative insights
        with open('generated/outputs/comparative_insights.json', 'w') as f:
            json.dump(self.comparative_insights, f, indent=2)
        
        print("✅ All results saved!")
        
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("📊 Creating comprehensive visualizations...")
        
        # 1. Model vulnerability heatmap
        self._create_vulnerability_heatmap()
        
        # 2. Subject performance comparison
        self._create_subject_comparison()
        
        # 3. Risk profile comparison
        self._create_risk_profile_comparison()
        
        print("✅ All visualizations created!")
    
    def _create_vulnerability_heatmap(self):
        """Create heatmap showing vulnerability patterns across models."""
        # Extract key vulnerability metrics for each model
        vulnerability_data = []
        
        for model, profile in self.model_profiles.items():
            max_hr = 0
            min_hr = 1.0
            death_trap_count = len(profile['death_traps'])
            safe_haven_count = len(profile['safe_havens'])
            
            if profile['death_traps']:
                max_hr = max([trap['hazard_ratio'] for trap in profile['death_traps']])
            if profile['safe_havens']:
                min_hr = min([haven['hazard_ratio'] for haven in profile['safe_havens']])
            
            vulnerability_data.append({
                'Model': model,
                'Max_HR_Log': np.log10(max_hr + 0.001),
                'Min_HR_Log': np.log10(min_hr + 0.001),
                'Death_Traps': death_trap_count,
                'Safe_Havens': safe_haven_count,
                'Risk_Balance': death_trap_count - safe_haven_count
            })
        
        df_vuln = pd.DataFrame(vulnerability_data).set_index('Model')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_vuln.T, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0)
        plt.title('Model Vulnerability Profile Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Vulnerability Metrics', fontweight='bold')
        plt.xlabel('Models', fontweight='bold')
        plt.tight_layout()
        plt.savefig('generated/figs/model_vulnerability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_subject_comparison(self):
        """Create subject domain performance comparison."""
        if len(self.subject_df) == 0:
            return
            
        # Pivot data for heatmap
        subject_pivot = self.subject_df.pivot(index='subject_cluster', columns='model', values='time_to_failure_mean')
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(subject_pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=3.5)
        plt.title('Subject Domain Performance Across All Models\n(Mean Survival Time)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontweight='bold')
        plt.ylabel('Subject Domains', fontweight='bold')
        plt.tight_layout()
        plt.savefig('generated/figs/subject_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_risk_profile_comparison(self):
        """Create risk profile comparison chart."""
        risk_data = []
        
        for model, profile in self.model_profiles.items():
            death_count = len(profile['death_traps'])
            safe_count = len(profile['safe_havens'])
            
            # Get max and min HRs
            max_hr = 1.0
            min_hr = 1.0
            if profile['death_traps']:
                max_hr = max([trap['hazard_ratio'] for trap in profile['death_traps']])
            if profile['safe_havens']:
                min_hr = min([haven['hazard_ratio'] for haven in profile['safe_havens']])
            
            risk_data.append({
                'Model': model,
                'Death_Traps': death_count,
                'Safe_Havens': safe_count,
                'Max_HR': max_hr,
                'Min_HR': min_hr,
                'Risk_Type': 'Binary' if (death_count + safe_count) > 10 else 'Moderate'
            })
        
        df_risk = pd.DataFrame(risk_data)
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Death traps vs safe havens
        ax1.scatter(df_risk['Death_Traps'], df_risk['Safe_Havens'], s=100, alpha=0.7)
        for i, model in enumerate(df_risk['Model']):
            ax1.annotate(model, (df_risk.iloc[i]['Death_Traps'], df_risk.iloc[i]['Safe_Havens']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax1.set_xlabel('Number of Death Traps (HR > 10)', fontweight='bold')
        ax1.set_ylabel('Number of Safe Havens (HR < 0.1)', fontweight='bold')
        ax1.set_title('Death Traps vs Safe Havens', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Max HR comparison
        bars1 = ax2.bar(range(len(df_risk)), df_risk['Max_HR'], alpha=0.7)
        ax2.set_yscale('log')
        ax2.set_xlabel('Models', fontweight='bold')
        ax2.set_ylabel('Maximum Hazard Ratio (Log Scale)', fontweight='bold')
        ax2.set_title('Highest Vulnerability Spikes', fontweight='bold')
        ax2.set_xticks(range(len(df_risk)))
        ax2.set_xticklabels(df_risk['Model'], rotation=45)
        
        # Min HR comparison
        bars2 = ax3.bar(range(len(df_risk)), df_risk['Min_HR'], alpha=0.7, color='green')
        ax3.set_yscale('log')
        ax3.set_xlabel('Models', fontweight='bold')
        ax3.set_ylabel('Minimum Hazard Ratio (Log Scale)', fontweight='bold')
        ax3.set_title('Strongest Protection Zones', fontweight='bold')
        ax3.set_xticks(range(len(df_risk)))
        ax3.set_xticklabels(df_risk['Model'], rotation=45)
        
        # Risk type distribution
        risk_counts = df_risk['Risk_Type'].value_counts()
        ax4.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.0f%%')
        ax4.set_title('Risk Profile Types', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('generated/figs/risk_profile_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function."""
    print("🚀 Starting Comprehensive LLM Vulnerability Analysis...")
    print("=" * 60)
    
    analyzer = LLMVulnerabilityAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Analyze all models
    analyzer.analyze_all_models()
    
    # Generate comparative insights
    analyzer.generate_comparative_insights()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Save results
    analyzer.save_results()
    
    print("\n🎉 Comprehensive analysis complete!")
    print("📁 Files created:")
    print("  - comprehensive_model_analysis.md")
    print("  - generated/outputs/model_vulnerability_profiles.json")
    print("  - generated/outputs/comparative_insights.json")
    print("  - generated/figs/model_vulnerability_heatmap.png")
    print("  - generated/figs/subject_performance_comparison.png")
    print("  - generated/figs/risk_profile_comparison.png")

if __name__ == "__main__":
    main() 