#!/usr/bin/env python3
"""
All Models Vulnerability Analysis
=================================
Clean, simple analysis of all 10 models' vulnerability patterns.
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict

def load_and_analyze_all_models():
    """Load data and analyze all models."""
    print("🚀 Loading data for all models...")
    
    # Load interaction effects
    interaction_df = pd.read_csv('generated/outputs/combined_interaction_effects_summary.csv')
    
    # Load subject analysis  
    subject_df = pd.read_csv('generated/outputs/time_varying_subject_cluster_analysis_combined.csv')
    
    print(f"✅ Loaded {len(interaction_df)} interaction effects")
    print(f"✅ Loaded {len(subject_df)} subject analyses")
    
    # Get unique models
    models = interaction_df['Model'].unique()
    print(f"📊 Found models: {models}")
    
    results = {}
    
    for model in models:
        print(f"\n🔍 Analyzing {model}...")
        
        # Get model-specific data
        model_interactions = interaction_df[interaction_df['Model'] == model]
        model_subjects = subject_df[subject_df['model'] == model]
        
        profile = analyze_single_model(model, model_interactions, model_subjects)
        results[model] = profile
    
    return results

def analyze_single_model(model_name, interactions_df, subjects_df):
    """Analyze a single model's patterns."""
    profile = {
        'model': model_name,
        'extreme_vulnerabilities': [],
        'extreme_protections': [],
        'subject_performance': {},
        'key_insights': []
    }
    
    # Analyze interaction effects
    for _, row in interactions_df.iterrows():
        hr = row['Hazard_Ratio']
        term = row['Term']
        
        # Extreme vulnerabilities (HR > 50)
        if hr > 50:
            profile['extreme_vulnerabilities'].append({
                'term': term,
                'hr': hr,
                'level': 'CATASTROPHIC' if hr > 1000 else 'EXTREME'
            })
        
        # Extreme protections (HR < 0.05)
        elif hr < 0.05:
            profile['extreme_protections'].append({
                'term': term, 
                'hr': hr,
                'protection_pct': round((1-hr)*100, 1)
            })
    
    # Analyze subject performance
    if len(subjects_df) > 0:
        for _, row in subjects_df.iterrows():
            subject = row['subject']
            mean_survival = row['time_to_failure_mean']
            profile['subject_performance'][subject] = mean_survival
    
    # Generate insights
    profile['key_insights'] = generate_insights(profile)
    
    return profile

def generate_insights(profile):
    """Generate key insights for a model."""
    insights = []
    
    # Vulnerability analysis
    if profile['extreme_vulnerabilities']:
        max_hr = max([v['hr'] for v in profile['extreme_vulnerabilities']])
        insights.append(f"MAXIMUM VULNERABILITY: {max_hr:.0f}x failure risk")
        
        if max_hr > 1000:
            insights.append("CATASTROPHIC SPIKES: >1000x failure risk in specific scenarios")
    
    # Protection analysis  
    if profile['extreme_protections']:
        min_hr = min([p['hr'] for p in profile['extreme_protections']])
        max_protection = max([p['protection_pct'] for p in profile['extreme_protections']])
        insights.append(f"MAXIMUM PROTECTION: {max_protection}% risk reduction")
    
    # Balance analysis
    vuln_count = len(profile['extreme_vulnerabilities'])
    prot_count = len(profile['extreme_protections'])
    
    if prot_count > vuln_count * 2:
        insights.append("PREDOMINANTLY PROTECTIVE profile")
    elif vuln_count > prot_count * 2:
        insights.append("HIGH RISK profile with many vulnerabilities")
    else:
        insights.append("BALANCED risk profile")
    
    # Subject analysis
    if profile['subject_performance']:
        subjects = profile['subject_performance']
        best_subject = max(subjects.keys(), key=lambda k: subjects[k])
        worst_subject = min(subjects.keys(), key=lambda k: subjects[k])
        best_score = subjects[best_subject]
        worst_score = subjects[worst_subject]
        
        insights.append(f"BEST DOMAIN: {best_subject} ({best_score:.2f} turns)")
        insights.append(f"WORST DOMAIN: {worst_subject} ({worst_score:.2f} turns)")
        
        if best_score - worst_score > 1.5:
            insights.append(f"HIGHLY SPECIALIZED: {best_score - worst_score:.2f} turn gap")
    
    return insights

def create_comparative_analysis(all_results):
    """Create comparative analysis across all models."""
    print("\n🔄 Creating comparative analysis...")
    
    comparison = {
        'most_vulnerable_model': None,
        'most_protected_model': None, 
        'most_specialized_model': None,
        'model_rankings': {}
    }
    
    # Find most vulnerable
    max_vulnerability = 0
    for model, profile in all_results.items():
        if profile['extreme_vulnerabilities']:
            max_hr = max([v['hr'] for v in profile['extreme_vulnerabilities']])
            if max_hr > max_vulnerability:
                max_vulnerability = max_hr
                comparison['most_vulnerable_model'] = {
                    'model': model,
                    'max_hr': max_hr
                }
    
    # Find most protected
    max_protection = 0
    for model, profile in all_results.items():
        if profile['extreme_protections']:
            best_protection = max([p['protection_pct'] for p in profile['extreme_protections']])
            if best_protection > max_protection:
                max_protection = best_protection
                comparison['most_protected_model'] = {
                    'model': model,
                    'protection_pct': best_protection
                }
    
    # Find most specialized
    max_specialization = 0
    for model, profile in all_results.items():
        if profile['subject_performance']:
            subjects = profile['subject_performance']
            if len(subjects) > 1:
                gap = max(subjects.values()) - min(subjects.values())
                if gap > max_specialization:
                    max_specialization = gap
                    comparison['most_specialized_model'] = {
                        'model': model,
                        'specialization_gap': gap
                    }
    
    return comparison

def create_comprehensive_report(all_results, comparison):
    """Create comprehensive markdown report."""
    report = """# 🔬 ALL MODELS VULNERABILITY ANALYSIS
## Time-Varying Advanced Models - Complete Analysis

### 📊 Executive Summary

This analysis reveals the unique vulnerability patterns of all 10 LLMs using time-varying advanced models. Each model shows distinct strategic profiles and specialized strengths/weaknesses.

---

"""
    
    # Add individual model profiles
    for model, profile in all_results.items():
        report += f"## 🤖 {model.upper()}\n\n"
        
        # Key insights
        if profile['key_insights']:
            report += "### 🎯 Key Insights\n"
            for insight in profile['key_insights']:
                report += f"- **{insight}**\n"
            report += "\n"
        
        # Extreme vulnerabilities
        if profile['extreme_vulnerabilities']:
            report += "### ⚠️ Extreme Vulnerabilities\n"
            for vuln in sorted(profile['extreme_vulnerabilities'], key=lambda x: x['hr'], reverse=True)[:3]:
                emoji = "💀" if vuln['level'] == 'CATASTROPHIC' else "🔴"
                report += f"- {emoji} **{vuln['term']}**: HR = {vuln['hr']:.1f}\n"
            report += "\n"
        
        # Extreme protections
        if profile['extreme_protections']:
            report += "### 🛡️ Extreme Protections\n"
            for prot in sorted(profile['extreme_protections'], key=lambda x: x['protection_pct'], reverse=True)[:3]:
                report += f"- 🟢 **{prot['term']}**: {prot['protection_pct']}% protection (HR = {prot['hr']:.4f})\n"
            report += "\n"
        
        # Subject performance
        if profile['subject_performance']:
            report += "### 📚 Subject Performance\n"
            subjects = profile['subject_performance']
            sorted_subjects = sorted(subjects.items(), key=lambda x: x[1], reverse=True)
            for subject, score in sorted_subjects:
                report += f"- **{subject}**: {score:.2f} turns\n"
            report += "\n"
        
        report += "---\n\n"
    
    # Add comparative analysis
    if comparison:
        report += "## 🏆 COMPARATIVE ANALYSIS\n\n"
        
        if comparison['most_vulnerable_model']:
            mv = comparison['most_vulnerable_model']
            report += f"### ⚠️ Most Vulnerable: {mv['model']}\n"
            report += f"Maximum hazard ratio: **{mv['max_hr']:.0f}x**\n\n"
        
        if comparison['most_protected_model']:
            mp = comparison['most_protected_model']
            report += f"### 🛡️ Most Protected: {mp['model']}\n"
            report += f"Maximum protection: **{mp['protection_pct']}%**\n\n"
        
        if comparison['most_specialized_model']:
            ms = comparison['most_specialized_model']
            report += f"### 🎯 Most Specialized: {ms['model']}\n"
            report += f"Specialization gap: **{ms['specialization_gap']:.2f} turns**\n\n"
    
    return report

def main():
    """Main execution."""
    print("🚀 Starting All Models Vulnerability Analysis...")
    print("=" * 60)
    
    # Analyze all models
    all_results = load_and_analyze_all_models()
    
    # Create comparative analysis
    comparison = create_comparative_analysis(all_results)
    
    # Create comprehensive report
    report = create_comprehensive_report(all_results, comparison)
    
    # Save results
    print("\n💾 Saving results...")
    
    with open('all_models_vulnerability_analysis.md', 'w') as f:
        f.write(report)
    
    with open('generated/outputs/all_model_profiles.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    with open('generated/outputs/model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print("✅ Analysis complete!")
    print("📁 Files created:")
    print("  - all_models_vulnerability_analysis.md")
    print("  - generated/outputs/all_model_profiles.json")
    print("  - generated/outputs/model_comparison.json")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"  - Analyzed {len(all_results)} models")
    
    if comparison['most_vulnerable_model']:
        mv = comparison['most_vulnerable_model']
        print(f"  - Most vulnerable: {mv['model']} (HR: {mv['max_hr']:.0f})")
    
    if comparison['most_protected_model']:
        mp = comparison['most_protected_model']
        print(f"  - Most protected: {mp['model']} ({mp['protection_pct']}% protection)")

if __name__ == "__main__":
    main() 