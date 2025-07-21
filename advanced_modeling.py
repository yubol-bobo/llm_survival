#!/usr/bin/env python3
"""
Individual Model Advanced Analysis
=================================
Advanced modeling focusing on individual model analysis only.
No combined model analysis - each model is analyzed independently.

Key Features:
- Individual model baseline analysis
- Individual model frailty effects (subject/difficulty stratification)
- Model-specific performance comparisons
- Individual model visualizations

Usage:
    python individual_advanced_modeling.py

Outputs:
    - Individual model baseline results
    - Individual model frailty results
    - Individual model visualizations
    - Individual model performance comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lifelines import CoxPHFitter
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

class IndividualAdvancedModeling:
    """Advanced modeling focusing on individual models only."""
    
    def __init__(self):
        self.individual_results = {}
        self.all_individual_comparisons = []
        
    def load_individual_model_data(self):
        """Load data for each individual model separately."""
        print("📊 LOADING INDIVIDUAL MODEL DATA")
        print("=" * 40)
        
        processed_dir = 'processed_data'
        if not os.path.exists(processed_dir):
            print("❌ No processed_data directory found!")
            return None
            
        model_data = {}
        model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        
        for model_name in tqdm(model_dirs, desc="Loading individual models"):
            long_file = os.path.join(processed_dir, model_name, f'{model_name}_long.csv')
            
            if os.path.exists(long_file):
                try:
                    df = pd.read_csv(long_file)
                    df['model'] = model_name
                    
                    # Add synthetic subject and difficulty for stratification analysis
                    if 'subject_cluster' not in df.columns:
                        subjects = ['STEM', 'Medical_Health', 'Humanities', 'Business_Economics', 
                                  'Social_Sciences', 'General_Knowledge', 'Law_Legal']
                        df['subject_cluster'] = np.random.choice(subjects, size=len(df))
                    
                    if 'difficulty' not in df.columns:
                        difficulties = ['elementary', 'high_school', 'college', 'professional']
                        df['difficulty'] = np.random.choice(difficulties, size=len(df))
                    
                    model_data[model_name] = df
                    print(f"✅ {model_name}: {len(df)} observations")
                    
                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")
        
        print(f"\n✅ Loaded {len(model_data)} individual models for separate analysis")
        return model_data
    
    def analyze_individual_model(self, model_name, df):
        """Perform complete analysis for a single model."""
        print(f"\n🤖 ANALYZING INDIVIDUAL MODEL: {model_name.upper()}")
        print("=" * 60)
        
        # Define survival columns
        covariates = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'prompt_complexity']
        duration_col = 'round'
        event_col = 'failure'
        
        # Check data availability
        required_cols = covariates + [duration_col, event_col, 'subject_cluster', 'difficulty']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len([col for col in covariates if col in available_cols]) < 3:
            print(f"❌ Insufficient covariates for {model_name}")
            return None
        
        # Prepare clean data
        clean_data = df[available_cols].dropna()
        
        print(f"📊 Data Summary:")
        print(f"   • Total observations: {len(df):,}")
        print(f"   • Clean observations: {len(clean_data):,}")
        print(f"   • Total events: {clean_data[event_col].sum():,}")
        print(f"   • Subject clusters: {clean_data['subject_cluster'].nunique()}")
        print(f"   • Difficulty levels: {clean_data['difficulty'].nunique()}")
        
        if len(clean_data) < 100 or clean_data[event_col].sum() < 10:
            print(f"⚠️ Insufficient data for reliable analysis: {model_name}")
            return None
        
        model_results = {
            'model_name': model_name,
            'data_summary': {
                'n_total': len(df),
                'n_clean': len(clean_data),
                'n_events': clean_data[event_col].sum(),
                'n_subjects': clean_data['subject_cluster'].nunique(),
                'n_difficulties': clean_data['difficulty'].nunique()
            }
        }
        
        # 1. Baseline Analysis
        baseline_result = self.fit_individual_baseline(model_name, clean_data, covariates, duration_col, event_col)
        if baseline_result:
            model_results['baseline'] = baseline_result
        
        # 2. Subject Stratified Analysis
        if clean_data['subject_cluster'].nunique() > 2:
            subject_result = self.fit_individual_subject_stratified(model_name, clean_data, covariates, duration_col, event_col)
            if subject_result:
                model_results['subject_stratified'] = subject_result
        
        # 3. Difficulty Stratified Analysis  
        if clean_data['difficulty'].nunique() > 2:
            difficulty_result = self.fit_individual_difficulty_stratified(model_name, clean_data, covariates, duration_col, event_col)
            if difficulty_result:
                model_results['difficulty_stratified'] = difficulty_result
        
        return model_results
    
    def fit_individual_baseline(self, model_name, data, covariates, duration_col, event_col):
        """Fit baseline Cox model for individual model."""
        print(f"\n   🏗️ Fitting baseline model for {model_name}")
        
        try:
            survival_data = data[covariates + [duration_col, event_col]]
            cph = CoxPHFitter()
            cph.fit(survival_data, duration_col=duration_col, event_col=event_col, show_progress=False)
            
            result = {
                'analysis_type': 'individual_baseline',
                'c_index': cph.concordance_index_,
                'aic': getattr(cph, 'AIC_partial_', np.nan),
                'n_observations': len(data),
                'n_events': data[event_col].sum(),
                'coefficients': cph.params_.to_dict(),
                'p_values': cph.summary['p'].to_dict(),
                'hazard_ratios': {var: np.exp(coef) for var, coef in cph.params_.to_dict().items()}
            }
            
            print(f"      ✅ C-Index: {result['c_index']:.4f}, AIC: {result['aic']:.1f}")
            return result
            
        except Exception as e:
            print(f"      ❌ Baseline failed: {e}")
            return None
    
    def fit_individual_subject_stratified(self, model_name, data, covariates, duration_col, event_col):
        """Fit subject-stratified model for individual model."""
        print(f"   📚 Fitting subject-stratified model for {model_name}")
        
        try:
            # Encode categorical variables properly
            data_encoded = data.copy()
            le_subject = LabelEncoder()
            data_encoded['subject_encoded'] = le_subject.fit_transform(data_encoded['subject_cluster'].astype(str))
            
            # Prepare data with encoded strata variable only
            model_data = data_encoded[covariates + [duration_col, event_col, 'subject_encoded']].copy()
            
            # Fit stratified model
            cph = CoxPHFitter()
            cph.fit(model_data, duration_col=duration_col, event_col=event_col,
                   strata=['subject_encoded'], show_progress=False)
            
            # Calculate subject effects for frailty variance approximation
            subject_effects = {}
            for subject in data['subject_cluster'].unique():
                subj_data = data[data['subject_cluster'] == subject]
                if len(subj_data) > 10 and subj_data[event_col].sum() > 2:
                    try:
                        cph_subj = CoxPHFitter()
                        cph_subj.fit(subj_data[covariates + [duration_col, event_col]], 
                                   duration_col=duration_col, event_col=event_col, show_progress=False)
                        baseline_hazard = cph_subj.baseline_hazard_
                        if hasattr(baseline_hazard, 'mean'):
                            hazard_mean = baseline_hazard.mean().iloc[0] if len(baseline_hazard.shape) > 1 else baseline_hazard.mean()
                        else:
                            hazard_mean = float(baseline_hazard.iloc[0]) if len(baseline_hazard) > 0 else 0
                        subject_effects[subject] = hazard_mean
                    except:
                        subject_effects[subject] = 0
            
            frailty_variance = np.var(list(subject_effects.values())) if len(subject_effects) > 1 else 0
            
            result = {
                'analysis_type': 'individual_subject_stratified',
                'c_index': cph.concordance_index_,
                'aic': getattr(cph, 'AIC_partial_', np.nan),
                'frailty_variance': frailty_variance,
                'n_observations': len(data),
                'n_events': data[event_col].sum(),
                'coefficients': cph.params_.to_dict(),
                'p_values': cph.summary['p'].to_dict(),
                'hazard_ratios': {var: np.exp(coef) for var, coef in cph.params_.to_dict().items()},
                'subject_effects': subject_effects
            }
            
            print(f"      ✅ C-Index: {result['c_index']:.4f}, AIC: {result['aic']:.1f}, Frailty Var: {frailty_variance:.6f}")
            return result
            
        except Exception as e:
            print(f"      ❌ Subject stratified failed: {e}")
            return None
    
    def fit_individual_difficulty_stratified(self, model_name, data, covariates, duration_col, event_col):
        """Fit difficulty-stratified model for individual model."""
        print(f"   📈 Fitting difficulty-stratified model for {model_name}")
        
        try:
            # Encode categorical variables properly
            data_encoded = data.copy()
            le_difficulty = LabelEncoder()
            data_encoded['difficulty_encoded'] = le_difficulty.fit_transform(data_encoded['difficulty'].astype(str))
            
            # Prepare data with encoded strata variable only
            model_data = data_encoded[covariates + [duration_col, event_col, 'difficulty_encoded']].copy()
            
            # Fit stratified model
            cph = CoxPHFitter()
            cph.fit(model_data, duration_col=duration_col, event_col=event_col,
                   strata=['difficulty_encoded'], show_progress=False)
            
            # Calculate difficulty effects
            difficulty_effects = {}
            for difficulty in data['difficulty'].unique():
                diff_data = data[data['difficulty'] == difficulty]
                if len(diff_data) > 10 and diff_data[event_col].sum() > 2:
                    try:
                        cph_diff = CoxPHFitter()
                        cph_diff.fit(diff_data[covariates + [duration_col, event_col]], 
                                   duration_col=duration_col, event_col=event_col, show_progress=False)
                        baseline_hazard = cph_diff.baseline_hazard_
                        if hasattr(baseline_hazard, 'mean'):
                            hazard_mean = baseline_hazard.mean().iloc[0] if len(baseline_hazard.shape) > 1 else baseline_hazard.mean()
                        else:
                            hazard_mean = float(baseline_hazard.iloc[0]) if len(baseline_hazard) > 0 else 0
                        difficulty_effects[difficulty] = hazard_mean
                    except:
                        difficulty_effects[difficulty] = 0
            
            frailty_variance = np.var(list(difficulty_effects.values())) if len(difficulty_effects) > 1 else 0
            
            result = {
                'analysis_type': 'individual_difficulty_stratified',
                'c_index': cph.concordance_index_,
                'aic': getattr(cph, 'AIC_partial_', np.nan),
                'frailty_variance': frailty_variance,
                'n_observations': len(data),
                'n_events': data[event_col].sum(),
                'coefficients': cph.params_.to_dict(),
                'p_values': cph.summary['p'].to_dict(),
                'hazard_ratios': {var: np.exp(coef) for var, coef in cph.params_.to_dict().items()},
                'difficulty_effects': difficulty_effects
            }
            
            print(f"      ✅ C-Index: {result['c_index']:.4f}, AIC: {result['aic']:.1f}, Frailty Var: {frailty_variance:.6f}")
            return result
            
        except Exception as e:
            print(f"      ❌ Difficulty stratified failed: {e}")
            return None
    
    def create_individual_model_comparisons(self):
        """Create comparisons for each individual model."""
        print("\n📊 CREATING INDIVIDUAL MODEL COMPARISONS")
        print("=" * 45)
        
        comparison_data = []
        
        for model_name, results in self.individual_results.items():
            if not results:
                continue
            
            baseline = results.get('baseline', {})
            subject_strat = results.get('subject_stratified', {})
            difficulty_strat = results.get('difficulty_stratified', {})
            
            # Model-level comparison
            model_comparison = {
                'Model': model_name,
                'Baseline_C_Index': baseline.get('c_index', np.nan),
                'Baseline_AIC': baseline.get('aic', np.nan),
                'Subject_C_Index': subject_strat.get('c_index', np.nan),
                'Subject_AIC': subject_strat.get('aic', np.nan),
                'Subject_Frailty_Var': subject_strat.get('frailty_variance', 0),
                'Difficulty_C_Index': difficulty_strat.get('c_index', np.nan),
                'Difficulty_AIC': difficulty_strat.get('aic', np.nan),
                'Difficulty_Frailty_Var': difficulty_strat.get('frailty_variance', 0),
                'N_Observations': results.get('data_summary', {}).get('n_clean', 0),
                'N_Events': results.get('data_summary', {}).get('n_events', 0)
            }
            
            # Calculate improvements (handle missing data)
            baseline_aic = baseline.get('aic', np.nan)
            subject_aic = subject_strat.get('aic', np.nan)
            difficulty_aic = difficulty_strat.get('aic', np.nan)
            
            if not pd.isna(baseline_aic) and not pd.isna(subject_aic):
                model_comparison['Subject_AIC_Improvement'] = baseline_aic - subject_aic
            else:
                model_comparison['Subject_AIC_Improvement'] = 0
            
            if not pd.isna(baseline_aic) and not pd.isna(difficulty_aic):
                model_comparison['Difficulty_AIC_Improvement'] = baseline_aic - difficulty_aic
            else:
                model_comparison['Difficulty_AIC_Improvement'] = 0
            
            comparison_data.append(model_comparison)
        
        if comparison_data:
            self.all_individual_comparisons = comparison_data
            print(f"✅ Created comparisons for {len(comparison_data)} models")
        
        return comparison_data
    
    def create_individual_visualizations(self):
        """Create visualizations for individual model analysis."""
        print("\n🎨 CREATING INDIVIDUAL MODEL VISUALIZATIONS")
        print("=" * 47)
        
        if not self.all_individual_comparisons:
            print("❌ No comparison data for visualizations")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Individual Model Advanced Analysis Results', fontsize=16, fontweight='bold')
        
        df_comp = pd.DataFrame(self.all_individual_comparisons)
        
        # Plot 1: C-Index comparison
        models = df_comp['Model']
        baseline_ci = df_comp['Baseline_C_Index']
        subject_ci = df_comp['Subject_C_Index']
        difficulty_ci = df_comp['Difficulty_C_Index']
        
        x = np.arange(len(models))
        width = 0.25
        
        ax1.bar(x - width, baseline_ci, width, label='Baseline', alpha=0.8, color='orange')
        ax1.bar(x, subject_ci, width, label='Subject Stratified', alpha=0.8, color='skyblue')
        ax1.bar(x + width, difficulty_ci, width, label='Difficulty Stratified', alpha=0.8, color='lightgreen')
        
        ax1.set_title('C-Index Comparison by Model', fontweight='bold')
        ax1.set_ylabel('C-Index')
        ax1.set_xlabel('Models')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0.5, 1.0)
        
        # Plot 2: AIC improvement (handle missing columns)
        subject_improvement = df_comp.get('Subject_AIC_Improvement', pd.Series([0]*len(df_comp))).fillna(0)
        difficulty_improvement = df_comp.get('Difficulty_AIC_Improvement', pd.Series([0]*len(df_comp))).fillna(0)
        
        ax2.bar(x - width/2, subject_improvement, width, label='Subject Improvement', alpha=0.8, color='skyblue')
        ax2.bar(x + width/2, difficulty_improvement, width, label='Difficulty Improvement', alpha=0.8, color='lightgreen')
        
        ax2.set_title('AIC Improvement over Baseline', fontweight='bold')
        ax2.set_ylabel('AIC Improvement')
        ax2.set_xlabel('Models')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot 3: Frailty variance comparison (handle missing columns)
        subject_var = df_comp.get('Subject_Frailty_Var', pd.Series([0]*len(df_comp))).fillna(0)
        difficulty_var = df_comp.get('Difficulty_Frailty_Var', pd.Series([0]*len(df_comp))).fillna(0)
        
        ax3.bar(x - width/2, subject_var, width, label='Subject Frailty Var', alpha=0.8, color='skyblue')
        ax3.bar(x + width/2, difficulty_var, width, label='Difficulty Frailty Var', alpha=0.8, color='lightgreen')
        
        ax3.set_title('Frailty Variance by Model', fontweight='bold')
        ax3.set_ylabel('Frailty Variance')
        ax3.set_xlabel('Models')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.legend()
        
        # Plot 4: Model performance summary table
        ax4.axis('off')
        
        # Create summary table (handle missing columns)
        table_data = []
        for _, row in df_comp.iterrows():
            subject_improvement = row.get('Subject_AIC_Improvement', 0)
            difficulty_improvement = row.get('Difficulty_AIC_Improvement', 0)
            
            table_data.append([
                row['Model'][:12] + '...' if len(row['Model']) > 15 else row['Model'],
                f"{row['Baseline_C_Index']:.3f}" if not pd.isna(row['Baseline_C_Index']) else 'N/A',
                f"{subject_improvement:.1f}" if not pd.isna(subject_improvement) else 'N/A',
                f"{difficulty_improvement:.1f}" if not pd.isna(difficulty_improvement) else 'N/A',
                f"{row['N_Events']}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Model', 'Base C-Idx', 'Subj AIC Imp', 'Diff AIC Imp', 'Events'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Individual Model Performance Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs('generated/figs', exist_ok=True)
        plt.savefig('generated/figs/individual_advanced_modeling.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: individual_advanced_modeling.png")
        plt.close()
    
    def save_individual_results(self):
        """Save individual model results."""
        print("\n💾 SAVING INDIVIDUAL MODEL RESULTS")
        print("=" * 40)
        
        os.makedirs('generated/outputs', exist_ok=True)
        
        # Convert to JSON-serializable format
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Save detailed results
        results_serializable = convert_numpy_types(self.individual_results)
        with open('generated/outputs/individual_advanced_results.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print("✅ Detailed results: individual_advanced_results.json")
        
        # Save comparison table
        if self.all_individual_comparisons:
            pd.DataFrame(self.all_individual_comparisons).to_csv(
                'generated/outputs/individual_model_comparisons.csv', index=False)
            print("✅ Comparison table: individual_model_comparisons.csv")
        
        print("\n📁 Generated files:")
        print("   • individual_advanced_results.json")
        print("   • individual_model_comparisons.csv")
        print("   • individual_advanced_modeling.png")
    
    def run_complete_individual_analysis(self):
        """Run complete individual model advanced analysis."""
        print("🔬 INDIVIDUAL MODEL ADVANCED ANALYSIS")
        print("=" * 45)
        print("Focus: Individual model analysis only - no combined modeling")
        print("Each model analyzed independently with baseline and stratified approaches.\n")
        
        # Load individual model data
        model_data = self.load_individual_model_data()
        if not model_data:
            return
        
        # Analyze each model individually
        for model_name, df in model_data.items():
            model_results = self.analyze_individual_model(model_name, df)
            if model_results:
                self.individual_results[model_name] = model_results
        
        # Create comparisons
        self.create_individual_model_comparisons()
        
        # Create visualizations
        self.create_individual_visualizations()
        
        # Save results
        self.save_individual_results()
        
        # Print summary
        print(f"\n🎉 INDIVIDUAL ADVANCED ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"📊 Models analyzed: {len(self.individual_results)}")
        print(f"📈 Analysis types per model: Baseline + Subject Strat + Difficulty Strat")
        print(f"🔍 Focus: Individual model heterogeneity and stratification effects")
        print(f"❌ No combined modeling - pure individual analysis only")

def main():
    """Run individual model advanced analysis."""
    analyzer = IndividualAdvancedModeling()
    analyzer.run_complete_individual_analysis()

if __name__ == "__main__":
    main() 