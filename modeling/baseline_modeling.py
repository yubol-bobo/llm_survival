#!/usr/bin/env python3
"""
Baseline Modeling for LLM Robustness Analysis Pipeline
-----------------------------------------------------
This script performs baseline count (Negative Binomial) and Cox PH survival modeling
for each LLM, using subject clusters and difficulty levels. It is designed to be run
as the baseline step in the analysis pipeline, before advanced modeling.

- Always uses Negative Binomial regression for count models (no user input required)
- Loads processed data from processed_data/<model>/
- Enhances data with deterministic subject cluster and difficulty level assignment
- Outputs results to generated/outputs/ and generated/figs/
- Can be run standalone or imported as a module
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import CoxPHFitter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set beautiful plot style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'

def create_subject_clusters():
    """Create 7 meaningful subject clusters from 39 individual subjects."""
    subject_clusters = {
        'STEM': [
            'mathematics', 'statistics', 'abstract_algebra',
            'physics', 'conceptual_physics', 'astronomy', 
            'chemistry', 'computer_science', 'computer_security', 
            'machine_learning', 'electrical_engineering'
        ],
        'Medical_Health': [
            'medicine', 'clinical_knowledge', 'medical_genetics',
            'biology', 'anatomy', 'virology',
            'nutrition', 'human_sexuality'
        ],
        'Social_Sciences': [
            'psychology', 'sociology',
            'moral_scenarios', 'global_facts'
        ],
        'Humanities': [
            'philosophy', 'formal_logic', 'world_religions',
            'world_history', 'us_history', 'prehistory'
        ],
        'Business_Economics': [
            'microeconomics', 'econometrics',
            'accounting', 'marketing', 'management'
        ],
        'Law_Legal': [
            'law', 'jurisprudence', 'international_law'
        ],
        'General_Knowledge': [
            'truthful', 'common sense'
        ]
    }
    return subject_clusters

def map_subject_to_cluster(subject, subject_clusters):
    """Map individual subject to cluster."""
    for cluster_name, subjects in subject_clusters.items():
        if subject in subjects:
            return cluster_name
    return 'Other'

def process_subject_clustering():
    """Process and create subject clusters deterministically."""
    print("🏷️ CREATING SUBJECT CLUSTERS")
    print("=" * 50)
    try:
        cleaned_data = pd.read_csv('./raw data/cleaned_data - cleaned_data.csv')
        print(f"✅ Loaded {len(cleaned_data)} questions")
        subject_clusters = create_subject_clusters()
        cleaned_data['subject_cluster'] = cleaned_data['subject'].apply(
            lambda x: map_subject_to_cluster(x, subject_clusters)
        )
        cluster_counts = cleaned_data['subject_cluster'].value_counts()
        print("\n📊 Cluster distribution:")
        for cluster, count in cluster_counts.items():
            percentage = count / len(cleaned_data) * 100
            print(f"   • {cluster}: {count} questions ({percentage:.1f}%)")
        cleaned_data.to_csv('./raw data/cleaned_data_with_clusters.csv', index=False)
        print("✅ Subject clustering complete!")
        return cleaned_data, subject_clusters
    except Exception as e:
        print(f"❌ Error in subject clustering: {e}")
        return None, None

class LLMRobustnessAnalyzer:
    def __init__(self):
        self.models_data = {}
        self.count_results = {}
        self.survival_results = {}
        self.load_existing_data()
    def load_existing_data(self):
        print("\n🔍 LOADING EXISTING DATA")
        print("=" * 50)
        processed_dir = './processed_data'
        if not os.path.exists(processed_dir):
            print("❌ No processed_data directory found!")
            return
        model_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
        for model_name in model_dirs:
            static_file = os.path.join(processed_dir, model_name, f'{model_name}_static.csv')
            long_file = os.path.join(processed_dir, model_name, f'{model_name}_long.csv')
            if os.path.exists(static_file) and os.path.exists(long_file):
                try:
                    static_df = pd.read_csv(static_file)
                    long_df = pd.read_csv(long_file)
                    self.models_data[model_name] = {
                        'static': static_df,
                        'long': long_df
                    }
                    print(f"✅ Loaded {model_name}: {len(static_df)} conversations, {len(long_df)} turns")
                except Exception as e:
                    print(f"⚠️ Error loading {model_name}: {e}")
        print(f"📊 Total models loaded: {len(self.models_data)}")
    def enhance_with_clusters(self, cleaned_data_with_clusters):
        print("\n🔄 ENHANCING DATA WITH CLUSTERS")
        print("=" * 50)
        # Build lookup for question → (subject_cluster, difficulty_level)
        question_lookup = {row['question'].strip(): {
            'subject_cluster': row['subject_cluster'],
            'difficulty_level': row['level']
        } for _, row in cleaned_data_with_clusters.iterrows()}
        for model_name in self.models_data.keys():
            static_df = self.models_data[model_name]['static']
            long_df = self.models_data[model_name]['long']
            # Deterministically assign clusters/levels based on question text
            if 'question' in static_df.columns:
                static_df['subject_cluster'] = static_df['question'].map(
                    lambda q: question_lookup.get(str(q).strip(), {}).get('subject_cluster', 'Other'))
                static_df['difficulty_level'] = static_df['question'].map(
                    lambda q: question_lookup.get(str(q).strip(), {}).get('difficulty_level', 'Unknown'))
            if 'question' in long_df.columns:
                long_df['subject_cluster'] = long_df['question'].map(
                    lambda q: question_lookup.get(str(q).strip(), {}).get('subject_cluster', 'Other'))
                long_df['difficulty_level'] = long_df['question'].map(
                    lambda q: question_lookup.get(str(q).strip(), {}).get('difficulty_level', 'Unknown'))
            self.models_data[model_name]['static'] = static_df
            self.models_data[model_name]['long'] = long_df
            print(f"✅ Enhanced {model_name} with cluster information")
    def fit_count_model(self, static_df, model_name):
        try:
            potential_cols = ['time_to_failure', 'avg_prompt_to_prompt_drift', 'avg_context_to_prompt_drift', 
                            'avg_prompt_complexity', 'subject_cluster', 'difficulty_level']
            available_cols = []
            for col in potential_cols:
                if col in static_df.columns:
                    if col in ['avg_prompt_to_prompt_drift', 'avg_context_to_prompt_drift']:
                        if not static_df[col].isna().all():
                            available_cols.append(col)
                    else:
                        available_cols.append(col)
            if 'time_to_failure' not in available_cols:
                return None, f"Missing required column: time_to_failure"
            df_clean = static_df[available_cols].dropna()
            if len(df_clean) < 10:
                return None, f"Insufficient data ({len(df_clean)} observations)"
            formula_parts = []
            if 'avg_prompt_to_prompt_drift' in df_clean.columns:
                formula_parts.append('avg_prompt_to_prompt_drift')
            if 'avg_context_to_prompt_drift' in df_clean.columns:
                formula_parts.append('avg_context_to_prompt_drift')
            if 'avg_prompt_complexity' in df_clean.columns:
                formula_parts.append('avg_prompt_complexity')
            if 'subject_cluster' in df_clean.columns:
                formula_parts.append('C(subject_cluster)')
            if 'difficulty_level' in df_clean.columns:
                formula_parts.append('C(difficulty_level)')
            formula = f"time_to_failure ~ {' + '.join(formula_parts)}"
            # Always use Negative Binomial
            model = smf.glm(formula=formula, data=df_clean, family=sm.families.NegativeBinomial()).fit()
            return model, None
        except Exception as e:
            return None, f"Error: {str(e)}"
    def fit_survival_model(self, long_df, model_name):
        try:
            drift_covariates = [col for col in ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'prompt_complexity'] if col in long_df.columns]
            required_cols = drift_covariates + ['round', 'failure']
            available_cols = [col for col in required_cols if col in long_df.columns]
            if len(available_cols) < 3:
                return None, f"Missing required columns"
            df_clean = long_df[available_cols].dropna()
            if len(df_clean) < 10 or df_clean['failure'].sum() < 2:
                return None, f"Insufficient events"
            cph = CoxPHFitter()
            model_df = df_clean[drift_covariates + ['round', 'failure']]
            cph.fit(model_df, duration_col='round', event_col='failure', show_progress=False)
            return cph, None
        except Exception as e:
            return None, f"Error: {str(e)}"
    def analyze_all_models(self):
        print(f"\n🔍 ANALYZING MODELS (NEGATIVE BINOMIAL + COX PH)")
        print("=" * 60)
        count_summary = []
        survival_summary = []
        for model_name in tqdm(self.models_data.keys(), desc="Analyzing models"):
            static_df = self.models_data[model_name]['static']
            long_df = self.models_data[model_name]['long']
            count_model, count_error = self.fit_count_model(static_df, model_name)
            if count_model is not None:
                self.count_results[model_name] = count_model
                count_summary.append({
                    'Model': model_name,
                    'N_conversations': len(static_df),
                    'AIC': count_model.aic,
                    'Drift_coef': count_model.params.get('avg_prompt_to_prompt_drift', np.nan),
                    'Drift_pval': count_model.pvalues.get('avg_prompt_to_prompt_drift', np.nan)
                })
            else:
                print(f"❌ Count model failed for {model_name}: {count_error}")
            survival_model, survival_error = self.fit_survival_model(long_df, model_name)
            if survival_model is not None:
                self.survival_results[model_name] = survival_model
                try:
                    c_index = survival_model.concordance_index_
                except:
                    c_index = np.nan
                survival_summary.append({
                    'Model': model_name,
                    'N_turns': len(long_df),
                    'N_failures': long_df['failure'].sum(),
                    'C_index': c_index
                })
            else:
                print(f"❌ Survival model failed for {model_name}: {survival_error}")
        self.count_summary = pd.DataFrame(count_summary)
        self.survival_summary = pd.DataFrame(survival_summary)
        return self.count_summary, self.survival_summary
    def create_beautiful_visualizations(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import os
        # Ensure output directory exists
        os.makedirs('generated/figs', exist_ok=True)
        # Load the CSVs
        model_df = pd.read_csv('generated/outputs/model_analysis_results.csv')
        surv_df = pd.read_csv('generated/outputs/survival_analysis_results.csv')
        # Merge for joint plots
        merged = pd.merge(model_df, surv_df, on='Model')
        # Set style
        sns.set(style="whitegrid", palette="husl", font_scale=1.2)
        # 1. Bar plot: C-index by model
        plt.figure(figsize=(10,6))
        order = merged.sort_values('C_index', ascending=False)['Model']
        sns.barplot(x='C_index', y='Model', data=merged, order=order, palette='viridis')
        plt.title('C-index by Model (Survival Discrimination)')
        plt.xlabel('C-index')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig('generated/figs/llm_cindex_by_model.png', dpi=300)
        plt.close()
        # 2. Bar plot: Drift_coef by model, color by p-value significance
        plt.figure(figsize=(10,6))
        # Add a significance column for coloring
        def pval_to_sig(p):
            if p < 0.05:
                return 'p < 0.05'
            elif p < 0.1:
                return 'p < 0.1'
            else:
                return 'n.s.'
        merged['Significance'] = merged['Drift_pval'].apply(pval_to_sig)
        sns.barplot(x='Drift_coef', y='Model', data=merged, order=order, hue='Significance', dodge=False, palette={'p < 0.05':'#d62728', 'p < 0.1':'#ff7f0e', 'n.s.':'#1f77b4'})
        plt.title('Drift Coefficient by Model (Negative Binomial)')
        plt.xlabel('Drift Coefficient')
        plt.ylabel('Model')
        plt.legend(title='Drift p-value', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('generated/figs/llm_drift_coef_by_model.png', dpi=300)
        plt.close()
        # 3. Bar plot: AIC by model
        plt.figure(figsize=(10,6))
        order_aic = merged.sort_values('AIC')['Model']
        sns.barplot(x='AIC', y='Model', data=merged, order=order_aic, palette='mako')
        plt.title('AIC by Model (Lower is Better)')
        plt.xlabel('AIC')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig('generated/figs/llm_aic_by_model.png', dpi=300)
        plt.close()
        # 4. Scatter plot: Drift_coef vs C-index
        plt.figure(figsize=(8,6))
        sns.scatterplot(x='Drift_coef', y='C_index', hue='Model', data=merged, s=120, palette='tab10')
        plt.title('Drift Coefficient vs C-index')
        plt.xlabel('Drift Coefficient (NegBin)')
        plt.ylabel('C-index (Survival)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('generated/figs/llm_drift_vs_cindex.png', dpi=300)
        plt.close()
        # 5. Scatter plot: AIC vs C-index
        plt.figure(figsize=(8,6))
        sns.scatterplot(x='AIC', y='C_index', hue='Model', data=merged, s=120, palette='tab20')
        plt.title('AIC vs C-index by Model')
        plt.xlabel('AIC (NegBin)')
        plt.ylabel('C-index (Survival)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('generated/figs/llm_aic_vs_cindex.png', dpi=300)
        plt.close()
    def generate_report(self):
        # ... existing code ...
        pass  # (No change needed here, keep as is)

def main():
    print("🚀 BASELINE MODELING: LLM ROBUSTNESS ANALYSIS")
    print("="*60)
    os.makedirs('./generated/outputs', exist_ok=True)
    os.makedirs('./generated/figs', exist_ok=True)
    cleaned_data, subject_clusters = process_subject_clustering()
    if cleaned_data is None:
        print("❌ Subject clustering failed!")
        return
    analyzer = LLMRobustnessAnalyzer()
    if not analyzer.models_data:
        print("❌ No model data loaded!")
        return
    analyzer.enhance_with_clusters(cleaned_data)
    print(f"\n🔍 Running baseline analysis with NEGATIVE BINOMIAL + Cox PH...")
    count_summary, survival_summary = analyzer.analyze_all_models()
    analyzer.generate_report()
    analyzer.create_beautiful_visualizations()
    count_summary.to_csv('./generated/outputs/model_analysis_results.csv', index=False)
    survival_summary.to_csv('./generated/outputs/survival_analysis_results.csv', index=False)
    print(f"\n✅ BASELINE ANALYSIS COMPLETE!")
    print(f"📁 Results saved:")
    print(f"   • generated/outputs/model_analysis_results.csv")
    print(f"   • generated/outputs/survival_analysis_results.csv")
    print(f"   • generated/figs/llm_robustness_analysis.png")
    return analyzer
if __name__ == '__main__':
    analyzer = main() 