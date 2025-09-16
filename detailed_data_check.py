#!/usr/bin/env python3
"""
Detailed data check to see what's happening with rounds 5-8
"""

import pandas as pd
import sys
import os
sys.path.append('src')

def check_individual_files():
    """Check individual long.csv files"""
    print("🔍 CHECKING INDIVIDUAL MODEL FILES")
    print("=" * 60)
    
    processed_dir = 'data/processed'
    models = ['claude_35', 'deepseek_r1', 'gemini_25', 'gpt_4o', 'gpt_5']
    
    for model in models:
        file_path = f"{processed_dir}/{model}/{model}_long.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"\n📊 {model}:")
            print(f"   Total rows: {len(df)}")
            print(f"   Rounds: {sorted(df['round'].unique())}")
            
            # Check drift data by round
            drift_cols = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift']
            for col in drift_cols:
                if col in df.columns:
                    non_null_by_round = df.groupby('round')[col].count()
                    print(f"   {col} non-null counts:")
                    for round_num in sorted(df['round'].unique()):
                        count = non_null_by_round.get(round_num, 0)
                        print(f"     Round {round_num}: {count}")
            
            # Show sample data for rounds 5-8
            late_rounds = df[df['round'] >= 5]
            if len(late_rounds) > 0:
                print(f"   Sample from rounds 5-8 (first 3 rows):")
                sample_cols = ['round', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'failure']
                available_cols = [col for col in sample_cols if col in late_rounds.columns]
                print(late_rounds[available_cols].head(3).to_string())
        else:
            print(f"❌ File not found: {file_path}")

def check_combined_data():
    """Check the combined data that the visualization uses"""
    print("\n🔍 CHECKING COMBINED DATA FROM VISUALIZER")
    print("=" * 60)
    
    from visualization.aft import AFTVisualizer
    
    viz = AFTVisualizer()
    combined_df = viz._load_round_level_data()
    
    if combined_df is not None:
        print(f"📊 Combined data shape: {combined_df.shape}")
        print(f"📊 Rounds in combined data: {sorted(combined_df['round'].unique())}")
        print(f"📊 Models in combined data: {sorted(combined_df['model'].unique()) if 'model' in combined_df.columns else 'No model column'}")
        
        # Check specific rounds 5-8
        for round_num in [5, 6, 7, 8]:
            round_data = combined_df[combined_df['round'] == round_num]
            print(f"\n📊 Round {round_num}:")
            print(f"   Total observations: {len(round_data)}")
            
            if len(round_data) > 0:
                # Check drift columns
                drift_cols = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift']
                for col in drift_cols:
                    if col in round_data.columns:
                        non_null = round_data[col].notna().sum()
                        print(f"   {col} non-null: {non_null}")
                
                # Show sample
                print("   Sample data (first 3 rows):")
                sample_cols = ['model', 'round', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'failure']
                available_cols = [col for col in sample_cols if col in round_data.columns]
                print(round_data[available_cols].head(3).to_string())

if __name__ == "__main__":
    check_individual_files()
    check_combined_data()