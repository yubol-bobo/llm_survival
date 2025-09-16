#!/usr/bin/env python3
"""
Check drift data patterns across different models
"""
import pandas as pd
import os

def check_drift_availability(model_name):
    file_path = f'data/processed/{model_name}/{model_name}_long.csv'
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name.upper()} - Drift Data Availability")
    print(f"{'='*50}")
    
    df = pd.read_csv(file_path)
    
    for r in range(1, 9):
        round_data = df[df['round'] == r]
        non_null_p2p = round_data['prompt_to_prompt_drift'].notna().sum()
        non_null_c2p = round_data['context_to_prompt_drift'].notna().sum()
        non_null_cum = round_data['cumulative_drift'].notna().sum()
        total = len(round_data)
        
        print(f"Round {r}: p2p={non_null_p2p}/{total} ({non_null_p2p/total*100:.1f}%), "
              f"c2p={non_null_c2p}/{total} ({non_null_c2p/total*100:.1f}%), "
              f"cum={non_null_cum}/{total} ({non_null_cum/total*100:.1f}%)")
    
    # Check a few sample conversations to understand the pattern
    print(f"\n🔍 Sample conversation patterns:")
    sample_convs = df['conversation_id'].unique()[:3]
    for conv_id in sample_convs:
        conv_data = df[df['conversation_id'] == conv_id]
        drift_rounds = conv_data[conv_data['prompt_to_prompt_drift'].notna()]['round'].tolist()
        failure_round = conv_data[conv_data['failure'] == 1]['round'].tolist()
        failure_round = failure_round[0] if failure_round else "No failure"
        print(f"  Conv {conv_id}: drift in rounds {drift_rounds}, failed at round {failure_round}")

# Check multiple models
models_to_check = ['deepseek_r1', 'gemini_25', 'gpt_4o', 'claude_35']

for model in models_to_check:
    check_drift_availability(model)