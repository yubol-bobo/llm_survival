#!/usr/bin/env python3
"""
Check failure patterns to understand why drift stops being calculated
"""

import pandas as pd
import sys
import os
sys.path.append('src')

def check_failure_patterns():
    """Check how failures relate to missing drift data"""
    print("🔍 CHECKING FAILURE PATTERNS")
    print("=" * 60)
    
    from visualization.aft import AFTVisualizer
    
    viz = AFTVisualizer()
    combined_df = viz._load_round_level_data()
    
    if combined_df is None:
        print("❌ No data loaded")
        return
    
    print("📊 Failure patterns by round:")
    for round_num in range(1, 9):
        round_data = combined_df[combined_df['round'] == round_num]
        if len(round_data) > 0:
            total_obs = len(round_data)
            failures = round_data['failure'].sum() if 'failure' in round_data.columns else 0
            failure_rate = failures / total_obs * 100
            
            # Check drift availability
            drift_available = round_data['prompt_to_prompt_drift'].notna().sum() if 'prompt_to_prompt_drift' in round_data.columns else 0
            drift_rate = drift_available / total_obs * 100
            
            print(f"Round {round_num}: {total_obs:,} obs, {failures:,} failures ({failure_rate:.1f}%), {drift_available:,} with drift ({drift_rate:.1f}%)")
    
    print("\n📊 Checking individual conversation survival:")
    
    # Look at individual conversations to see survival patterns
    if 'conversation_id' in combined_df.columns:
        print("Sample conversation survival patterns (first 10 conversations):")
        for conv_id in combined_df['conversation_id'].unique()[:10]:
            conv_data = combined_df[combined_df['conversation_id'] == conv_id].sort_values('round')
            failure_rounds = conv_data[conv_data['failure'] == 1]['round'].tolist()
            last_drift_round = conv_data[conv_data['prompt_to_prompt_drift'].notna()]['round'].max() if not conv_data[conv_data['prompt_to_prompt_drift'].notna()].empty else 0
            
            print(f"  Conv {conv_id}: Failure in rounds {failure_rounds}, Last drift data: Round {last_drift_round}")
    
    print("\n📊 Checking by model:")
    if 'model' in combined_df.columns:
        for model in sorted(combined_df['model'].unique()):
            model_data = combined_df[combined_df['model'] == model]
            print(f"\n{model}:")
            for round_num in range(1, 9):
                round_data = model_data[model_data['round'] == round_num]
                if len(round_data) > 0:
                    total_obs = len(round_data)
                    failures = round_data['failure'].sum() if 'failure' in round_data.columns else 0
                    drift_available = round_data['prompt_to_prompt_drift'].notna().sum() if 'prompt_to_prompt_drift' in round_data.columns else 0
                    
                    print(f"  R{round_num}: {total_obs} obs, {failures} failures, {drift_available} with drift")

if __name__ == "__main__":
    check_failure_patterns()