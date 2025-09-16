#!/usr/bin/env python3
"""
Debug script to check round distribution
"""

import sys
import os
sys.path.append('src')

from visualization.aft import AFTVisualizer

def main():
    """Debug round distribution"""
    print("🔍 DEBUGGING ROUND DISTRIBUTION")
    print("=" * 50)
    
    # Create visualizer and load data
    viz = AFTVisualizer()
    combined_df = viz._load_round_level_data()
    
    if combined_df is None:
        print("❌ No data loaded")
        return
    
    print(f"📊 Total observations: {len(combined_df)}")
    print(f"📊 Round range: {combined_df['round'].min()} to {combined_df['round'].max()}")
    print()
    
    print("Round distribution in raw data:")
    print(combined_df['round'].value_counts().sort_index())
    print()
    
    # Check drift features availability by round
    drift_features = ['prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'prompt_complexity']
    
    for feature in drift_features:
        if feature in combined_df.columns:
            print(f"{feature} non-null counts by round:")
            round_counts = combined_df.groupby('round')[feature].count()
            print(round_counts)
            print()
            
    # Check what happens after dropna
    clean_df = combined_df.dropna(subset=drift_features)
    print(f"After removing NaN values:")
    print(f"📊 Clean observations: {len(clean_df)}")
    print(f"📊 Rounds remaining: {sorted(clean_df['round'].unique())}")
    print("Round distribution after cleaning:")
    print(clean_df['round'].value_counts().sort_index())

if __name__ == "__main__":
    main()