#!/usr/bin/env python3
"""
Test the fixed preprocessing by processing a small subset
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def cosine_dist(a, b):
    if a is None or b is None:
        return None
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_fixed_processing():
    """Test the fixed preprocessing on a small sample"""
    print("🧪 TESTING FIXED PREPROCESSING")
    print("=" * 50)
    
    # Load embedding model
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load metadata
    meta_df = pd.read_csv('data/raw/cleaned_data_with_clusters.csv')
    
    # Process just one model (claude_35) and first 3 conversations
    model = 'claude_35'
    json_file = f'data/processed/{model}/{model}.json'
    csv_file = f'data/processed/{model}/{model}.csv'
    
    with open(json_file, 'r', encoding='utf-8') as f:
        conv_data = json.load(f)
    
    csv_data = pd.read_csv(csv_file)
    
    # Process first 3 conversations
    long_rows = []
    conv_ids_to_test = ['0', '1', '2']
    
    for conv_id in conv_ids_to_test:
        if conv_id not in conv_data:
            continue
            
        conv = conv_data[conv_id]
        rounds = csv_data.loc[int(conv_id)]
        
        print(f"\n📊 Processing conversation {conv_id}")
        
        # Get metadata
        try:
            meta_row = meta_df.iloc[int(conv_id)]
            level = meta_row['level']
            subject = meta_row['subject']
            subject_cluster = meta_row['subject_cluster']
        except Exception as e:
            print(f"Metadata lookup failed: {e}")
            continue
        
        # Convert round indicators
        round_indicators = []
        for val in rounds:
            if isinstance(val, str) and val.startswith('('):
                indicator = int(val.split(',')[0].replace('(', '').strip())
            elif pd.isna(val) or val == '' or val == 0:
                indicator = 0
            else:
                indicator = int(val)
            round_indicators.append(indicator)
        
        # Skip if round_0 != 1
        if len(round_indicators) == 0 or round_indicators[0] != 1:
            continue
        
        max_followup_round = 8
        
        # Calculate time to failure
        time_to_failure = None
        censored = 0
        for i in range(1, max_followup_round + 1):
            if i >= len(round_indicators) or round_indicators[i] == 0:
                time_to_failure = i if i <= max_followup_round else max_followup_round
                break
        if time_to_failure is None:
            time_to_failure = max_followup_round
            censored = 1
        
        # FIXED: Process embeddings correctly
        prompt_embeddings = []
        context_embeddings = []
        prompt_complexities = []
        
        user_round = 0
        for i, msg in enumerate(conv):
            if msg['role'] == 'user':
                if user_round <= max_followup_round:
                    prompt = msg['content']
                    prompt_embeddings.append(model_embed.encode(prompt, show_progress_bar=False))
                    prompt_complexities.append(len(prompt.split()))
                    
                    # Context calculation
                    context = ''
                    for m in conv[:i]:
                        context += m['content'] + ' '
                    context += prompt
                    context_embeddings.append(model_embed.encode(context, show_progress_bar=False))
                user_round += 1
        
        print(f"  Generated {len(prompt_embeddings)} embeddings (should be 9 for rounds 0-8)")
        
        # Calculate drifts
        prompt_to_prompt_drifts = [None]
        for i in range(1, len(prompt_embeddings)):
            prompt_to_prompt_drifts.append(cosine_dist(prompt_embeddings[i-1], prompt_embeddings[i]))
        
        context_to_prompt_drifts = []
        for i in range(len(prompt_embeddings)):
            context_to_prompt_drifts.append(cosine_dist(context_embeddings[i], prompt_embeddings[i]))
        
        cumulative_drifts = []
        cum = 0
        for d in prompt_to_prompt_drifts:
            if d is not None:
                cum += d
            cumulative_drifts.append(cum)
        
        # Generate long table rows for rounds 1-8
        for i in range(1, max_followup_round + 1):
            if i >= len(round_indicators):
                break
            
            label = round_indicators[i]
            failure = 1 if (label == 0 and censored == 0 and i == time_to_failure) else 0
            
            # Get drift values (now should have data for all rounds)
            p2p_drift = prompt_to_prompt_drifts[i] if i < len(prompt_to_prompt_drifts) else None
            c2p_drift = context_to_prompt_drifts[i] if i < len(context_to_prompt_drifts) else None
            cum_drift = cumulative_drifts[i] if i < len(cumulative_drifts) else None
            complexity = prompt_complexities[i] if i < len(prompt_complexities) else None
            
            long_rows.append({
                'conversation_id': conv_id,
                'model': model,
                'round': i,
                'prompt_to_prompt_drift': p2p_drift,
                'context_to_prompt_drift': c2p_drift,
                'cumulative_drift': cum_drift,
                'prompt_complexity': complexity,
                'failure': failure,
                'censored': 1 if (i == max_followup_round and censored == 1) else 0,
                'level': level,
                'subject': subject,
                'subject_cluster': subject_cluster,
            })
    
    # Create DataFrame and show results
    test_df = pd.DataFrame(long_rows)
    
    print(f"\n✅ FIXED PROCESSING RESULTS:")
    print(f"📊 Total rows generated: {len(test_df)}")
    print(f"📊 Rounds covered: {sorted(test_df['round'].unique())}")
    
    # Check drift availability by round
    print(f"\n📊 Drift data availability by round:")
    for round_num in sorted(test_df['round'].unique()):
        round_data = test_df[test_df['round'] == round_num]
        non_null_drifts = round_data['prompt_to_prompt_drift'].notna().sum()
        print(f"  Round {round_num}: {non_null_drifts}/{len(round_data)} conversations have drift data")
    
    # Show sample data
    print(f"\n📊 Sample of first conversation:")
    conv_0_data = test_df[test_df['conversation_id'] == '0'][['round', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'failure']]
    print(conv_0_data.to_string())
    
    # Save test results
    test_df.to_csv('test_fixed_processing.csv', index=False)
    print(f"\n💾 Test results saved to test_fixed_processing.csv")

if __name__ == "__main__":
    test_fixed_processing()