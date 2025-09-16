#!/usr/bin/env python3
"""
Test the array indexing logic specifically
"""

def test_array_indexing():
    """Test array indexing logic"""
    print("🔍 TESTING ARRAY INDEXING LOGIC")
    print("=" * 50)
    
    # Simulate the arrays
    max_followup_round = 8
    
    # Simulated arrays (like in preprocessing)
    prompt_embeddings = [f"emb_round_{i}" for i in range(9)]  # rounds 0-8
    prompt_to_prompt_drifts = [None] + [f"p2p_drift_{i}" for i in range(1, 9)]  # [None, drift1, drift2, ..., drift8]
    context_to_prompt_drifts = [f"c2p_drift_{i}" for i in range(9)]  # [drift0, drift1, ..., drift8]
    cumulative_drifts = [f"cum_drift_{i}" for i in range(9)]  # [cum0, cum1, ..., cum8]
    
    print("📊 ARRAY CONTENTS:")
    print(f"prompt_embeddings: {prompt_embeddings}")
    print(f"prompt_to_prompt_drifts: {prompt_to_prompt_drifts}")
    print(f"context_to_prompt_drifts: {context_to_prompt_drifts}")
    print(f"cumulative_drifts: {cumulative_drifts}")
    
    print(f"\n📊 LONG TABLE GENERATION (rounds 1-8):")
    for i in range(1, max_followup_round + 1):
        p2p = prompt_to_prompt_drifts[i] if i < len(prompt_to_prompt_drifts) else None
        c2p = context_to_prompt_drifts[i] if i < len(context_to_prompt_drifts) else None
        cum = cumulative_drifts[i] if i < len(cumulative_drifts) else None
        
        print(f"   Round {i}: p2p={p2p}, c2p={c2p}, cum={cum}")
    
    print(f"\n📊 STATIC TABLE AVERAGING:")
    # Check the static table calculation
    print("prompt_to_prompt_drifts[1:max_followup_round+1]:")
    slice_p2p = prompt_to_prompt_drifts[1:max_followup_round+1]
    print(f"   Slice [1:9]: {slice_p2p}")
    print(f"   Length: {len(slice_p2p)} (should be 8)")
    
    print("context_to_prompt_drifts[1:max_followup_round+1]:")
    slice_c2p = context_to_prompt_drifts[1:max_followup_round+1]
    print(f"   Slice [1:9]: {slice_c2p}")
    print(f"   Length: {len(slice_c2p)} (should be 8)")
    
    print("\n🔍 ISSUE ANALYSIS:")
    print("The arrays have correct length (9) and indexing looks correct.")
    print("The static table slicing [1:9] gives indices 1,2,3,4,5,6,7,8 - correct!")
    print("This gives us 8 values for rounds 1-8, which is what we want.")

def test_real_preprocessing_output():
    """Test with actual preprocessing output"""
    print("\n🔍 TESTING WITH REAL DATA")
    print("=" * 50)
    
    # Load our test results
    import pandas as pd
    test_df = pd.read_csv('test_fixed_processing.csv')
    
    print("📊 REAL PROCESSING RESULTS:")
    conv_0 = test_df[test_df['conversation_id'] == '0'].sort_values('round')
    print(conv_0[['round', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift']].to_string())
    
    print(f"\n📊 VALIDATION:")
    print(f"All rounds 1-8 have data: {conv_0['prompt_to_prompt_drift'].notna().all()}")
    print(f"Cumulative drift increases: {conv_0['cumulative_drift'].is_monotonic_increasing}")

if __name__ == "__main__":
    test_array_indexing()
    test_real_preprocessing_output()