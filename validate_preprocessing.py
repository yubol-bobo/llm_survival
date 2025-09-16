#!/usr/bin/env python3
"""
Comprehensive validation of the preprocessing logic
"""

import json
import pandas as pd
import numpy as np

def validate_preprocessing_logic():
    """Validate the preprocessing logic step by step"""
    print("🔍 COMPREHENSIVE PREPROCESSING VALIDATION")
    print("=" * 60)
    
    # Load test data
    json_file = "data/processed/claude_35/claude_35.json"
    csv_file = "data/processed/claude_35/claude_35.csv"
    
    with open(json_file, 'r', encoding='utf-8') as f:
        conv_data = json.load(f)
    
    csv_data = pd.read_csv(csv_file)
    
    # Test with first few conversations
    test_conv_ids = ['0', '1', '2']
    
    for conv_id in test_conv_ids:
        print(f"\n📊 VALIDATING CONVERSATION {conv_id}")
        print("-" * 40)
        
        conv = conv_data[conv_id]
        rounds = csv_data.loc[int(conv_id)]
        
        print(f"Total messages: {len(conv)}")
        
        # Check message structure
        user_positions = []
        assistant_positions = []
        for i, msg in enumerate(conv):
            if msg['role'] == 'user':
                user_positions.append(i)
            elif msg['role'] == 'assistant':
                assistant_positions.append(i)
        
        print(f"User message positions: {user_positions}")
        print(f"Assistant message positions: {assistant_positions}")
        
        # Validate round indicators
        round_indicators = []
        for val in rounds:
            if isinstance(val, str) and val.startswith('('):
                indicator = int(val.split(',')[0].replace('(', '').strip())
            elif pd.isna(val) or val == '' or val == 0:
                indicator = 0
            else:
                indicator = int(val)
            round_indicators.append(indicator)
        
        print(f"Round indicators (round_0 to round_8): {round_indicators}")
        
        # Check key assumptions
        print(f"\n🔍 VALIDATION CHECKS:")
        
        # 1. Message alternation check
        alternates_correctly = True
        for i in range(0, len(conv)-1, 2):
            if i < len(conv) and conv[i]['role'] != 'user':
                alternates_correctly = False
                break
            if i+1 < len(conv) and conv[i+1]['role'] != 'assistant':
                alternates_correctly = False
                break
        
        print(f"✅ Messages alternate user/assistant: {alternates_correctly}")
        
        # 2. Expected number of user messages (should be 9 for rounds 0-8)
        expected_user_messages = 9
        actual_user_messages = len(user_positions)
        print(f"✅ User messages count: {actual_user_messages} (expected: {expected_user_messages})")
        
        # 3. Round mapping validation
        print(f"✅ Round mapping validation:")
        for round_num in range(9):  # rounds 0-8
            expected_position = round_num * 2  # should be at even positions
            if round_num < len(user_positions):
                actual_position = user_positions[round_num]
                print(f"   Round {round_num}: pos {actual_position} (expected: {expected_position}) {'✅' if actual_position == expected_position else '❌'}")
            else:
                print(f"   Round {round_num}: MISSING")
        
        # 4. Time to failure calculation validation
        max_followup_round = 8
        time_to_failure = None
        censored = 0
        
        for i in range(1, max_followup_round + 1):
            if i >= len(round_indicators) or round_indicators[i] == 0:
                time_to_failure = i if i <= max_followup_round else max_followup_round
                break
        if time_to_failure is None:
            time_to_failure = max_followup_round
            censored = 1
        
        print(f"✅ Time to failure: {time_to_failure}, Censored: {censored}")
        
        # 5. Embedding collection simulation
        print(f"✅ Embedding collection simulation:")
        collected_rounds = []
        user_round = 0
        for i, msg in enumerate(conv):
            if msg['role'] == 'user':
                if user_round <= max_followup_round:
                    collected_rounds.append(user_round)
                user_round += 1
        
        print(f"   Rounds with embeddings: {collected_rounds}")
        print(f"   Total embeddings: {len(collected_rounds)} (expected: 9)")
        
        # 6. Drift array lengths validation
        expected_lengths = {
            'prompt_to_prompt_drifts': 9,  # [None, drift1, drift2, ..., drift8]
            'context_to_prompt_drifts': 9,  # [drift0, drift1, ..., drift8]
            'cumulative_drifts': 9  # [cum0, cum1, ..., cum8]
        }
        
        print(f"✅ Expected array lengths: {expected_lengths}")
        
        # 7. Long table row generation validation
        print(f"✅ Long table validation:")
        for round_num in range(1, max_followup_round + 1):
            if round_num < len(round_indicators):
                label = round_indicators[round_num]
                failure = 1 if (label == 0 and censored == 0 and round_num == time_to_failure) else 0
                
                # Check if we have drift data for this round
                has_p2p_drift = round_num < len(collected_rounds)
                has_c2p_drift = round_num < len(collected_rounds)
                has_cum_drift = round_num < len(collected_rounds)
                
                print(f"   Round {round_num}: label={label}, failure={failure}, has_drift={has_p2p_drift}")

def identify_issues():
    """Identify potential issues in the current logic"""
    print(f"\n🚨 POTENTIAL ISSUES IDENTIFIED:")
    print("=" * 60)
    
    issues = [
        "1. ARRAY INDEXING MISMATCH:",
        "   - prompt_embeddings[0] = round 0 prompt",
        "   - prompt_embeddings[1] = round 1 prompt",
        "   - But long table expects drift for round 1 at index 1",
        "   - prompt_to_prompt_drifts[1] = drift between round 0 and 1",
        "",
        "2. STATIC TABLE DRIFT CALCULATION:",
        "   - Uses prompt_to_prompt_drifts[1:max_followup_round+1]",
        "   - Should be [1:9] = indices 1,2,3,4,5,6,7,8",
        "   - But array only has indices 0,1,2,3,4,5,6,7,8",
        "   - Index 8 contains round 8 drift, but we need 8 elements",
        "",
        "3. CONTEXT DRIFT INDEXING:",
        "   - context_to_prompt_drifts[0] = round 0 context vs prompt",
        "   - context_to_prompt_drifts[1] = round 1 context vs prompt",
        "   - This seems correct",
        "",
        "4. CUMULATIVE DRIFT CALCULATION:",
        "   - cumulative_drifts[0] = 0 (no prior drift)",
        "   - cumulative_drifts[1] = prompt_to_prompt_drifts[1]",
        "   - This seems correct",
    ]
    
    for issue in issues:
        print(issue)

if __name__ == "__main__":
    validate_preprocessing_logic()
    identify_issues()