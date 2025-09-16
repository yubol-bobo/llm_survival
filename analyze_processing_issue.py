#!/usr/bin/env python3
"""
Analyze the processing issue by checking the actual data generation
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_dist(a, b):
    if a is None or b is None:
        return None
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def debug_single_conversation():
    """Debug processing for a single conversation"""
    print("🔍 DEBUGGING SINGLE CONVERSATION PROCESSING")
    print("=" * 60)
    
    # Load one conversation from Claude 35
    json_file = "data/processed/claude_35/claude_35.json"
    csv_file = "data/processed/claude_35/claude_35.csv"
    
    with open(json_file, 'r', encoding='utf-8') as f:
        conv_data = json.load(f)
    
    csv_data = pd.read_csv(csv_file)
    
    # Take first conversation
    conv_id = '0'
    conv = conv_data[conv_id]
    rounds = csv_data.loc[int(conv_id)]
    
    print(f"Conversation {conv_id}:")
    print(f"  Total messages: {len(conv)}")
    print(f"  Round indicators: {list(rounds)}")
    
    # Extract user messages
    user_messages = []
    for i, msg in enumerate(conv):
        if msg['role'] == 'user':
            user_messages.append((i, msg['content']))
    
    print(f"  User messages at positions: {[pos for pos, _ in user_messages]}")
    print(f"  Total user messages: {len(user_messages)}")
    
    # Check what round indicators say
    round_indicators = []
    for val in rounds:
        if isinstance(val, str) and val.startswith('('):
            indicator = int(val.split(',')[0].replace('(', '').strip())
        elif pd.isna(val) or val == '' or val == 0:
            indicator = 0
        else:
            indicator = int(val)
        round_indicators.append(indicator)
    
    print(f"  Processed round indicators: {round_indicators}")
    
    # Simulate embedding calculation for all rounds
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    print("\n📊 Simulating drift calculation:")
    
    max_followup_round = 8
    prompt_embeddings = []
    context_embeddings = []
    
    # Process each user message (rounds 0-8) - FIXED VERSION
    user_round = 0
    for i, msg in enumerate(conv):
        if msg['role'] == 'user':
            if user_round <= max_followup_round:
                prompt = msg['content']
                print(f"  Round {user_round}: Processing user message at position {i}")
                
                # Generate embeddings
                prompt_emb = model_embed.encode(prompt, show_progress_bar=False)
                prompt_embeddings.append(prompt_emb)
                
                # Context calculation
                context = ''
                for m in conv[:i]:
                    context += m['content'] + ' '
                context += prompt
                context_emb = model_embed.encode(context, show_progress_bar=False)
                context_embeddings.append(context_emb)
            user_round += 1
    
    print(f"  Total prompt embeddings generated: {len(prompt_embeddings)}")
    
    # Calculate drifts
    prompt_to_prompt_drifts = [None]
    for i in range(1, len(prompt_embeddings)):
        drift = cosine_dist(prompt_embeddings[i-1], prompt_embeddings[i])
        prompt_to_prompt_drifts.append(drift)
        print(f"  Prompt-to-prompt drift round {i}: {drift}")
    
    # Check why existing data stops at round 4
    print(f"\n📊 Expected vs Actual results:")
    print(f"  Expected rounds with drift: 1-8")
    print(f"  Actual rounds with drift in file: 1-4")
    
    # Check if the issue is in the loop condition
    print(f"\n🔍 Checking loop conditions:")
    print(f"  max_followup_round = {max_followup_round}")
    print(f"  range(1, max_followup_round + 1) = {list(range(1, max_followup_round + 1))}")
    
    # Check conversation length vs available data
    print(f"\n🔍 Data availability check:")
    for i in range(1, 9):
        if i < len(round_indicators):
            success = round_indicators[i]
            has_message = i * 2 < len(conv)  # User messages are at even positions
            print(f"  Round {i}: success={success}, has_message={has_message}")

if __name__ == "__main__":
    debug_single_conversation()