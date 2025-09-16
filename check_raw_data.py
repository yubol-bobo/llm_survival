#!/usr/bin/env python3
"""
Check raw conversation data to see if the issue is there
"""

import json
import pandas as pd
import os

def check_raw_conversation_data():
    """Check raw JSON conversation data"""
    print("🔍 CHECKING RAW CONVERSATION DATA")
    print("=" * 60)
    
    # Check one model's JSON data
    json_file = "data/processed/claude_35/claude_35.json"
    csv_file = "data/processed/claude_35/claude_35.csv"
    
    if os.path.exists(json_file):
        print(f"📊 Loading {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            conv_data = json.load(f)
        
        print(f"Total conversations: {len(conv_data)}")
        
        # Check first few conversations
        for conv_id in list(conv_data.keys())[:3]:
            conv = conv_data[conv_id]
            print(f"\nConversation {conv_id}:")
            print(f"  Total messages: {len(conv)}")
            
            user_messages = [i for i, msg in enumerate(conv) if msg['role'] == 'user']
            assistant_messages = [i for i, msg in enumerate(conv) if msg['role'] == 'assistant']
            
            print(f"  User messages at positions: {user_messages}")
            print(f"  Assistant messages at positions: {assistant_messages}")
            print(f"  Max user message rounds: {len(user_messages)}")
    
    if os.path.exists(csv_file):
        print(f"\n📊 Loading {csv_file}")
        csv_data = pd.read_csv(csv_file)
        print(f"CSV shape: {csv_data.shape}")
        print(f"CSV columns: {list(csv_data.columns)}")
        
        # Show first few rows
        print("\nFirst 3 rows of CSV:")
        print(csv_data.head(3))

def check_existing_long_files():
    """Check what's actually in the _long.csv files"""
    print("\n🔍 CHECKING EXISTING LONG FILES")
    print("=" * 60)
    
    long_file = "data/processed/claude_35/claude_35_long.csv"
    if os.path.exists(long_file):
        df = pd.read_csv(long_file)
        print(f"Long file shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check one conversation in detail
        first_conv = df['conversation_id'].iloc[0]
        conv_data = df[df['conversation_id'] == first_conv].sort_values('round')
        print(f"\nFirst conversation ({first_conv}) data:")
        print(conv_data[['round', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'failure']].to_string())
        
        # Check another conversation that might have more rounds
        conv_ids = df['conversation_id'].unique()
        if len(conv_ids) > 1:
            second_conv = conv_ids[1]
            conv_data2 = df[df['conversation_id'] == second_conv].sort_values('round')
            print(f"\nSecond conversation ({second_conv}) data:")
            print(conv_data2[['round', 'prompt_to_prompt_drift', 'context_to_prompt_drift', 'cumulative_drift', 'failure']].to_string())

if __name__ == "__main__":
    check_raw_conversation_data()
    check_existing_long_files()