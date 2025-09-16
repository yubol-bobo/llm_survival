import os
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Load embedding model
model_embed = SentenceTransformer(EMBEDDING_MODEL)

# Load question metadata for subject/level info
meta_df = pd.read_csv('data/raw/cleaned_data_with_clusters.csv')

def cosine_dist(a, b):
    if a is None or b is None:
        return None
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Process all models found in data/raw
print("🚀 Processing ALL models from data/raw directory")
print("=" * 60)

# Get all model directories from data/raw (filter out non-model directories)
all_dirs = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
# Filter to only include directories that don't start with 'cleaned_data'
raw_models = [d for d in all_dirs if not d.startswith('cleaned_data')]
print(f"🔍 Found model directories in {RAW_DATA_DIR}: {raw_models}")

for model in raw_models:
    print(f"\n📊 Processing model: {model}")
    model_path = os.path.join(RAW_DATA_DIR, model)
    if not os.path.isdir(model_path):
        print(f"⚠️ Directory not found: {model_path}")
        continue
    out_dir = os.path.join(PROCESSED_DATA_DIR, model)
    os.makedirs(out_dir, exist_ok=True)

    # Merge CSVs
    csv_files = sorted(glob.glob(os.path.join(model_path, '*.csv')))
    if csv_files:
        dfs = [pd.read_csv(f) for f in csv_files]
        merged_csv = pd.concat(dfs, ignore_index=True)
        merged_csv.to_csv(os.path.join(out_dir, f'{model}.csv'), index=False)
    else:
        continue

    # Merge JSONs
    json_files = sorted(glob.glob(os.path.join(model_path, '*.json')))
    merged_json = {}
    idx = 0
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for k in sorted(data.keys(), key=lambda x: int(x)):
                merged_json[str(idx)] = data[k]
                idx += 1
    if merged_json:
        with open(os.path.join(out_dir, f'{model}.json'), 'w', encoding='utf-8') as f:
            json.dump(merged_json, f, indent=2)
    else:
        continue

    # --- Generate static and long tables with embeddings and drift ---
    static_rows = []
    long_rows = []
    merged_csv = pd.read_csv(os.path.join(out_dir, f'{model}.csv'))
    with open(os.path.join(out_dir, f'{model}.json'), 'r', encoding='utf-8') as f:
        merged_json = json.load(f)

    for conv_id, conv in tqdm(merged_json.items(), desc=f'Processing {model}'):
        # Extract Prompt0 and all prompts/responses
        prompt0 = None
        all_user_msgs = []
        all_assistant_msgs = []
        for msg in conv:
            if msg['role'] == 'user':
                if prompt0 is None:
                    prompt0 = msg['content']
                all_user_msgs.append(msg['content'])
            elif msg['role'] == 'assistant':
                all_assistant_msgs.append(msg['content'])
        prompt0_id = hash(prompt0) if prompt0 else None
        model_name = model
        # --- Merge subject/difficulty info by row index ---
        try:
            meta_row = meta_df.iloc[int(conv_id)]
            level = meta_row['level']
            subject = meta_row['subject']
            subject_cluster = meta_row['subject_cluster']
        except Exception as e:
            print(f"Metadata lookup failed for conv_id {conv_id}: {e}")
            continue
        rounds = merged_csv.loc[int(conv_id)]
        # Convert all round values to indicator (0 or 1)
        round_indicators = []
        for val in rounds:
            if isinstance(val, str) and val.startswith('('):
                indicator = int(val.split(',')[0].replace('(', '').strip())
            elif pd.isna(val) or val == '' or val == 0:
                indicator = 0
            else:
                indicator = int(val)
            round_indicators.append(indicator)
        # Only keep conversations where round_0 == 1
        if len(round_indicators) == 0 or round_indicators[0] != 1:
            continue
        # Only keep rounds 1 to 8 (follow-up rounds)
        max_followup_round = 8
        # For static table, we care about time to failure in rounds 1-8
        time_to_failure = None
        censored = 0
        for i in range(1, max_followup_round + 1):
            if i >= len(round_indicators) or round_indicators[i] == 0:
                time_to_failure = i if i <= max_followup_round else max_followup_round
                break
        if time_to_failure is None:
            time_to_failure = max_followup_round
            censored = 1
        prompt_embeddings = []
        context_embeddings = []
        prompt_complexities = []
        context_texts = []
        # Prepare for drift calculations (for rounds 0 to 8)
        user_round = 0
        for i, msg in enumerate(conv):
            if msg['role'] == 'user':
                if user_round <= max_followup_round:
                    prompt = msg['content']
                    prompt_embeddings.append(model_embed.encode(prompt, show_progress_bar=False))
                    prompt_complexities.append(len(prompt.split()))
                    # Context is prompt0 + all prior prompts & responses up to this round
                    context = ''
                    for m in conv[:i]:
                        context += m['content'] + ' '
                    context += prompt
                    context_texts.append(context)
                    context_embeddings.append(model_embed.encode(context, show_progress_bar=False))
                user_round += 1
        # Calculate drifts (for rounds 0 to 8)
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
        # Static table row (only for conversations that passed round_0)
        static_rows.append({
            'conversation_id': conv_id,
            'model': model_name,
            'prompt0_id': prompt0_id,
            'model_prompt0_interaction': f'{model_name}_{prompt0_id}',
            'time_to_failure': time_to_failure,
            'censored': censored,
            'avg_prompt_to_prompt_drift': np.nanmean([d for d in prompt_to_prompt_drifts[1:max_followup_round+1] if d is not None]) if len(prompt_to_prompt_drifts) > 1 else None,
            'avg_context_to_prompt_drift': np.nanmean([d for d in context_to_prompt_drifts[1:max_followup_round+1] if d is not None]) if len(context_to_prompt_drifts) > 1 else None,
            'avg_prompt_complexity': np.mean(prompt_complexities[1:max_followup_round+1]) if len(prompt_complexities) > 1 else None,
            'level': level,
            'subject': subject,
            'subject_cluster': subject_cluster,
        })
        # Long table rows (only for rounds 1 to 8)
        for i in range(1, max_followup_round + 1):
            if i >= len(round_indicators):
                break
            label = round_indicators[i]
            failure = 1 if (label == 0 and censored == 0 and i == time_to_failure) else 0
            long_rows.append({
                'conversation_id': conv_id,
                'model': model_name,
                'prompt0_id': prompt0_id,
                'model_prompt0_interaction': f'{model_name}_{prompt0_id}',
                'round': i,
                'prompt_to_prompt_drift': prompt_to_prompt_drifts[i] if i < len(prompt_to_prompt_drifts) else None,
                'context_to_prompt_drift': context_to_prompt_drifts[i] if i < len(context_to_prompt_drifts) else None,
                'cumulative_drift': cumulative_drifts[i] if i < len(cumulative_drifts) else None,
                'prompt_complexity': prompt_complexities[i] if i < len(prompt_complexities) else None,
                'failure': failure,
                'censored': 1 if (i == max_followup_round and censored == 1) else 0,
                'level': level,
                'subject': subject,
                'subject_cluster': subject_cluster,
            })

    pd.DataFrame(static_rows).to_csv(os.path.join(out_dir, f'{model}_static.csv'), index=False)
    pd.DataFrame(long_rows).to_csv(os.path.join(out_dir, f'{model}_long.csv'), index=False)
    
    print(f"✅ {model} processing completed!")
    print(f"   • Static data: {len(static_rows)} conversations")
    print(f"   • Long data: {len(long_rows)} turns")
    print(f"   • Output directory: {out_dir}")

print("\n🎉 All models processed successfully!")
print("=" * 60)