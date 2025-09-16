#!/usr/bin/env python3
"""
Test the model discovery in the updated preprocessing script
"""

import os

def test_model_discovery():
    """Test what models will be discovered"""
    print("🔍 TESTING MODEL DISCOVERY")
    print("=" * 50)
    
    RAW_DATA_DIR = 'data/raw'
    
    # Get all directories
    all_dirs = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    print(f"All directories in {RAW_DATA_DIR}: {all_dirs}")
    
    # Filter to model directories
    raw_models = [d for d in all_dirs if not d.startswith('cleaned_data')]
    print(f"Model directories to process: {raw_models}")
    
    # Check if each has required files
    print(f"\n📊 Checking file availability for each model:")
    for model in raw_models:
        model_path = os.path.join(RAW_DATA_DIR, model)
        json_files = [f for f in os.listdir(model_path) if f.endswith('.json')]
        csv_files = [f for f in os.listdir(model_path) if f.endswith('.csv')]
        
        print(f"  {model}: {len(json_files)} JSON files, {len(csv_files)} CSV files")
        
        if len(json_files) == 0 or len(csv_files) == 0:
            print(f"    ⚠️ Missing required files!")
        else:
            print(f"    ✅ Ready for processing")

if __name__ == "__main__":
    test_model_discovery()