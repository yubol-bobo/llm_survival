#!/usr/bin/env python3
"""
Run Baseline Analysis
====================
Execute baseline modeling analysis including negative binomial and Cox PH models.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Run baseline survival analysis"""
    
    print("🚀 RUNNING BASELINE LLM SURVIVAL ANALYSIS")
    print("=" * 60)
    
    try:
        # Import and run baseline modeling
        from modeling.baseline_modeling import main as run_baseline_modeling
        
        print("📊 Running baseline modeling (Negative Binomial + Cox PH)...")
        run_baseline_modeling()
        
        print("\n" + "=" * 60)
        print("✅ BASELINE ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📁 Check results/outputs/ for analysis results")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Make sure you have activated the 'survival_analysis' conda environment")
        sys.exit(1)

if __name__ == "__main__":
    main() 