#!/usr/bin/env python3
"""
Test Cumulative Risk Calculation
================================
Test script to verify that the cumulative risk is truly cumulative
and show the data values for debugging.
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.core import load_model_data

def test_cumulative_calculation():
    """Test and display the cumulative risk calculation"""
    
    print("🧪 TESTING CUMULATIVE RISK CALCULATION")
    print("=" * 50)
    
    data = load_model_data()
    models = data['Model']
    failures = data['N_failures']
    
    # Test with a few representative models
    test_models = ['CARG', 'GPT-4', 'Claude-3.5']
    turns = np.arange(1, 9)  # 8 turns
    
    for model in test_models:
        model_idx = models.index(model)
        n_fail = failures[model_idx]
        
        print(f"\n📊 MODEL: {model} (Failures: {n_fail})")
        print("-" * 40)
        
        # Get hazard ratios (same logic as in visualization)
        if n_fail < 100:  # Robust models
            hazard = np.array([1, 1.2, 1.5, 2, 3, 5, 8, 12]) * (n_fail / 68)
        elif n_fail < 300:  # Moderate models  
            hazard = np.array([1, 1.5, 3, 8, 20, 50, 150, 400]) * (n_fail / 200)
        else:  # Vulnerable models
            hazard = np.array([1, 2, 10, 50, 500, 5000, 50000, 200000]) * (n_fail / 400)
        
        # Convert to cumulative risk
        base_hazard_rate = 0.05
        hazard_rates = hazard * base_hazard_rate
        cumulative_hazard = np.cumsum(hazard_rates)
        cumulative_risk = 1 - np.exp(-cumulative_hazard)
        cumulative_risk = np.clip(cumulative_risk, 0, 1)
        
        print("Turn | Hazard Ratio | Hazard Rate | Cum. Hazard | Cum. Risk | Risk %")
        print("-" * 70)
        for i, turn in enumerate(turns):
            print(f"{turn:4d} | {hazard[i]:11.2f} | {hazard_rates[i]:10.4f} | "
                  f"{cumulative_hazard[i]:10.4f} | {cumulative_risk[i]:8.4f} | {cumulative_risk[i]*100:5.1f}%")
        
        # Verify monotonic increase
        is_monotonic = all(cumulative_risk[i] <= cumulative_risk[i+1] for i in range(len(cumulative_risk)-1))
        print(f"\n✅ Monotonic increasing: {is_monotonic}")
        print(f"📈 Final cumulative risk: {cumulative_risk[-1]*100:.1f}%")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"   • Cumulative hazard = sum of all previous hazard rates")
    print(f"   • Cumulative risk = 1 - exp(-cumulative_hazard)")
    print(f"   • Risk can only increase (monotonic)")
    print(f"   • Cliff models show exponential risk growth in later turns")

if __name__ == "__main__":
    test_cumulative_calculation()
