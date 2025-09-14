#!/usr/bin/env python3
"""
Test Monotonic Cumulative Risk
==============================
Simple test to verify monotonic behavior without matplotlib dependencies.
"""

import numpy as np

def test_monotonic():
    # Same data as in visualization
    models = ['CARG', 'GPT-4', 'Claude-3.5']
    failures = [68, 134, 453]
    
    for i, (model, n_fail) in enumerate(zip(models, failures)):
        print(f"\n🔍 Testing {model} (failures: {n_fail})")
        
        # Same hazard logic
        if n_fail < 100:
            hazard = np.array([1, 1.2, 1.5, 2, 3, 5, 8, 12]) * (n_fail / 68)
        elif n_fail < 300:
            hazard = np.array([1, 1.5, 3, 8, 20, 50, 150, 400]) * (n_fail / 200)
        else:
            hazard = np.array([1, 2, 10, 50, 500, 5000, 50000, 200000]) * (n_fail / 400)
        
        # Convert to cumulative risk
        base_hazard_rate = 0.05
        hazard_rates = hazard * base_hazard_rate
        cumulative_hazard = np.cumsum(hazard_rates)
        cumulative_risk = 1 - np.exp(-cumulative_hazard)
        cumulative_risk = np.clip(cumulative_risk, 0, 1)
        
        # Check monotonic
        is_monotonic = all(cumulative_risk[j] >= cumulative_risk[j-1] for j in range(1, len(cumulative_risk)))
        
        print(f"Cumulative risk: {[f'{r:.3f}' for r in cumulative_risk]}")
        print(f"Monotonic: {is_monotonic}")
        
        if not is_monotonic:
            print("❌ NOT MONOTONIC - This should never happen!")
            for j in range(1, len(cumulative_risk)):
                if cumulative_risk[j] < cumulative_risk[j-1]:
                    print(f"   Turn {j+1}: {cumulative_risk[j]:.3f} < {cumulative_risk[j-1]:.3f}")
        else:
            print("✅ Correctly monotonic")

if __name__ == "__main__":
    test_monotonic()
