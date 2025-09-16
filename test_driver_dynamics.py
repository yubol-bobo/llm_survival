#!/usr/bin/env python3
"""
Test script for driver dynamics visualization
Runs only the new driver dynamics function
"""

import sys
import os
sys.path.append('src')

from visualization.aft import AFTVisualizer

def main():
    """Test the driver dynamics visualization"""
    print("🧪 TESTING DRIVER DYNAMICS VISUALIZATION")
    print("=" * 50)
    
    # Create visualizer
    visualizer = AFTVisualizer()
    
    # Load results
    if not visualizer.load_results():
        print("❌ Failed to load AFT results")
        return
    
    # Run only the driver dynamics plot
    visualizer.plot_driver_dynamics_over_time()
    
    print("✅ Driver dynamics test completed!")

if __name__ == "__main__":
    main()