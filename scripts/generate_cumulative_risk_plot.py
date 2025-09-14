#!/usr/bin/env python3
"""
Generate Cumulative Risk Plot
=============================
Standalone script to generate the cumulative risk accumulation plot
based on the extreme cliff cascade dynamics data.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.cliffs import create_cumulative_risk_dynamics

def main():
    """Generate the cumulative risk dynamics plot"""
    
    print("🚀 GENERATING CUMULATIVE RISK DYNAMICS PLOT")
    print("=" * 50)
    
    try:
        # Create the cumulative risk plot
        create_cumulative_risk_dynamics()
        
        print("\n✅ SUCCESS!")
        print("📁 Plot saved as: results/figures/cumulative_risk_dynamic_cliffs.pdf")
        print("📁 Plot saved as: results/figures/cumulative_risk_dynamic_cliffs.png")
        print("\n🎯 This plot shows DYNAMIC cumulative risk accumulation:")
        print("   - Risk can only increase (monotonic)")
        print("   - Phases determined by ACTUAL slope analysis")
        print("   - Top 3 steepest transitions marked as cliff phases")
        print("   - Individual model cliff regions highlighted")
        print("   - Vertical lines show steepest risk increases")
        print("   - Y-axis shows cumulative failure probability (0-100%)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Make sure you have activated the 'survival_analysis' conda environment")
        sys.exit(1)

if __name__ == "__main__":
    main()
