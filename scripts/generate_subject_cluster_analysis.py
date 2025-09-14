#!/usr/bin/env python3
"""
Generate Subject Cluster Analysis
=================================
Generate cumulative risk plots separated by subject clusters to analyze
domain-specific vulnerability patterns.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.cliffs import create_cumulative_risk_by_subject_clusters

def main():
    """Generate the subject cluster cumulative risk analysis"""
    
    print("🚀 GENERATING SUBJECT CLUSTER CUMULATIVE RISK ANALYSIS")
    print("=" * 60)
    
    try:
        # Create the subject cluster analysis
        create_cumulative_risk_by_subject_clusters()
        
        print("\n✅ SUCCESS!")
        print("📁 Plot saved as: results/figures/cumulative_risk_by_subject_clusters.pdf")
        print("📁 Plot saved as: results/figures/cumulative_risk_by_subject_clusters.png")
        
        print("\n🎯 SUBJECT CLUSTER ANALYSIS COMPLETE!")
        print("=" * 60)
        print("📊 This analysis shows how cumulative risk patterns differ across:")
        print("   • STEM (×0.9 vulnerability) - Most stable domain")
        print("   • Business Economics (×0.85) - Very stable")
        print("   • General Knowledge (×1.0) - Baseline")
        print("   • Social Sciences (×1.1) - Moderate challenge")
        print("   • Humanities (×1.15) - Slightly challenging")
        print("   • Law Legal (×1.25) - Challenging domain")
        print("   • Medical Health (×1.3) - Most challenging domain")
        print("\n🔍 Key insights:")
        print("   - Medical domains show steeper cliff patterns")
        print("   - STEM and Business domains provide more stability")
        print("   - Legal domains create moderate additional risk")
        print("   - Each subplot shows domain-specific model rankings")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Make sure you have activated the 'survival_analysis' conda environment")
        sys.exit(1)

if __name__ == "__main__":
    main()
