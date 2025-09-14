#!/usr/bin/env python3
"""
Generate All Visualizations
===========================
Unified script to generate all visualization plots for the LLM survival analysis.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.core import create_model_performance_comparison, create_semantic_drift_effects
from visualization.cliffs import (create_drift_cliff_visualization, create_cliff_cascade_dynamics, 
                                 create_3d_cliff_landscape, create_dramatic_cliff_profiles, create_cumulative_risk_dynamics,
                                 create_cumulative_risk_by_subject_clusters, create_cumulative_risk_by_difficulty_levels)
from visualization.trajectories import create_all_trajectory_plots
from visualization.heatmaps import create_all_heatmaps
from visualization.profiles import create_all_profile_visualizations

def main():
    """Generate all visualizations for the LLM survival analysis"""
    
    print("🚀 GENERATING LLM SURVIVAL ANALYSIS VISUALIZATIONS")
    print("=" * 60)
    
    try:
        # Core visualizations
        print("\n📊 CORE VISUALIZATIONS")
        print("-" * 30)
        create_model_performance_comparison()
        print("✅ Model performance comparison plots created")
        
        create_semantic_drift_effects()
        print("✅ Semantic drift effects plot created")
        
        # Cliff visualizations
        print("\n🏔️  CLIFF PHENOMENON VISUALIZATIONS")
        print("-" * 40)
        create_drift_cliff_visualization()
        print("✅ Drift cliff phenomenon plots created")
        
        create_cliff_cascade_dynamics()
        print("✅ Cliff cascade dynamics plot created")
        
        create_3d_cliff_landscape()
        print("✅ 3D cliff landscape plot created")
        
        create_dramatic_cliff_profiles()
        print("✅ Dramatic cliff profiles plot created")
        
        create_cumulative_risk_dynamics()
        print("✅ Cumulative risk dynamics plot created")
        
        create_cumulative_risk_by_subject_clusters()
        print("✅ Subject cluster cumulative risk analysis created")
        
        create_cumulative_risk_by_difficulty_levels()
        print("✅ Difficulty level cumulative risk analysis created")
        
        # Trajectory visualizations
        print("\n🛤️  TRAJECTORY VISUALIZATIONS")
        print("-" * 35)
        create_all_trajectory_plots()
        
        # Heatmap visualizations
        print("\n🔥 HEATMAP VISUALIZATIONS")
        print("-" * 30)
        create_all_heatmaps()
        
        # Profile visualizations
        print("\n👤 PROFILE VISUALIZATIONS")
        print("-" * 30)
        create_all_profile_visualizations()
        
        print("\n" + "=" * 60)
        print("🎉 ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("📁 Check results/figures/ for all PDF files")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 