#!/usr/bin/env python3
"""
LLM Survival Analysis Pipeline
=============================

Clean, organized pipeline for LLM robustness survival analysis.
Runs modeling stages in proper order with consistent data handling.

Stages:
1. Baseline modeling - Basic Cox PH survival analysis
2. Advanced modeling - Interaction effects with drift×model terms
3. AFT modeling - Accelerated Failure Time models
4. RSF modeling - Random Survival Forest with hyperparameter tuning
5. Visualization generation - Publication-ready plots

Usage:
    python run_analysis.py [--stage STAGE]

    --stage: Run specific stage only (baseline, advanced, aft, rsf, visualization, all)
             Each stage automatically includes its visualizations

Outputs:
    - results/outputs/baseline/*.csv
    - results/outputs/advanced/*.csv
    - results/outputs/aft/*.csv
    - results/outputs/rsf/*.csv
    - results/figures/*/*.png/.pdf (if --viz flag used)
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from modeling.baseline import BaselineModeling
from modeling.advanced import AdvancedModeling
from modeling.aft import AFTModeling
from modeling.rsf import RSFModeling
from visualization.baseline import BaselineCombinedVisualizer
from visualization.baseline_summary import create_baseline_summary
from visualization.advanced import AdvancedModelVisualizer
from visualization.aft import AFTVisualizer


class LLMSurvivalPipeline:
    """Main pipeline for LLM survival analysis."""
    
    def __init__(self):
        self.baseline_model = None
        self.advanced_model = None
        self.aft_model = None
        self.rsf_model = None
        
    def run_baseline_stage(self, generate_viz=True):
        """Run baseline modeling stage."""
        print("🚀 STAGE 1: BASELINE MODELING")
        print("=" * 40)
        
        try:
            self.baseline_model = BaselineModeling()
            baseline_results = self.baseline_model.run_complete_analysis()
            
            print("✅ Baseline modeling completed successfully")
            
            # Generate visualizations if requested
            if generate_viz:
                print("\n🎨 GENERATING BASELINE VISUALIZATIONS")
                print("=" * 40)
                try:
                    visualizer = BaselineCombinedVisualizer()
                    visualizer.generate_all_visualizations()
                    create_baseline_summary()
                    print("✅ Baseline visualizations completed")
                except Exception as e:
                    print(f"⚠️  Visualization generation failed: {e}")
            
            return baseline_results
            
        except Exception as e:
            print(f"❌ Baseline modeling failed: {e}")
            return None
    
    def run_advanced_stage(self, generate_viz=True):
        """Run advanced modeling stage."""
        print("\n🚀 STAGE 2: ADVANCED MODELING")
        print("=" * 40)

        try:
            self.advanced_model = AdvancedModeling()
            advanced_results = self.advanced_model.run_complete_analysis()

            print("✅ Advanced modeling completed successfully")

            # Generate visualizations if requested
            if generate_viz:
                print("\n🎨 GENERATING ADVANCED VISUALIZATIONS")
                print("=" * 40)
                try:
                    visualizer = AdvancedModelVisualizer()
                    visualizer.generate_all_visualizations()
                    print("✅ Advanced visualizations completed")
                except Exception as e:
                    print(f"⚠️  Advanced visualization generation failed: {e}")

            return advanced_results

        except Exception as e:
            print(f"❌ Advanced modeling failed: {e}")
            return None
    
    def run_aft_stage(self, generate_viz=True):
        """Run AFT modeling stage."""
        print("\n🚀 STAGE 3: AFT MODELING")
        print("=" * 40)
        
        try:
            self.aft_model = AFTModeling()
            aft_results = self.aft_model.run_complete_analysis()
            
            print("✅ AFT modeling completed successfully")
            
            # Generate visualizations if requested
            if generate_viz:
                print("\n🎨 GENERATING AFT VISUALIZATIONS")
                print("=" * 40)
                try:
                    visualizer = AFTVisualizer()
                    visualizer.create_all_visualizations()
                    print("✅ AFT visualizations completed")
                except Exception as e:
                    print(f"⚠️  AFT visualization generation failed: {e}")
            
            return aft_results

        except Exception as e:
            print(f"❌ AFT modeling failed: {e}")
            return None

    def run_rsf_stage(self):
        """Run RSF modeling stage."""
        print("\n🚀 STAGE 4: RSF MODELING")
        print("=" * 40)

        try:
            self.rsf_model = RSFModeling()
            rsf_results = self.rsf_model.run_complete_analysis()

            print("✅ RSF modeling completed successfully")
            return rsf_results

        except Exception as e:
            print(f"❌ RSF modeling failed: {e}")
            return None

    def run_visualization_stage(self):
        """Run visualization generation stage."""
        print("\n🚀 STAGE 5: VISUALIZATION GENERATION")
        print("=" * 40)
        
        try:
            # Generate baseline visualizations
            if os.path.exists('results/outputs/baseline'):
                baseline_viz = BaselineCombinedVisualizer()
                baseline_viz.generate_all_visualizations()
                create_baseline_summary()
            else:
                print("⚠️  No baseline results found, skipping baseline visualizations")
            
            # Generate advanced visualizations
            if os.path.exists('results/outputs/advanced'):
                print("\n🎨 GENERATING ADVANCED VISUALIZATIONS")
                advanced_viz = AdvancedModelVisualizer()
                advanced_viz.generate_all_visualizations()
            else:
                print("⚠️  No advanced results found, skipping advanced visualizations")
            
            # Generate AFT visualizations
            if os.path.exists('results/outputs/aft'):
                print("\n🎨 GENERATING AFT VISUALIZATIONS")
                aft_viz = AFTVisualizer()
                aft_viz.create_all_visualizations()
            else:
                print("⚠️  No AFT results found, skipping AFT visualizations")

            # Generate RSF visualizations
            if os.path.exists('results/outputs/rsf'):
                print("\n🎨 GENERATING RSF VISUALIZATIONS")
                print("📊 RSF visualization integration coming soon...")
            else:
                print("⚠️  No RSF results found, skipping RSF visualizations")

            print("✅ Visualization generation completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete analysis pipeline."""
        print("🔬 STARTING COMPLETE LLM SURVIVAL ANALYSIS PIPELINE")
        print("=" * 60)
        
        results = {}
        
        # Stage 1: Baseline
        results['baseline'] = self.run_baseline_stage()

        # Stage 2: Advanced
        results['advanced'] = self.run_advanced_stage()

        # Stage 3: AFT
        results['aft'] = self.run_aft_stage()

        # Stage 4: RSF
        results['rsf'] = self.run_rsf_stage()

        # Final summary
        print("\n" + "=" * 60)
        print("📊 PIPELINE SUMMARY")
        print("=" * 60)

        for stage, result in results.items():
            status = "✅ SUCCESS" if result is not None else "❌ FAILED"
            print(f"{stage.upper():.<20} {status}")

        successful_stages = sum(1 for r in results.values() if r is not None)
        print(f"\n🎯 Completed {successful_stages}/4 modeling stages successfully")
        
        return results
    
    def run_stage(self, stage_name):
        """Run a specific modeling stage with its visualizations."""
        stage_functions = {
            'baseline': lambda: self.run_baseline_stage(generate_viz=True),
            'advanced': lambda: self.run_advanced_stage(generate_viz=True),
            'aft': lambda: self.run_aft_stage(generate_viz=True),
            'rsf': self.run_rsf_stage,
            'visualization': self.run_visualization_stage
        }

        if stage_name in stage_functions:
            return stage_functions[stage_name]()
        else:
            print(f"❌ Unknown stage: {stage_name}")
            print(f"Available stages: {', '.join(stage_functions.keys())}")
            return None


def main():
    """Main execution function with command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Survival Analysis Pipeline')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['baseline', 'advanced', 'aft', 'rsf', 'visualization', 'all'],
                       help='Specific stage to run (default: all)')

    args = parser.parse_args()

    # Create pipeline
    pipeline = LLMSurvivalPipeline()

    # Run requested stage(s)
    if args.stage == 'all':
        results = pipeline.run_complete_pipeline()
    else:
        result = pipeline.run_stage(args.stage)
    
    print("\n🏁 PIPELINE EXECUTION COMPLETED")


if __name__ == "__main__":
    main()