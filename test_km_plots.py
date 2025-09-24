#!/usr/bin/env python3
"""
Test script for Kaplan-Meier plotting functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from evaluation.test_evaluation import TestEvaluator

    print("🧪 Testing Kaplan-Meier plotting functionality...")

    # Create evaluator instance
    evaluator = TestEvaluator()

    # Test data loading
    print("📊 Testing data loading...")
    try:
        evaluator.load_test_data()
        print("✅ Test data loaded successfully")
    except Exception as e:
        print(f"❌ Test data loading failed: {e}")
        sys.exit(1)

    # Test feature preparation
    print("🔧 Testing feature preparation...")
    try:
        evaluator.prepare_test_features()
        print("✅ Test features prepared successfully")
    except Exception as e:
        print(f"❌ Feature preparation failed: {e}")
        sys.exit(1)

    # Test Kaplan-Meier plotting
    print("📈 Testing Kaplan-Meier plotting...")
    try:
        evaluator.create_comprehensive_kaplan_meier_plots()
        print("✅ Kaplan-Meier plots created successfully")
    except Exception as e:
        print(f"❌ Kaplan-Meier plotting failed: {e}")
        sys.exit(1)

    print("🎉 All tests passed! Check results/figures/test_evaluation/ for plots.")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure all dependencies are installed")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)