#!/usr/bin/env python3
"""
Test failure detection and censoring logic
"""

def test_failure_detection():
    """Test failure detection logic with various scenarios"""
    print("🔍 TESTING FAILURE DETECTION LOGIC")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Early failure (conv 0)',
            'round_indicators': [1, 1, 1, 0, 0, 1, 1, 1, 0],  # fails at round 3
            'expected_ttf': 3,
            'expected_censored': 0,
            'expected_failures': {1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        },
        {
            'name': 'Complete success (conv 1)',
            'round_indicators': [1, 1, 1, 1, 1, 1, 1, 1, 1],  # never fails
            'expected_ttf': 8,
            'expected_censored': 1,
            'expected_failures': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        },
        {
            'name': 'Immediate failure (conv 2)',
            'round_indicators': [1, 0, 0, 1, 1, 0, 0, 0, 1],  # fails at round 1
            'expected_ttf': 1,
            'expected_censored': 0,
            'expected_failures': {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        },
        {
            'name': 'Late failure',
            'round_indicators': [1, 1, 1, 1, 1, 1, 1, 0, 0],  # fails at round 7
            'expected_ttf': 7,
            'expected_censored': 0,
            'expected_failures': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0}
        }
    ]
    
    max_followup_round = 8
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        round_indicators = test_case['round_indicators']
        print(f"   Round indicators: {round_indicators}")
        
        # Calculate time to failure
        time_to_failure = None
        censored = 0
        for i in range(1, max_followup_round + 1):
            if i >= len(round_indicators) or round_indicators[i] == 0:
                time_to_failure = i if i <= max_followup_round else max_followup_round
                break
        if time_to_failure is None:
            time_to_failure = max_followup_round
            censored = 1
        
        print(f"   Calculated TTF: {time_to_failure} (expected: {test_case['expected_ttf']})")
        print(f"   Calculated censored: {censored} (expected: {test_case['expected_censored']})")
        
        # Check failure assignment for each round
        print(f"   Failure assignments:")
        actual_failures = {}
        for round_num in range(1, max_followup_round + 1):
            if round_num < len(round_indicators):
                label = round_indicators[round_num]
                failure = 1 if (label == 0 and censored == 0 and round_num == time_to_failure) else 0
                actual_failures[round_num] = failure
                expected = test_case['expected_failures'][round_num]
                status = "✅" if failure == expected else "❌"
                print(f"     Round {round_num}: {failure} (expected: {expected}) {status}")
        
        # Overall validation
        ttf_correct = time_to_failure == test_case['expected_ttf']
        censored_correct = censored == test_case['expected_censored']
        failures_correct = actual_failures == test_case['expected_failures']
        
        overall_status = "✅" if (ttf_correct and censored_correct and failures_correct) else "❌"
        print(f"   Overall: {overall_status}")

def test_edge_cases():
    """Test edge cases"""
    print(f"\n🔍 TESTING EDGE CASES")
    print("=" * 50)
    
    edge_cases = [
        {
            'name': 'Truncated conversation (missing rounds)',
            'round_indicators': [1, 1, 1],  # only 3 rounds
            'note': 'Should handle gracefully'
        },
        {
            'name': 'All zeros after round 0',
            'round_indicators': [1, 0, 0, 0, 0, 0, 0, 0, 0],
            'note': 'Immediate failure scenario'
        },
        {
            'name': 'Alternating pattern',
            'round_indicators': [1, 0, 1, 0, 1, 0, 1, 0, 1],
            'note': 'Multiple failure points'
        }
    ]
    
    max_followup_round = 8
    
    for case in edge_cases:
        print(f"\n📊 Edge case: {case['name']}")
        print(f"   Note: {case['note']}")
        round_indicators = case['round_indicators']
        print(f"   Round indicators: {round_indicators}")
        
        # Calculate time to failure
        time_to_failure = None
        censored = 0
        for i in range(1, max_followup_round + 1):
            if i >= len(round_indicators) or round_indicators[i] == 0:
                time_to_failure = i if i <= max_followup_round else max_followup_round
                break
        if time_to_failure is None:
            time_to_failure = max_followup_round
            censored = 1
        
        print(f"   Result: TTF={time_to_failure}, Censored={censored}")
        
        # Show failure assignments
        for round_num in range(1, min(max_followup_round + 1, len(round_indicators))):
            label = round_indicators[round_num]
            failure = 1 if (label == 0 and censored == 0 and round_num == time_to_failure) else 0
            print(f"     Round {round_num}: label={label}, failure={failure}")

if __name__ == "__main__":
    test_failure_detection()
    test_edge_cases()