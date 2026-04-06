#!/usr/bin/env python3
"""Manual test script for the grader module."""

import numpy as np
from env.grader import EpisodeGrader

def test_basic_grading():
    """Test basic grader functionality."""
    print("Testing EpisodeGrader...")
    
    grader = EpisodeGrader()
    
    # Create simple trajectory
    states = [
        np.array([50.0, 50.0, 50, 3], dtype=np.float32)
        for _ in range(10)
    ]
    actions = [1] * 10
    rewards = [0.5] * 10
    
    result = grader.grade_episode(states, actions, rewards)
    
    print(f"Result: {result}")
    print(f"Score: {result['score']:.4f}")
    print(f"Passed: {result['passed']}")
    print(f"Stability: {result['stability_score']:.4f}")
    print(f"Efficiency: {result['efficiency_score']:.4f}")
    print(f"Avg Reward: {result['avg_reward']:.4f}")
    
    # Verify structure
    assert 'score' in result
    assert 'passed' in result
    assert 'stability_score' in result
    assert 'efficiency_score' in result
    assert 'avg_reward' in result
    
    print("\n✓ Basic grading test passed!")

def test_stability_comparison():
    """Test that lower variance produces higher stability score."""
    print("\nTesting stability assessment...")
    
    grader = EpisodeGrader()
    
    # Stable episode (low variance)
    states_stable = [
        np.array([50.0 + i*0.1, 50.0 + i*0.1, 50, 3], dtype=np.float32)
        for i in range(10)
    ]
    
    # Unstable episode (high variance)
    states_unstable = [
        np.array([50.0 + i*5.0, 50.0 + i*5.0, 50, 3], dtype=np.float32)
        for i in range(10)
    ]
    
    actions = [1] * 10
    rewards = [0.5] * 10
    
    result_stable = grader.grade_episode(states_stable, actions, rewards)
    result_unstable = grader.grade_episode(states_unstable, actions, rewards)
    
    print(f"Stable stability score: {result_stable['stability_score']:.4f}")
    print(f"Unstable stability score: {result_unstable['stability_score']:.4f}")
    
    assert result_stable['stability_score'] > result_unstable['stability_score'], \
        "Stable episode should have higher stability score"
    
    print("✓ Stability assessment test passed!")

def test_efficiency_comparison():
    """Test that lower resources produces higher efficiency score."""
    print("\nTesting efficiency assessment...")
    
    grader = EpisodeGrader()
    
    # Efficient episode (low resources)
    states_efficient = [
        np.array([50.0, 50.0, 50, 2], dtype=np.float32)
        for _ in range(10)
    ]
    
    # Inefficient episode (high resources)
    states_inefficient = [
        np.array([50.0, 50.0, 50, 8], dtype=np.float32)
        for _ in range(10)
    ]
    
    actions = [1] * 10
    rewards = [0.5] * 10
    
    result_efficient = grader.grade_episode(states_efficient, actions, rewards)
    result_inefficient = grader.grade_episode(states_inefficient, actions, rewards)
    
    print(f"Efficient efficiency score: {result_efficient['efficiency_score']:.4f}")
    print(f"Inefficient efficiency score: {result_inefficient['efficiency_score']:.4f}")
    
    assert result_efficient['efficiency_score'] > result_inefficient['efficiency_score'], \
        "Efficient episode should have higher efficiency score"
    
    print("✓ Efficiency assessment test passed!")

if __name__ == '__main__':
    test_basic_grading()
    test_stability_comparison()
    test_efficiency_comparison()
    print("\n✅ All manual tests passed!")
