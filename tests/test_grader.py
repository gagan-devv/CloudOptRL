"""
Unit tests for the EpisodeGrader.

These tests validate specific examples and edge cases for the grader module.
"""

import pytest
import numpy as np
from env.grader import EpisodeGrader


def test_grader_with_empty_trajectory():
    """
    Test grader with empty trajectory (should raise ValueError).
    
    Requirements: 6.1, 6.2, 6.5
    """
    grader = EpisodeGrader()
    
    # Test with empty states
    with pytest.raises(ValueError, match="Trajectory data cannot be empty"):
        grader.grade_episode([], [1], [0.5])
    
    # Test with empty actions
    with pytest.raises(ValueError, match="Trajectory data cannot be empty"):
        grader.grade_episode([np.array([50.0, 50.0, 50, 3])], [], [0.5])
    
    # Test with empty rewards
    with pytest.raises(ValueError, match="Trajectory data cannot be empty"):
        grader.grade_episode([np.array([50.0, 50.0, 50, 3])], [1], [])


def test_grader_with_mismatched_list_lengths():
    """
    Test grader with mismatched list lengths (should raise ValueError).
    
    Requirements: 6.1, 6.2, 6.5
    """
    grader = EpisodeGrader()
    
    states = [np.array([50.0, 50.0, 50, 3], dtype=np.float32) for _ in range(10)]
    actions = [1] * 5  # Mismatched length
    rewards = [0.5] * 10
    
    with pytest.raises(ValueError, match="Mismatched trajectory lengths"):
        grader.grade_episode(states, actions, rewards)
    
    # Test another mismatch
    states = [np.array([50.0, 50.0, 50, 3], dtype=np.float32) for _ in range(10)]
    actions = [1] * 10
    rewards = [0.5] * 8  # Mismatched length
    
    with pytest.raises(ValueError, match="Mismatched trajectory lengths"):
        grader.grade_episode(states, actions, rewards)


def test_grader_with_single_step_episode():
    """
    Test grader with single-step episode.
    
    Requirements: 6.1, 6.2, 6.5
    """
    grader = EpisodeGrader()
    
    # Single-step episode
    states = [np.array([50.0, 50.0, 50, 3], dtype=np.float32)]
    actions = [1]
    rewards = [0.5]
    
    result = grader.grade_episode(states, actions, rewards)
    
    # Verify result structure
    assert 'score' in result
    assert 'passed' in result
    assert 'stability_score' in result
    assert 'efficiency_score' in result
    assert 'avg_reward' in result
    
    # Verify values are reasonable
    assert 0.0 <= result['score'] <= 1.0
    assert 0.0 <= result['stability_score'] <= 1.0
    assert 0.0 <= result['efficiency_score'] <= 1.0
    assert result['avg_reward'] == 0.5
    
    # Single-step episode should have perfect stability (variance = 0)
    assert result['stability_score'] == 1.0


def test_grader_pass_fail_threshold_boundary():
    """
    Test grader pass/fail threshold boundary.
    
    Requirements: 6.1, 6.2, 6.5
    """
    # Test with threshold at 0.5
    grader = EpisodeGrader(pass_threshold=0.5)
    
    # Create episode that should score around 0.5
    # We'll use moderate stability, moderate efficiency, and moderate rewards
    states = [
        np.array([50.0 + i*2.0, 50.0 + i*2.0, 50, 5], dtype=np.float32)
        for i in range(10)
    ]
    actions = [1] * 10
    rewards = [-1.0] * 10  # Negative rewards to lower score
    
    result = grader.grade_episode(states, actions, rewards)
    
    # Verify passed status is consistent with score
    if result['score'] >= 0.5:
        assert result['passed'] is True, (
            f"Episode should pass when score ({result['score']:.4f}) >= threshold (0.5)"
        )
    else:
        assert result['passed'] is False, (
            f"Episode should fail when score ({result['score']:.4f}) < threshold (0.5)"
        )
    
    # Test with very low threshold (should always pass)
    grader_easy = EpisodeGrader(pass_threshold=-1.0)
    result_easy = grader_easy.grade_episode(states, actions, rewards)
    assert result_easy['passed'] is True, "Episode should pass with very low threshold"
    
    # Test with very high threshold (should always fail)
    grader_hard = EpisodeGrader(pass_threshold=2.0)
    result_hard = grader_hard.grade_episode(states, actions, rewards)
    assert result_hard['passed'] is False, "Episode should fail with very high threshold"


def test_grader_with_extreme_values():
    """
    Test grader with extreme state values.
    
    Requirements: 6.1, 6.2, 6.3, 6.4
    """
    grader = EpisodeGrader()
    
    # Episode with extreme CPU/memory utilization
    states = [
        np.array([100.0, 100.0, 200, 1], dtype=np.float32),  # Max utilization, min resources
        np.array([0.0, 0.0, 0, 20], dtype=np.float32),  # Min utilization, max resources
        np.array([50.0, 50.0, 50, 5], dtype=np.float32),  # Normal
    ]
    actions = [0, 2, 1]
    rewards = [-2.0, -1.0, 0.5]
    
    result = grader.grade_episode(states, actions, rewards)
    
    # Should handle extreme values without crashing
    assert 'score' in result
    assert 0.0 <= result['score'] <= 1.0
    assert 0.0 <= result['stability_score'] <= 1.0
    assert 0.0 <= result['efficiency_score'] <= 1.0
    
    # High variance should result in lower stability score
    assert result['stability_score'] < 0.5, "High variance should reduce stability score"


def test_grader_perfect_episode():
    """
    Test grader with a perfect episode (stable, efficient, high rewards).
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    grader = EpisodeGrader()
    
    # Perfect episode: stable utilization, low resources, positive rewards
    states = [
        np.array([55.0, 55.0, 50, 2], dtype=np.float32)
        for _ in range(20)
    ]
    actions = [1] * 20  # All maintain actions
    rewards = [1.5] * 20  # High positive rewards
    
    result = grader.grade_episode(states, actions, rewards)
    
    # Perfect episode should have high scores
    assert result['stability_score'] > 0.95, "Perfect stability should score high"
    assert result['efficiency_score'] > 0.4, "Low resources should score high efficiency"
    assert result['score'] > 0.7, "Perfect episode should have high overall score"
    assert result['passed'] is True, "Perfect episode should pass"


def test_grader_poor_episode():
    """
    Test grader with a poor episode (unstable, inefficient, low rewards).
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    # Use a higher pass threshold to ensure poor episode fails
    grader = EpisodeGrader(pass_threshold=0.3)
    
    # Poor episode: highly variable utilization, high resources, negative rewards
    states = [
        np.array([20.0 + i*8.0, 30.0 + i*7.0, 50, 10], dtype=np.float32)
        for i in range(10)
    ]
    actions = [0, 2, 0, 2, 1, 0, 2, 1, 0, 2]  # Erratic actions
    rewards = [-3.0] * 10  # Very negative rewards
    
    result = grader.grade_episode(states, actions, rewards)
    
    # Poor episode should have low scores
    assert result['stability_score'] < 0.5, "High variance should score low stability"
    assert result['efficiency_score'] < 0.15, "High resources should score low efficiency"
    assert result['score'] < 0.3, "Poor episode should have low overall score"
    assert result['passed'] is False, "Poor episode should fail with threshold 0.3"


def test_grader_component_weights():
    """
    Test that grader correctly applies component weights.
    
    Requirements: 6.2, 6.3, 6.4
    """
    # Create grader with custom weights
    grader = EpisodeGrader(
        stability_weight=0.5,
        efficiency_weight=0.3,
        reward_weight=0.2
    )
    
    states = [
        np.array([50.0, 50.0, 50, 3], dtype=np.float32)
        for _ in range(10)
    ]
    actions = [1] * 10
    rewards = [0.0] * 10
    
    result = grader.grade_episode(states, actions, rewards)
    
    # Manually calculate expected score
    stability_score = result['stability_score']
    efficiency_score = result['efficiency_score']
    avg_reward = result['avg_reward']
    
    # Normalize reward: (avg_reward + 5.0) / 7.0
    normalized_reward = (avg_reward + 5.0) / 7.0
    normalized_reward = max(0.0, min(1.0, normalized_reward))
    
    expected_score = (
        0.5 * stability_score +
        0.3 * efficiency_score +
        0.2 * normalized_reward
    )
    
    assert abs(result['score'] - expected_score) < 0.001, (
        f"Score calculation mismatch: expected {expected_score:.4f}, got {result['score']:.4f}"
    )
