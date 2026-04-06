#!/usr/bin/env python3
"""Integration test for grader with environment."""

import numpy as np
from env.environment import CloudResourceEnv
from env.grader import EpisodeGrader

def test_grader_with_real_episode():
    """Test grader with a real episode from the environment."""
    print("Testing grader integration with environment...")
    
    # Create environment and grader
    env = CloudResourceEnv()
    grader = EpisodeGrader()
    
    # Run a short episode
    env.reset()
    
    for _ in range(20):
        action = np.random.choice([0, 1, 2])
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    # Grade the episode using environment's stored trajectory
    result = grader.grade_episode(
        env.episode_states,
        env.episode_actions,
        env.episode_rewards
    )
    
    print(f"\nEpisode Results:")
    print(f"  Steps: {len(env.episode_states)}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Passed: {result['passed']}")
    print(f"  Stability: {result['stability_score']:.4f}")
    print(f"  Efficiency: {result['efficiency_score']:.4f}")
    print(f"  Avg Reward: {result['avg_reward']:.4f}")
    print(f"  Cumulative Reward: {env.cumulative_reward:.4f}")
    
    # Verify structure
    assert 'score' in result
    assert 'passed' in result
    assert isinstance(result['passed'], bool)
    assert 0.0 <= result['score'] <= 1.0
    
    print("\n✅ Integration test passed!")

if __name__ == '__main__':
    test_grader_with_real_episode()
