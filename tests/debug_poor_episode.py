#!/usr/bin/env python3
"""Debug script for poor episode test."""

import numpy as np
from env.grader import EpisodeGrader

grader = EpisodeGrader()

# Poor episode: highly variable utilization, high resources, negative rewards
states = [
    np.array([20.0 + i*8.0, 30.0 + i*7.0, 50, 10], dtype=np.float32)
    for i in range(10)
]
actions = [0, 2, 0, 2, 1, 0, 2, 1, 0, 2]  # Erratic actions
rewards = [-3.0] * 10  # Very negative rewards

result = grader.grade_episode(states, actions, rewards)

print(f"Result: {result}")
print(f"Score: {result['score']:.4f}")
print(f"Passed: {result['passed']}")
print(f"Pass threshold: {grader.pass_threshold}")
print(f"Stability: {result['stability_score']:.4f}")
print(f"Efficiency: {result['efficiency_score']:.4f}")
print(f"Avg Reward: {result['avg_reward']:.4f}")
