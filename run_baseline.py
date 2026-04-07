"""
Baseline heuristic policy for cloud resource allocation.

This script implements a simple rule-based policy that makes resource allocation
decisions based on CPU utilization thresholds. It serves as a baseline for
comparing RL agent performance.
"""

import numpy as np
from env.environment import CloudResourceEnv
from env.grader import EpisodeGrader


def baseline_policy(cpu_util: float) -> int:
    """
    Simple heuristic policy based on CPU utilization thresholds.
    
    Decision rules:
    - CPU > 70%: Increase resources (action 2)
    - CPU < 40%: Decrease resources (action 0)
    - 40% <= CPU <= 70%: Maintain resources (action 1)
    
    Args:
        cpu_util: Current CPU utilization percentage (0-100)
    
    Returns:
        Action: 0 (decrease), 1 (maintain), or 2 (increase)
    """
    if cpu_util > 70.0:
        return 2  # Increase resources
    elif cpu_util < 40.0:
        return 0  # Decrease resources
    else:
        return 1  # Maintain resources


def main():
    """
    Run baseline policy for one episode and display metrics.
    
    Creates environment with fixed seed for reproducibility, runs episode
    for 100 steps using baseline policy, and displays performance metrics
    including cumulative reward, average CPU/memory utilization, and final score.
    """
    # Create environment with fixed seed for reproducibility
    env = CloudResourceEnv()
    np.random.seed(42)
    env.rng.seed(42)
    
    # Reset environment to start episode
    state = env.reset()
    
    # Run episode for 100 steps
    done = False
    step = 0
    
    while not done and step < 100:
        # Extract CPU utilization from state
        cpu_util = state[0]
        
        # Get action from baseline policy
        action = baseline_policy(cpu_util)
        
        # Execute action in environment
        state, reward, done, info = env.step(action)
        step += 1
    
    # Grade episode using EpisodeGrader
    grader = EpisodeGrader()
    results = grader.grade_episode(
        env.episode_states,
        env.episode_actions,
        env.episode_rewards
    )
    
    # Display episode metrics
    print("=" * 60)
    print("BASELINE POLICY EPISODE RESULTS")
    print("=" * 60)
    print(f"Cumulative Reward:    {env.cumulative_reward:.2f}")
    print(f"Average CPU:          {results['avg_cpu']:.2f}%")
    print(f"Average Memory:       {results['avg_memory']:.2f}%")
    print(f"Final Score:          {results['final_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
