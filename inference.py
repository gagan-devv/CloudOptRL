"""
Inference script with heuristic policy for hackathon submission compliance.

This script runs environment episodes with strict logging format for validator compliance.
It uses a simple heuristic policy based on CPU thresholds and outputs only [START], [STEP],
and [END] logs to stdout.
"""

import os
import asyncio
from openai import OpenAI
from env.environment import CloudResourceEnv
from env.async_wrapper import AsyncEnvWrapper

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK_NAME", "medium")
BENCHMARK = os.getenv("BENCHMARK", "cloud_resource_env")

# Action Map
ACTION_MAP = {
    0: "decrease",
    1: "maintain",
    2: "increase"
}

# Constants
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5


def heuristic_policy(state) -> int:
    """
    Simple CPU threshold-based policy.
    
    Decision rules:
    - CPU > 70%: Increase resources (action 2)
    - CPU < 40%: Decrease resources (action 0)
    - 40% <= CPU <= 70%: Maintain resources (action 1)
    
    Args:
        state: Environment state with cpu attribute
    
    Returns:
        Action: 0 (decrease), 1 (maintain), or 2 (increase)
    """
    cpu = state.cpu
    if cpu > 70.0:
        return 2  # increase
    elif cpu < 40.0:
        return 0  # decrease
    else:
        return 1  # maintain


async def main():
    """
    Main execution function.
    
    Runs one episode with strict logging format:
    - [START] log at episode beginning
    - [STEP] log for each step
    - [END] log at episode completion
    """
    # Initialize OpenAI client (structure for future use)
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    
    # Create environment with async wrapper
    env = CloudResourceEnv(task_name=TASK_NAME)
    async_env = AsyncEnvWrapper(env)
    
    # Log [START]
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)    
    # Run episode
    state = await async_env.reset()
    rewards = []
    
    for step in range(MAX_STEPS):
        # Get action from heuristic policy
        action_int = heuristic_policy(env.state())
        action = ACTION_MAP[action_int]
        
        # Execute step
        try:
            obs, reward, done, info = await async_env.step(action_int)
            rewards.append(reward)
            # Log [STEP]
            print(f"[STEP] step={step+1} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            if done:
                break
        except Exception as e:
            # Log error
            print(f"[STEP] step={step+1} action={action} reward=0.00 done=True error={str(e)}")
            break
    
    # Calculate success
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    success = avg_reward >= SUCCESS_SCORE_THRESHOLD
    
    # Log [END]
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    score = min(max(avg_reward, 0.0), 1.0)
    print(f"[END] success={str(success).lower()} steps={len(rewards)} score={score:.2f} rewards={rewards_str}", flush=True)
    
    await async_env.close()


if __name__ == "__main__":
    asyncio.run(main())
