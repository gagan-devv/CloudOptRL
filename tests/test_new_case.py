#!/usr/bin/env python
"""Test the new failing case."""

import numpy as np
from env.environment import CloudResourceEnv
from env.config import EnvConfig

# New failing case
action = 0  # DECREASE
initial_resources = 2
base_request_rate = 54
seed1 = 0
seed2 = 3

config = EnvConfig(
    initial_resources=initial_resources,
    base_request_rate=base_request_rate,
    request_rate_std=10.0
)

print("=== ENV1 (seed=0) ===")
env1 = CloudResourceEnv(config=config)
env1.rng.seed(seed1)
env1.reset()
print(f"After reset: request_rate={env1.request_rate}, resources={env1.allocated_resources}")

for i in range(10):
    obs1, _, done1, info1 = env1.step(action)
    print(f"Step {i+1}: request_rate={info1['request_rate']}, resources={info1['allocated_resources']}, obs={obs1}")
    if done1:
        print(f"  Episode terminated at step {i+1}")
        break

print()
print("=== ENV2 (seed=3) ===")
env2 = CloudResourceEnv(config=config)
env2.rng.seed(seed2)
env2.reset()
print(f"After reset: request_rate={env2.request_rate}, resources={env2.allocated_resources}")

for i in range(10):
    obs2, _, done2, info2 = env2.step(action)
    print(f"Step {i+1}: request_rate={info2['request_rate']}, resources={info2['allocated_resources']}, obs={obs2}")
    if done2:
        print(f"  Episode terminated at step {i+1}")
        break
