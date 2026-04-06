#!/usr/bin/env python
"""Test the specific failing case."""

import numpy as np
from env.environment import CloudResourceEnv
from env.config import EnvConfig

# Exact failing case from Hypothesis
action = 0  # DECREASE
initial_resources = 2
base_request_rate = 30
seed1 = 2
seed2 = 50

config = EnvConfig(
    initial_resources=initial_resources,
    base_request_rate=base_request_rate,
    request_rate_std=10.0
)

# First execution with seed1
env1 = CloudResourceEnv(config=config)
env1.rng.seed(seed1)
env1.reset()
print(f"Env1 after reset (seed={seed1}):")
print(f"  State: {env1._get_observation()}")
print(f"  Request rate: {env1.request_rate}")
print(f"  Allocated resources: {env1.allocated_resources}")

obs1, reward1, done1, info1 = env1.step(action)
print(f"Env1 after step (action={action}):")
print(f"  State: {obs1}")
print(f"  Request rate: {info1['request_rate']}")
print(f"  Allocated resources: {info1['allocated_resources']}")

print()

# Second execution with seed2
env2 = CloudResourceEnv(config=config)
env2.rng.seed(seed2)
env2.reset()
print(f"Env2 after reset (seed={seed2}):")
print(f"  State: {env2._get_observation()}")
print(f"  Request rate: {env2.request_rate}")
print(f"  Allocated resources: {env2.allocated_resources}")

obs2, reward2, done2, info2 = env2.step(action)
print(f"Env2 after step (action={action}):")
print(f"  State: {obs2}")
print(f"  Request rate: {info2['request_rate']}")
print(f"  Allocated resources: {info2['allocated_resources']}")

print()
print(f"States equal: {np.allclose(obs1, obs2, rtol=1e-5)}")
print(f"Request rates equal: {info1['request_rate'] == info2['request_rate']}")
print(f"States differ: {not np.allclose(obs1, obs2, rtol=1e-5)}")
print(f"Request rates differ: {info1['request_rate'] != info2['request_rate']}")
