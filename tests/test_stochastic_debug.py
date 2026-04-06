#!/usr/bin/env python
"""Debug script to test stochastic behavior."""

import numpy as np
from env.environment import CloudResourceEnv
from env.config import EnvConfig

# Test with two different seeds
config = EnvConfig(
    initial_resources=3,
    base_request_rate=50,
    request_rate_std=10.0
)

env1 = CloudResourceEnv(config=config)
env1.rng.seed(42)
obs1_reset = env1.reset()
print(f"Env1 after reset (seed=42): {obs1_reset}")
print(f"  Request rate: {env1.request_rate}")

obs1_step, _, _, _ = env1.step(1)  # MAINTAIN action
print(f"Env1 after step: {obs1_step}")
print(f"  Request rate: {env1.request_rate}")

print()

env2 = CloudResourceEnv(config=config)
env2.rng.seed(99)
obs2_reset = env2.reset()
print(f"Env2 after reset (seed=99): {obs2_reset}")
print(f"  Request rate: {env2.request_rate}")

obs2_step, _, _, _ = env2.step(1)  # MAINTAIN action
print(f"Env2 after step: {obs2_step}")
print(f"  Request rate: {env2.request_rate}")

print()
print(f"Reset observations equal: {np.allclose(obs1_reset, obs2_reset)}")
print(f"Step observations equal: {np.allclose(obs1_step, obs2_step)}")
print(f"Request rates after step equal: {env1.request_rate == env2.request_rate}")
