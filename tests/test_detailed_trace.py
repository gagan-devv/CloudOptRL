#!/usr/bin/env python
"""Detailed trace of RNG usage."""

import numpy as np
from env.environment import CloudResourceEnv
from env.config import EnvConfig

action = 0  # DECREASE
initial_resources = 2
base_request_rate = 30

config = EnvConfig(
    initial_resources=initial_resources,
    base_request_rate=base_request_rate,
    request_rate_std=10.0
)

print("=== ENV1 (seed=2) ===")
env1 = CloudResourceEnv(config=config)
env1.rng.seed(2)

# Manually trace reset
print("During reset():")
fluctuation_reset = env1.rng.normal(0, config.request_rate_std)
print(f"  RNG call 1 (reset): {fluctuation_reset:.4f}")
request_rate_reset = int(config.base_request_rate + fluctuation_reset)
request_rate_reset = max(0, request_rate_reset)
print(f"  Request rate after reset: {request_rate_reset}")

# Now actually reset to verify
env1.rng.seed(2)  # Re-seed
env1.reset()
print(f"  Actual request rate after reset: {env1.request_rate}")

# Manually trace step
print("During step():")
fluctuation_step = env1.rng.normal(0, config.request_rate_std)
print(f"  RNG call 2 (step): {fluctuation_step:.4f}")
request_rate_step = int(config.base_request_rate + fluctuation_step)
request_rate_step = max(0, request_rate_step)
print(f"  Request rate after step: {request_rate_step}")

# Actually do the step
env1.rng.seed(2)  # Re-seed
env1.reset()
obs1, _, _, info1 = env1.step(action)
print(f"  Actual request rate after step: {info1['request_rate']}")

print()
print("=== ENV2 (seed=50) ===")
env2 = CloudResourceEnv(config=config)
env2.rng.seed(50)

# Manually trace reset
print("During reset():")
fluctuation_reset = env2.rng.normal(0, config.request_rate_std)
print(f"  RNG call 1 (reset): {fluctuation_reset:.4f}")
request_rate_reset = int(config.base_request_rate + fluctuation_reset)
request_rate_reset = max(0, request_rate_reset)
print(f"  Request rate after reset: {request_rate_reset}")

# Now actually reset to verify
env2.rng.seed(50)  # Re-seed
env2.reset()
print(f"  Actual request rate after reset: {env2.request_rate}")

# Manually trace step
print("During step():")
fluctuation_step = env2.rng.normal(0, config.request_rate_std)
print(f"  RNG call 2 (step): {fluctuation_step:.4f}")
request_rate_step = int(config.base_request_rate + fluctuation_step)
request_rate_step = max(0, request_rate_step)
print(f"  Request rate after step: {request_rate_step}")

# Actually do the step
env2.rng.seed(50)  # Re-seed
env2.reset()
obs2, _, _, info2 = env2.step(action)
print(f"  Actual request rate after step: {info2['request_rate']}")

print()
print(f"Request rates after step: {info1['request_rate']} vs {info2['request_rate']}")
print(f"Are they equal? {info1['request_rate'] == info2['request_rate']}")
