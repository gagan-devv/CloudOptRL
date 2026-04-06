#!/usr/bin/env python
"""Compare RNG sequences for seeds 0 and 3."""

import numpy as np

base_request_rate = 54
request_rate_std = 10.0

print("=== Seed 0 ===")
rng0 = np.random.RandomState()
rng0.seed(0)
for i in range(5):
    fluctuation = rng0.normal(0, request_rate_std)
    request_rate = int(base_request_rate + fluctuation)
    request_rate = max(0, request_rate)
    print(f"Step {i}: fluctuation={fluctuation:.4f}, request_rate={request_rate}")

print()
print("=== Seed 3 ===")
rng3 = np.random.RandomState()
rng3.seed(3)
for i in range(5):
    fluctuation = rng3.normal(0, request_rate_std)
    request_rate = int(base_request_rate + fluctuation)
    request_rate = max(0, request_rate)
    print(f"Step {i}: fluctuation={fluctuation:.4f}, request_rate={request_rate}")
