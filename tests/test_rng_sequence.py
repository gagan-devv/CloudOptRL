#!/usr/bin/env python
"""Test RNG sequence."""

import numpy as np

# Test RNG sequences
rng1 = np.random.RandomState()
rng1.seed(2)
print("RNG1 (seed=2) sequence:")
for i in range(5):
    val = rng1.normal(0, 10.0)
    print(f"  {i}: {val:.4f}")

print()

rng2 = np.random.RandomState()
rng2.seed(50)
print("RNG2 (seed=50) sequence:")
for i in range(5):
    val = rng2.normal(0, 10.0)
    print(f"  {i}: {val:.4f}")
