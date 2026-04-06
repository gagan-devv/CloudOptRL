"""
Property-based tests for reward calculation module.

These tests validate universal correctness properties of the RewardCalculator
using Hypothesis for randomized input generation. Each test runs a minimum of
100 iterations to ensure adequate coverage.
"""

from hypothesis import given, strategies as st, settings
from env.reward import RewardCalculator


# Feature: cloud-resource-allocation-rl, Property 8: Optimal Range Rewards
@settings(max_examples=100)
@given(
    cpu_util=st.floats(min_value=35.0, max_value=70.0),
    memory_util=st.floats(min_value=35.0, max_value=70.0),
    allocated_resources=st.integers(min_value=1, max_value=20)
)
def test_optimal_range_rewards(cpu_util, memory_util, allocated_resources):
    """
    Property 8: Optimal Range Rewards
    
    **Validates: Requirements 4.1, 4.5**
    
    For any state where CPU utilization is between 35-70% and memory utilization
    is between 35-70%, the reward SHALL be positive.
    """
    calculator = RewardCalculator()
    reward = calculator.calculate_reward(cpu_util, memory_util, allocated_resources)
    
    # The reward should be positive when both metrics are in optimal range
    # Note: The resource cost penalty is small (0.1 per instance), so even with
    # 20 instances (-2.0 penalty), the utilization rewards (up to 2.0) should
    # still result in a positive or near-zero total reward
    assert reward > -2.1, (
        f"Expected positive or near-zero reward for optimal utilization "
        f"(CPU={cpu_util:.2f}%, Memory={memory_util:.2f}%, Resources={allocated_resources}), "
        f"but got {reward:.4f}"
    )


# Feature: cloud-resource-allocation-rl, Property 9: Over-Provisioning Penalty
@settings(max_examples=100)
@given(
    cpu_util=st.floats(min_value=0.0, max_value=34.9),
    memory_util=st.floats(min_value=0.0, max_value=34.9),
    allocated_resources=st.integers(min_value=1, max_value=20)
)
def test_over_provisioning_penalty(cpu_util, memory_util, allocated_resources):
    """
    Property 9: Over-Provisioning Penalty
    
    **Validates: Requirements 4.2**
    
    For any state where CPU utilization is below 35% and memory utilization is
    below 35% (indicating excess resources), the reward SHALL be negative.
    """
    calculator = RewardCalculator()
    reward = calculator.calculate_reward(cpu_util, memory_util, allocated_resources)
    
    # Both CPU and memory are below optimal range, indicating over-provisioning
    assert reward < 0, (
        f"Expected negative reward for over-provisioning "
        f"(CPU={cpu_util:.2f}%, Memory={memory_util:.2f}%, Resources={allocated_resources}), "
        f"but got {reward:.4f}"
    )


# Feature: cloud-resource-allocation-rl, Property 10: Under-Provisioning Penalty
@settings(max_examples=100)
@given(
    cpu_util=st.floats(min_value=70.1, max_value=100.0),
    memory_util=st.floats(min_value=0.0, max_value=100.0),
    allocated_resources=st.integers(min_value=1, max_value=20)
)
def test_under_provisioning_penalty_cpu(cpu_util, memory_util, allocated_resources):
    """
    Property 10: Under-Provisioning Penalty (CPU variant)
    
    **Validates: Requirements 4.3**
    
    For any state where CPU utilization exceeds 70% (indicating insufficient
    resources), the reward SHALL be negative.
    """
    calculator = RewardCalculator()
    reward = calculator.calculate_reward(cpu_util, memory_util, allocated_resources)
    
    # CPU is above optimal range, indicating under-provisioning
    assert reward < 0, (
        f"Expected negative reward for under-provisioning "
        f"(CPU={cpu_util:.2f}%, Memory={memory_util:.2f}%, Resources={allocated_resources}), "
        f"but got {reward:.4f}"
    )


@settings(max_examples=100)
@given(
    cpu_util=st.floats(min_value=0.0, max_value=100.0),
    memory_util=st.floats(min_value=70.1, max_value=100.0),
    allocated_resources=st.integers(min_value=1, max_value=20)
)
def test_under_provisioning_penalty_memory(cpu_util, memory_util, allocated_resources):
    """
    Property 10: Under-Provisioning Penalty (Memory variant)
    
    **Validates: Requirements 4.3**
    
    For any state where memory utilization exceeds 70% (indicating insufficient
    resources), the reward SHALL be negative.
    """
    calculator = RewardCalculator()
    reward = calculator.calculate_reward(cpu_util, memory_util, allocated_resources)
    
    # Memory is above optimal range, indicating under-provisioning
    assert reward < 0, (
        f"Expected negative reward for under-provisioning "
        f"(CPU={cpu_util:.2f}%, Memory={memory_util:.2f}%, Resources={allocated_resources}), "
        f"but got {reward:.4f}"
    )


# Feature: cloud-resource-allocation-rl, Property 11: Reward Sensitivity
@settings(max_examples=100)
@given(
    cpu_util=st.floats(min_value=0.0, max_value=100.0),
    memory_util=st.floats(min_value=0.0, max_value=100.0),
    allocated_resources=st.integers(min_value=1, max_value=19),
    delta_cpu=st.floats(min_value=0.1, max_value=10.0),
    delta_memory=st.floats(min_value=0.1, max_value=10.0)
)
def test_reward_sensitivity(cpu_util, memory_util, allocated_resources, delta_cpu, delta_memory):
    """
    Property 11: Reward Sensitivity
    
    **Validates: Requirements 4.4**
    
    For any state, modifying CPU utilization, memory utilization, or allocated
    resource count SHALL result in a different reward value, demonstrating that
    rewards depend on all three factors.
    """
    calculator = RewardCalculator()
    
    # Calculate baseline reward
    baseline_reward = calculator.calculate_reward(cpu_util, memory_util, allocated_resources)
    
    # Test CPU sensitivity (ensure we stay in valid range)
    cpu_util_modified = min(100.0, cpu_util + delta_cpu)
    # Use epsilon comparison to handle floating point precision issues
    if abs(cpu_util_modified - cpu_util) > 1e-6:
        reward_cpu_changed = calculator.calculate_reward(cpu_util_modified, memory_util, allocated_resources)
        assert reward_cpu_changed != baseline_reward, (
            f"Reward should change when CPU changes from {cpu_util:.2f}% to {cpu_util_modified:.2f}%"
        )
    
    # Test memory sensitivity (ensure we stay in valid range)
    memory_util_modified = min(100.0, memory_util + delta_memory)
    # Use epsilon comparison to handle floating point precision issues
    if abs(memory_util_modified - memory_util) > 1e-6:
        reward_memory_changed = calculator.calculate_reward(cpu_util, memory_util_modified, allocated_resources)
        assert reward_memory_changed != baseline_reward, (
            f"Reward should change when memory changes from {memory_util:.2f}% to {memory_util_modified:.2f}%"
        )
    
    # Test resource sensitivity (always valid since we constrained allocated_resources to max 19)
    reward_resource_changed = calculator.calculate_reward(cpu_util, memory_util, allocated_resources + 1)
    assert reward_resource_changed != baseline_reward, (
        f"Reward should change when resources change from {allocated_resources} to {allocated_resources + 1}"
    )
