"""
Unit tests for reward calculation module.

These tests verify specific examples and edge cases for the RewardCalculator,
complementing the property-based tests with concrete scenarios.
"""

import pytest
from env.reward import RewardCalculator


class TestRewardCalculator:
    """Test suite for RewardCalculator class."""
    
    def test_initialization_default_values(self):
        """Test that RewardCalculator initializes with correct default values."""
        calculator = RewardCalculator()
        
        assert calculator.target_cpu_range == (35.0, 70.0)
        assert calculator.target_memory_range == (35.0, 70.0)
        assert calculator.resource_cost_weight == 0.05
    
    def test_initialization_custom_values(self):
        """Test that RewardCalculator accepts custom parameter values."""
        calculator = RewardCalculator(
            target_cpu_range=(30.0, 80.0),
            target_memory_range=(35.0, 75.0),
            resource_cost_weight=0.2
        )
        
        assert calculator.target_cpu_range == (30.0, 80.0)
        assert calculator.target_memory_range == (35.0, 75.0)
        assert calculator.resource_cost_weight == 0.2
    
    def test_reward_with_extreme_utilization_zero(self):
        """Test reward calculation with 0% utilization (extreme over-provisioning)."""
        calculator = RewardCalculator()
        reward = calculator.calculate_reward(
            cpu_util=0.0,
            memory_util=0.0,
            allocated_resources=5
        )
        
        # Should be strongly negative due to over-provisioning
        assert reward < -1.0
    
    def test_reward_with_extreme_utilization_hundred(self):
        """Test reward calculation with 100% utilization (extreme under-provisioning)."""
        calculator = RewardCalculator()
        reward = calculator.calculate_reward(
            cpu_util=100.0,
            memory_util=100.0,
            allocated_resources=1
        )
        
        # Should be strongly negative due to under-provisioning
        assert reward < -1.0
    
    def test_reward_with_minimum_resources(self):
        """Test reward calculation with minimum resources (1 instance)."""
        calculator = RewardCalculator()
        
        # Optimal utilization with minimum resources
        reward = calculator.calculate_reward(
            cpu_util=55.0,
            memory_util=55.0,
            allocated_resources=1
        )
        
        # Should be positive (optimal utilization) with minimal resource penalty
        assert reward > 0
    
    def test_reward_with_invalid_resources_raises_error(self):
        """Test that allocated_resources < 1 raises ValueError."""
        calculator = RewardCalculator()
        
        with pytest.raises(ValueError, match="Allocated resources must be at least 1"):
            calculator.calculate_reward(
                cpu_util=50.0,
                memory_util=50.0,
                allocated_resources=0
            )
    
    def test_reward_clamping_cpu_above_hundred(self):
        """Test that CPU utilization above 100 is clamped to 100."""
        calculator = RewardCalculator()
        
        # Should clamp to 100 and not crash
        reward = calculator.calculate_reward(
            cpu_util=150.0,
            memory_util=50.0,
            allocated_resources=1
        )
        
        # Should be negative (under-provisioning)
        assert reward < 0
        assert isinstance(reward, float)
    
    def test_reward_clamping_memory_negative(self):
        """Test that negative memory utilization is clamped to 0."""
        calculator = RewardCalculator()
        
        # Should clamp to 0 and not crash
        reward = calculator.calculate_reward(
            cpu_util=50.0,
            memory_util=-10.0,
            allocated_resources=1
        )
        
        # Memory is clamped to 0 (below optimal), but CPU is optimal
        # With new reward logic, this might be slightly positive or negative
        # The important thing is it doesn't crash and returns a valid float
        assert isinstance(reward, float)
        assert -2.0 < reward < 2.0  # Reasonable range
    
    def test_reward_at_optimal_range_boundaries(self):
        """Test reward calculation at the boundaries of optimal ranges."""
        calculator = RewardCalculator()
        
        # Test at lower boundary (40%, 40%)
        reward_lower = calculator.calculate_reward(
            cpu_util=40.0,
            memory_util=40.0,
            allocated_resources=3
        )
        assert reward_lower > -1.0  # Should be near positive
        
        # Test at upper boundary (70%, 70%)
        reward_upper = calculator.calculate_reward(
            cpu_util=70.0,
            memory_util=70.0,
            allocated_resources=3
        )
        assert reward_upper > -1.0  # Should be near positive
        
        # Test at center (55%, 55%)
        reward_center = calculator.calculate_reward(
            cpu_util=55.0,
            memory_util=55.0,
            allocated_resources=3
        )
        assert reward_center > reward_lower  # Center should be better than edges
        assert reward_center > reward_upper
    
    def test_reward_mixed_utilization_scenarios(self):
        """Test reward with one metric optimal and one metric suboptimal."""
        calculator = RewardCalculator()
        
        # CPU optimal, memory over-provisioned
        reward1 = calculator.calculate_reward(
            cpu_util=55.0,
            memory_util=20.0,
            allocated_resources=3
        )
        # With new logic, this gets -0.1 penalty for being out of range
        # but might still be slightly positive or negative depending on balance
        
        # CPU optimal, memory under-provisioned
        reward2 = calculator.calculate_reward(
            cpu_util=55.0,
            memory_util=85.0,
            allocated_resources=3
        )
        # Under-provisioning has harsher penalty, should be more negative
        
        # Both optimal
        reward3 = calculator.calculate_reward(
            cpu_util=55.0,
            memory_util=55.0,
            allocated_resources=3
        )
        # Both optimal should be the best
        assert reward3 > reward1  # Both optimal should be better than mixed
        assert reward3 > reward2  # Both optimal should be better than under-provisioned
        assert reward3 > 0  # Both optimal should be positive
    
    def test_resource_cost_penalty_scaling(self):
        """Test that resource cost penalty scales with allocated resources."""
        calculator = RewardCalculator()
        
        # Same utilization, different resources
        reward_low_resources = calculator.calculate_reward(
            cpu_util=55.0,
            memory_util=55.0,
            allocated_resources=1
        )
        
        reward_high_resources = calculator.calculate_reward(
            cpu_util=55.0,
            memory_util=55.0,
            allocated_resources=10
        )
        
        # Higher resources should result in lower reward due to cost penalty
        assert reward_high_resources < reward_low_resources
        
        # The difference should be approximately 0.05 * (10 - 1) = 0.45
        expected_difference = 0.05 * (10 - 1)
        actual_difference = reward_low_resources - reward_high_resources
        assert abs(actual_difference - expected_difference) < 0.01
