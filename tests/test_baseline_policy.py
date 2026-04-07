"""
Unit tests for baseline policy logic.

This module tests the baseline_policy function to ensure it returns
the correct action based on CPU utilization thresholds.
"""

import pytest
from hypothesis import given, strategies as st, settings


class TestBaselinePolicyLogic:
    """Test suite for baseline policy decision logic."""
    
    def test_high_cpu_returns_increase(self):
        """Test that CPU > 70% returns action 2 (increase)."""
        from run_baseline import baseline_policy
        
        # Test exact boundary
        assert baseline_policy(70.01) == 2
        
        # Test well above threshold
        assert baseline_policy(80.0) == 2
        assert baseline_policy(90.0) == 2
        assert baseline_policy(100.0) == 2
    
    def test_low_cpu_returns_decrease(self):
        """Test that CPU < 40% returns action 0 (decrease)."""
        from run_baseline import baseline_policy
        
        # Test exact boundary
        assert baseline_policy(39.99) == 0
        
        # Test well below threshold
        assert baseline_policy(30.0) == 0
        assert baseline_policy(20.0) == 0
        assert baseline_policy(0.0) == 0
    
    def test_medium_cpu_returns_maintain(self):
        """Test that 40% <= CPU <= 70% returns action 1 (maintain)."""
        from run_baseline import baseline_policy
        
        # Test exact boundaries
        assert baseline_policy(40.0) == 1
        assert baseline_policy(70.0) == 1
        
        # Test middle range
        assert baseline_policy(50.0) == 1
        assert baseline_policy(55.0) == 1
        assert baseline_policy(60.0) == 1


class TestBaselinePolicyProperties:
    """
    Property 7: Baseline Policy Behavior
    **Validates: Requirements 7.3, 7.4, 7.5**
    
    Property-based tests verifying baseline policy behavior across all CPU values.
    """
    
    @given(cpu=st.floats(min_value=70.01, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_high_cpu_always_increases(self, cpu):
        """
        Property: For all CPU > 70%, baseline_policy returns 2 (increase).
        Validates: Requirement 7.3
        """
        from run_baseline import baseline_policy
        
        action = baseline_policy(cpu)
        assert action == 2, f"Expected action 2 for CPU={cpu:.2f}%, got {action}"
    
    @given(cpu=st.floats(min_value=0.0, max_value=39.99, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_low_cpu_always_decreases(self, cpu):
        """
        Property: For all CPU < 40%, baseline_policy returns 0 (decrease).
        Validates: Requirement 7.4
        """
        from run_baseline import baseline_policy
        
        action = baseline_policy(cpu)
        assert action == 0, f"Expected action 0 for CPU={cpu:.2f}%, got {action}"
    
    @given(cpu=st.floats(min_value=40.0, max_value=70.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_medium_cpu_always_maintains(self, cpu):
        """
        Property: For all 40% <= CPU <= 70%, baseline_policy returns 1 (maintain).
        Validates: Requirement 7.5
        """
        from run_baseline import baseline_policy
        
        action = baseline_policy(cpu)
        assert action == 1, f"Expected action 1 for CPU={cpu:.2f}%, got {action}"
