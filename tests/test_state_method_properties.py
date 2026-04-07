"""
Property-based tests for CloudResourceEnv state() method.

This module validates that the state() method returns consistent EnvState
instances that match the internal state values.
"""

import pytest
from hypothesis import given, strategies as st, settings

from env.environment import CloudResourceEnv
from env.config import EnvState


class TestStateMethodConsistency:
    """Property 3: State Method Consistency"""
    
    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=50)
    def test_state_matches_internal_state_after_reset(self, seed):
        """Test that state() returns EnvState matching internal state after reset()."""
        env = CloudResourceEnv()
        env.rng.seed(seed)
        env.reset()
        
        state = env.state()
        
        assert isinstance(state, EnvState)
        assert state.cpu == env.cpu_util
        assert state.memory == env.memory_util
        assert state.request_rate == float(env.request_rate)
        assert state.resources == env.allocated_resources
    
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        action=st.integers(min_value=0, max_value=2)
    )
    @settings(max_examples=50)
    def test_state_matches_internal_state_after_step(self, seed, action):
        """Test that state() returns EnvState matching internal state after step()."""
        env = CloudResourceEnv()
        env.rng.seed(seed)
        env.reset()
        env.step(action)
        
        state = env.state()
        
        assert isinstance(state, EnvState)
        assert state.cpu == env.cpu_util
        assert state.memory == env.memory_util
        assert state.request_rate == float(env.request_rate)
        assert state.resources == env.allocated_resources
    
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        actions=st.lists(st.integers(min_value=0, max_value=2), min_size=1, max_size=10)
    )
    @settings(max_examples=30)
    def test_state_consistency_across_multiple_steps(self, seed, actions):
        """Test that state() remains consistent across multiple steps."""
        env = CloudResourceEnv()
        env.rng.seed(seed)
        env.reset()
        
        for action in actions:
            if env.done:
                break  # Stop if episode terminates
            env.step(action)
            state = env.state()
            
            # Verify state matches internal values
            assert state.cpu == env.cpu_util
            assert state.memory == env.memory_util
            assert state.request_rate == float(env.request_rate)
            assert state.resources == env.allocated_resources


class TestEnvStateStructureCompleteness:
    """Property 1: EnvState Structure Completeness"""
    
    def test_state_returns_envstate_with_all_fields(self):
        """Test that state() returns EnvState with all four required fields."""
        env = CloudResourceEnv()
        env.rng.seed(42)
        env.reset()
        
        state = env.state()
        
        assert isinstance(state, EnvState)
        assert hasattr(state, 'cpu')
        assert hasattr(state, 'memory')
        assert hasattr(state, 'request_rate')
        assert hasattr(state, 'resources')
    
    @given(
        seed=st.integers(min_value=0, max_value=10000),
        task_name=st.sampled_from(["easy", "medium", "hard"])
    )
    @settings(max_examples=30)
    def test_state_fields_have_correct_types(self, seed, task_name):
        """Test that state() returns EnvState with correctly typed fields."""
        env = CloudResourceEnv(task_name=task_name)
        env.rng.seed(seed)
        env.reset()
        
        state = env.state()
        
        assert isinstance(state.cpu, float)
        assert isinstance(state.memory, float)
        assert isinstance(state.request_rate, float)
        assert isinstance(state.resources, int)
    
    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=50)
    def test_state_fields_within_valid_ranges(self, seed):
        """Test that state() returns EnvState with fields in valid ranges."""
        env = CloudResourceEnv()
        env.rng.seed(seed)
        env.reset()
        
        state = env.state()
        
        # EnvState validation ensures these constraints
        assert 0.0 <= state.cpu <= 100.0
        assert 0.0 <= state.memory <= 100.0
        assert state.request_rate >= 0.0
        assert state.resources >= 1
