"""
Unit tests for the CloudResourceEnv environment.

These tests validate specific examples and edge cases for environment behavior,
including termination conditions and error handling.
"""

import pytest
import numpy as np
from env.environment import CloudResourceEnv
from env.config import EnvConfig


class TestTerminationConditions:
    """Test suite for episode termination conditions."""
    
    def test_episode_terminates_at_max_steps(self):
        """
        Test that episode terminates when step count reaches max_steps.
        
        Requirements: 5.1
        """
        config = EnvConfig(max_steps=10)
        env = CloudResourceEnv(config=config)
        env.reset()
        
        # Execute steps up to max_steps
        for i in range(10):
            obs, reward, done, info = env.step(CloudResourceEnv.ACTION_MAINTAIN)
            
            if i < 9:
                # Should not be done before max_steps
                assert not done, f"Episode should not terminate at step {i+1}"
            else:
                # Should be done at max_steps
                assert done, f"Episode should terminate at step {i+1} (max_steps={config.max_steps})"
    
    def test_episode_terminates_when_cpu_exceeds_threshold(self):
        """
        Test that episode terminates when CPU utilization exceeds 95%.
        
        Requirements: 5.2
        """
        # Configure environment to create high CPU utilization
        config = EnvConfig(
            initial_resources=1,  # Minimal resources
            base_request_rate=200,  # High request rate
            request_rate_std=0.0,  # No randomness
            termination_threshold=95.0
        )
        env = CloudResourceEnv(config=config)
        env.reset()
        
        # Check if CPU is already above threshold after reset
        if env.cpu_util > config.termination_threshold:
            # Episode should terminate on first step
            obs, reward, done, info = env.step(CloudResourceEnv.ACTION_MAINTAIN)
            assert done, f"Episode should terminate when CPU={env.cpu_util:.2f}% > {config.termination_threshold}%"
        else:
            # Decrease resources to increase CPU utilization
            obs, reward, done, info = env.step(CloudResourceEnv.ACTION_DECREASE)
            
            # Check if termination occurred
            if env.cpu_util > config.termination_threshold:
                assert done, f"Episode should terminate when CPU={env.cpu_util:.2f}% > {config.termination_threshold}%"
    
    def test_episode_terminates_when_memory_exceeds_threshold(self):
        """
        Test that episode terminates when memory utilization exceeds 95%.
        
        Requirements: 5.3
        """
        # Configure environment to create high memory utilization
        config = EnvConfig(
            initial_resources=1,  # Minimal resources
            base_request_rate=200,  # High request rate
            request_rate_std=0.0,  # No randomness
            termination_threshold=95.0
        )
        env = CloudResourceEnv(config=config)
        env.reset()
        
        # Check if memory is already above threshold after reset
        if env.memory_util > config.termination_threshold:
            # Episode should terminate on first step
            obs, reward, done, info = env.step(CloudResourceEnv.ACTION_MAINTAIN)
            assert done, f"Episode should terminate when Memory={env.memory_util:.2f}% > {config.termination_threshold}%"
        else:
            # Decrease resources to increase memory utilization
            obs, reward, done, info = env.step(CloudResourceEnv.ACTION_DECREASE)
            
            # Check if termination occurred
            if env.memory_util > config.termination_threshold:
                assert done, f"Episode should terminate when Memory={env.memory_util:.2f}% > {config.termination_threshold}%"
    
    def test_error_raised_when_step_called_after_termination(self):
        """
        Test that RuntimeError is raised when step() is called after episode termination.
        
        Requirements: 5.4
        """
        config = EnvConfig(max_steps=5)
        env = CloudResourceEnv(config=config)
        env.reset()
        
        # Execute steps until termination
        for _ in range(5):
            obs, reward, done, info = env.step(CloudResourceEnv.ACTION_MAINTAIN)
        
        # Verify episode is terminated
        assert env.done, "Episode should be terminated"
        
        # Attempting to call step() should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot call step.*after episode termination"):
            env.step(CloudResourceEnv.ACTION_MAINTAIN)


class TestInvalidActions:
    """Test suite for invalid action handling."""
    
    def test_error_raised_for_negative_action(self):
        """
        Test that ValueError is raised for action < 0.
        
        Requirements: 2.1, 2.2, 2.3
        """
        env = CloudResourceEnv()
        env.reset()
        
        # Attempting to execute negative action should raise ValueError
        with pytest.raises(ValueError, match="Invalid action.*Must be 0.*1.*or 2"):
            env.step(-1)
    
    def test_error_raised_for_action_greater_than_2(self):
        """
        Test that ValueError is raised for action > 2.
        
        Requirements: 2.1, 2.2, 2.3
        """
        env = CloudResourceEnv()
        env.reset()
        
        # Attempting to execute action > 2 should raise ValueError
        with pytest.raises(ValueError, match="Invalid action.*Must be 0.*1.*or 2"):
            env.step(3)
    
    def test_state_unchanged_after_invalid_action(self):
        """
        Test that state remains unchanged when invalid action is rejected.
        
        Requirements: 2.1, 2.2, 2.3
        """
        env = CloudResourceEnv()
        obs_initial = env.reset()
        
        # Capture initial state
        cpu_before = env.cpu_util
        memory_before = env.memory_util
        request_rate_before = env.request_rate
        resources_before = env.allocated_resources
        step_before = env.current_step
        
        # Attempt invalid action (should raise ValueError)
        try:
            env.step(5)
        except ValueError:
            pass  # Expected
        
        # Verify state is unchanged
        assert env.cpu_util == cpu_before, "CPU utilization should be unchanged"
        assert env.memory_util == memory_before, "Memory utilization should be unchanged"
        assert env.request_rate == request_rate_before, "Request rate should be unchanged"
        assert env.allocated_resources == resources_before, "Allocated resources should be unchanged"
        assert env.current_step == step_before, "Step count should be unchanged"
        assert not env.done, "Done flag should remain False"


class TestStepMethod:
    """Test suite for step() method behavior."""
    
    def test_step_increments_counter(self):
        """Test that step() increments the step counter."""
        env = CloudResourceEnv()
        env.reset()
        
        assert env.current_step == 0, "Step counter should be 0 after reset"
        
        env.step(CloudResourceEnv.ACTION_MAINTAIN)
        assert env.current_step == 1, "Step counter should be 1 after first step"
        
        env.step(CloudResourceEnv.ACTION_MAINTAIN)
        assert env.current_step == 2, "Step counter should be 2 after second step"
    
    def test_step_stores_transition_in_history(self):
        """Test that step() stores state, action, and reward in episode history."""
        env = CloudResourceEnv()
        env.reset()
        
        # Execute a step
        action = CloudResourceEnv.ACTION_INCREASE
        obs, reward, done, info = env.step(action)
        
        # Verify history is updated
        assert len(env.episode_states) == 1, "Should have 1 state in history"
        assert len(env.episode_actions) == 1, "Should have 1 action in history"
        assert len(env.episode_rewards) == 1, "Should have 1 reward in history"
        
        # Verify stored values
        assert np.allclose(env.episode_states[0], obs), "Stored state should match observation"
        assert env.episode_actions[0] == action, "Stored action should match executed action"
        assert env.episode_rewards[0] == reward, "Stored reward should match returned reward"
    
    def test_step_updates_cumulative_reward(self):
        """Test that step() updates cumulative reward correctly."""
        env = CloudResourceEnv()
        env.reset()
        
        assert env.cumulative_reward == 0.0, "Cumulative reward should be 0 after reset"
        
        # Execute first step
        obs1, reward1, done1, info1 = env.step(CloudResourceEnv.ACTION_MAINTAIN)
        assert env.cumulative_reward == reward1, f"Cumulative reward should be {reward1}"
        assert info1['cumulative_reward'] == reward1, "Info should contain cumulative reward"
        
        # Execute second step
        obs2, reward2, done2, info2 = env.step(CloudResourceEnv.ACTION_MAINTAIN)
        expected_cumulative = reward1 + reward2
        assert abs(env.cumulative_reward - expected_cumulative) < 1e-6, (
            f"Cumulative reward should be {expected_cumulative}"
        )
        assert abs(info2['cumulative_reward'] - expected_cumulative) < 1e-6, (
            "Info should contain updated cumulative reward"
        )
    
    def test_step_calls_reward_calculator(self):
        """Test that step() calls the reward calculator with correct parameters."""
        env = CloudResourceEnv()
        env.reset()
        
        # Execute step
        obs, reward, done, info = env.step(CloudResourceEnv.ACTION_MAINTAIN)
        
        # Verify reward is calculated (should be a float)
        assert isinstance(reward, (float, np.floating)), "Reward should be a float"
        
        # Verify reward matches what we'd expect from the reward calculator
        expected_reward = env.reward_calculator.calculate_reward(
            env.cpu_util,
            env.memory_util,
            env.allocated_resources
        )
        assert abs(reward - expected_reward) < 1e-6, (
            f"Reward should match reward calculator output: {expected_reward} vs {reward}"
        )
