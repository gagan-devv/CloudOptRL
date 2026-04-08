"""
Property-based tests for AsyncEnvWrapper.

This module validates that AsyncEnvWrapper correctly wraps CloudResourceEnv
with async methods while preserving all return values and behavior.

Feature: final-submission-compliance
Properties tested:
- Property 1: AsyncEnvWrapper Reset Equivalence
"""

import pytest
import asyncio
import numpy as np
from hypothesis import given, strategies as st, settings
from env.environment import CloudResourceEnv
from env.async_wrapper import AsyncEnvWrapper


# Feature: final-submission-compliance, Property 1: AsyncEnvWrapper Reset Equivalence
@given(
    task_name=st.sampled_from(["easy", "medium", "hard"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1)
)
@settings(max_examples=100, deadline=None)
def test_async_reset_equivalence(task_name, seed):
    """
    Property 1: For any CloudResourceEnv instance, calling async reset() through
    AsyncEnvWrapper should return the same observation as calling synchronous
    reset() directly on the environment.
    
    Validates: Requirements 2.2
    """
    # Create two identical environments with same seed
    env1 = CloudResourceEnv(task_name=task_name)
    env1.rng = np.random.RandomState(seed)
    
    env2 = CloudResourceEnv(task_name=task_name)
    env2.rng = np.random.RandomState(seed)
    
    # Wrap second environment with AsyncEnvWrapper
    async_env = AsyncEnvWrapper(env2)
    
    # Get synchronous reset observation
    sync_obs = env1.reset()
    
    # Get async reset observation
    async_obs = asyncio.run(async_env.reset())
    
    # Verify observations are identical
    np.testing.assert_array_equal(
        sync_obs,
        async_obs,
        err_msg=f"Async reset observation differs from sync reset for task={task_name}, seed={seed}"
    )
    
    # Verify observation shape and dtype
    assert sync_obs.shape == async_obs.shape, "Observation shapes differ"
    assert sync_obs.dtype == async_obs.dtype, "Observation dtypes differ"
    
    # Verify observation has expected structure [cpu, memory, request_rate, resources]
    assert len(async_obs) == 4, "Observation should have 4 elements"
    
    # Verify all values are finite
    assert np.all(np.isfinite(async_obs)), "Observation contains non-finite values"


@pytest.mark.parametrize("task_name", ["easy", "medium", "hard"])
def test_async_reset_preserves_state_structure(task_name):
    """
    Verify that async reset returns observation with correct structure.
    
    This test ensures AsyncEnvWrapper.reset() returns a numpy array with
    the expected shape, dtype, and value ranges.
    """
    env = CloudResourceEnv(task_name=task_name)
    async_env = AsyncEnvWrapper(env)
    
    obs = asyncio.run(async_env.reset())
    
    # Verify type and shape
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (4,), "Observation should have shape (4,)"
    assert obs.dtype == np.float32, "Observation should be float32"
    
    # Verify value ranges
    cpu_util, memory_util, request_rate, allocated_resources = obs
    
    assert 0.0 <= cpu_util <= 100.0, f"CPU utilization {cpu_util} out of range [0, 100]"
    assert 0.0 <= memory_util <= 100.0, f"Memory utilization {memory_util} out of range [0, 100]"
    assert request_rate >= 0.0, f"Request rate {request_rate} should be non-negative"
    assert allocated_resources >= 1.0, f"Allocated resources {allocated_resources} should be >= 1"


def test_async_reset_multiple_calls():
    """
    Verify that multiple async reset calls produce different observations
    due to stochastic initialization.
    
    This test ensures AsyncEnvWrapper correctly propagates the stochastic
    behavior of CloudResourceEnv.reset().
    """
    env = CloudResourceEnv(task_name="medium")
    async_env = AsyncEnvWrapper(env)
    
    observations = []
    for _ in range(10):
        obs = asyncio.run(async_env.reset())
        observations.append(obs.copy())
    
    # Verify at least some observations differ (stochastic initialization)
    # We check request_rate (index 2) which has stochastic variation
    request_rates = [obs[2] for obs in observations]
    unique_rates = set(request_rates)
    
    # With 10 resets, we expect some variation in request rates
    assert len(unique_rates) > 1, "Expected stochastic variation in reset observations"


def test_async_reset_after_episode():
    """
    Verify that async reset works correctly after an episode completes.
    
    This test ensures AsyncEnvWrapper.reset() can be called after step()
    to start a new episode.
    """
    env = CloudResourceEnv(task_name="medium")
    async_env = AsyncEnvWrapper(env)
    
    # Run initial episode
    obs1 = asyncio.run(async_env.reset())
    
    # Take a few steps
    for _ in range(5):
        obs, reward, done, info = asyncio.run(async_env.step(1))  # maintain action
        if done:
            break
    
    # Reset for new episode
    obs2 = asyncio.run(async_env.reset())
    
    # Verify reset returns valid observation
    assert isinstance(obs2, np.ndarray), "Reset should return numpy array"
    assert obs2.shape == (4,), "Reset observation should have shape (4,)"
    assert np.all(np.isfinite(obs2)), "Reset observation should contain finite values"


# Feature: final-submission-compliance, Property 2: AsyncEnvWrapper Step Equivalence
@given(
    task_name=st.sampled_from(["easy", "medium", "hard"]),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    action=st.integers(min_value=0, max_value=2)
)
@settings(max_examples=100, deadline=None)
def test_async_step_equivalence(task_name, seed, action):
    """
    Property 2: For any CloudResourceEnv instance and any valid action, calling
    async step(action) through AsyncEnvWrapper should return the same
    (observation, reward, done, info) tuple as calling synchronous step(action)
    directly on the environment.
    
    Validates: Requirements 2.3
    """
    # Create two identical environments with same seed
    env1 = CloudResourceEnv(task_name=task_name)
    env1.rng = np.random.RandomState(seed)
    
    env2 = CloudResourceEnv(task_name=task_name)
    env2.rng = np.random.RandomState(seed)
    
    # Reset both environments to same initial state
    env1.reset()
    env2.reset()
    
    # Wrap second environment with AsyncEnvWrapper
    async_env = AsyncEnvWrapper(env2)
    
    # Execute synchronous step
    sync_obs, sync_reward, sync_done, sync_info = env1.step(action)
    
    # Execute async step
    async_obs, async_reward, async_done, async_info = asyncio.run(async_env.step(action))
    
    # Verify observations are identical
    np.testing.assert_array_equal(
        sync_obs,
        async_obs,
        err_msg=f"Async step observation differs from sync step for task={task_name}, seed={seed}, action={action}"
    )
    
    # Verify rewards are identical
    assert sync_reward == async_reward, \
        f"Async step reward {async_reward} differs from sync step reward {sync_reward}"
    
    # Verify done flags are identical
    assert sync_done == async_done, \
        f"Async step done {async_done} differs from sync step done {sync_done}"
    
    # Verify info dictionaries have same keys
    assert set(sync_info.keys()) == set(async_info.keys()), \
        f"Info dictionary keys differ: sync={set(sync_info.keys())}, async={set(async_info.keys())}"
    
    # Verify info dictionary values are identical
    for key in sync_info.keys():
        if isinstance(sync_info[key], float):
            # Use approximate equality for floating point values
            assert abs(sync_info[key] - async_info[key]) < 1e-6, \
                f"Info[{key}] differs: sync={sync_info[key]}, async={async_info[key]}"
        else:
            assert sync_info[key] == async_info[key], \
                f"Info[{key}] differs: sync={sync_info[key]}, async={async_info[key]}"


@pytest.mark.parametrize("task_name", ["easy", "medium", "hard"])
@pytest.mark.parametrize("action", [0, 1, 2])
def test_async_step_preserves_tuple_structure(task_name, action):
    """
    Verify that async step returns tuple with correct structure.
    
    This test ensures AsyncEnvWrapper.step() returns a 4-tuple with
    (observation, reward, done, info) matching expected types.
    """
    env = CloudResourceEnv(task_name=task_name)
    async_env = AsyncEnvWrapper(env)
    
    asyncio.run(async_env.reset())
    obs, reward, done, info = asyncio.run(async_env.step(action))
    
    # Verify observation type and shape
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (4,), "Observation should have shape (4,)"
    assert obs.dtype == np.float32, "Observation should be float32"
    
    # Verify reward type
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert np.isfinite(reward), "Reward should be finite"
    
    # Verify done type
    assert isinstance(done, bool), "Done should be boolean"
    
    # Verify info type and structure
    assert isinstance(info, dict), "Info should be dictionary"
    expected_keys = {'step', 'cumulative_reward', 'cpu_util', 'memory_util', 
                     'request_rate', 'allocated_resources', 'latency'}
    assert set(info.keys()) == expected_keys, \
        f"Info keys {set(info.keys())} differ from expected {expected_keys}"


def test_async_step_sequence_equivalence():
    """
    Verify that a sequence of async steps produces same results as sync steps.
    
    This test ensures AsyncEnvWrapper maintains equivalence across multiple
    steps in an episode, not just a single step.
    """
    seed = 42
    task_name = "medium"
    actions = [1, 2, 1, 0, 1]  # sequence of actions
    
    # Create two identical environments
    env1 = CloudResourceEnv(task_name=task_name)
    env1.rng = np.random.RandomState(seed)
    
    env2 = CloudResourceEnv(task_name=task_name)
    env2.rng = np.random.RandomState(seed)
    
    async_env = AsyncEnvWrapper(env2)
    
    # Reset both environments
    sync_obs = env1.reset()
    async_obs = asyncio.run(async_env.reset())
    
    np.testing.assert_array_equal(sync_obs, async_obs, 
                                   err_msg="Initial observations differ")
    
    # Execute sequence of steps
    for i, action in enumerate(actions):
        sync_obs, sync_reward, sync_done, sync_info = env1.step(action)
        async_obs, async_reward, async_done, async_info = asyncio.run(async_env.step(action))
        
        # Verify equivalence at each step
        np.testing.assert_array_equal(
            sync_obs, async_obs,
            err_msg=f"Observations differ at step {i+1}"
        )
        
        assert sync_reward == async_reward, \
            f"Rewards differ at step {i+1}: sync={sync_reward}, async={async_reward}"
        
        assert sync_done == async_done, \
            f"Done flags differ at step {i+1}: sync={sync_done}, async={async_done}"
        
        # If episode terminated, stop
        if sync_done:
            break


def test_async_step_invalid_action():
    """
    Verify that async step raises ValueError for invalid actions.
    
    This test ensures AsyncEnvWrapper correctly propagates exceptions
    from the wrapped environment.
    """
    env = CloudResourceEnv(task_name="medium")
    async_env = AsyncEnvWrapper(env)
    
    asyncio.run(async_env.reset())
    
    # Test invalid action values
    invalid_actions = [-1, 3, 10, -5]
    
    for invalid_action in invalid_actions:
        with pytest.raises(ValueError, match="Invalid action"):
            asyncio.run(async_env.step(invalid_action))


def test_async_step_after_done():
    """
    Verify that async step raises RuntimeError when called after episode termination.
    
    This test ensures AsyncEnvWrapper correctly propagates the done state
    and prevents steps after termination.
    """
    env = CloudResourceEnv(task_name="medium")
    async_env = AsyncEnvWrapper(env)
    
    asyncio.run(async_env.reset())
    
    # Run until episode terminates
    max_steps = 100
    for _ in range(max_steps):
        obs, reward, done, info = asyncio.run(async_env.step(1))
        if done:
            break
    
    # Verify we reached done state
    assert done, "Episode should have terminated within max_steps"
    
    # Attempt to step after done should raise RuntimeError
    with pytest.raises(RuntimeError, match="Cannot call step.*after episode termination"):
        asyncio.run(async_env.step(1))


def test_async_step_preserves_episode_history():
    """
    Verify that async step correctly updates episode history in wrapped environment.
    
    This test ensures AsyncEnvWrapper doesn't interfere with the environment's
    internal state tracking.
    """
    env = CloudResourceEnv(task_name="medium")
    async_env = AsyncEnvWrapper(env)
    
    asyncio.run(async_env.reset())
    
    # Execute several steps
    actions = [1, 2, 0, 1]
    for action in actions:
        asyncio.run(async_env.step(action))
    
    # Verify episode history was updated
    assert len(env.episode_actions) == len(actions), \
        f"Expected {len(actions)} actions in history, got {len(env.episode_actions)}"
    
    assert env.episode_actions == actions, \
        f"Episode actions {env.episode_actions} differ from expected {actions}"
    
    assert len(env.episode_rewards) == len(actions), \
        f"Expected {len(actions)} rewards in history, got {len(env.episode_rewards)}"
    
    # Environment stores states after each step (not including initial state from reset)
    assert len(env.episode_states) == len(actions), \
        f"Expected {len(actions)} states in history, got {len(env.episode_states)}"
