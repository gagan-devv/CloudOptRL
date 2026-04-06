"""
Unit tests for PyTorch integration in CloudResourceEnv.

This module tests the minimal PyTorch tensor-based state representation
added to the environment while ensuring NumPy compatibility is maintained.
"""

import pytest
import numpy as np
import torch
from env.environment import CloudResourceEnv
from env.config import EnvConfig


def test_get_state_tensor_returns_torch_tensor():
    """Test that get_state_tensor returns a PyTorch tensor."""
    env = CloudResourceEnv()
    env.reset()
    
    state_tensor = env.get_state_tensor()
    
    assert isinstance(state_tensor, torch.Tensor), "get_state_tensor should return a torch.Tensor"


def test_get_state_tensor_matches_numpy_observation():
    """Test that PyTorch tensor state matches NumPy observation."""
    env = CloudResourceEnv()
    observation = env.reset()
    
    state_tensor = env.get_state_tensor()
    
    # Convert tensor to numpy for comparison
    tensor_as_numpy = state_tensor.numpy()
    
    # Should match the observation exactly
    np.testing.assert_array_equal(
        tensor_as_numpy,
        observation,
        err_msg="PyTorch tensor state should match NumPy observation"
    )


def test_get_state_tensor_after_step():
    """Test that get_state_tensor works correctly after environment steps."""
    env = CloudResourceEnv()
    env.reset()
    
    # Take a step
    observation, reward, done, info = env.step(CloudResourceEnv.ACTION_INCREASE)
    
    # Get tensor representation
    state_tensor = env.get_state_tensor()
    
    # Verify it matches the observation
    tensor_as_numpy = state_tensor.numpy()
    np.testing.assert_array_equal(
        tensor_as_numpy,
        observation,
        err_msg="Tensor state should match observation after step"
    )


def test_get_state_tensor_shape():
    """Test that state tensor has correct shape."""
    env = CloudResourceEnv()
    env.reset()
    
    state_tensor = env.get_state_tensor()
    
    assert state_tensor.shape == (4,), f"State tensor should have shape (4,), got {state_tensor.shape}"


def test_get_state_tensor_dtype():
    """Test that state tensor has correct dtype (float32)."""
    env = CloudResourceEnv()
    env.reset()
    
    state_tensor = env.get_state_tensor()
    
    assert state_tensor.dtype == torch.float32, f"State tensor should be float32, got {state_tensor.dtype}"


def test_numpy_compatibility_maintained():
    """Test that existing NumPy-based functionality still works."""
    env = CloudResourceEnv()
    
    # Reset should still return NumPy array
    observation = env.reset()
    assert isinstance(observation, np.ndarray), "reset() should still return NumPy array"
    
    # Step should still return NumPy array
    observation, reward, done, info = env.step(CloudResourceEnv.ACTION_MAINTAIN)
    assert isinstance(observation, np.ndarray), "step() should still return NumPy array"


def test_get_state_tensor_multiple_calls():
    """Test that multiple calls to get_state_tensor return consistent results."""
    env = CloudResourceEnv()
    env.reset()
    
    tensor1 = env.get_state_tensor()
    tensor2 = env.get_state_tensor()
    
    # Should be equal since state hasn't changed
    assert torch.equal(tensor1, tensor2), "Multiple calls should return equal tensors"


def test_get_state_tensor_updates_with_state():
    """Test that state tensor updates when environment state changes."""
    env = CloudResourceEnv()
    env.reset()
    
    tensor_before = env.get_state_tensor().clone()
    
    # Take multiple steps to ensure state changes
    for _ in range(5):
        observation, reward, done, info = env.step(CloudResourceEnv.ACTION_INCREASE)
        if done:
            break
    
    tensor_after = env.get_state_tensor()
    
    # Tensors should be different (state has changed)
    assert not torch.equal(tensor_before, tensor_after), "State tensor should update with environment state"
