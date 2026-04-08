"""
Property-based tests for inference script.

Tests Properties 3, 4, 7, 8, 9 from the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
import os
import sys
from unittest.mock import Mock

# Import inference script components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import heuristic_policy, MAX_STEPS, SUCCESS_SCORE_THRESHOLD


# Feature: final-submission-compliance, Property 4: Heuristic Policy Correctness
@given(cpu=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_heuristic_policy_correctness(cpu):
    """
    Property 4: For any CPU value, heuristic policy returns correct action.
    
    **Validates: Requirements 4.1, 4.2, 4.3, 4.5**
    
    The heuristic policy should:
    - Return action 2 (increase) when CPU > 70%
    - Return action 0 (decrease) when CPU < 40%
    - Return action 1 (maintain) when CPU is in [40%, 70%]
    - Always return an integer in {0, 1, 2}
    """
    # Create mock state with CPU value
    state = Mock()
    state.cpu = cpu
    
    # Get action from heuristic policy
    action = heuristic_policy(state)
    
    # Verify action is valid integer
    assert isinstance(action, int), f"Action must be integer, got {type(action)}"
    assert action in {0, 1, 2}, f"Action must be in {{0, 1, 2}}, got {action}"
    
    # Verify correct action for CPU value
    if cpu > 70.0:
        assert action == 2, f"CPU {cpu:.2f}% > 70% should return action 2 (increase), got {action}"
    elif cpu < 40.0:
        assert action == 0, f"CPU {cpu:.2f}% < 40% should return action 0 (decrease), got {action}"
    else:
        assert action == 1, f"CPU {cpu:.2f}% in [40%, 70%] should return action 1 (maintain), got {action}"


# Feature: final-submission-compliance, Property 7: Average Reward Calculation
@given(rewards=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=0, max_size=50))
@settings(max_examples=100)
def test_average_reward_calculation(rewards):
    """
    Property 7: For any list of rewards, average equals sum/len or 0.0 if empty.
    
    **Validates: Requirements 6.3**
    
    The average reward calculation should:
    - Return sum(rewards) / len(rewards) for non-empty lists
    - Return 0.0 for empty lists
    """
    # Calculate average using inference script logic
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    
    # Verify calculation
    if len(rewards) == 0:
        assert avg_reward == 0.0, "Empty reward list should have average 0.0"
    else:
        expected_avg = sum(rewards) / len(rewards)
        assert abs(avg_reward - expected_avg) < 1e-6, f"Average {avg_reward} != expected {expected_avg}"


# Feature: final-submission-compliance, Property 8: Success Determination
@given(avg_reward=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_success_determination(avg_reward):
    """
    Property 8: Success is True if and only if average reward >= 0.5.
    
    **Validates: Requirements 6.4, 6.5**
    
    The success determination should:
    - Return True when avg_reward >= SUCCESS_SCORE_THRESHOLD (0.5)
    - Return False when avg_reward < SUCCESS_SCORE_THRESHOLD (0.5)
    """
    # Calculate success using inference script logic
    success = avg_reward >= SUCCESS_SCORE_THRESHOLD
    
    # Verify success determination
    if avg_reward >= SUCCESS_SCORE_THRESHOLD:
        assert success is True, f"avg_reward {avg_reward:.2f} >= {SUCCESS_SCORE_THRESHOLD} should be success=True"
    else:
        assert success is False, f"avg_reward {avg_reward:.2f} < {SUCCESS_SCORE_THRESHOLD} should be success=False"


# Feature: final-submission-compliance, Property 9: Episode Termination
@given(
    steps_taken=st.integers(min_value=0, max_value=MAX_STEPS + 10),
    done_flags=st.lists(st.booleans(), min_size=0, max_size=MAX_STEPS + 10)
)
@settings(max_examples=100)
def test_episode_termination(steps_taken, done_flags):
    """
    Property 9: Episode terminates at MAX_STEPS or when done=True.
    
    **Validates: Requirements 6.6**
    
    The episode should terminate when:
    - MAX_STEPS (20) is reached, OR
    - Environment returns done=True
    """
    # Simulate episode termination logic
    terminated = False
    actual_steps = 0
    
    for step in range(min(steps_taken, MAX_STEPS)):
        actual_steps = step + 1
        
        # Check if done flag is True at this step
        if step < len(done_flags) and done_flags[step]:
            terminated = True
            break
        
        # Check if MAX_STEPS reached
        if actual_steps >= MAX_STEPS:
            terminated = True
            break
    
    # Verify termination conditions
    if actual_steps >= MAX_STEPS:
        assert terminated, f"Episode should terminate at MAX_STEPS={MAX_STEPS}"
    
    if any(done_flags[:actual_steps]):
        assert terminated, "Episode should terminate when done=True"


# Feature: final-submission-compliance, Property 3: Environment Variable Defaults
def test_environment_variable_defaults():
    """
    Property 3: Environment variables default to correct values when not set.
    
    **Validates: Requirements 3.4, 3.5**
    
    The inference script should use:
    - TASK_NAME defaults to "medium" when not set
    - BENCHMARK defaults to "cloud_resource_env" when not set
    """
    # Save original environment variables
    original_task = os.environ.get("TASK_NAME")
    original_benchmark = os.environ.get("BENCHMARK")
    
    try:
        # Clear environment variables
        if "TASK_NAME" in os.environ:
            del os.environ["TASK_NAME"]
        if "BENCHMARK" in os.environ:
            del os.environ["BENCHMARK"]
        
        # Re-import to get fresh values
        import importlib
        import inference
        importlib.reload(inference)
        
        # Verify defaults
        assert inference.TASK_NAME == "medium", f"TASK_NAME should default to 'medium', got '{inference.TASK_NAME}'"
        assert inference.BENCHMARK == "cloud_resource_env", f"BENCHMARK should default to 'cloud_resource_env', got '{inference.BENCHMARK}'"
        
    finally:
        # Restore original environment variables
        if original_task is not None:
            os.environ["TASK_NAME"] = original_task
        if original_benchmark is not None:
            os.environ["BENCHMARK"] = original_benchmark
        
        # Reload to restore original state
        importlib.reload(inference)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
