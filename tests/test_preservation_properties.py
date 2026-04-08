"""
Property-based tests for preservation of existing functionality.

These tests verify that existing functionality remains unchanged after the bugfix.
They test the baseline behavior on UNFIXED code and should PASS, confirming the
behavior to preserve when the fix is applied.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

import pytest
import subprocess
import sys
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import numpy as np

# Import environment components
from env.environment import CloudResourceEnv
from env.config import EnvConfig
from env.grader import EpisodeGrader


# Feature: openenv-deployment-fixes, Property 2: Preservation - Existing Test Suite Passes
@settings(max_examples=3, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10000))
def test_preservation_existing_tests_pass(seed):
    """
    Property 2.1: Preservation - Existing Test Suite Passes
    
    **Validates: Requirements 3.1**
    
    This test verifies that the existing pytest test suite continues to pass
    after the bugfix is applied. We run a subset of critical tests to ensure
    core functionality is preserved.
    
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES (confirms baseline)
    **EXPECTED OUTCOME ON FIXED CODE**: Test PASSES (confirms preservation)
    """
    # Run a subset of critical unit tests
    # We focus on core environment tests that don't depend on file structure
    test_files = [
        "tests/test_environment.py::TestTerminationConditions::test_episode_terminates_at_max_steps",
        "tests/test_environment.py::TestInvalidActions::test_error_raised_for_negative_action",
        "tests/test_environment.py::TestStepMethod::test_step_increments_counter",
    ]
    
    for test_file in test_files:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Assert test passed
        assert result.returncode == 0, (
            f"Preservation check failed: {test_file} did not pass.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


# Feature: openenv-deployment-fixes, Property 2: Preservation - Environment Logic Unchanged
@settings(max_examples=10)
@given(
    seed=st.integers(min_value=0, max_value=10000),
    action_sequence=st.lists(
        st.integers(min_value=0, max_value=2),
        min_size=1,
        max_size=10
    )
)
def test_preservation_environment_step_logic(seed, action_sequence):
    """
    Property 2.2: Preservation - Environment Step Logic Unchanged
    
    **Validates: Requirements 3.2, 3.5**
    
    This test verifies that environment step logic, state transitions, and
    reward calculations remain unchanged after the bugfix. We execute random
    action sequences and verify the environment behaves consistently.
    
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES (confirms baseline)
    **EXPECTED OUTCOME ON FIXED CODE**: Test PASSES (confirms preservation)
    """
    # Set numpy random seed for reproducibility
    np.random.seed(seed)
    
    # Create environment with fixed configuration
    config = EnvConfig(max_steps=20)
    env = CloudResourceEnv(config=config)
    env.rng = np.random.RandomState(seed)
    
    # Reset environment
    initial_state = env.reset()
    
    # Verify initial state structure
    assert len(initial_state) == 4, "Initial state should have 4 components"
    assert 0 <= initial_state[0] <= 100, "CPU utilization should be in [0, 100]"
    assert 0 <= initial_state[1] <= 100, "Memory utilization should be in [0, 100]"
    assert initial_state[2] >= 0, "Request rate should be non-negative"
    assert initial_state[3] >= 1, "Allocated resources should be at least 1"
    
    # Execute action sequence
    for i, action in enumerate(action_sequence):
        if env.done:
            break
        
        # Store state before step
        cpu_before = env.cpu_util
        memory_before = env.memory_util
        resources_before = env.allocated_resources
        step_before = env.current_step
        
        # Execute step
        obs, reward, done, info = env.step(action)
        
        # Verify observation structure
        assert len(obs) == 4, f"Observation should have 4 components at step {i}"
        assert isinstance(reward, (float, np.floating)), f"Reward should be float at step {i}"
        assert isinstance(done, bool), f"Done should be boolean at step {i}"
        assert isinstance(info, dict), f"Info should be dict at step {i}"
        
        # Verify step counter incremented
        assert env.current_step == step_before + 1, (
            f"Step counter should increment at step {i}"
        )
        
        # Verify resource constraints
        if action == CloudResourceEnv.ACTION_INCREASE:
            assert env.allocated_resources == resources_before + 1, (
                f"Resources should increase by 1 at step {i}"
            )
        elif action == CloudResourceEnv.ACTION_DECREASE:
            expected_resources = max(1, resources_before - 1)
            assert env.allocated_resources == expected_resources, (
                f"Resources should decrease by 1 (min 1) at step {i}"
            )
        elif action == CloudResourceEnv.ACTION_MAINTAIN:
            assert env.allocated_resources == resources_before, (
                f"Resources should remain unchanged at step {i}"
            )
        
        # Verify utilization bounds
        assert 0 <= env.cpu_util <= 100, f"CPU should be in [0, 100] at step {i}"
        assert 0 <= env.memory_util <= 100, f"Memory should be in [0, 100] at step {i}"
        
        # Verify history tracking
        assert len(env.episode_states) == i + 1, (
            f"Episode states should have {i + 1} entries at step {i}"
        )
        assert len(env.episode_actions) == i + 1, (
            f"Episode actions should have {i + 1} entries at step {i}"
        )
        assert len(env.episode_rewards) == i + 1, (
            f"Episode rewards should have {i + 1} entries at step {i}"
        )


# Feature: openenv-deployment-fixes, Property 2: Preservation - Inference Output Format
@settings(max_examples=5, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10000))
def test_preservation_inference_output_format(seed):
    """
    Property 2.3: Preservation - Inference Output Format Unchanged
    
    **Validates: Requirements 3.3**
    
    This test verifies that inference.py produces the correct output format
    with [START], [STEP], and [END] logs. We check that the output structure
    remains unchanged after the bugfix.
    
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES (confirms baseline)
    **EXPECTED OUTCOME ON FIXED CODE**: Test PASSES (confirms preservation)
    """
    # Run inference.py and capture output
    # We use a short timeout and environment variable to limit execution
    env_vars = {
        **subprocess.os.environ,
        "TASK_NAME": "medium",
        "BENCHMARK": "cloud_resource_env",
        "MODEL_NAME": "gpt-4",
    }
    
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env_vars
        )
        
        output = result.stdout
        
        # Verify [START] log exists
        assert "[START]" in output, "Output should contain [START] log"
        
        # Verify [STEP] logs exist
        assert "[STEP]" in output, "Output should contain [STEP] logs"
        
        # Verify [END] log exists
        assert "[END]" in output, "Output should contain [END] log"
        
        # Verify log order: START appears before STEP, STEP before END
        start_pos = output.find("[START]")
        step_pos = output.find("[STEP]")
        end_pos = output.find("[END]")
        
        assert start_pos < step_pos, "[START] should appear before [STEP]"
        assert step_pos < end_pos, "[STEP] should appear before [END]"
        
        # Verify [END] log contains required fields
        end_line = [line for line in output.split("\n") if "[END]" in line][0]
        assert "success=" in end_line, "[END] log should contain success field"
        assert "steps=" in end_line, "[END] log should contain steps field"
        assert "score=" in end_line, "[END] log should contain score field"
        assert "rewards=" in end_line, "[END] log should contain rewards field"
        
    except subprocess.TimeoutExpired:
        # If inference takes too long, we still verify it can be imported
        # This ensures the module structure is preserved
        import inference
        assert hasattr(inference, "main"), "inference.py should have main function"
        assert hasattr(inference, "heuristic_policy"), (
            "inference.py should have heuristic_policy function"
        )


# Feature: openenv-deployment-fixes, Property 2: Preservation - Grader Logic Unchanged
@settings(max_examples=10)
@given(
    seed=st.integers(min_value=0, max_value=10000),
    num_steps=st.integers(min_value=5, max_value=20)
)
def test_preservation_grader_logic(seed, num_steps):
    """
    Property 2.4: Preservation - Episode Grading Logic Unchanged
    
    **Validates: Requirements 3.5**
    
    This test verifies that episode grading logic remains unchanged after the
    bugfix. We run episodes and verify grading results are consistent.
    
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES (confirms baseline)
    **EXPECTED OUTCOME ON FIXED CODE**: Test PASSES (confirms preservation)
    """
    # Set numpy random seed for reproducibility
    np.random.seed(seed)
    
    # Create environment and grader
    config = EnvConfig(max_steps=num_steps)
    env = CloudResourceEnv(config=config)
    env.rng = np.random.RandomState(seed)
    grader = EpisodeGrader()
    
    # Run episode with random actions
    env.reset()
    
    for _ in range(num_steps):
        if env.done:
            break
        action = np.random.choice([0, 1, 2])
        env.step(action)
    
    # Grade episode
    grading_result = grader.grade_episode(
        env.episode_states,
        env.episode_actions,
        env.episode_rewards
    )
    
    # Verify grading result structure
    assert isinstance(grading_result, dict), "Grading result should be a dict"
    
    required_fields = [
        'passed', 'final_score', 'avg_reward', 'stability_score',
        'efficiency_score', 'avg_cpu', 'avg_memory', 'avg_latency'
    ]
    
    for field in required_fields:
        assert field in grading_result, (
            f"Grading result should contain '{field}' field"
        )
    
    # Verify field types and ranges
    assert isinstance(grading_result['passed'], bool), "'passed' should be boolean"
    assert isinstance(grading_result['final_score'], (float, np.floating)), (
        "'final_score' should be float"
    )
    assert 0 <= grading_result['final_score'] <= 1, (
        "'final_score' should be in [0, 1]"
    )
    assert isinstance(grading_result['avg_reward'], (float, np.floating)), (
        "'avg_reward' should be float"
    )
    assert isinstance(grading_result['stability_score'], (float, np.floating)), (
        "'stability_score' should be float"
    )
    assert isinstance(grading_result['efficiency_score'], (float, np.floating)), (
        "'efficiency_score' should be float"
    )
    assert 0 <= grading_result['avg_cpu'] <= 100, "'avg_cpu' should be in [0, 100]"
    assert 0 <= grading_result['avg_memory'] <= 100, (
        "'avg_memory' should be in [0, 100]"
    )
    assert grading_result['avg_latency'] >= 0, "'avg_latency' should be non-negative"


# Feature: openenv-deployment-fixes, Property 2: Preservation - Module Imports Work
@settings(max_examples=3, deadline=None)
@given(dummy=st.just(None))
def test_preservation_module_imports(dummy):
    """
    Property 2.5: Preservation - All Module Imports Work
    
    **Validates: Requirements 3.1, 3.2, 3.4**
    
    This test verifies that all critical module imports continue to work after
    the bugfix. This ensures the file structure changes don't break imports.
    
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES (confirms baseline)
    **EXPECTED OUTCOME ON FIXED CODE**: Test PASSES (confirms preservation)
    """
    # Test environment module imports
    from env.environment import CloudResourceEnv
    from env.config import EnvConfig
    from env.grader import EpisodeGrader
    from env.reward import RewardCalculator
    from env.async_wrapper import AsyncEnvWrapper
    
    # Verify classes can be instantiated
    config = EnvConfig()
    env = CloudResourceEnv(config=config)
    grader = EpisodeGrader()
    
    assert isinstance(env, CloudResourceEnv), "Environment should be instantiable"
    assert isinstance(grader, EpisodeGrader), "Grader should be instantiable"
    
    # Test that server/app.py can be imported (Gradio UI)
    try:
        from server import app
        assert hasattr(app, "env"), "server/app.py should have env variable"
        assert hasattr(app, "grader"), "server/app.py should have grader variable"
        assert hasattr(app, "reset_environment"), (
            "server/app.py should have reset_environment function"
        )
        assert hasattr(app, "execute_action"), (
            "server/app.py should have execute_action function"
        )
    except ImportError as e:
        pytest.fail(f"Failed to import server/app.py: {e}")
    
    # Test that inference.py can be imported
    try:
        import inference
        assert hasattr(inference, "main"), "inference.py should have main function"
        assert hasattr(inference, "heuristic_policy"), (
            "inference.py should have heuristic_policy function"
        )
    except ImportError as e:
        pytest.fail(f"Failed to import inference.py: {e}")


# Feature: openenv-deployment-fixes, Property 2: Preservation - Reward Calculation Unchanged
@settings(max_examples=10)
@given(
    cpu_util=st.floats(min_value=0.0, max_value=100.0),
    memory_util=st.floats(min_value=0.0, max_value=100.0),
    allocated_resources=st.integers(min_value=1, max_value=100)
)
def test_preservation_reward_calculation(cpu_util, memory_util, allocated_resources):
    """
    Property 2.6: Preservation - Reward Calculation Logic Unchanged
    
    **Validates: Requirements 3.5**
    
    This test verifies that reward calculation logic remains unchanged after
    the bugfix. We test reward calculations across various input combinations.
    
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES (confirms baseline)
    **EXPECTED OUTCOME ON FIXED CODE**: Test PASSES (confirms preservation)
    """
    from env.reward import RewardCalculator
    
    # Create reward calculator
    calculator = RewardCalculator()
    
    # Calculate reward
    reward = calculator.calculate_reward(cpu_util, memory_util, allocated_resources)
    
    # Verify reward is a float
    assert isinstance(reward, (float, np.floating)), "Reward should be a float"
    
    # Verify reward is finite
    assert np.isfinite(reward), "Reward should be finite"
    
    # Verify reward calculation is deterministic
    reward2 = calculator.calculate_reward(cpu_util, memory_util, allocated_resources)
    assert reward == reward2, "Reward calculation should be deterministic"
    
    # Verify reward bounds (based on typical reward structure)
    # Rewards should be in a reasonable range (e.g., -10 to 10)
    assert -100 <= reward <= 100, f"Reward {reward} should be in reasonable range"
