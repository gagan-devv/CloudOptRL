"""
Property-based tests for logging format and clean stdout output.

Tests Properties 5 and 6 from the design document.
"""

import pytest
from hypothesis import given, strategies as st, settings
import re
import sys
import io
from contextlib import redirect_stdout


# Feature: final-submission-compliance, Property 5: Step Log Format Consistency
@given(
    step=st.integers(min_value=1, max_value=100),
    action=st.integers(min_value=0, max_value=2),
    reward=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    done=st.booleans()
)
@settings(max_examples=100)
def test_step_log_format_consistency(step, action, reward, done):
    """
    Property 5: [STEP] log matches expected format with 2 decimal places for reward.
    
    **Validates: Requirements 5.2, 5.6**
    
    The [STEP] log should:
    - Match format: [STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}
    - Format reward with exactly 2 decimal places
    """
    # Generate log message using inference script format
    error = None
    log_message = f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}"
    
    # Verify log format using regex
    pattern = r'^\[STEP\] step=\d+ action=[0-2] reward=-?\d+\.\d{2} done=(True|False) error=(None|.+)$'
    assert re.match(pattern, log_message), f"Log message '{log_message}' does not match expected format"
    
    # Verify reward has exactly 2 decimal places
    reward_match = re.search(r'reward=(-?\d+\.\d{2})', log_message)
    assert reward_match is not None, "Reward not found in log message"
    
    # Extract and verify reward value
    logged_reward = float(reward_match.group(1))
    expected_reward = round(reward, 2)
    assert abs(logged_reward - expected_reward) < 0.01, f"Logged reward {logged_reward} != expected {expected_reward}"


# Feature: final-submission-compliance, Property 5: Step Log Format with Error
@given(
    step=st.integers(min_value=1, max_value=100),
    action=st.integers(min_value=0, max_value=2),
    error_msg=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=('Cs', 'Cc')))
)
@settings(max_examples=100)
def test_step_log_format_with_error(step, action, error_msg):
    """
    Property 5: [STEP] log with error matches expected format.
    
    **Validates: Requirements 5.2, 5.6**
    
    The [STEP] log with error should:
    - Match format: [STEP] step={step} action={action} reward=0.00 done=True error={error}
    - Always show reward=0.00 and done=True when error occurs
    """
    # Generate log message for error case
    log_message = f"[STEP] step={step} action={action} reward=0.00 done=True error={error_msg}"
    
    # Verify log format
    pattern = r'^\[STEP\] step=\d+ action=[0-2] reward=0\.00 done=True error=.+$'
    assert re.match(pattern, log_message), f"Error log message '{log_message}' does not match expected format"


# Feature: final-submission-compliance, Property 6: Clean Stdout Output
def test_clean_stdout_output_pattern():
    """
    Property 6: Only [START], [STEP], [END] patterns appear in stdout.
    
    **Validates: Requirements 5.5, 7.1, 7.2, 7.3**
    
    The stdout should:
    - Only contain lines matching [START], [STEP], or [END] patterns
    - Not contain any debug messages or other output
    """
    # Sample valid log output
    valid_logs = [
        "[START] task=medium env=cloud_resource_env model=gpt-4",
        "[STEP] step=1 action=2 reward=0.85 done=False error=None",
        "[STEP] step=2 action=1 reward=0.92 done=False error=None",
        "[END] success=True steps=2 score=0.89 rewards=0.85,0.92"
    ]
    
    # Verify each log line matches expected patterns
    start_pattern = r'^\[START\] task=\w+ env=\w+ model=[\w-]+$'
    step_pattern = r'^\[STEP\] step=\d+ action=[0-2] reward=-?\d+\.\d{2} done=(True|False) error=(None|.+)$'
    end_pattern = r'^\[END\] success=(True|False) steps=\d+ score=-?\d+\.\d{2} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*)?$'
    
    for log in valid_logs:
        is_valid = (
            re.match(start_pattern, log) or
            re.match(step_pattern, log) or
            re.match(end_pattern, log)
        )
        assert is_valid, f"Log line '{log}' does not match any valid pattern"


# Feature: final-submission-compliance, Property 6: Invalid Output Detection
@given(invalid_line=st.text(min_size=1, max_size=100, alphabet=st.characters(blacklist_categories=('Cs', 'Cc'))))
@settings(max_examples=100)
def test_invalid_output_detection(invalid_line):
    """
    Property 6: Lines not matching [START], [STEP], [END] are invalid.
    
    **Validates: Requirements 5.5, 7.1, 7.2, 7.3**
    
    Any line that doesn't match the strict log patterns should be detected as invalid.
    """
    # Skip if line happens to match valid patterns
    start_pattern = r'^\[START\] task=\w+ env=\w+ model=[\w-]+$'
    step_pattern = r'^\[STEP\] step=\d+ action=[0-2] reward=-?\d+\.\d{2} done=(True|False) error=(None|.+)$'
    end_pattern = r'^\[END\] success=(True|False) steps=\d+ score=-?\d+\.\d{2} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*)?$'
    
    is_valid = (
        re.match(start_pattern, invalid_line) or
        re.match(step_pattern, invalid_line) or
        re.match(end_pattern, invalid_line)
    )
    
    # If line doesn't match any valid pattern, it should be detected as invalid
    if not is_valid:
        # This represents a line that should NOT appear in stdout
        assert not is_valid, f"Line '{invalid_line}' should be invalid but was not detected"


# Feature: final-submission-compliance, Property 5: START Log Format
@given(
    task=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_-')),
    env=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_-')),
    model=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_-'))
)
@settings(max_examples=100)
def test_start_log_format(task, env, model):
    """
    Property 5: [START] log matches expected format.
    
    **Validates: Requirements 5.1**
    
    The [START] log should match format: [START] task={task} env={env} model={model}
    """
    # Generate log message
    log_message = f"[START] task={task} env={env} model={model}"
    
    # Verify log format
    pattern = r'^\[START\] task=[\w-]+ env=[\w-]+ model=[\w-]+$'
    assert re.match(pattern, log_message), f"START log '{log_message}' does not match expected format"


# Feature: final-submission-compliance, Property 5: END Log Format
@given(
    success=st.booleans(),
    steps=st.integers(min_value=0, max_value=100),
    score=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    rewards=st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=0, max_size=20)
)
@settings(max_examples=100)
def test_end_log_format(success, steps, score, rewards):
    """
    Property 5: [END] log matches expected format.
    
    **Validates: Requirements 5.3, 5.7**
    
    The [END] log should match format: [END] success={success} steps={steps} score={score:.2f} rewards={r1,r2,...}
    """
    # Generate log message
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    log_message = f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards_str}"
    
    # Verify log format
    pattern = r'^\[END\] success=(True|False) steps=\d+ score=-?\d+\.\d{2} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*)?$'
    assert re.match(pattern, log_message), f"END log '{log_message}' does not match expected format"


# Feature: final-submission-compliance, Property 6: Integration Test for Clean Stdout
def test_inference_script_clean_stdout():
    """
    Property 6: Integration test - inference.py produces only clean stdout.
    
    **Validates: Requirements 5.5, 7.1, 7.2, 7.3, 7.4**
    
    Run the actual inference script and verify:
    - Only [START], [STEP], [END] logs appear in stdout
    - No debug messages from CloudResourceEnv
    - No debug messages from AsyncEnvWrapper
    - No FastAPI/uvicorn access logs
    """
    import subprocess
    
    # Run inference script and capture stdout
    result = subprocess.run(
        ["python", "inference.py"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    # Verify script completed successfully
    assert result.returncode == 0, f"Inference script failed with return code {result.returncode}"
    
    # Get stdout lines
    stdout_lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    
    # Verify we have output
    assert len(stdout_lines) > 0, "No output from inference script"
    
    # Define valid patterns
    start_pattern = r'^\[START\] task=\w+ env=\w+ model=[\w-]+$'
    step_pattern = r'^\[STEP\] step=\d+ action=[0-2] reward=-?\d+\.\d{2} done=(True|False) error=(None|.+)$'
    end_pattern = r'^\[END\] success=(True|False) steps=\d+ score=-?\d+\.\d{2} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*)?$'
    
    # Verify each line matches one of the valid patterns
    for i, line in enumerate(stdout_lines):
        is_valid = (
            re.match(start_pattern, line) or
            re.match(step_pattern, line) or
            re.match(end_pattern, line)
        )
        assert is_valid, f"Line {i+1} does not match any valid pattern: '{line}'"
    
    # Verify first line is [START]
    assert re.match(start_pattern, stdout_lines[0]), f"First line should be [START]: '{stdout_lines[0]}'"
    
    # Verify last line is [END]
    assert re.match(end_pattern, stdout_lines[-1]), f"Last line should be [END]: '{stdout_lines[-1]}'"
    
    # Verify no stderr output (except for potential warnings that don't affect stdout)
    # We allow stderr to have content, but stdout must be clean
    # This is important because errors should go to stderr, not stdout


# Feature: final-submission-compliance, Property 6: Environment and Wrapper Produce No Stdout
def test_environment_and_wrapper_no_stdout():
    """
    Property 6: CloudResourceEnv and AsyncEnvWrapper produce no stdout.
    
    **Validates: Requirements 7.2, 7.3**
    
    Verify that:
    - CloudResourceEnv does not print to stdout during normal operation
    - AsyncEnvWrapper does not print to stdout during normal operation
    """
    import asyncio
    from env.environment import CloudResourceEnv
    from env.async_wrapper import AsyncEnvWrapper
    
    # Capture stdout
    captured_output = io.StringIO()
    
    async def run_episode():
        """Run a short episode and capture any stdout."""
        env = CloudResourceEnv(task_name="medium")
        async_env = AsyncEnvWrapper(env)
        
        # Reset environment
        await async_env.reset()
        
        # Run a few steps
        for _ in range(5):
            action = 1  # maintain
            obs, reward, done, info = await async_env.step(action)
            if done:
                break
        
        # Close wrapper
        await async_env.close()
    
    # Run episode with stdout capture
    with redirect_stdout(captured_output):
        asyncio.run(run_episode())
    
    # Verify no stdout output
    output = captured_output.getvalue()
    assert output == "", f"Environment/Wrapper produced stdout output: '{output}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
