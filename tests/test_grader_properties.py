"""
Property-based tests for the EpisodeGrader.

These tests validate universal correctness properties of the grader
using Hypothesis for randomized input generation. Each test runs a minimum of
100 iterations to ensure adequate coverage.
"""

from hypothesis import given, strategies as st, settings
import numpy as np
from env.grader import EpisodeGrader


# Hypothesis strategies for generating valid trajectory data
@st.composite
def trajectory_data(draw):
    """Generate valid episode trajectory data."""
    # Generate episode length
    episode_length = draw(st.integers(min_value=1, max_value=100))
    
    # Generate states (each state is [cpu, memory, request_rate, resources])
    states = []
    for _ in range(episode_length):
        cpu = draw(st.floats(min_value=0.0, max_value=100.0))
        memory = draw(st.floats(min_value=0.0, max_value=100.0))
        request_rate = draw(st.integers(min_value=0, max_value=200))
        resources = draw(st.integers(min_value=1, max_value=20))
        states.append(np.array([cpu, memory, request_rate, resources], dtype=np.float32))
    
    # Generate actions
    actions = [draw(st.integers(min_value=0, max_value=2)) for _ in range(episode_length)]
    
    # Generate rewards
    rewards = [draw(st.floats(min_value=-10.0, max_value=5.0)) for _ in range(episode_length)]
    
    return states, actions, rewards


# Feature: cloud-resource-allocation-rl, Property 12: Grader Trajectory Processing
@settings(max_examples=100)
@given(trajectory=trajectory_data())
def test_grader_trajectory_processing(trajectory):
    """
    Property 12: Grader Trajectory Processing
    
    **Validates: Requirements 6.1, 6.2, 6.5**
    
    For any complete episode trajectory containing lists of states, actions, and
    rewards, the grader SHALL accept these inputs and return a dictionary containing
    a numerical score and a boolean passed status.
    """
    states, actions, rewards = trajectory
    
    grader = EpisodeGrader()
    result = grader.grade_episode(states, actions, rewards)
    
    # Verify result is a dictionary
    assert isinstance(result, dict), f"Grader should return dict, got {type(result)}"
    
    # Verify required keys are present
    required_keys = {'score', 'passed', 'stability_score', 'efficiency_score', 'avg_reward'}
    assert required_keys.issubset(result.keys()), (
        f"Result missing required keys: {required_keys - result.keys()}"
    )
    
    # Verify score is a numerical value
    assert isinstance(result['score'], (int, float, np.number)), (
        f"Score should be numerical, got {type(result['score'])}"
    )
    
    # Verify passed is a boolean
    assert isinstance(result['passed'], bool), (
        f"Passed status should be boolean, got {type(result['passed'])}"
    )
    
    # Verify stability_score is numerical
    assert isinstance(result['stability_score'], (int, float, np.number)), (
        f"Stability score should be numerical, got {type(result['stability_score'])}"
    )
    
    # Verify efficiency_score is numerical
    assert isinstance(result['efficiency_score'], (int, float, np.number)), (
        f"Efficiency score should be numerical, got {type(result['efficiency_score'])}"
    )
    
    # Verify avg_reward is numerical
    assert isinstance(result['avg_reward'], (int, float, np.number)), (
        f"Average reward should be numerical, got {type(result['avg_reward'])}"
    )
    
    # Verify score is in reasonable range [0, 1] (since it's a weighted average of normalized metrics)
    assert 0.0 <= result['score'] <= 1.0, (
        f"Score should be in range [0, 1], got {result['score']}"
    )
    
    # Verify stability and efficiency scores are in [0, 1] range
    assert 0.0 <= result['stability_score'] <= 1.0, (
        f"Stability score should be in range [0, 1], got {result['stability_score']}"
    )
    assert 0.0 <= result['efficiency_score'] <= 1.0, (
        f"Efficiency score should be in range [0, 1], got {result['efficiency_score']}"
    )
    
    # Verify passed status is consistent with score and threshold
    if result['score'] >= grader.pass_threshold:
        assert result['passed'] is True, (
            f"Episode should pass when score ({result['score']}) >= threshold ({grader.pass_threshold})"
        )
    else:
        assert result['passed'] is False, (
            f"Episode should fail when score ({result['score']}) < threshold ({grader.pass_threshold})"
        )


# Feature: cloud-resource-allocation-rl, Property 13: Stability Assessment
@settings(max_examples=100)
@given(
    episode_length=st.integers(min_value=10, max_value=50),
    base_cpu=st.floats(min_value=30.0, max_value=70.0),
    base_memory=st.floats(min_value=30.0, max_value=70.0),
    variance_multiplier=st.floats(min_value=0.1, max_value=5.0),
    seed=st.integers(min_value=0, max_value=1000000)
)
def test_stability_assessment(episode_length, base_cpu, base_memory, variance_multiplier, seed):
    """
    Property 13: Stability Assessment
    
    **Validates: Requirements 6.3**
    
    For any two episodes with the same average reward, the episode with lower
    variance in CPU and memory utilization SHALL receive a higher stability score
    from the grader.
    """
    rng = np.random.RandomState(seed)
    
    # Create two episodes with same average reward but different variance
    # Episode 1: Low variance (stable)
    states_stable = []
    for _ in range(episode_length):
        # Small random fluctuation around base values
        cpu = base_cpu + rng.normal(0, 2.0)  # Low variance
        memory = base_memory + rng.normal(0, 2.0)  # Low variance
        cpu = max(0.0, min(100.0, cpu))
        memory = max(0.0, min(100.0, memory))
        request_rate = rng.randint(40, 60)
        resources = rng.randint(3, 5)
        states_stable.append(np.array([cpu, memory, request_rate, resources], dtype=np.float32))
    
    # Episode 2: High variance (unstable)
    states_unstable = []
    for _ in range(episode_length):
        # Large random fluctuation around base values
        cpu = base_cpu + rng.normal(0, 10.0 * variance_multiplier)  # High variance
        memory = base_memory + rng.normal(0, 10.0 * variance_multiplier)  # High variance
        cpu = max(0.0, min(100.0, cpu))
        memory = max(0.0, min(100.0, memory))
        request_rate = rng.randint(40, 60)
        resources = rng.randint(3, 5)
        states_unstable.append(np.array([cpu, memory, request_rate, resources], dtype=np.float32))
    
    # Use same actions and rewards for both episodes
    actions = [rng.randint(0, 3) for _ in range(episode_length)]
    rewards = [rng.uniform(-2.0, 1.0) for _ in range(episode_length)]
    
    # Grade both episodes
    grader = EpisodeGrader()
    result_stable = grader.grade_episode(states_stable, actions, rewards)
    result_unstable = grader.grade_episode(states_unstable, actions, rewards)
    
    # Calculate actual variance for verification
    cpu_variance_stable = np.var([s[0] for s in states_stable])
    cpu_variance_unstable = np.var([s[0] for s in states_unstable])
    memory_variance_stable = np.var([s[1] for s in states_stable])
    memory_variance_unstable = np.var([s[1] for s in states_unstable])
    
    avg_variance_stable = (cpu_variance_stable + memory_variance_stable) / 2.0
    avg_variance_unstable = (cpu_variance_unstable + memory_variance_unstable) / 2.0
    
    # If stable episode actually has lower variance, it should have higher stability score
    if avg_variance_stable < avg_variance_unstable:
        assert result_stable['stability_score'] > result_unstable['stability_score'], (
            f"Episode with lower variance ({avg_variance_stable:.2f}) should have higher "
            f"stability score than episode with higher variance ({avg_variance_unstable:.2f}). "
            f"Got {result_stable['stability_score']:.4f} vs {result_unstable['stability_score']:.4f}"
        )


# Feature: cloud-resource-allocation-rl, Property 14: Efficiency Assessment
@settings(max_examples=100)
@given(
    episode_length=st.integers(min_value=10, max_value=50),
    low_resources=st.integers(min_value=1, max_value=3),
    high_resources=st.integers(min_value=5, max_value=10),
    seed=st.integers(min_value=0, max_value=1000000)
)
def test_efficiency_assessment(episode_length, low_resources, high_resources, seed):
    """
    Property 14: Efficiency Assessment
    
    **Validates: Requirements 6.4**
    
    For any two episodes with the same average reward, the episode with lower
    average allocated resources SHALL receive a higher efficiency score from
    the grader.
    """
    rng = np.random.RandomState(seed)
    
    # Ensure high_resources is actually higher
    if high_resources <= low_resources:
        high_resources = low_resources + 3
    
    # Create two episodes with same rewards but different resource allocation
    # Episode 1: Low resource usage (efficient)
    states_efficient = []
    for _ in range(episode_length):
        cpu = rng.uniform(40.0, 70.0)
        memory = rng.uniform(40.0, 70.0)
        request_rate = rng.randint(40, 60)
        resources = low_resources  # Consistently low resources
        states_efficient.append(np.array([cpu, memory, request_rate, resources], dtype=np.float32))
    
    # Episode 2: High resource usage (inefficient)
    states_inefficient = []
    for _ in range(episode_length):
        cpu = rng.uniform(40.0, 70.0)
        memory = rng.uniform(40.0, 70.0)
        request_rate = rng.randint(40, 60)
        resources = high_resources  # Consistently high resources
        states_inefficient.append(np.array([cpu, memory, request_rate, resources], dtype=np.float32))
    
    # Use same actions and rewards for both episodes
    actions = [rng.randint(0, 3) for _ in range(episode_length)]
    rewards = [rng.uniform(-2.0, 1.0) for _ in range(episode_length)]
    
    # Grade both episodes
    grader = EpisodeGrader()
    result_efficient = grader.grade_episode(states_efficient, actions, rewards)
    result_inefficient = grader.grade_episode(states_inefficient, actions, rewards)
    
    # Calculate actual average resources for verification
    avg_resources_efficient = np.mean([s[3] for s in states_efficient])
    avg_resources_inefficient = np.mean([s[3] for s in states_inefficient])
    
    # Episode with lower average resources should have higher efficiency score
    assert avg_resources_efficient < avg_resources_inefficient, (
        f"Efficient episode should have lower average resources: "
        f"{avg_resources_efficient:.2f} vs {avg_resources_inefficient:.2f}"
    )
    
    assert result_efficient['efficiency_score'] > result_inefficient['efficiency_score'], (
        f"Episode with lower average resources ({avg_resources_efficient:.2f}) should have "
        f"higher efficiency score than episode with higher resources ({avg_resources_inefficient:.2f}). "
        f"Got {result_efficient['efficiency_score']:.4f} vs {result_inefficient['efficiency_score']:.4f}"
    )
