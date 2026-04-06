# Cloud Resource Allocation RL

A hands-on reinforcement learning project that simulates cloud infrastructure resource allocation. Learn RL concepts by training an agent to dynamically allocate server instances based on system load, CPU utilization, and memory demand.

## Overview

This project provides a beginner-friendly introduction to reinforcement learning through a practical cloud resource management scenario. An RL agent observes system metrics (CPU utilization, memory utilization, request rate) and learns to make optimal resource allocation decisions (increase, decrease, or maintain server instances).

### Learning Objectives

By working with this project, you will:
- Understand core RL concepts: states, actions, rewards, and episodes
- Learn how reward shaping influences agent behavior
- Experience the challenges of balancing exploration vs. exploitation
- See how stochastic environments affect learning dynamics
- Practice implementing and testing RL environments
- Gain intuition for resource allocation problems in cloud systems

### Key Features

- **Realistic Simulation**: Cloud infrastructure with stochastic request patterns and resource dynamics
- **Interactive UI**: Gradio-based interface for manual exploration before agent training
- **Modular Architecture**: Clean separation of environment, reward, and evaluation logic
- **Property-Based Testing**: Comprehensive test coverage using Hypothesis
- **Beginner-Friendly**: Clear documentation, type hints, and inline comments explaining RL concepts

**Target Audience:** Developers new to RL but experienced with Python and PyTorch.

## Architecture

The system follows a modular design with four primary components:

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI (app.py)                     │
│  - Display system metrics (CPU, memory, requests)           │
│  - Action buttons (increase, decrease, maintain)            │
│  - Episode progress and grading results                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            Environment (env/environment.py)                 │
│  - Maintains state (CPU, memory, requests, resources)       │
│  - Processes actions and updates state                      │
│  - Manages episode lifecycle (reset, step, termination)     │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────────┐   ┌─────────────────────────────┐
│  Reward Calculator       │   │   Episode Grader            │
│  (env/reward.py)         │   │   (env/grader.py)           │
│  - Evaluates decisions   │   │   - Evaluates episodes      │
│  - Encourages efficiency │   │   - Calculates scores       │
│  - Penalizes waste       │   │   - Assesses stability      │
└──────────────────────────┘   └─────────────────────────────┘
```

### Component Responsibilities

- **Environment**: Simulates cloud system dynamics, manages state transitions, coordinates with reward module
- **Reward Calculator**: Provides feedback signals to guide learning, rewards optimal utilization (35-70%), penalizes over/under-provisioning
- **Episode Grader**: Evaluates complete trajectories, calculates stability and efficiency scores, determines pass/fail status
- **UI**: Enables manual interaction, visualizes system behavior, displays grading results

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **pip** package manager (usually included with Python)
- **Basic Python knowledge**: Classes, functions, numpy arrays
- **Terminal/command line** familiarity

To check your Python version:
```bash
python --version
# or
python3 --version
```

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd cloud-resource-allocation-rl
```

### Step 2: (Optional) Create a Virtual Environment

Using a virtual environment is recommended to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install:
- **numpy**: Numerical computations and state representation
- **gradio**: Interactive web UI framework
- **hypothesis**: Property-based testing library
- **pytest**: Testing framework

### Step 4: Verify Installation

Run a quick test to ensure everything is set up correctly:

```bash
pytest tests/test_environment.py -v
```

If tests pass, you're ready to go!

## Running the Project

### Launch the Interactive UI

Start the Gradio interface to manually explore the environment:

```bash
python app.py
```

The UI will launch in your default browser (typically at `http://127.0.0.1:7860`). You'll see:
- Real-time system metrics (CPU, memory, request rate, allocated resources)
- Action buttons to control resource allocation
- Episode progress tracking (cumulative reward, step count)
- System status indicators
- Episode completion report with grading results

### Run Tests

Execute the full test suite:

```bash
pytest tests/
```

Run tests with verbose output:

```bash
pytest tests/ -v
```

Run only property-based tests:

```bash
pytest tests/test_environment_properties.py tests/test_reward_properties.py tests/test_grader_properties.py -v
```

Run a specific test file:

```bash
pytest tests/test_environment.py -v
```

## Usage Examples

### Example 1: Manual Interaction via UI

1. **Launch the UI**: Run `python app.py`
2. **Click "Reset Environment"**: Initializes a new episode with random starting conditions
3. **Observe Initial State**: Note the CPU utilization, memory utilization, and request rate
4. **Take Actions**:
   - If CPU/memory > 70%: Click "Increase Resources" to add a server instance
   - If CPU/memory < 35%: Click "Decrease Resources" to remove an instance
   - If CPU/memory is 35-70%: Click "Maintain Resources" to keep current allocation
5. **Watch the Dynamics**: Request rate fluctuates randomly each step, affecting utilization
6. **Complete the Episode**: Continue until the episode terminates (100 steps or critical utilization)
7. **Review Results**: Check the grading report showing stability, efficiency, and overall score

### Example 2: Programmatic Environment Usage

```python
from env.environment import CloudResourceEnv

# Create environment
env = CloudResourceEnv(max_steps=100, initial_resources=3)

# Reset to start new episode
state = env.reset()
print(f"Initial state: CPU={state[0]:.1f}%, Memory={state[1]:.1f}%, "
      f"Requests={state[2]}, Resources={state[3]}")

# Run episode
done = False
cumulative_reward = 0

while not done:
    # Simple policy: increase if CPU > 70%, decrease if CPU < 40%, else maintain
    if state[0] > 70:
        action = CloudResourceEnv.ACTION_INCREASE
    elif state[0] < 40:
        action = CloudResourceEnv.ACTION_DECREASE
    else:
        action = CloudResourceEnv.ACTION_MAINTAIN
    
    # Execute action
    state, reward, done, info = env.step(action)
    cumulative_reward += reward
    
    print(f"Step {info['step']}: Action={action}, Reward={reward:.2f}, "
          f"CPU={state[0]:.1f}%, Resources={state[3]}")

print(f"Episode finished! Total reward: {cumulative_reward:.2f}")
```

### Example 3: Evaluating an Episode

```python
from env.environment import CloudResourceEnv
from env.grader import EpisodeGrader

# Run an episode
env = CloudResourceEnv()
grader = EpisodeGrader()

state = env.reset()
done = False

while not done:
    action = 1  # Maintain resources (simple baseline)
    state, reward, done, info = env.step(action)

# Grade the episode
results = grader.grade_episode(
    env.episode_states,
    env.episode_actions,
    env.episode_rewards
)

print(f"Episode Score: {results['score']:.3f}")
print(f"Passed: {results['passed']}")
print(f"Stability: {results['stability_score']:.3f}")
print(f"Efficiency: {results['efficiency_score']:.3f}")
print(f"Avg Reward: {results['avg_reward']:.3f}")
```

## Understanding RL Concepts

### State

The **state** represents the current observation of the system. In this environment, the state is a 4-dimensional vector:

```python
state = [cpu_utilization, memory_utilization, request_rate, allocated_resources]
# Example: [65.3, 58.2, 47, 3]
```

- **CPU Utilization**: Percentage (0-100) indicating how much CPU capacity is being used
- **Memory Utilization**: Percentage (0-100) indicating how much memory is being used
- **Request Rate**: Number of incoming requests per time step
- **Allocated Resources**: Number of server instances currently allocated (minimum 1)

### Action

An **action** is a decision the agent makes to modify resource allocation. There are three discrete actions:

- **Action 0 (Decrease)**: Remove one server instance (minimum 1 instance maintained)
- **Action 1 (Maintain)**: Keep current allocation unchanged
- **Action 2 (Increase)**: Add one server instance

### Reward

The **reward** is a numerical signal indicating how good the current state is. The reward function encourages:

- **Optimal Utilization (35-70%)**: Positive rewards when CPU and memory are in this range
- **Avoiding Over-Provisioning**: Negative rewards when utilization is too low (wasting resources)
- **Avoiding Under-Provisioning**: Negative rewards when utilization is too high (risking instability)

Reward calculation considers:
1. CPU utilization relative to target range (35-70%)
2. Memory utilization relative to target range (35-70%)
3. Resource cost penalty (encourages using fewer instances when possible)

### Episode

An **episode** is a complete simulation run from initialization to termination. Episodes terminate when:
- Maximum steps reached (default: 100 steps)
- CPU utilization exceeds 95% (system instability)
- Memory utilization exceeds 95% (system instability)

### System Dynamics

The environment simulates realistic cloud behavior:

- **Stochastic Request Rate**: Incoming requests fluctuate randomly each step, simulating real-world variability
- **Resource-Utilization Relationship**: More resources → lower utilization; fewer resources → higher utilization
- **Request Impact**: Higher request rate → higher CPU and memory utilization

## Tips for Understanding System Dynamics

### Observation Tips

1. **Start with Manual Exploration**: Use the UI to get intuition before writing agent code
2. **Watch for Patterns**: Notice how request rate changes affect utilization
3. **Test Edge Cases**: Try maintaining 1 resource with high requests, or 10 resources with low requests
4. **Observe Termination**: See what happens when you let CPU/memory exceed 95%

### Common Behaviors

- **Over-Provisioning**: Allocating too many resources leads to low utilization and negative rewards (wasted money)
- **Under-Provisioning**: Allocating too few resources leads to high utilization and negative rewards (instability risk)
- **Optimal Zone**: Keeping utilization between 35-70% yields positive rewards
- **Stochastic Challenges**: Random request fluctuations make perfect control impossible

### Debugging Tips

- **Print State Transitions**: Add print statements to see how state evolves
- **Track Reward Components**: Modify reward calculator to log CPU reward, memory reward, and resource penalty separately
- **Visualize Episodes**: Plot CPU/memory utilization over time to identify patterns
- **Compare Policies**: Run multiple episodes with different strategies and compare results

## Extending the Project

### Beginner Extensions

1. **Modify Reward Function**: Adjust target utilization ranges in `env/reward.py`
   ```python
   reward_calc = RewardCalculator(
       target_cpu_range=(50.0, 80.0),  # Change optimal range
       target_memory_range=(50.0, 80.0),
       resource_cost_weight=0.2  # Increase cost penalty
   )
   ```

2. **Change Environment Parameters**: Adjust dynamics in `env/environment.py`
   ```python
   env = CloudResourceEnv(
       max_steps=200,  # Longer episodes
       initial_resources=5,  # Start with more resources
       base_request_rate=100  # Higher baseline load
   )
   ```

3. **Implement Simple Policies**: Create rule-based agents to test different strategies
   ```python
   def aggressive_policy(state):
       """Always increase resources when utilization > 60%"""
       return 2 if state[0] > 60 or state[1] > 60 else 1
   
   def conservative_policy(state):
       """Only increase when utilization > 80%"""
       return 2 if state[0] > 80 or state[1] > 80 else 0
   ```

### Intermediate Extensions

4. **Add New State Features**: Include additional metrics like network bandwidth or disk I/O
5. **Implement Multi-Step Actions**: Allow increasing/decreasing by 2 or 3 instances at once
6. **Add Cost Models**: Introduce different instance types with varying costs and capacities
7. **Create Visualization Tools**: Plot episode trajectories, reward distributions, or policy heatmaps

### Advanced Extensions

8. **Train an RL Agent**: Implement Q-learning, DQN, or PPO to learn optimal policies
9. **Add Constraints**: Introduce budget limits, SLA requirements, or resource quotas
10. **Multi-Objective Optimization**: Balance cost, performance, and reliability simultaneously
11. **Realistic Workload Patterns**: Model daily/weekly traffic patterns instead of random fluctuations

### Testing Extensions

12. **Add Property Tests**: Write new Hypothesis tests for custom features
13. **Benchmark Policies**: Create a test suite comparing different strategies
14. **Stress Testing**: Test environment behavior under extreme conditions

## Project Structure

```
cloud-resource-allocation-rl/
├── env/                          # Core environment logic
│   ├── __init__.py
│   ├── environment.py            # CloudResourceEnv class (state, actions, dynamics)
│   ├── reward.py                 # RewardCalculator class (reward shaping)
│   ├── grader.py                 # EpisodeGrader class (performance evaluation)
│   └── config.py                 # Configuration dataclasses
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_environment.py       # Unit tests for environment
│   ├── test_environment_properties.py  # Property tests for environment
│   ├── test_reward.py            # Unit tests for reward calculator
│   ├── test_reward_properties.py # Property tests for reward calculator
│   ├── test_grader.py            # Unit tests for grader
│   └── test_grader_properties.py # Property tests for grader
├── app.py                        # Gradio interactive UI
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'gradio'`
- **Solution**: Run `pip install -r requirements.txt` to install dependencies

**Issue**: UI doesn't open in browser
- **Solution**: Manually navigate to `http://127.0.0.1:7860` or check terminal for the correct URL

**Issue**: Tests fail with `hypothesis` errors
- **Solution**: Ensure Hypothesis is installed: `pip install hypothesis`

**Issue**: Python version error
- **Solution**: Upgrade to Python 3.8+: `python --version` to check current version

## Dependencies

- **numpy**: Numerical computations and state representation
- **gradio**: Interactive web UI for manual environment control
- **hypothesis**: Property-based testing framework
- **pytest**: Unit testing framework

## License

This project is provided for educational purposes. 