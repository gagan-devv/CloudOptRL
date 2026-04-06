# Cloud Resource Allocation RL

A reinforcement learning environment for cloud infrastructure resource allocation, built with PyTorch and Gradio. This project demonstrates intelligent resource management through interactive visualization and property-based testing.

**This project uses PyTorch for tensor-based state representation and is compatible with OpenAI Gym-style interfaces.**

## Problem Statement

Cloud infrastructure providers face a critical challenge: dynamically allocating computational resources to meet fluctuating demand while minimizing costs and maintaining system stability. Over-provisioning wastes money on idle resources, while under-provisioning risks system crashes and poor user experience.

This project simulates this real-world problem as a reinforcement learning environment where an agent must learn to:
- Balance resource utilization between 40-70% for optimal efficiency
- Respond to stochastic request rate fluctuations
- Minimize infrastructure costs while maintaining service quality
- Avoid system instability from resource exhaustion

## Solution Overview

We model cloud resource allocation as a Markov Decision Process (MDP) where:
- **State**: System metrics (CPU utilization, memory utilization, request rate, allocated resources, latency)
- **Actions**: Discrete resource adjustments (increase, decrease, or maintain server instances)
- **Rewards**: Shaped to encourage efficient operation and penalize waste or instability
- **Dynamics**: Stochastic request patterns simulate real-world variability

The solution provides:
1. **Interactive Gradio UI** with real-time visualizations for manual exploration
2. **PyTorch-based state representation** for seamless integration with deep RL algorithms
3. **Comprehensive grading system** that evaluates stability, efficiency, and performance
4. **Property-based testing** ensuring correctness across diverse scenarios

## Tech Stack

- **Python 3.8+**: Core programming language
- **PyTorch**: Tensor-based state representation for neural network integration
- **Gradio**: Interactive web UI with real-time plotting
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization of system metrics over time
- **Hypothesis**: Property-based testing for robust validation
- **Pytest**: Unit and integration testing framework

### Why These Technologies?

- **PyTorch**: Industry-standard deep learning framework, enables easy integration with RL algorithms (DQN, PPO, A3C)
- **Gradio**: Rapid prototyping of interactive demos, perfect for showcasing RL environments
- **Hypothesis**: Discovers edge cases automatically through property-based testing
- **Matplotlib**: Publication-quality plots for analyzing agent behavior

## Environment Design

### State Space

The environment state is a 5-dimensional continuous vector:

```python
state = [cpu_util, memory_util, request_rate, allocated_resources, latency]
# Example: [65.3, 58.2, 47.0, 3.0, 15.67]
```

**State Components:**
- **CPU Utilization** (0-100%): Percentage of CPU capacity being used
- **Memory Utilization** (0-100%): Percentage of memory capacity being used  
- **Request Rate** (0-∞): Number of incoming requests per timestep
- **Allocated Resources** (1-∞): Number of server instances currently allocated
- **Latency** (0-∞ ms): Average request latency calculated as `request_rate / allocated_resources`

**PyTorch Integration:**
```python
# Get state as PyTorch tensor for neural network input
state_tensor = env.get_state_tensor()  # Returns torch.Tensor
```

### Action Space

Three discrete actions control resource allocation:

| Action | Value | Effect |
|--------|-------|--------|
| Decrease | 0 | Remove 1 server instance (minimum 1) |
| Maintain | 1 | Keep current allocation unchanged |
| Increase | 2 | Add 1 server instance |

### System Dynamics

The environment simulates realistic cloud behavior:

1. **Resource-Utilization Relationship**:
   ```
   CPU_util = (request_rate × cpu_per_request) / (resources × capacity) × 100
   Memory_util = (request_rate × memory_per_request) / (resources × capacity) × 100
   ```

2. **Stochastic Request Patterns**:
   - Request rate fluctuates with Gaussian noise: `N(base_rate, std_dev)`
   - Simulates real-world traffic variability

3. **Latency Calculation**:
   ```
   Latency = request_rate / max(allocated_resources, 1)
   ```
   - Higher request rate → higher latency
   - More resources → lower latency

### Termination Conditions

Episodes terminate when:
- **Max steps reached** (default: 100 timesteps)
- **CPU utilization > 95%** (system instability)
- **Memory utilization > 95%** (system instability)

## Reward Function Design

The reward function shapes agent behavior through three components:

### 1. Utilization Rewards

Encourages keeping CPU and memory in optimal range (40-70%):

```python
if 40% ≤ utilization ≤ 70%:
    reward = +1.0 (scaled by distance from center)
elif utilization < 40%:
    reward = -0.3 - (distance_below / 100)  # Over-provisioning penalty
else:  # utilization > 70%
    reward = -1.0 - (distance_above / 100)  # Under-provisioning penalty
```

### 2. Resource Cost Penalty

Encourages minimizing resource usage:

```python
cost_penalty = -0.05 × allocated_resources
```

### 3. Combined Reward

```python
total_reward = cpu_reward + memory_reward + cost_penalty
if not (cpu_optimal and memory_optimal):
    total_reward -= 0.1  # Additional penalty for mixed states
```

**Reward Range**: Typically -5.0 to +2.0

**Design Rationale**:
- Positive rewards only when both CPU and memory are optimal
- Stronger penalties for under-provisioning (instability risk) than over-provisioning
- Resource cost encourages efficiency without sacrificing performance

## Grader Explanation

The `EpisodeGrader` evaluates complete trajectories using three metrics:

### Stability Score (30% weight)

Measures consistency of utilization levels:

```python
stability = exp(-avg_variance / 100)
```

- Lower variance → higher stability → better score
- Penalizes erratic resource allocation patterns

### Efficiency Score (30% weight)

Measures resource utilization efficiency:

```python
efficiency = 1.0 / avg_allocated_resources
```

- Fewer resources → higher efficiency → better score
- Encourages lean resource usage

### Performance Score (40% weight)

Normalized average reward:

```python
performance = (avg_reward + 5.0) / 7.0  # Map [-5, 2] to [0, 1]
```

### Final Score

```python
final_score = 0.3×stability + 0.3×efficiency + 0.4×performance
passed = final_score ≥ threshold (default: 0.0)
```

**Grader Output**:
```python
{
    'final_score': 0.523,
    'passed': True,
    'stability_score': 0.847,
    'efficiency_score': 0.333,
    'avg_reward': 0.156,
    'avg_cpu': 52.3,
    'avg_memory': 48.7,
    'avg_latency': 12.4
}
```

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

### Prerequisites

- **Python 3.8+** (Python 3.9+ recommended)
- **pip** package manager
- **Virtual environment** (recommended)

Check your Python version:
```bash
python --version  # or python3 --version
```

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd cloud-resource-allocation-rl
```

### Step 2: Create Virtual Environment (Recommended)

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

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `torch>=2.0.0` - PyTorch for tensor operations
- `gradio` - Interactive web UI
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `hypothesis` - Property-based testing
- `pytest` - Testing framework

### Step 4: Verify Installation

```bash
# Run tests to verify setup
pytest tests/test_environment.py -v

# Launch demo
python app.py
```
pytest tests/test_environment.py -v
```

If tests pass, you're ready to go!

## Demo Instructions

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the interactive demo**:
   ```bash
   python app.py
   ```

3. **Open your browser** to `http://127.0.0.1:7860`

### Using the Demo

#### Step 1: Reset Environment
Click **"Reset Environment"** to start a new episode with randomized initial conditions.

#### Step 2: Observe System Metrics
Monitor the left panel:
- CPU Utilization (target: 40-70%)
- Memory Utilization (target: 40-70%)
- Request Rate (stochastic)
- Allocated Resources (your control)
- Latency (lower is better)

#### Step 3: Take Actions
Use action buttons to control resources:
- **⬆️ Increase Resources**: Add 1 server instance
- **➡️ Maintain Resources**: Keep current allocation
- **⬇️ Decrease Resources**: Remove 1 instance (min 1)

#### Step 4: Watch Real-Time Plots
Observe how your actions affect:
- **CPU Utilization Over Time**: Shows target zones (40-70% optimal, 95% critical)
- **Resource Allocation Over Time**: Tracks your resource decisions

#### Step 5: Use Multi-Step Mode
For faster exploration:
1. Select number of steps (1-20)
2. Choose action to repeat
3. Click **"▶️ Run Multiple Steps"**

#### Step 6: Review Episode Results
When the episode ends, review:
- Final Score and Pass/Fail status
- Stability, Efficiency, and Performance metrics
- Average CPU, Memory, and Latency
- Total steps and cumulative reward

### Demo Tips

- **Optimal Strategy**: Keep CPU and memory between 40-70%
- **Watch for Spikes**: Request rate changes randomly each step
- **Latency Matters**: More resources = lower latency
- **Cost vs Performance**: Balance efficiency with stability

### Expected Behavior

- **Good Episode**: Stable utilization around 50-60%, minimal resource changes, positive cumulative reward
- **Poor Episode**: Erratic utilization, frequent resource adjustments, negative cumulative reward
- **Failed Episode**: CPU or memory exceeds 95% (early termination)

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

## Deployment

### Hugging Face Spaces Deployment

This project is ready for deployment on Hugging Face Spaces:

#### Step 1: Prepare Repository

Ensure your repository contains:
- `app.py` - Main Gradio application
- `requirements.txt` - All dependencies
- `env/` directory - Environment modules
- `README.md` - This documentation

#### Step 2: Create Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **SDK**: Gradio
   - **Hardware**: CPU Basic (free tier works fine)
   - **Visibility**: Public or Private

#### Step 3: Upload Files

Option A - Git:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push hf main
```

Option B - Web Interface:
- Upload files directly through the Hugging Face web interface

#### Step 4: Configure Space

The space will automatically:
- Install dependencies from `requirements.txt`
- Run `app.py`
- Launch the Gradio interface

#### Step 5: Verify Deployment

- Check build logs for any errors
- Test the deployed app at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

### Local Deployment

For local deployment:

```bash
# Standard launch
python app.py

# Custom port
python app.py --server-port 8080

# Share publicly (temporary link)
# Modify app.py: demo.launch(share=True)
```

### Environment Variables

No environment variables required. All configuration is in code.

### Troubleshooting Deployment

**Issue**: Build fails on Hugging Face
- **Solution**: Check `requirements.txt` has all dependencies
- **Solution**: Ensure Python 3.8+ compatibility

**Issue**: App crashes on startup
- **Solution**: Check logs for import errors
- **Solution**: Verify all `env/` modules are uploaded

**Issue**: Slow performance
- **Solution**: Upgrade to GPU hardware (if needed for RL training)
- **Solution**: Current CPU tier is sufficient for demo

## Dependencies

- **torch>=2.0.0**: PyTorch for tensor operations and neural network integration
- **numpy**: Numerical computations and state representation
- **gradio**: Interactive web UI for manual environment control
- **matplotlib**: Real-time plotting and visualization
- **hypothesis**: Property-based testing framework
- **pytest**: Unit testing framework

## License

This project is provided for educational purposes. 