# Cloud Resource Allocation RL

A hands-on reinforcement learning project that simulates cloud infrastructure resource allocation. Learn RL concepts by training an agent to dynamically allocate server instances based on system load, CPU utilization, and memory demand.

## Overview

This project provides a beginner-friendly introduction to reinforcement learning through a practical cloud resource management scenario. An RL agent observes system metrics (CPU utilization, memory utilization, request rate) and learns to make optimal resource allocation decisions (increase, decrease, or maintain server instances).

**Key Features:**
- Realistic cloud infrastructure simulation with stochastic dynamics
- Interactive Gradio UI for manual exploration before agent training
- Modular architecture separating environment, reward, and evaluation logic
- Comprehensive property-based testing using Hypothesis
- Clear documentation and type hints for learning

**Target Audience:** Developers new to RL but experienced with Python and PyTorch.

## Project Structure

```
cloud-resource-allocation-rl/
├── env/                    # Environment and core logic
│   ├── __init__.py
│   ├── environment.py      # CloudResourceEnv class (state, actions, dynamics)
│   ├── reward.py           # RewardCalculator class (reward shaping)
│   ├── grader.py           # EpisodeGrader class (performance evaluation)
│   └── config.py           # Configuration dataclasses
├── tests/                  # Unit and property-based tests
│   ├── __init__.py
│   └── test_properties.py  # Hypothesis property tests
├── app.py                  # Gradio interactive UI
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cloud-resource-allocation-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Tests

Run all tests with pytest:
```bash
pytest tests/
```

Run property-based tests specifically:
```bash
pytest tests/test_properties.py -v
```

### Launching the Interactive UI

Start the Gradio interface:
```bash
python app.py
```

The UI will open in your browser, allowing you to manually control resource allocation and observe system dynamics.

## Quick Start

1. **Install dependencies** (see Installation above)
2. **Explore the UI** to understand system behavior manually
3. **Review the code** starting with `env/environment.py` to understand the RL interface
4. **Run tests** to see property-based testing in action
5. **Extend the project** by modifying reward functions or adding new features

## RL Concepts

- **State**: System observation (CPU %, memory %, request rate, allocated resources)
- **Action**: Resource allocation decision (increase, decrease, maintain instances)
- **Reward**: Feedback signal encouraging efficient allocation (positive for optimal utilization, negative for over/under-provisioning)
- **Episode**: Complete simulation run from initialization to termination
- **Policy**: Strategy for selecting actions based on observed state (learned by the agent)

## Dependencies

- **numpy**: Numerical computations and state representation
- **gradio**: Interactive web UI for manual environment control
- **hypothesis**: Property-based testing framework
- **pytest**: Unit testing framework

## License

This project is provided for educational purposes. 