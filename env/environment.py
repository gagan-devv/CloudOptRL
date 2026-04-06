"""
Cloud Resource Allocation RL Environment.

This module implements the core environment class for simulating cloud infrastructure
resource allocation. The environment maintains system state (CPU, memory, request rate,
allocated resources) and processes actions to dynamically adjust resource allocation.
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from env.config import EnvConfig
from env.reward import RewardCalculator


class CloudResourceEnv:
    """
    Simulates a cloud infrastructure environment for RL-based resource allocation.
    
    The environment models a cloud system where an agent must dynamically allocate
    server instances based on observed system metrics. The state includes CPU utilization,
    memory utilization, incoming request rate, and currently allocated resources.
    
    State space: [cpu_util, memory_util, request_rate, allocated_resources]
    Action space: {0: decrease, 1: maintain, 2: increase}
    
    The environment incorporates stochastic dynamics to simulate real-world variability
    in request patterns and system behavior.
    """
    
    # Action constants
    ACTION_DECREASE = 0
    ACTION_MAINTAIN = 1
    ACTION_INCREASE = 2
    
    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        task_name: str = "medium"
    ):
        """
        Initialize the cloud resource allocation environment.
        
        Args:
            config: Environment configuration parameters. If None, uses default EnvConfig.
            reward_calculator: Reward calculator instance. If None, creates default RewardCalculator.
            task_name: Task difficulty level ("easy", "medium", or "hard"). Defaults to "medium".
        
        Raises:
            ValueError: If task_name is not in TASKS dictionary.
        """
        # Import TASKS here to avoid circular imports
        from env.config import TASKS
        
        # Validate task_name
        if task_name not in TASKS:
            raise ValueError(
                f"Invalid task_name '{task_name}'. Must be one of: {list(TASKS.keys())}"
            )
        
        # Use default config if not provided
        self.config = config if config is not None else EnvConfig()
        
        # Override config with task-specific parameters
        task_config = TASKS[task_name]
        self.config.base_request_rate = task_config["base_request_rate"]
        self.config.request_rate_std = task_config["request_rate_std"]
        self.task_name = task_name
        
        # Initialize reward calculator
        self.reward_calculator = reward_calculator if reward_calculator is not None else RewardCalculator()
        
        # Initialize state variables
        self.cpu_util: float = 0.0
        self.memory_util: float = 0.0
        self.request_rate: int = 0
        self.allocated_resources: int = self.config.initial_resources
        self.latency: float = 0.0  # Latency in ms
        
        # Initialize episode tracking
        self.current_step: int = 0
        self.episode_states: List[np.ndarray] = []
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        self.cumulative_reward: float = 0.0
        
        # Episode termination flag
        self.done: bool = False
        
        # Random number generator for reproducibility
        self.rng = np.random.RandomState()
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        This method reinitializes the environment for a new episode. It resets all
        state variables to their initial values, clears episode history, and introduces
        stochastic variation in the initial request rate to ensure diverse starting
        conditions across episodes.
        
        Returns:
            Initial observation as numpy array [cpu_util, memory_util, request_rate, allocated_resources]
        """
        # Reset episode tracking
        self.current_step = 0
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.cumulative_reward = 0.0
        self.done = False
        
        # Reset allocated resources to initial value
        self.allocated_resources = self.config.initial_resources
        
        # Initialize request rate with stochastic variation
        # This ensures different starting conditions across episodes
        self.request_rate = int(
            self.config.base_request_rate + 
            self.rng.normal(0, self.config.request_rate_std)
        )
        # Ensure request rate is non-negative
        self.request_rate = max(0, self.request_rate)
        
        # Calculate initial CPU and memory utilization
        # Utilization depends on request rate and allocated resources
        self.cpu_util = (self.request_rate * self.config.cpu_per_request) / (
            self.allocated_resources * self.config.resource_capacity
        ) * 100.0
        
        self.memory_util = (self.request_rate * self.config.memory_per_request) / (
            self.allocated_resources * self.config.resource_capacity
        ) * 100.0
        
        # Clamp utilization values to valid range [0, 100]
        self.cpu_util = max(0.0, min(100.0, self.cpu_util))
        self.memory_util = max(0.0, min(100.0, self.memory_util))
        
        # Calculate latency: higher request rate and fewer resources increase latency
        self.latency = self.request_rate / max(self.allocated_resources, 1)
        
        # Return initial observation
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current state as observation array.
        
        Returns:
            Numpy array [cpu_util, memory_util, request_rate, allocated_resources]
        """
        return np.array([
            self.cpu_util,
            self.memory_util,
            float(self.request_rate),
            float(self.allocated_resources)
        ], dtype=np.float32)
    
    def get_state_tensor(self) -> torch.Tensor:
        """
        Get current state as PyTorch tensor.
        
        This method provides a PyTorch tensor representation of the current state,
        enabling seamless integration with PyTorch-based RL algorithms and neural networks.
        The tensor maintains the same structure as the NumPy observation array.
        
        Returns:
            PyTorch tensor [cpu_util, memory_util, request_rate, allocated_resources]
        """
        observation = self._get_observation()
        return torch.from_numpy(observation)
    
    def state(self):
        """
        Get current state as EnvState instance for OpenEnv compliance.
        
        This method provides typed, validated access to the environment state
        using the Pydantic EnvState model. It can be called after reset() or step()
        to retrieve the current state in a structured format.
        
        Returns:
            EnvState: Pydantic model with cpu, memory, request_rate, and resources fields
        """
        from env.config import EnvState
        
        return EnvState(
            cpu=self.cpu_util,
            memory=self.memory_util,
            request_rate=float(self.request_rate),
            resources=self.allocated_resources
        )
    
    def _update_state(self, action: int) -> None:
        """
        Update system state based on action and stochastic dynamics.
        
        This method applies the selected action to modify allocated resources,
        updates the request rate with stochastic fluctuation, and recalculates
        CPU and memory utilization based on the new system configuration.
        
        Args:
            action: Action to execute (0: decrease, 1: maintain, 2: increase)
        """
        # Apply action to allocated resources
        if action == self.ACTION_INCREASE:
            self.allocated_resources += 1
        elif action == self.ACTION_DECREASE:
            # Maintain minimum of 1 resource
            self.allocated_resources = max(1, self.allocated_resources - 1)
        elif action == self.ACTION_MAINTAIN:
            # Keep resources unchanged
            pass
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2.")
        
        # Update request rate with stochastic fluctuation
        # This simulates real-world variability in incoming traffic
        fluctuation = self.rng.normal(0, self.config.request_rate_std)
        self.request_rate = int(self.config.base_request_rate + fluctuation)
        # Ensure request rate is non-negative
        self.request_rate = max(0, self.request_rate)
        
        # Calculate CPU utilization based on request rate and allocated resources
        # Higher request rate increases utilization
        # More allocated resources decrease utilization
        self.cpu_util = (self.request_rate * self.config.cpu_per_request) / (
            self.allocated_resources * self.config.resource_capacity
        ) * 100.0
        
        # Calculate memory utilization based on request rate and allocated resources
        self.memory_util = (self.request_rate * self.config.memory_per_request) / (
            self.allocated_resources * self.config.resource_capacity
        ) * 100.0
        
        # Clamp all values to valid ranges
        self.cpu_util = max(0.0, min(100.0, self.cpu_util))
        self.memory_util = max(0.0, min(100.0, self.memory_util))
        
        # Calculate latency: higher request rate and fewer resources increase latency
        self.latency = self.request_rate / max(self.allocated_resources, 1)
    
    def _check_termination(self) -> bool:
        """
        Check if episode should terminate based on state or step count.
        
        Termination conditions:
        1. Step count exceeds max_steps
        2. CPU utilization exceeds termination threshold (default 95%)
        3. Memory utilization exceeds termination threshold (default 95%)
        
        Returns:
            True if episode should terminate, False otherwise
        """
        # Check if max steps exceeded
        if self.current_step >= self.config.max_steps:
            return True
        
        # Check if CPU utilization exceeds threshold
        if self.cpu_util > self.config.termination_threshold:
            return True
        
        # Check if memory utilization exceeds threshold
        if self.memory_util > self.config.termination_threshold:
            return True
        
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step with the given action.
        
        This method processes the action, updates the environment state, calculates
        the reward, checks for termination, and stores the transition in episode
        history. It returns the standard RL tuple (observation, reward, done, info).
        
        Args:
            action: Action to execute (0: decrease, 1: maintain, 2: increase)
        
        Returns:
            observation: Current state after action as numpy array
            reward: Reward signal from reward calculator
            done: Whether episode has terminated
            info: Additional diagnostic information dictionary
        
        Raises:
            ValueError: If action is not in valid range {0, 1, 2}
            RuntimeError: If step is called after episode termination
        """
        # Check if episode already terminated
        if self.done:
            raise RuntimeError(
                "Cannot call step() after episode termination. Call reset() to start a new episode."
            )
        
        # Validate action is in valid range
        if action not in {self.ACTION_DECREASE, self.ACTION_MAINTAIN, self.ACTION_INCREASE}:
            raise ValueError(
                f"Invalid action: {action}. Must be 0 (decrease), 1 (maintain), or 2 (increase)."
            )
        
        # Update state based on action
        self._update_state(action)
        
        # Increment step counter
        self.current_step += 1
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            self.cpu_util,
            self.memory_util,
            self.allocated_resources
        )
        
        # Check termination status
        self.done = self._check_termination()
        
        # Get current observation
        observation = self._get_observation()
        
        # Store transition in episode history
        self.episode_states.append(observation.copy())
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.cumulative_reward += reward
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'cumulative_reward': self.cumulative_reward,
            'cpu_util': self.cpu_util,
            'memory_util': self.memory_util,
            'request_rate': self.request_rate,
            'allocated_resources': self.allocated_resources,
            'latency': self.latency
        }
        
        return observation, reward, self.done, info
