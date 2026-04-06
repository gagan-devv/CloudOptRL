"""
Cloud Resource Allocation RL Environment.

This module implements the core environment class for simulating cloud infrastructure
resource allocation. The environment maintains system state (CPU, memory, request rate,
allocated resources) and processes actions to dynamically adjust resource allocation.
"""

import numpy as np
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
        reward_calculator: Optional[RewardCalculator] = None
    ):
        """
        Initialize the cloud resource allocation environment.
        
        Args:
            config: Environment configuration parameters. If None, uses default EnvConfig.
            reward_calculator: Reward calculator instance. If None, creates default RewardCalculator.
        """
        # Use default config if not provided
        self.config = config if config is not None else EnvConfig()
        
        # Initialize reward calculator
        self.reward_calculator = reward_calculator if reward_calculator is not None else RewardCalculator()
        
        # Initialize state variables
        self.cpu_util: float = 0.0
        self.memory_util: float = 0.0
        self.request_rate: int = 0
        self.allocated_resources: int = self.config.initial_resources
        
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
