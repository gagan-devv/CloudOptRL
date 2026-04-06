"""
Configuration data models for the Cloud Resource Allocation RL environment.

This module defines dataclasses for environment and reward configuration parameters,
providing type-safe configuration with sensible defaults for the RL simulation.
"""

from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class EnvConfig:
    """
    Environment configuration parameters.
    
    This dataclass encapsulates all parameters controlling the cloud resource
    allocation environment's behavior, including episode length, initial conditions,
    request dynamics, and resource constraints.
    
    Attributes:
        max_steps: Maximum episode length (number of time steps before termination)
        initial_resources: Starting number of server instances at episode start
        base_request_rate: Baseline incoming request rate (requests per time step)
        request_rate_std: Standard deviation for stochastic request rate fluctuation
        cpu_per_request: CPU utilization cost per request (percentage points)
        memory_per_request: Memory utilization cost per request (percentage points)
        resource_capacity: Processing capacity per server instance
        termination_threshold: Utilization percentage triggering episode termination
    """
    max_steps: int = 100
    initial_resources: int = 2
    base_request_rate: int = 50
    request_rate_std: float = 10.0
    cpu_per_request: float = 0.5
    memory_per_request: float = 0.45
    resource_capacity: float = 30.0
    termination_threshold: float = 95.0


@dataclass
class RewardConfig:
    """
    Reward calculation configuration parameters.
    
    This dataclass defines the reward shaping parameters that guide the RL agent
    toward efficient resource allocation. It specifies optimal utilization ranges
    and penalty weights for over-provisioning and under-provisioning.
    
    Attributes:
        target_cpu_min: Lower bound of optimal CPU utilization range (percentage)
        target_cpu_max: Upper bound of optimal CPU utilization range (percentage)
        target_memory_min: Lower bound of optimal memory utilization range (percentage)
        target_memory_max: Upper bound of optimal memory utilization range (percentage)
        resource_cost_weight: Penalty weight for resource usage (cost per instance)
        over_provision_penalty: Penalty applied when resources exceed system needs
        under_provision_penalty: Penalty applied when resources are insufficient
    """
    target_cpu_min: float = 40.0
    target_cpu_max: float = 70.0
    target_memory_min: float = 40.0
    target_memory_max: float = 70.0
    resource_cost_weight: float = 0.1
    over_provision_penalty: float = -0.5
    under_provision_penalty: float = -1.0


class EnvState(BaseModel):
    """
    Typed environment state representation for OpenEnv compliance.
    
    This Pydantic model provides validated, typed access to environment state
    with automatic field validation ensuring values are within valid ranges.
    
    Attributes:
        cpu: CPU utilization percentage in range [0.0, 100.0]
        memory: Memory utilization percentage in range [0.0, 100.0]
        request_rate: Incoming request rate, non-negative float
        resources: Allocated server instances, minimum 1
    """
    cpu: float = Field(ge=0.0, le=100.0, description="CPU utilization percentage")
    memory: float = Field(ge=0.0, le=100.0, description="Memory utilization percentage")
    request_rate: float = Field(ge=0.0, description="Incoming request rate")
    resources: int = Field(ge=1, description="Allocated server instances")


# Task difficulty configurations for OpenEnv compliance
TASKS = {
    "easy": {
        "base_request_rate": 40,
        "request_rate_std": 5.0,
        "description": "Low request rate with minimal variance - ideal for learning basic resource allocation"
    },
    "medium": {
        "base_request_rate": 60,
        "request_rate_std": 15.0,
        "description": "Moderate request rate with medium variance - balanced challenge for standard training"
    },
    "hard": {
        "base_request_rate": 80,
        "request_rate_std": 25.0,
        "description": "High request rate with high variance - challenging scenario requiring robust policies"
    }
}
