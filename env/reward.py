"""
Reward calculation module for the Cloud Resource Allocation RL environment.

This module implements the RewardCalculator class that evaluates resource allocation
decisions based on CPU utilization, memory utilization, and resource consumption.
The reward structure encourages efficient operation within optimal utilization ranges
while penalizing both over-provisioning and under-provisioning.
"""


class RewardCalculator:
    """
    Calculates reward signals for resource allocation decisions.
    
    Reward structure:
    - Positive rewards for efficient operation (moderate utilization)
    - Negative rewards for over-provisioning (wasted resources)
    - Negative rewards for under-provisioning (high utilization risk)
    
    The reward calculation considers three factors:
    1. CPU utilization relative to target range
    2. Memory utilization relative to target range
    3. Resource cost penalty based on allocated instances
    """
    
    def __init__(
        self,
        target_cpu_range: tuple[float, float] = (40.0, 70.0),
        target_memory_range: tuple[float, float] = (40.0, 70.0),
        resource_cost_weight: float = 0.1
    ):
        """
        Initialize reward calculator with target ranges and weights.
        
        Args:
            target_cpu_range: Optimal CPU utilization range (min, max) in percentage
            target_memory_range: Optimal memory utilization range (min, max) in percentage
            resource_cost_weight: Weight for resource cost penalty (cost per instance)
        """
        self.target_cpu_range = target_cpu_range
        self.target_memory_range = target_memory_range
        self.resource_cost_weight = resource_cost_weight
    
    def calculate_reward(
        self,
        cpu_util: float,
        memory_util: float,
        allocated_resources: int
    ) -> float:
        """
        Calculate reward for current state.
        
        The reward combines utilization rewards for CPU and memory with a resource
        cost penalty. States within optimal utilization ranges receive positive rewards,
        while over-provisioning and under-provisioning receive negative rewards.
        
        If either CPU or memory is outside the optimal range, the total reward is
        capped at a small negative value to ensure suboptimal states are penalized.
        
        Args:
            cpu_util: Current CPU utilization percentage (0-100)
            memory_util: Current memory utilization percentage (0-100)
            allocated_resources: Current number of allocated instances (minimum 1)
        
        Returns:
            Reward value (positive for good allocation, negative for poor allocation)
        
        Raises:
            ValueError: If allocated_resources is less than 1
        """
        # Validate inputs
        if allocated_resources < 1:
            raise ValueError("Allocated resources must be at least 1")
        
        # Clamp utilization values to valid range [0, 100]
        cpu_util = max(0.0, min(100.0, cpu_util))
        memory_util = max(0.0, min(100.0, memory_util))
        
        # Calculate utilization rewards
        cpu_reward = self._utilization_reward(cpu_util, self.target_cpu_range)
        memory_reward = self._utilization_reward(memory_util, self.target_memory_range)
        
        # Calculate resource cost penalty
        resource_penalty = self._resource_cost_penalty(allocated_resources)
        
        # Combine components
        total_reward = cpu_reward + memory_reward + resource_penalty
        
        # If either metric is outside optimal range, cap reward at negative value
        # This ensures that mixed scenarios (one optimal, one suboptimal) are always negative
        cpu_in_range = self.target_cpu_range[0] <= cpu_util <= self.target_cpu_range[1]
        memory_in_range = self.target_memory_range[0] <= memory_util <= self.target_memory_range[1]
        
        if not (cpu_in_range and memory_in_range):
            # At least one metric is out of range - cap at small negative value
            total_reward = min(total_reward, -0.01)
        
        return total_reward
    
    def _utilization_reward(self, util: float, target_range: tuple[float, float]) -> float:
        """
        Calculate reward component for a single utilization metric.
        
        Reward structure:
        - Positive reward when utilization is within target range (optimal)
        - Negative reward when utilization is below target range (over-provisioning)
        - Negative reward when utilization is above target range (under-provisioning)
        
        Args:
            util: Current utilization percentage (0-100)
            target_range: Optimal utilization range (min, max)
        
        Returns:
            Reward component for this utilization metric
        """
        target_min, target_max = target_range
        
        if target_min <= util <= target_max:
            # Within optimal range - positive reward
            # Reward is higher when closer to the middle of the range
            range_center = (target_min + target_max) / 2
            distance_from_center = abs(util - range_center)
            range_width = target_max - target_min
            # Normalize to [0, 1] where 1 is at center, 0 is at edges
            normalized_position = 1.0 - (distance_from_center / (range_width / 2))
            return normalized_position
        elif util < target_min:
            # Below optimal range - over-provisioning penalty
            # Penalty increases as utilization gets further below target
            distance_below = target_min - util
            return -0.5 - (distance_below / 100.0)
        else:  # util > target_max
            # Above optimal range - under-provisioning penalty
            # Penalty increases as utilization gets further above target
            distance_above = util - target_max
            return -1.0 - (distance_above / 100.0)
    
    def _resource_cost_penalty(self, allocated_resources: int) -> float:
        """
        Calculate penalty for resource consumption.
        
        This penalty encourages the agent to minimize resource usage while
        maintaining acceptable utilization levels. The penalty scales linearly
        with the number of allocated instances.
        
        Args:
            allocated_resources: Current number of allocated instances
        
        Returns:
            Resource cost penalty (always negative or zero)
        """
        return -self.resource_cost_weight * allocated_resources
