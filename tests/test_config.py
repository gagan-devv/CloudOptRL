"""
Unit tests for configuration data models.

Tests verify that EnvConfig and RewardConfig dataclasses correctly initialize
with default values and accept custom parameters.
"""

import pytest
from env.config import EnvConfig, RewardConfig


class TestEnvConfig:
    """Test suite for EnvConfig dataclass."""
    
    def test_default_values(self):
        """Test that EnvConfig initializes with correct default values."""
        config = EnvConfig()
        
        assert config.max_steps == 100
        assert config.initial_resources == 3
        assert config.base_request_rate == 50
        assert config.request_rate_std == 10.0
        assert config.cpu_per_request == 0.5
        assert config.memory_per_request == 0.3
        assert config.resource_capacity == 30.0
        assert config.termination_threshold == 95.0
    
    def test_custom_parameters(self):
        """Test that EnvConfig accepts custom parameter values."""
        config = EnvConfig(
            max_steps=200,
            initial_resources=5,
            base_request_rate=100,
            request_rate_std=15.0,
            cpu_per_request=0.7,
            memory_per_request=0.4,
            resource_capacity=50.0,
            termination_threshold=90.0
        )
        
        assert config.max_steps == 200
        assert config.initial_resources == 5
        assert config.base_request_rate == 100
        assert config.request_rate_std == 15.0
        assert config.cpu_per_request == 0.7
        assert config.memory_per_request == 0.4
        assert config.resource_capacity == 50.0
        assert config.termination_threshold == 90.0
    
    def test_partial_custom_parameters(self):
        """Test that EnvConfig allows partial parameter override."""
        config = EnvConfig(max_steps=150, initial_resources=4)
        
        assert config.max_steps == 150
        assert config.initial_resources == 4
        # Verify other parameters use defaults
        assert config.base_request_rate == 50
        assert config.request_rate_std == 10.0


class TestRewardConfig:
    """Test suite for RewardConfig dataclass."""
    
    def test_default_values(self):
        """Test that RewardConfig initializes with correct default values."""
        config = RewardConfig()
        
        assert config.target_cpu_min == 40.0
        assert config.target_cpu_max == 70.0
        assert config.target_memory_min == 40.0
        assert config.target_memory_max == 70.0
        assert config.resource_cost_weight == 0.1
        assert config.over_provision_penalty == -0.5
        assert config.under_provision_penalty == -1.0
    
    def test_custom_parameters(self):
        """Test that RewardConfig accepts custom parameter values."""
        config = RewardConfig(
            target_cpu_min=30.0,
            target_cpu_max=80.0,
            target_memory_min=35.0,
            target_memory_max=75.0,
            resource_cost_weight=0.2,
            over_provision_penalty=-0.8,
            under_provision_penalty=-1.5
        )
        
        assert config.target_cpu_min == 30.0
        assert config.target_cpu_max == 80.0
        assert config.target_memory_min == 35.0
        assert config.target_memory_max == 75.0
        assert config.resource_cost_weight == 0.2
        assert config.over_provision_penalty == -0.8
        assert config.under_provision_penalty == -1.5
    
    def test_partial_custom_parameters(self):
        """Test that RewardConfig allows partial parameter override."""
        config = RewardConfig(
            target_cpu_min=35.0,
            target_cpu_max=65.0
        )
        
        assert config.target_cpu_min == 35.0
        assert config.target_cpu_max == 65.0
        # Verify other parameters use defaults
        assert config.target_memory_min == 40.0
        assert config.resource_cost_weight == 0.1
