"""
Property-based tests for task configuration loading in CloudResourceEnv.

This module validates that task configurations are correctly loaded and applied
when creating environment instances with different difficulty levels.
"""

import pytest
from hypothesis import given, strategies as st

from env.environment import CloudResourceEnv
from env.config import TASKS


class TestTaskConfigurationLoading:
    """Property 4: Task Configuration Loading"""
    
    @given(task_name=st.sampled_from(["easy", "medium", "hard"]))
    def test_task_configuration_loading(self, task_name):
        """Test that creating environment with each task name results in correct configuration."""
        env = CloudResourceEnv(task_name=task_name)
        
        expected_config = TASKS[task_name]
        
        assert env.config.base_request_rate == expected_config["base_request_rate"]
        assert env.config.request_rate_std == expected_config["request_rate_std"]
        assert env.task_name == task_name
    
    def test_easy_task_configuration(self):
        """Test that easy task loads correct configuration."""
        env = CloudResourceEnv(task_name="easy")
        
        assert env.config.base_request_rate == 40
        assert env.config.request_rate_std == 5.0
        assert env.task_name == "easy"
    
    def test_medium_task_configuration(self):
        """Test that medium task loads correct configuration."""
        env = CloudResourceEnv(task_name="medium")
        
        assert env.config.base_request_rate == 60
        assert env.config.request_rate_std == 15.0
        assert env.task_name == "medium"
    
    def test_hard_task_configuration(self):
        """Test that hard task loads correct configuration."""
        env = CloudResourceEnv(task_name="hard")
        
        assert env.config.base_request_rate == 80
        assert env.config.request_rate_std == 25.0
        assert env.task_name == "hard"
    
    def test_default_task_is_medium(self):
        """Test that default task_name is 'medium'."""
        env = CloudResourceEnv()
        
        assert env.task_name == "medium"
        assert env.config.base_request_rate == 60
        assert env.config.request_rate_std == 15.0
    
    def test_invalid_task_name_raises_error(self):
        """Test that invalid task_name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CloudResourceEnv(task_name="invalid")
        
        assert "invalid" in str(exc_info.value).lower()
        assert "task_name" in str(exc_info.value).lower()
