"""
Unit tests for task configuration system.

This module validates the TASKS dictionary structure and ensures all
required fields are present for each difficulty level.
"""

import pytest
from env.config import TASKS


class TestTaskConfigurations:
    """Tests for task difficulty configurations."""
    
    def test_tasks_contains_all_difficulty_levels(self):
        """Test that TASKS dictionary contains all three difficulty levels."""
        assert "easy" in TASKS
        assert "medium" in TASKS
        assert "hard" in TASKS
        assert len(TASKS) == 3
    
    def test_easy_task_has_required_fields(self):
        """Test that easy task has all required fields."""
        easy = TASKS["easy"]
        assert "base_request_rate" in easy
        assert "request_rate_std" in easy
        assert "description" in easy
        
        assert easy["base_request_rate"] == 40
        assert easy["request_rate_std"] == 5.0
        assert isinstance(easy["description"], str)
    
    def test_medium_task_has_required_fields(self):
        """Test that medium task has all required fields."""
        medium = TASKS["medium"]
        assert "base_request_rate" in medium
        assert "request_rate_std" in medium
        assert "description" in medium
        
        assert medium["base_request_rate"] == 60
        assert medium["request_rate_std"] == 15.0
        assert isinstance(medium["description"], str)
    
    def test_hard_task_has_required_fields(self):
        """Test that hard task has all required fields."""
        hard = TASKS["hard"]
        assert "base_request_rate" in hard
        assert "request_rate_std" in hard
        assert "description" in hard
        
        assert hard["base_request_rate"] == 80
        assert hard["request_rate_std"] == 25.0
        assert isinstance(hard["description"], str)
    
    def test_all_tasks_have_required_fields(self):
        """Test that each task has required fields."""
        required_fields = {"base_request_rate", "request_rate_std", "description"}
        
        for task_name, task_config in TASKS.items():
            assert required_fields.issubset(task_config.keys()), \
                f"Task '{task_name}' missing required fields"
