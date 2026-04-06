"""
Property-based tests for EnvState validation.

This module validates EnvState field constraints using property-based testing
to ensure robust validation across a wide range of inputs.
"""

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from env.config import EnvState


class TestEnvStateValidation:
    """Property 2: EnvState Field Validation"""
    
    @given(
        cpu=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        memory=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        request_rate=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        resources=st.integers(min_value=1, max_value=100)
    )
    def test_valid_values_create_envstate(self, cpu, memory, request_rate, resources):
        """Test that valid values create EnvState without errors."""
        state = EnvState(
            cpu=cpu,
            memory=memory,
            request_rate=request_rate,
            resources=resources
        )
        
        assert state.cpu == cpu
        assert state.memory == memory
        assert state.request_rate == request_rate
        assert state.resources == resources
    
    @given(cpu=st.floats(min_value=100.01, max_value=200.0, allow_nan=False, allow_infinity=False))
    def test_invalid_cpu_raises_validation_error(self, cpu):
        """Test that cpu > 100 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EnvState(cpu=cpu, memory=50.0, request_rate=100.0, resources=3)
        
        assert "cpu" in str(exc_info.value).lower()
    
    @given(memory=st.floats(min_value=100.01, max_value=200.0, allow_nan=False, allow_infinity=False))
    def test_invalid_memory_raises_validation_error(self, memory):
        """Test that memory > 100 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EnvState(cpu=50.0, memory=memory, request_rate=100.0, resources=3)
        
        assert "memory" in str(exc_info.value).lower()
    
    @given(resources=st.integers(min_value=-10, max_value=0))
    def test_invalid_resources_raises_validation_error(self, resources):
        """Test that resources < 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EnvState(cpu=50.0, memory=50.0, request_rate=100.0, resources=resources)
        
        assert "resources" in str(exc_info.value).lower()
    
    @given(request_rate=st.floats(min_value=-100.0, max_value=-0.01, allow_nan=False, allow_infinity=False))
    def test_negative_request_rate_raises_validation_error(self, request_rate):
        """Test that negative request_rate raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            EnvState(cpu=50.0, memory=50.0, request_rate=request_rate, resources=3)
        
        assert "request_rate" in str(exc_info.value).lower()
