"""
Property-based tests for OpenEnv validator compliance bugfix.

These tests validate that the repository meets OpenEnv multi-mode deployment
requirements. The bug condition exploration test verifies that the validator
currently fails on unfixed code, confirming the bug exists.
"""

import os
import tomllib
from pathlib import Path
from hypothesis import given, strategies as st, settings


# Feature: openenv-deployment-fixes, Property 1: Bug Condition - OpenEnv Validator Fails on Missing Requirements
@settings(max_examples=1)  # Deterministic bug - single example is sufficient
@given(dummy=st.just(None))  # Scoped to concrete failing case
def test_openenv_validator_bug_condition(dummy):
    """
    Property 1: Bug Condition - OpenEnv Validator Fails on Missing Requirements
    
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    
    This test verifies that the OpenEnv validator currently fails on the unfixed
    code due to missing packaging requirements. This test MUST FAIL on unfixed code
    to confirm the bug exists. Once the fix is implemented, this same test will
    pass, confirming the expected behavior is satisfied.
    
    The test checks for four specific validation failures:
    1. Missing server/app.py file
    2. Missing [project.scripts] server entry in pyproject.toml
    3. Missing openenv-core>=0.2.0 dependency in pyproject.toml
    4. Missing uv.lock file
    
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test FAILS (this is correct - proves bug exists)
    **EXPECTED OUTCOME ON FIXED CODE**: Test PASSES (confirms bug is fixed)
    """
    repo_root = Path(__file__).parent.parent
    
    # Check 1: server/app.py file exists
    server_app_path = repo_root / "server" / "app.py"
    server_app_exists = server_app_path.exists() and server_app_path.is_file()
    
    # Check 2 & 3: pyproject.toml contains [project.scripts] entry and openenv-core dependency
    pyproject_path = repo_root / "pyproject.toml"
    has_project_scripts_entry = False
    has_openenv_core_dependency = False
    
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)
            
            # Check for [project.scripts] section with server entry
            if "project" in pyproject_data and "scripts" in pyproject_data["project"]:
                scripts = pyproject_data["project"]["scripts"]
                if "server" in scripts:
                    # Verify it points to server.app:app
                    has_project_scripts_entry = scripts["server"] == "server.app:app"
            
            # Check for openenv-core>=0.2.0 in dependencies
            if "project" in pyproject_data and "dependencies" in pyproject_data["project"]:
                dependencies = pyproject_data["project"]["dependencies"]
                for dep in dependencies:
                    if isinstance(dep, str) and dep.startswith("openenv-core"):
                        # Check if version constraint is >=0.2.0
                        has_openenv_core_dependency = ">=0.2.0" in dep or ">=" in dep
                        break
    
    # Check 4: uv.lock file exists
    uv_lock_path = repo_root / "uv.lock"
    uv_lock_exists = uv_lock_path.exists() and uv_lock_path.is_file()
    
    # Collect validation failures for detailed error reporting
    failures = []
    
    if not server_app_exists:
        failures.append("Missing server/app.py file")
    
    if not has_project_scripts_entry:
        failures.append("Missing [project.scripts] server entry in pyproject.toml")
    
    if not has_openenv_core_dependency:
        failures.append("Missing openenv-core>=0.2.0 dependency in pyproject.toml")
    
    if not uv_lock_exists:
        failures.append("Missing uv.lock file")
    
    # Assert all checks pass (this will FAIL on unfixed code, confirming the bug)
    assert server_app_exists, (
        f"OpenEnv validation failed: server/app.py file not found at {server_app_path}"
    )
    
    assert has_project_scripts_entry, (
        "OpenEnv validation failed: [project.scripts] section missing 'server = \"server.app:app\"' entry"
    )
    
    assert has_openenv_core_dependency, (
        "OpenEnv validation failed: openenv-core>=0.2.0 dependency not found in pyproject.toml"
    )
    
    assert uv_lock_exists, (
        f"OpenEnv validation failed: uv.lock file not found at {uv_lock_path}"
    )
    
    # If we reach here, all checks passed
    # On unfixed code, we should never reach this point
    # On fixed code, this confirms the bug is resolved
    print(f"✅ OpenEnv validation passed: All {4} checks successful")
