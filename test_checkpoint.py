#!/usr/bin/env python
"""
Checkpoint verification script for task 10.

This script verifies that all validator compliance requirements are met:
1. All tests pass
2. FastAPI app has /reset endpoint
3. inference.py exists and runs
4. Strict logging format is correct
"""

import subprocess
import sys
import re
import json
from pathlib import Path


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_command(cmd, description, capture_output=True, timeout=30):
    """Run a command and return result."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: Command exceeded {timeout} seconds")
        return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None


def check_tests():
    """Check that all tests pass."""
    print_section("CHECKPOINT 1: Running Test Suite")
    
    result = run_command(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-x"],
        "Running all tests",
        timeout=120
    )
    
    if result is None:
        return False
    
    if result.returncode == 0:
        print("✅ All tests passed!")
        return True
    else:
        print(f"❌ Tests failed with return code {result.returncode}")
        print("\nSTDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return False


def check_fastapi_reset_endpoint():
    """Check that FastAPI app has /reset endpoint."""
    print_section("CHECKPOINT 2: Verifying FastAPI /reset Endpoint")
    
    # Check that app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found")
        return False
    
    # Read app.py and verify it has FastAPI and /reset endpoint
    with open("app.py", "r") as f:
        content = f.read()
    
    checks = {
        "FastAPI import": "from fastapi import FastAPI" in content,
        "/reset endpoint": '@app.post("/reset")' in content or "@app.post('/reset')" in content,
        "Returns status ok": '{"status": "ok"}' in content or "{'status': 'ok'}" in content,
        "Port 7860": "port=7860" in content or "port = 7860" in content,
        "Gradio mounting": "gr.mount_gradio_app" in content
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ FastAPI integration verified!")
    else:
        print("\n❌ FastAPI integration incomplete")
    
    return all_passed


def check_inference_script():
    """Check that inference.py exists and runs correctly."""
    print_section("CHECKPOINT 3: Verifying inference.py")
    
    # Check that inference.py exists
    if not Path("inference.py").exists():
        print("❌ inference.py not found")
        return False
    
    print("✅ inference.py exists")
    
    # Run inference.py and capture output
    print("\nRunning inference.py...")
    result = run_command(
        ["python", "inference.py"],
        "Running inference script",
        timeout=30
    )
    
    if result is None or result.returncode != 0:
        print(f"❌ inference.py failed to run")
        if result:
            print(f"Return code: {result.returncode}")
            print(f"STDERR: {result.stderr}")
        return False
    
    print("✅ inference.py ran successfully")
    
    # Verify output format
    stdout = result.stdout
    lines = [line.strip() for line in stdout.strip().split('\n') if line.strip()]
    
    if not lines:
        print("❌ No output from inference.py")
        return False
    
    print(f"\n✅ Got {len(lines)} output lines")
    
    # Check log patterns
    start_pattern = r'^\[START\] task=\w+ env=\w+ model=[\w-]+'
    step_pattern = r'^\[STEP\] step=\d+ action=[0-2] reward=-?\d+\.\d{2} done=(True|False) error=(None|.+)'
    end_pattern = r'^\[END\] success=(True|False) steps=\d+ score=-?\d+\.\d{2} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*)?'
    
    # Verify first line is [START]
    if not re.match(start_pattern, lines[0]):
        print(f"❌ First line is not [START]: {lines[0]}")
        return False
    print(f"✅ [START] log: {lines[0]}")
    
    # Verify last line is [END]
    if not re.match(end_pattern, lines[-1]):
        print(f"❌ Last line is not [END]: {lines[-1]}")
        return False
    print(f"✅ [END] log: {lines[-1]}")
    
    # Verify middle lines are [STEP]
    step_count = 0
    for i, line in enumerate(lines[1:-1], start=1):
        if not re.match(step_pattern, line):
            print(f"❌ Line {i+1} is not [STEP]: {line}")
            return False
        step_count += 1
    
    print(f"✅ All {step_count} [STEP] logs are correctly formatted")
    
    # Verify no other output
    print("\n✅ Clean stdout output verified (only [START], [STEP], [END])")
    
    return True


def check_async_wrapper():
    """Check that AsyncEnvWrapper exists."""
    print_section("CHECKPOINT 4: Verifying AsyncEnvWrapper")
    
    if not Path("env/async_wrapper.py").exists():
        print("❌ env/async_wrapper.py not found")
        return False
    
    print("✅ env/async_wrapper.py exists")
    
    # Try to import it
    try:
        from env.async_wrapper import AsyncEnvWrapper
        print("✅ AsyncEnvWrapper can be imported")
        
        # Check it has required methods
        required_methods = ['reset', 'step', 'close']
        for method in required_methods:
            if not hasattr(AsyncEnvWrapper, method):
                print(f"❌ AsyncEnvWrapper missing method: {method}")
                return False
        
        print(f"✅ AsyncEnvWrapper has all required methods: {', '.join(required_methods)}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to import AsyncEnvWrapper: {e}")
        return False


def check_requirements():
    """Check that requirements.txt has all dependencies."""
    print_section("CHECKPOINT 5: Verifying Dependencies")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    with open("requirements.txt", "r") as f:
        content = f.read()
    
    required_deps = ["fastapi", "uvicorn", "gradio", "pytest", "hypothesis", "numpy"]
    
    all_present = True
    for dep in required_deps:
        if dep in content:
            print(f"✅ {dep}")
        else:
            print(f"❌ {dep} missing")
            all_present = False
    
    return all_present


def main():
    """Run all checkpoint verifications."""
    print("\n" + "="*60)
    print("  FINAL CHECKPOINT - Task 10 Verification")
    print("  Spec: final-submission-compliance")
    print("="*60)
    
    results = {
        "Dependencies": check_requirements(),
        "AsyncEnvWrapper": check_async_wrapper(),
        "FastAPI /reset": check_fastapi_reset_endpoint(),
        "inference.py": check_inference_script(),
        "Test Suite": check_tests(),
    }
    
    # Print summary
    print_section("SUMMARY")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("  ✅ ALL CHECKS PASSED - READY FOR SUBMISSION")
    else:
        print("  ❌ SOME CHECKS FAILED - REVIEW REQUIRED")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
