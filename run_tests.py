#!/usr/bin/env python3
"""
Test runner for EcoNetToolkit.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py data               # Run only data tests
    python run_tests.py models trainer     # Run models and trainer tests
    python run_tests.py -v                 # Verbose mode
"""
import sys
import subprocess
import argparse


TEST_MODULES = {
    "data": "tests/test_data.py",
    "models": "tests/test_models.py",
    "trainer": "tests/test_trainer.py",
    "eval": "tests/test_eval.py",
    "config": "tests/test_config.py",
    "hyperopt": "tests/test_hyperopt.py",
    "integration": "tests/test_integration.py",
}


def main():
    parser = argparse.ArgumentParser(description="Run EcoNetToolkit tests")
    parser.add_argument(
        "modules",
        nargs="*",
        choices=list(TEST_MODULES.keys()) + ["all"],
        default=["all"],
        help="Test modules to run (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Determine which test files to run
    if "all" in args.modules or not args.modules:
        test_files = list(TEST_MODULES.values())
    else:
        test_files = [TEST_MODULES[m] for m in args.modules]

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"] + test_files

    if args.verbose:
        cmd.append("-v")

    # Run pytest
    print(f"Running tests: {', '.join([f.split('/')[-1] for f in test_files])}")
    print(f"Command: {' '.join(cmd[2:])}\n")  # Skip python executable in display

    result = subprocess.run(cmd)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
