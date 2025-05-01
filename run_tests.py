"""
Script to run tests for the sequence-to-sequence model.
"""

import os
import sys
import argparse


def main():
    """Run tests with pytest."""
    parser = argparse.ArgumentParser(description="Run tests for the sequence-to-sequence model")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Build command
    cmd = "pytest"

    if args.verbose:
        cmd += " -v"

    if args.unit:
        cmd += " -k 'not integration'"
    elif args.integration:
        cmd += " -k 'integration'"

    # Default to all tests if no specific option is selected
    if not (args.unit or args.integration or args.all):
        args.all = True

    # Skip slow tests by default unless all tests are requested
    if not args.all:
        cmd += " -m 'not slow'"

    # Run the command
    print(f"Running command: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    main()
