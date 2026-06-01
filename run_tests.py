#!/usr/bin/env python3
"""
Orchestrates and runs the entire AI Face Analytics test suite.
"""
import unittest
import sys
import os

def run_suite():
    print("="*60)
    print("Launching AI Face Analytics Platform - Automated Test Suite")
    print("="*60)
    
    # Set search path to project root
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Auto-discover all test modules under the 'tests' directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        print("\n[FAILED] Test suite encountered verification failures.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All 100% of the platform unit tests passed successfully!")
        sys.exit(0)

if __name__ == '__main__':
    run_suite()
