#!/usr/bin/env python3
"""
Test runner for integration and performance tests.
This script allows running specific test categories or all tests.
"""
import sys
import unittest
import argparse
from test_integration_performance import (
    TestEndToEndIntegration,
    TestLatencyValidation,
    TestAccuracyValidation,
    TestStressTesting
)


def run_specific_test_class(test_class, verbose=True):
    """Run a specific test class."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result


def run_all_tests(verbose=True):
    """Run all integration and performance tests."""
    from test_integration_performance import run_integration_performance_tests
    return run_integration_performance_tests()


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run gunshot localization integration tests')
    parser.add_argument('--test-type', choices=['all', 'integration', 'latency', 'accuracy', 'stress'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (skip stress tests)')
    
    args = parser.parse_args()
    
    print("Gunshot Localization System - Integration & Performance Tests")
    print("=" * 70)
    
    results = []
    
    if args.test_type == 'all' and not args.quick:
        print("Running all integration and performance tests...")
        result = run_all_tests(args.verbose)
        results.append(('All Tests', result))
    
    elif args.test_type == 'integration' or (args.test_type == 'all' and args.quick):
        print("Running end-to-end integration tests...")
        result = run_specific_test_class(TestEndToEndIntegration, args.verbose)
        results.append(('Integration Tests', result))
    
    elif args.test_type == 'latency':
        print("Running latency validation tests...")
        result = run_specific_test_class(TestLatencyValidation, args.verbose)
        results.append(('Latency Tests', result))
    
    elif args.test_type == 'accuracy':
        print("Running accuracy validation tests...")
        result = run_specific_test_class(TestAccuracyValidation, args.verbose)
        results.append(('Accuracy Tests', result))
    
    elif args.test_type == 'stress':
        print("Running stress tests...")
        result = run_specific_test_class(TestStressTesting, args.verbose)
        results.append(('Stress Tests', result))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_name, result in results:
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        status = "PASS" if (len(result.failures) == 0 and len(result.errors) == 0) else "FAIL"
        print(f"{test_name}: {result.testsRun} tests, {len(result.failures)} failures, "
              f"{len(result.errors)} errors - {status}")
    
    print("-" * 70)
    print(f"TOTAL: {total_tests} tests, {total_failures} failures, {total_errors} errors")
    
    overall_success = total_failures == 0 and total_errors == 0
    print(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == '__main__':
    main()