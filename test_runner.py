"""
Comprehensive test runner for gunshot localization system.
"""
import unittest
import sys
import os
import time
import importlib
from typing import List, Dict, Any
from pathlib import Path


class TestResult:
    """Container for test results."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.execution_time = 0.0
        self.module_results = {}
        self.failures = []
        self.errors = []


class ComprehensiveTestRunner:
    """Comprehensive test runner for all modules."""
    
    def __init__(self):
        self.test_modules = [
            'test_config_manager',
            'test_audio_capture',
            'test_gunshot_detector',
            'test_tdoa_calculation',
            'test_triangulation_solver',
            'test_solution_validation',
            'test_intensity_filter',
            'test_adaptive_channel_selector',
            'test_main_pipeline',
            'test_output_logging',
            'test_error_handling',
            'test_cli_interface',
            'test_diagnostics'
        ]
        
        self.results = TestResult()
    
    def discover_test_modules(self) -> List[str]:
        """Discover all test modules in the current directory."""
        test_files = []
        current_dir = Path('.')
        
        for file_path in current_dir.glob('test_*.py'):
            module_name = file_path.stem
            if module_name not in ['test_runner']:  # Exclude self
                test_files.append(module_name)
        
        return sorted(test_files)
    
    def run_module_tests(self, module_name: str) -> unittest.TestResult:
        """Run tests for a specific module."""
        try:
            # Import the test module
            test_module = importlib.import_module(module_name)
            
            # Create test suite
            if hasattr(test_module, 'create_test_suite'):
                suite = test_module.create_test_suite()
            else:
                # Fallback: load all tests from module
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromModule(test_module)
            
            # Run tests
            runner = unittest.TextTestRunner(
                verbosity=1,
                stream=open(os.devnull, 'w')  # Suppress output for now
            )
            
            result = runner.run(suite)
            return result
            
        except ImportError as e:
            print(f"⚠️  Could not import {module_name}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error running tests for {module_name}: {e}")
            return None
    
    def run_all_tests(self, verbose: bool = True) -> TestResult:
        """Run all tests and return comprehensive results."""
        print("🧪 Gunshot Localization System - Comprehensive Test Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Discover test modules
        discovered_modules = self.discover_test_modules()
        available_modules = [m for m in self.test_modules if m in discovered_modules]
        
        print(f"📋 Found {len(available_modules)} test modules")
        if verbose:
            for module in available_modules:
                print(f"   • {module}")
        print()
        
        # Run tests for each module
        for i, module_name in enumerate(available_modules, 1):
            print(f"🔍 [{i}/{len(available_modules)}] Testing {module_name}...")
            
            module_start = time.time()
            result = self.run_module_tests(module_name)
            module_time = time.time() - module_start
            
            if result:
                # Update overall results
                self.results.total_tests += result.testsRun
                self.results.passed_tests += result.testsRun - len(result.failures) - len(result.errors)
                self.results.failed_tests += len(result.failures)
                self.results.error_tests += len(result.errors)
                self.results.skipped_tests += len(getattr(result, 'skipped', []))
                
                # Store module results
                self.results.module_results[module_name] = {
                    'tests_run': result.testsRun,
                    'failures': len(result.failures),
                    'errors': len(result.errors),
                    'skipped': len(getattr(result, 'skipped', [])),
                    'execution_time': module_time,
                    'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
                }
                
                # Store failure details
                for test, traceback in result.failures:
                    self.results.failures.append({
                        'module': module_name,
                        'test': str(test),
                        'traceback': traceback
                    })
                
                for test, traceback in result.errors:
                    self.results.errors.append({
                        'module': module_name,
                        'test': str(test),
                        'traceback': traceback
                    })
                
                # Show module results
                success_rate = self.results.module_results[module_name]['success_rate']
                status_symbol = "✅" if success_rate == 1.0 else "⚠️" if success_rate > 0.8 else "❌"
                
                print(f"   {status_symbol} {result.testsRun} tests, "
                      f"{len(result.failures)} failures, "
                      f"{len(result.errors)} errors "
                      f"({success_rate:.1%} success) "
                      f"in {module_time:.2f}s")
            else:
                print(f"   ❌ Module could not be tested")
        
        self.results.execution_time = time.time() - start_time
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self) -> None:
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        # Overall statistics
        total = self.results.total_tests
        passed = self.results.passed_tests
        failed = self.results.failed_tests
        errors = self.results.error_tests
        
        success_rate = (passed / total) if total > 0 else 0
        
        print(f"Total Tests:     {total}")
        print(f"Passed:          {passed} ({passed/total:.1%})" if total > 0 else "Passed:          0")
        print(f"Failed:          {failed} ({failed/total:.1%})" if total > 0 else "Failed:          0")
        print(f"Errors:          {errors} ({errors/total:.1%})" if total > 0 else "Errors:          0")
        print(f"Success Rate:    {success_rate:.1%}")
        print(f"Execution Time:  {self.results.execution_time:.2f}s")
        
        # Overall status
        if success_rate == 1.0:
            print(f"\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        elif success_rate > 0.9:
            print(f"\n✅ Excellent! {success_rate:.1%} success rate - minor issues to address.")
        elif success_rate > 0.8:
            print(f"\n⚠️  Good progress! {success_rate:.1%} success rate - some issues need attention.")
        else:
            print(f"\n❌ Needs work! {success_rate:.1%} success rate - significant issues to resolve.")
        
        # Module breakdown
        if self.results.module_results:
            print(f"\n📋 MODULE BREAKDOWN:")
            print("-" * 40)
            
            for module, results in self.results.module_results.items():
                status_symbol = "✅" if results['success_rate'] == 1.0 else "⚠️" if results['success_rate'] > 0.8 else "❌"
                print(f"{status_symbol} {module:<25} {results['tests_run']:>3} tests "
                      f"({results['success_rate']:>5.1%}) {results['execution_time']:>6.2f}s")
        
        # Failure details
        if self.results.failures:
            print(f"\n❌ FAILURES ({len(self.results.failures)}):")
            print("-" * 40)
            for i, failure in enumerate(self.results.failures[:5], 1):  # Show first 5
                print(f"{i}. {failure['module']}: {failure['test']}")
            
            if len(self.results.failures) > 5:
                print(f"   ... and {len(self.results.failures) - 5} more failures")
        
        # Error details
        if self.results.errors:
            print(f"\n💥 ERRORS ({len(self.results.errors)}):")
            print("-" * 40)
            for i, error in enumerate(self.results.errors[:5], 1):  # Show first 5
                print(f"{i}. {error['module']}: {error['test']}")
            
            if len(self.results.errors) > 5:
                print(f"   ... and {len(self.results.errors) - 5} more errors")
        
        print("\n" + "=" * 60)
    
    def run_specific_modules(self, module_names: List[str], verbose: bool = True) -> TestResult:
        """Run tests for specific modules only."""
        print(f"🧪 Running tests for specific modules: {', '.join(module_names)}")
        print("=" * 60)
        
        start_time = time.time()
        
        for i, module_name in enumerate(module_names, 1):
            print(f"🔍 [{i}/{len(module_names)}] Testing {module_name}...")
            
            result = self.run_module_tests(module_name)
            
            if result:
                success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
                status_symbol = "✅" if success_rate == 1.0 else "⚠️" if success_rate > 0.8 else "❌"
                
                print(f"   {status_symbol} {result.testsRun} tests, "
                      f"{len(result.failures)} failures, "
                      f"{len(result.errors)} errors "
                      f"({success_rate:.1%} success)")
                
                # Update results
                self.results.total_tests += result.testsRun
                self.results.passed_tests += result.testsRun - len(result.failures) - len(result.errors)
                self.results.failed_tests += len(result.failures)
                self.results.error_tests += len(result.errors)
            else:
                print(f"   ❌ Module could not be tested")
        
        self.results.execution_time = time.time() - start_time
        self._print_summary()
        
        return self.results
    
    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate test coverage report."""
        coverage_data = {
            'timestamp': time.time(),
            'total_modules': len(self.test_modules),
            'tested_modules': len(self.results.module_results),
            'coverage_percentage': len(self.results.module_results) / len(self.test_modules) * 100,
            'module_coverage': {},
            'overall_health': 'excellent' if self.results.passed_tests / self.results.total_tests > 0.95 else 'good' if self.results.passed_tests / self.results.total_tests > 0.8 else 'needs_improvement'
        }
        
        for module, results in self.results.module_results.items():
            coverage_data['module_coverage'][module] = {
                'test_count': results['tests_run'],
                'success_rate': results['success_rate'],
                'execution_time': results['execution_time'],
                'status': 'passing' if results['success_rate'] == 1.0 else 'failing'
            }
        
        return coverage_data
    
    def export_results(self, format_type: str = 'json', filename: str = None) -> str:
        """Export test results to file."""
        if format_type.lower() == 'json':
            import json
            
            export_data = {
                'test_summary': {
                    'total_tests': self.results.total_tests,
                    'passed_tests': self.results.passed_tests,
                    'failed_tests': self.results.failed_tests,
                    'error_tests': self.results.error_tests,
                    'success_rate': self.results.passed_tests / self.results.total_tests if self.results.total_tests > 0 else 0,
                    'execution_time': self.results.execution_time
                },
                'module_results': self.results.module_results,
                'failures': self.results.failures,
                'errors': self.results.errors,
                'coverage_report': self.generate_coverage_report()
            }
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                return filename
            else:
                return json.dumps(export_data, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive test runner for gunshot localization system')
    parser.add_argument('--modules', '-m', nargs='+', help='Specific modules to test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--export', '-e', help='Export results to JSON file')
    parser.add_argument('--coverage', '-c', action='store_true', help='Generate coverage report')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    try:
        # Run tests
        if args.modules:
            results = runner.run_specific_modules(args.modules, verbose=args.verbose)
        else:
            results = runner.run_all_tests(verbose=args.verbose)
        
        # Export results if requested
        if args.export:
            runner.export_results('json', args.export)
            print(f"\n📄 Results exported to: {args.export}")
        
        # Generate coverage report if requested
        if args.coverage:
            coverage = runner.generate_coverage_report()
            print(f"\n📊 COVERAGE REPORT:")
            print(f"   Module Coverage: {coverage['coverage_percentage']:.1f}%")
            print(f"   Overall Health: {coverage['overall_health']}")
        
        # Exit with appropriate code
        if results.failed_tests > 0 or results.error_tests > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()