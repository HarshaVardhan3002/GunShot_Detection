"""
Tests for error handling and recovery system.
"""
import unittest
import time
import threading
from unittest.mock import Mock, patch
from error_handler import (
    ErrorHandlingSystem, ErrorSeverity, ErrorCategory, RecoveryAction,
    ErrorEvent, ComponentHealth, MicrophoneFailureHandler,
    ProcessingTimeoutHandler, ResourceExhaustionHandler,
    handle_component_error
)


class TestErrorEvent(unittest.TestCase):
    """Test cases for ErrorEvent data structure."""
    
    def test_error_event_creation(self):
        """Test error event creation."""
        error = ValueError("Test error")
        event = ErrorEvent(
            timestamp=time.time(),
            error_type="ValueError",
            error_message="Test error",
            component="test_component",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SOFTWARE,
            context={"test": "data"},
            exception=error
        )
        
        self.assertEqual(event.error_type, "ValueError")
        self.assertEqual(event.error_message, "Test error")
        self.assertEqual(event.component, "test_component")
        self.assertEqual(event.severity, ErrorSeverity.MEDIUM)
        self.assertEqual(event.category, ErrorCategory.SOFTWARE)
        self.assertFalse(event.recovery_attempted)
        self.assertFalse(event.recovery_successful)


class TestComponentHealth(unittest.TestCase):
    """Test cases for ComponentHealth tracking."""
    
    def test_component_health_initialization(self):
        """Test component health initialization."""
        health = ComponentHealth("test_component")
        
        self.assertEqual(health.component_name, "test_component")
        self.assertTrue(health.is_healthy)
        self.assertEqual(health.error_count, 0)
        self.assertEqual(health.consecutive_errors, 0)
        self.assertFalse(health.degraded_mode)
        self.assertEqual(health.health_score, 1.0)


class TestMicrophoneFailureHandler(unittest.TestCase):
    """Test cases for microphone failure handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = MicrophoneFailureHandler(min_microphones=4)
    
    def test_can_handle_microphone_error(self):
        """Test microphone error detection."""
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type="IOError",
            error_message="Microphone device not found",
            component="audio_capture",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.HARDWARE,
            context={}
        )
        
        self.assertTrue(self.handler.can_handle(error_event))
    
    def test_can_handle_non_microphone_error(self):
        """Test non-microphone error rejection."""
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type="ValueError",
            error_message="Invalid parameter",
            component="processor",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SOFTWARE,
            context={}
        )
        
        self.assertFalse(self.handler.can_handle(error_event))
    
    def test_handle_microphone_failure_success(self):
        """Test successful microphone failure handling."""
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type="IOError",
            error_message="Microphone 3 failed",
            component="audio_capture",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.HARDWARE,
            context={
                'microphone_id': 3,
                'total_microphones': 8
            }
        )
        
        result = self.handler.handle_error(error_event)
        
        self.assertTrue(result)
        self.assertIn(3, self.handler.failed_microphones)
    
    def test_handle_too_many_failures(self):
        """Test handling when too many microphones fail."""
        # Fail 5 microphones (leaving only 3, below minimum of 4)
        for mic_id in range(5):
            error_event = ErrorEvent(
                timestamp=time.time(),
                error_type="IOError",
                error_message=f"Microphone {mic_id} failed",
                component="audio_capture",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.HARDWARE,
                context={
                    'microphone_id': mic_id,
                    'total_microphones': 8
                }
            )
            
            if mic_id < 4:
                self.assertTrue(self.handler.handle_error(error_event))
            else:
                self.assertFalse(self.handler.handle_error(error_event))


class TestProcessingTimeoutHandler(unittest.TestCase):
    """Test cases for processing timeout handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ProcessingTimeoutHandler(max_retries=3, retry_delay=0.01)
    
    def test_can_handle_timeout_error(self):
        """Test timeout error detection."""
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type="TimeoutError",
            error_message="Processing timeout",
            component="processor",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
            context={}
        )
        
        self.assertTrue(self.handler.can_handle(error_event))
    
    def test_handle_timeout_with_retries(self):
        """Test timeout handling with retries."""
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type="TimeoutError",
            error_message="Processing timeout",
            component="processor",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
            context={}
        )
        
        # First few retries should succeed
        for i in range(3):
            result = self.handler.handle_error(error_event)
            self.assertTrue(result)
        
        # After max retries, should fail
        result = self.handler.handle_error(error_event)
        self.assertFalse(result)


class TestResourceExhaustionHandler(unittest.TestCase):
    """Test cases for resource exhaustion handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ResourceExhaustionHandler()
    
    def test_can_handle_memory_error(self):
        """Test memory error detection."""
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type="MemoryError",
            error_message="Out of memory",
            component="processor",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.RESOURCE,
            context={}
        )
        
        self.assertTrue(self.handler.can_handle(error_event))
    
    @patch('psutil.Process')
    def test_handle_memory_exhaustion(self, mock_process):
        """Test memory exhaustion handling."""
        # Mock process info
        mock_proc = Mock()
        mock_proc.memory_percent.return_value = 95.0
        mock_proc.cpu_percent.return_value = 80.0
        mock_process.return_value = mock_proc
        
        error_event = ErrorEvent(
            timestamp=time.time(),
            error_type="MemoryError",
            error_message="Out of memory",
            component="processor",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.RESOURCE,
            context={}
        )
        
        result = self.handler.handle_error(error_event)
        self.assertTrue(result)


class TestErrorHandlingSystem(unittest.TestCase):
    """Test cases for main error handling system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandlingSystem()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.error_handler.shutdown()
    
    def test_component_registration(self):
        """Test component registration."""
        component_name = "test_component"
        self.error_handler.register_component(component_name)
        
        self.assertIn(component_name, self.error_handler.component_health)
        health = self.error_handler.component_health[component_name]
        self.assertEqual(health.component_name, component_name)
        self.assertTrue(health.is_healthy)
    
    def test_error_reporting_success(self):
        """Test successful error reporting."""
        component = "test_component"
        error = ValueError("Test error")
        
        self.error_handler.register_component(component)
        
        result = self.error_handler.report_error(
            component=component,
            error=error,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SOFTWARE
        )
        
        # Should succeed for medium severity software error
        self.assertTrue(result)
        
        # Check error was recorded
        self.assertEqual(len(self.error_handler.error_history), 1)
        
        # Check component health was updated
        health = self.error_handler.component_health[component]
        self.assertEqual(health.error_count, 1)
    
    def test_error_reporting_with_recovery(self):
        """Test error reporting with recovery attempt."""
        component = "audio_capture"
        error = IOError("Microphone failed")
        context = {'microphone_id': 1, 'total_microphones': 8}
        
        self.error_handler.register_component(component)
        
        result = self.error_handler.report_error(
            component=component,
            error=error,
            context=context,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.HARDWARE
        )
        
        # Should succeed with microphone failure handler
        self.assertTrue(result)
        
        # Check recovery was attempted
        error_event = self.error_handler.error_history[0]
        self.assertTrue(error_event.recovery_attempted)
        self.assertTrue(error_event.recovery_successful)
    
    def test_system_health_tracking(self):
        """Test system health tracking."""
        # Register components
        components = ["comp1", "comp2", "comp3"]
        for comp in components:
            self.error_handler.register_component(comp)
        
        # Report some errors
        self.error_handler.report_error("comp1", ValueError("Error 1"))
        self.error_handler.report_error("comp2", IOError("Error 2"))
        
        # Get health status
        health = self.error_handler.get_system_health()
        
        self.assertIn('overall_health_score', health)
        self.assertIn('component_health', health)
        self.assertIn('total_errors', health)
        
        # Health score should be less than 1.0 due to errors
        self.assertLess(health['overall_health_score'], 1.0)
        
        # Should have component health for all registered components
        self.assertEqual(len(health['component_health']), 3)
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        component = "test_component"
        self.error_handler.register_component(component)
        
        # Report various types of errors
        errors = [
            (ValueError("Error 1"), ErrorCategory.SOFTWARE),
            (IOError("Error 2"), ErrorCategory.HARDWARE),
            (TimeoutError("Error 3"), ErrorCategory.TIMEOUT),
            (MemoryError("Error 4"), ErrorCategory.RESOURCE)
        ]
        
        for error, category in errors:
            self.error_handler.report_error(
                component=component,
                error=error,
                category=category
            )
        
        stats = self.error_handler.get_error_statistics()
        
        self.assertEqual(stats['total_errors'], 4)
        self.assertIn('errors_by_category', stats)
        self.assertIn('errors_by_component', stats)
        
        # Should have errors in different categories
        self.assertGreater(len(stats['errors_by_category']), 1)
    
    def test_health_monitoring_start_stop(self):
        """Test health monitoring thread management."""
        # Start monitoring
        self.error_handler.start_health_monitoring()
        self.assertTrue(self.error_handler.monitoring_active)
        self.assertIsNotNone(self.error_handler.health_monitor_thread)
        
        # Stop monitoring
        self.error_handler.stop_health_monitoring()
        self.assertFalse(self.error_handler.monitoring_active)
    
    def test_error_history_reset(self):
        """Test error history reset."""
        component = "test_component"
        self.error_handler.register_component(component)
        
        # Report some errors
        for i in range(5):
            self.error_handler.report_error(
                component=component,
                error=ValueError(f"Error {i}")
            )
        
        # Verify errors were recorded
        self.assertEqual(len(self.error_handler.error_history), 5)
        
        # Reset history
        self.error_handler.reset_error_history()
        
        # Verify reset
        self.assertEqual(len(self.error_handler.error_history), 0)
        self.assertEqual(self.error_handler.critical_errors, 0)
        self.assertFalse(self.error_handler.system_degraded)
        
        # Component health should be reset
        health = self.error_handler.component_health[component]
        self.assertEqual(health.error_count, 0)
        self.assertTrue(health.is_healthy)
        self.assertEqual(health.health_score, 1.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_handle_component_error(self):
        """Test handle_component_error convenience function."""
        error_handler = ErrorHandlingSystem()
        component = "test_component"
        error = ValueError("Test error")
        context = {"test": "data"}
        
        error_handler.register_component(component)
        
        result = handle_component_error(error_handler, component, error, context)
        
        # Should succeed for ValueError (medium severity)
        self.assertTrue(result)
        
        # Check error was recorded
        self.assertEqual(len(error_handler.error_history), 1)
        
        error_handler.shutdown()
    
    def test_error_severity_classification(self):
        """Test automatic error severity classification."""
        error_handler = ErrorHandlingSystem()
        component = "test_component"
        error_handler.register_component(component)
        
        # Test different error types
        test_cases = [
            (MemoryError("Out of memory"), ErrorSeverity.CRITICAL),
            (TimeoutError("Timeout"), ErrorSeverity.MEDIUM),
            (IOError("IO error"), ErrorSeverity.HIGH),
            (ValueError("Value error"), ErrorSeverity.MEDIUM),
            (RuntimeError("Runtime error"), ErrorSeverity.MEDIUM)
        ]
        
        for error, expected_severity in test_cases:
            # Clear history for clean test
            error_handler.reset_error_history()
            
            handle_component_error(error_handler, component, error)
            
            # Check that error was classified correctly
            if error_handler.error_history:
                recorded_error = error_handler.error_history[0]
                self.assertEqual(recorded_error.severity, expected_severity)
        
        error_handler.shutdown()


def create_test_suite():
    """Create test suite for error handling system."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestErrorEvent))
    suite.addTest(unittest.makeSuite(TestComponentHealth))
    suite.addTest(unittest.makeSuite(TestMicrophoneFailureHandler))
    suite.addTest(unittest.makeSuite(TestProcessingTimeoutHandler))
    suite.addTest(unittest.makeSuite(TestResourceExhaustionHandler))
    suite.addTest(unittest.makeSuite(TestErrorHandlingSystem))
    suite.addTest(unittest.makeSuite(TestConvenienceFunctions))
    
    return suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")