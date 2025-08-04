"""
Tests for output formatting and structured logging systems.
"""
import unittest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch
from output_formatter import (
    RealTimeOutputManager, OutputFormat, ConsoleFormatter, 
    JSONFormatter, CSVFormatter
)
from structured_logger import (
    StructuredLogger, PerformanceTracker, LogEventType
)


class TestConsoleFormatter(unittest.TestCase):
    """Test cases for console output formatter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ConsoleFormatter(use_colors=False)  # Disable colors for testing
    
    def test_detection_formatting(self):
        """Test detection event formatting."""
        timestamp = "12:34:56.789"
        confidence = 0.85
        method = "AmplitudeBasedDetector"
        processing_time = 0.015
        
        result = self.formatter.format_detection(timestamp, confidence, method, processing_time)
        
        self.assertIn("[DETECTION]", result)
        self.assertIn(timestamp, result)
        self.assertIn("0.850", result)
        self.assertIn(method, result)
        self.assertIn("15.0ms", result)
    
    def test_localization_formatting(self):
        """Test localization result formatting."""
        timestamp = "12:34:56.789"
        x, y, z = 1.23, 4.56, 0.0
        confidence = 0.75
        error = 0.5
        processing_time = 0.025
        channels_used = 6
        
        result = self.formatter.format_localization(
            timestamp, x, y, z, confidence, error, processing_time, channels_used
        )
        
        self.assertIn("[LOCATION]", result)
        self.assertIn(timestamp, result)
        self.assertIn("1.23", result)
        self.assertIn("4.56", result)
        self.assertIn("0.750", result)
        self.assertIn("0.50m", result)
        self.assertIn("25.0ms", result)
        self.assertIn("6", result)
    
    def test_system_status_formatting(self):
        """Test system status formatting."""
        timestamp = "12:34:56.789"
        status = {
            'pipeline_state': 'running',
            'metrics': {
                'total_detections': 42,
                'localization_accuracy': 0.85,
                'average_processing_time': 0.03
            }
        }
        
        result = self.formatter.format_system_status(timestamp, status)
        
        self.assertIn("[SYSTEM]", result)
        self.assertIn("RUNNING", result)
        self.assertIn("42", result)
        self.assertIn("85.0%", result)
        self.assertIn("30.0ms", result)
    
    def test_error_formatting(self):
        """Test error formatting."""
        timestamp = "12:34:56.789"
        error = "Connection failed"
        context = "audio_capture"
        
        result = self.formatter.format_error(timestamp, error, context)
        
        self.assertIn("[ERROR]", result)
        self.assertIn(timestamp, result)
        self.assertIn(error, result)
        self.assertIn(context, result)


class TestJSONFormatter(unittest.TestCase):
    """Test cases for JSON output formatter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = JSONFormatter()
    
    def test_detection_json_formatting(self):
        """Test detection JSON formatting."""
        timestamp = "2023-01-01T12:34:56.789"
        confidence = 0.85
        method = "AmplitudeBasedDetector"
        processing_time = 0.015
        
        result = self.formatter.format_detection(timestamp, confidence, method, processing_time)
        data = json.loads(result)
        
        self.assertEqual(data['event_type'], 'detection')
        self.assertEqual(data['timestamp'], timestamp)
        self.assertEqual(data['confidence'], confidence)
        self.assertEqual(data['detection_method'], method)
        self.assertEqual(data['processing_time_ms'], 15.0)
    
    def test_localization_json_formatting(self):
        """Test localization JSON formatting."""
        timestamp = "2023-01-01T12:34:56.789"
        x, y, z = 1.23, 4.56, 0.0
        confidence = 0.75
        error = 0.5
        processing_time = 0.025
        channels_used = 6
        
        result = self.formatter.format_localization(
            timestamp, x, y, z, confidence, error, processing_time, channels_used
        )
        data = json.loads(result)
        
        self.assertEqual(data['event_type'], 'localization')
        self.assertEqual(data['position']['x'], x)
        self.assertEqual(data['position']['y'], y)
        self.assertEqual(data['position']['z'], z)
        self.assertEqual(data['confidence'], confidence)
        self.assertEqual(data['error_meters'], error)
        self.assertEqual(data['channels_used'], channels_used)


class TestRealTimeOutputManager(unittest.TestCase):
    """Test cases for real-time output manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        self.temp_file.close()
        
        self.output_manager = RealTimeOutputManager(
            output_format=OutputFormat.JSON,
            enable_file_output=True,
            output_file=self.temp_file.name
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.output_manager.close()
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_detection_output(self):
        """Test detection output."""
        confidence = 0.85
        method = "TestDetector"
        processing_time = 0.015
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            self.output_manager.output_detection(confidence, method, processing_time)
            
            # Verify print was called
            mock_print.assert_called_once()
            
            # Verify JSON format
            output = mock_print.call_args[0][0]
            data = json.loads(output)
            self.assertEqual(data['event_type'], 'detection')
            self.assertEqual(data['confidence'], confidence)
    
    def test_localization_output(self):
        """Test localization output."""
        x, y, z = 1.0, 2.0, 0.0
        confidence = 0.75
        error = 0.5
        processing_time = 0.025
        channels_used = 6
        
        with patch('builtins.print') as mock_print:
            self.output_manager.output_localization(
                x, y, z, confidence, error, processing_time, channels_used
            )
            
            mock_print.assert_called_once()
            output = mock_print.call_args[0][0]
            data = json.loads(output)
            self.assertEqual(data['event_type'], 'localization')
            self.assertEqual(data['position']['x'], x)
    
    def test_file_output(self):
        """Test file output functionality."""
        confidence = 0.85
        method = "TestDetector"
        processing_time = 0.015
        
        self.output_manager.output_detection(confidence, method, processing_time)
        
        # Check file was written
        with open(self.temp_file.name, 'r') as f:
            content = f.read().strip()
            data = json.loads(content)
            self.assertEqual(data['event_type'], 'detection')
    
    def test_output_statistics(self):
        """Test output statistics."""
        # Generate some outputs
        for i in range(5):
            self.output_manager.output_detection(0.8, "TestDetector", 0.01)
        
        stats = self.output_manager.get_output_stats()
        
        self.assertEqual(stats['total_outputs'], 5)
        self.assertEqual(stats['output_format'], 'json')
        self.assertTrue(stats['file_output_enabled'])


class TestStructuredLogger(unittest.TestCase):
    """Test cases for structured logger."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = StructuredLogger(
            name="test_logger",
            log_dir=self.temp_dir,
            enable_console=False,  # Disable console for testing
            enable_file=True
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.logger.close()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detection_logging(self):
        """Test detection event logging."""
        confidence = 0.85
        method = "TestDetector"
        processing_time = 0.015
        channels = [1, 2, 3, 4]
        metadata = {"test": "data"}
        
        self.logger.log_detection(confidence, method, processing_time, channels, metadata)
        
        # Verify log was written
        self.assertEqual(self.logger.log_count, 2)  # system_start + detection
    
    def test_localization_logging(self):
        """Test localization event logging."""
        x, y, z = 1.0, 2.0, 0.0
        confidence = 0.75
        error = 0.5
        processing_time = 0.025
        channels_used = [1, 2, 3, 4]
        method = "cross_correlation"
        metadata = {"test": "data"}
        
        self.logger.log_localization(
            x, y, z, confidence, error, processing_time, 
            channels_used, method, metadata
        )
        
        self.assertEqual(self.logger.log_count, 2)  # system_start + localization
    
    def test_error_logging(self):
        """Test error logging."""
        error = "Test error"
        context = "test_context"
        component = "test_component"
        exception = ValueError("Test exception")
        
        self.logger.log_error(error, context, component, exception)
        
        self.assertEqual(self.logger.log_count, 2)  # system_start + error
    
    def test_performance_logging(self):
        """Test performance logging."""
        latency_ms = 25.0
        accuracy = 0.85
        throughput = 10.0
        resource_usage = {"cpu": 50.0, "memory": 100.0}
        component_times = {"detection": 10.0, "localization": 15.0}
        
        self.logger.log_performance(
            latency_ms, accuracy, throughput, resource_usage, component_times
        )
        
        self.assertEqual(self.logger.log_count, 2)  # system_start + performance
    
    def test_log_statistics(self):
        """Test log statistics."""
        # Generate some log entries
        for i in range(5):
            self.logger.log_detection(0.8, "TestDetector", 0.01, [1, 2], {})
        
        stats = self.logger.get_log_statistics()
        
        self.assertEqual(stats['total_logs'], 6)  # system_start + 5 detections
        self.assertGreater(stats['uptime_seconds'], 0)
        self.assertGreater(stats['logs_per_second'], 0)


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for performance tracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = StructuredLogger(
            name="test_logger",
            log_dir=self.temp_dir,
            enable_console=False
        )
        self.tracker = PerformanceTracker(self.logger)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.logger.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_timing_operations(self):
        """Test timing operations."""
        operation = "test_operation"
        
        self.tracker.start_timing(operation)
        time.sleep(0.01)  # 10ms
        duration = self.tracker.end_timing(operation)
        
        self.assertGreater(duration, 0.005)  # At least 5ms
        self.assertLess(duration, 0.05)     # Less than 50ms
    
    def test_metric_recording(self):
        """Test metric recording."""
        self.tracker.record_metric("test_metric", 42.0)
        
        self.assertEqual(self.tracker.metrics["test_metric"], 42.0)
    
    @patch('psutil.Process')
    def test_performance_summary(self, mock_process):
        """Test performance summary logging."""
        # Mock process info
        mock_proc = Mock()
        mock_proc.cpu_percent.return_value = 50.0
        mock_proc.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_proc.num_threads.return_value = 4
        mock_process.return_value = mock_proc
        
        # Record some component times
        self.tracker.component_times = {
            "detection": 0.01,
            "localization": 0.02
        }
        
        self.tracker.log_performance_summary(accuracy=0.85, throughput=10.0)
        
        # Verify performance was logged
        self.assertGreater(self.logger.log_count, 1)


def create_test_suite():
    """Create test suite for output and logging systems."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestConsoleFormatter))
    suite.addTest(unittest.makeSuite(TestJSONFormatter))
    suite.addTest(unittest.makeSuite(TestRealTimeOutputManager))
    suite.addTest(unittest.makeSuite(TestStructuredLogger))
    suite.addTest(unittest.makeSuite(TestPerformanceTracker))
    
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