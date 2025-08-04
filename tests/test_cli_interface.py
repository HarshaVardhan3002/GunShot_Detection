"""
Tests for command-line interface.
"""
import unittest
import tempfile
import os
import json
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from cli_interface import GunshotLocalizerCLI, CLIConfig, OutputFormat


class TestCLIArgumentParsing(unittest.TestCase):
    """Test cases for CLI argument parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = GunshotLocalizerCLI()
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        config_data = {
            "microphones": [
                {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0},
                {"id": 3, "x": 0.0, "y": 1.0, "z": 0.0},
                {"id": 4, "x": 1.0, "y": 1.0, "z": 0.0},
                {"id": 5, "x": 0.5, "y": 0.5, "z": 1.0},
                {"id": 6, "x": 1.5, "y": 0.5, "z": 0.0},
                {"id": 7, "x": 0.5, "y": 1.5, "z": 0.0},
                {"id": 8, "x": 1.5, "y": 1.5, "z": 0.0}
            ],
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -20.0,
                "buffer_duration": 1.0,
                "min_confidence": 0.6
            }
        }
        json.dump(config_data, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_basic_argument_parsing(self):
        """Test basic argument parsing."""
        args = ['--config', self.temp_config.name]
        config = self.cli.parse_arguments(args)
        
        self.assertIsInstance(config, CLIConfig)
        self.assertEqual(config.config_file, self.temp_config.name)
        self.assertEqual(config.output_format, OutputFormat.CONSOLE)
        self.assertEqual(config.log_level, 'INFO')
        self.assertFalse(config.verbose)
        self.assertFalse(config.quiet)
    
    def test_output_format_parsing(self):
        """Test output format parsing."""
        test_cases = [
            ('console', OutputFormat.CONSOLE),
            ('json', OutputFormat.JSON),
            ('csv', OutputFormat.CSV)
        ]
        
        for format_str, expected_format in test_cases:
            args = ['--config', self.temp_config.name, '--output-format', format_str]
            config = self.cli.parse_arguments(args)
            self.assertEqual(config.output_format, expected_format)
    
    def test_verbose_and_quiet_flags(self):
        """Test verbose and quiet flags."""
        # Test verbose
        args = ['--config', self.temp_config.name, '--verbose']
        config = self.cli.parse_arguments(args)
        self.assertTrue(config.verbose)
        self.assertFalse(config.quiet)
        
        # Test quiet
        args = ['--config', self.temp_config.name, '--quiet']
        config = self.cli.parse_arguments(args)
        self.assertFalse(config.verbose)
        self.assertTrue(config.quiet)
    
    def test_runtime_parameter_overrides(self):
        """Test runtime parameter overrides."""
        args = [
            '--config', self.temp_config.name,
            '--sample-rate', '44100',
            '--detection-threshold', '0.8',
            '--buffer-duration', '0.5',
            '--sound-speed', '340.0',
            '--min-confidence', '0.7'
        ]
        
        config = self.cli.parse_arguments(args)
        
        self.assertEqual(config.sample_rate, 44100)
        self.assertEqual(config.detection_threshold, 0.8)
        self.assertEqual(config.buffer_duration, 0.5)
        self.assertEqual(config.sound_speed, 340.0)
        self.assertEqual(config.min_confidence, 0.7)
    
    def test_operation_modes(self):
        """Test operation mode flags."""
        # Test test mode
        args = ['--config', self.temp_config.name, '--test-mode']
        config = self.cli.parse_arguments(args)
        self.assertTrue(config.test_mode)
        self.assertFalse(config.calibration_mode)
        self.assertFalse(config.benchmark_mode)
        
        # Test calibration mode
        args = ['--config', self.temp_config.name, '--calibration-mode']
        config = self.cli.parse_arguments(args)
        self.assertFalse(config.test_mode)
        self.assertTrue(config.calibration_mode)
        self.assertFalse(config.benchmark_mode)
        
        # Test benchmark mode
        args = ['--config', self.temp_config.name, '--benchmark-mode']
        config = self.cli.parse_arguments(args)
        self.assertFalse(config.test_mode)
        self.assertFalse(config.calibration_mode)
        self.assertTrue(config.benchmark_mode)
    
    def test_advanced_options(self):
        """Test advanced options."""
        args = [
            '--config', self.temp_config.name,
            '--enable-performance-monitoring',
            '--health-check-interval', '60.0',
            '--max-error-history', '500'
        ]
        
        config = self.cli.parse_arguments(args)
        
        self.assertTrue(config.enable_performance_monitoring)
        self.assertEqual(config.health_check_interval, 60.0)
        self.assertEqual(config.max_error_history, 500)
    
    def test_argument_validation_errors(self):
        """Test argument validation errors."""
        # Test conflicting verbosity options
        with self.assertRaises(SystemExit):
            args = ['--config', self.temp_config.name, '--verbose', '--quiet']
            self.cli.parse_arguments(args)
        
        # Test conflicting operation modes
        with self.assertRaises(SystemExit):
            args = ['--config', self.temp_config.name, '--test-mode', '--calibration-mode']
            self.cli.parse_arguments(args)
        
        # Test invalid threshold values
        with self.assertRaises(SystemExit):
            args = ['--config', self.temp_config.name, '--detection-threshold', '1.5']
            self.cli.parse_arguments(args)
        
        with self.assertRaises(SystemExit):
            args = ['--config', self.temp_config.name, '--min-confidence', '-0.1']
            self.cli.parse_arguments(args)
        
        # Test invalid positive values
        with self.assertRaises(SystemExit):
            args = ['--config', self.temp_config.name, '--sample-rate', '-1000']
            self.cli.parse_arguments(args)
    
    def test_missing_config_file(self):
        """Test missing configuration file error."""
        with self.assertRaises(SystemExit):
            args = ['--config', 'nonexistent_config.json']
            self.cli.parse_arguments(args)
    
    def test_daemon_mode_validation(self):
        """Test daemon mode validation."""
        # Daemon mode without PID file should fail
        with self.assertRaises(SystemExit):
            args = ['--config', self.temp_config.name, '--daemon']
            self.cli.parse_arguments(args)
        
        # Daemon mode with PID file should succeed
        args = ['--config', self.temp_config.name, '--daemon', '--pid-file', '/tmp/test.pid']
        config = self.cli.parse_arguments(args)
        self.assertTrue(config.daemon)
        self.assertEqual(config.pid_file, '/tmp/test.pid')


class TestCLISpecialCommands(unittest.TestCase):
    """Test cases for CLI special commands."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = GunshotLocalizerCLI()
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        config_data = {
            "microphones": [
                {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0},
                {"id": 3, "x": 0.0, "y": 1.0, "z": 0.0},
                {"id": 4, "x": 1.0, "y": 1.0, "z": 0.0},
                {"id": 5, "x": 0.5, "y": 0.5, "z": 1.0},
                {"id": 6, "x": 1.5, "y": 0.5, "z": 0.0},
                {"id": 7, "x": 0.5, "y": 1.5, "z": 0.0},
                {"id": 8, "x": 1.5, "y": 1.5, "z": 0.0}
            ],
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -20.0,
                "buffer_duration": 1.0,
                "min_confidence": 0.6
            }
        }
        json.dump(config_data, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    @patch('sys.exit')
    @patch('builtins.print')
    def test_check_config_command(self, mock_print, mock_exit):
        """Test --check-config command."""
        args = ['--config', self.temp_config.name, '--check-config']
        
        try:
            self.cli.parse_arguments(args)
        except SystemExit:
            pass
        
        # Should have called print and exit
        mock_print.assert_called()
        mock_exit.assert_called_with(0)  # Success exit
    
    @patch('sys.exit')
    @patch('builtins.print')
    @patch('sounddevice.query_devices')
    def test_list_devices_command(self, mock_query_devices, mock_print, mock_exit):
        """Test --list-devices command."""
        # Mock device list
        mock_devices = [
            {
                'name': 'Test Microphone',
                'max_inputs': 2,
                'max_outputs': 0,
                'default_samplerate': 48000.0
            },
            {
                'name': 'Test Speaker',
                'max_inputs': 0,
                'max_outputs': 2,
                'default_samplerate': 48000.0
            }
        ]
        mock_query_devices.return_value = mock_devices
        
        args = ['--config', self.temp_config.name, '--list-devices']
        
        try:
            self.cli.parse_arguments(args)
        except SystemExit:
            pass
        
        # Should have called print and exit
        mock_print.assert_called()
        mock_exit.assert_called_with(0)
    
    @patch('sys.exit')
    @patch('builtins.print')
    @patch('platform.platform')
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_system_info_command(self, mock_memory, mock_cpu_count, 
                                mock_platform, mock_print, mock_exit):
        """Test --system-info command."""
        # Mock system info
        mock_platform.return_value = "Test Platform"
        mock_cpu_count.return_value = 4
        mock_memory.return_value = Mock(total=8*1024**3, available=4*1024**3)
        
        args = ['--config', self.temp_config.name, '--system-info']
        
        try:
            self.cli.parse_arguments(args)
        except SystemExit:
            pass
        
        # Should have called print and exit
        mock_print.assert_called()
        mock_exit.assert_called_with(0)


class TestCLILogging(unittest.TestCase):
    """Test cases for CLI logging setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = GunshotLocalizerCLI()
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump({"microphones": [], "system": {}}, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    def test_logging_setup_verbose(self):
        """Test logging setup with verbose flag."""
        args = ['--config', self.temp_config.name, '--verbose']
        self.cli.parse_arguments(args)
        
        with patch('logging.basicConfig') as mock_basic_config:
            self.cli.setup_logging()
            
            # Should configure DEBUG level for verbose
            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            self.assertEqual(call_args[1]['level'], 10)  # DEBUG level
    
    def test_logging_setup_quiet(self):
        """Test logging setup with quiet flag."""
        args = ['--config', self.temp_config.name, '--quiet']
        self.cli.parse_arguments(args)
        
        with patch('logging.basicConfig') as mock_basic_config:
            self.cli.setup_logging()
            
            # Should configure WARNING level for quiet
            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            self.assertEqual(call_args[1]['level'], 30)  # WARNING level
    
    def test_logging_setup_custom_level(self):
        """Test logging setup with custom log level."""
        args = ['--config', self.temp_config.name, '--log-level', 'ERROR']
        self.cli.parse_arguments(args)
        
        with patch('logging.basicConfig') as mock_basic_config:
            self.cli.setup_logging()
            
            # Should configure ERROR level
            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            self.assertEqual(call_args[1]['level'], 40)  # ERROR level


class TestCLIOperationModes(unittest.TestCase):
    """Test cases for CLI operation modes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cli = GunshotLocalizerCLI()
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        config_data = {
            "microphones": [
                {"id": i, "x": i*0.5, "y": i*0.5, "z": 0.0} 
                for i in range(1, 9)
            ],
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -20.0,
                "buffer_duration": 1.0,
                "min_confidence": 0.6
            }
        }
        json.dump(config_data, self.temp_config)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
    
    @patch('cli_interface.GunshotLocalizationPipeline')
    def test_test_mode(self, mock_pipeline_class):
        """Test test mode execution."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.setup.return_value = True
        mock_pipeline_class.return_value = mock_pipeline
        
        args = ['--config', self.temp_config.name, '--test-mode']
        self.cli.parse_arguments(args)
        self.cli.setup_logging()
        self.cli.create_pipeline()
        
        result = self.cli.run_test_mode()
        
        self.assertEqual(result, 0)  # Success
        mock_pipeline.setup.assert_called_once()
        mock_pipeline.shutdown.assert_called_once()
    
    def test_calibration_mode(self):
        """Test calibration mode execution."""
        args = ['--config', self.temp_config.name, '--calibration-mode']
        self.cli.parse_arguments(args)
        self.cli.setup_logging()
        
        result = self.cli.run_calibration_mode()
        
        self.assertEqual(result, 0)  # Success (not implemented yet)
    
    @patch('cli_interface.GunshotLocalizationPipeline')
    def test_benchmark_mode(self, mock_pipeline_class):
        """Test benchmark mode execution."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.setup.return_value = True
        mock_pipeline_class.return_value = mock_pipeline
        
        args = ['--config', self.temp_config.name, '--benchmark-mode']
        self.cli.parse_arguments(args)
        self.cli.setup_logging()
        self.cli.create_pipeline()
        
        result = self.cli.run_benchmark_mode()
        
        self.assertEqual(result, 0)  # Success
        mock_pipeline.setup.assert_called_once()
        mock_pipeline.shutdown.assert_called_once()


def create_test_suite():
    """Create test suite for CLI interface."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestCLIArgumentParsing))
    suite.addTest(unittest.makeSuite(TestCLISpecialCommands))
    suite.addTest(unittest.makeSuite(TestCLILogging))
    suite.addTest(unittest.makeSuite(TestCLIOperationModes))
    
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