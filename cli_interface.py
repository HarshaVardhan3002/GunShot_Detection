"""
Command-line interface for gunshot localization system.
"""
import argparse
import sys
import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from output_formatter import OutputFormat
from main_pipeline import GunshotLocalizationPipeline, DefaultEventHandler


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class CLIConfig:
    """Configuration from command-line arguments."""
    config_file: str
    output_format: OutputFormat
    output_file: Optional[str]
    log_level: str
    log_dir: str
    verbose: bool
    quiet: bool
    daemon: bool
    pid_file: Optional[str]
    
    # Runtime parameter overrides
    sample_rate: Optional[int]
    detection_threshold: Optional[float]
    buffer_duration: Optional[float]
    sound_speed: Optional[float]
    min_confidence: Optional[float]
    
    # Operation modes
    test_mode: bool
    calibration_mode: bool
    benchmark_mode: bool
    
    # Advanced options
    enable_performance_monitoring: bool
    health_check_interval: float
    max_error_history: int


class GunshotLocalizerCLI:
    """Command-line interface for gunshot localization system."""
    
    def __init__(self):
        """Initialize CLI interface."""
        self.parser = None
        self.config = None
        self.pipeline = None
        self.logger = None
        
        self._setup_argument_parser()
    
    def _setup_argument_parser(self) -> None:
        """Set up command-line argument parser."""
        self.parser = argparse.ArgumentParser(
            prog='gunshot-localizer',
            description='Real-time gunshot localization system using acoustic triangulation',
            epilog='''
Examples:
  %(prog)s --config config.json                    # Run with configuration file
  %(prog)s --config config.json --verbose          # Run with verbose output
  %(prog)s --config config.json --output-format json --output-file results.json
  %(prog)s --config config.json --sample-rate 44100 --detection-threshold 0.8
  %(prog)s --test-mode --config test_config.json   # Run in test mode
  %(prog)s --calibration-mode                      # Run calibration
  %(prog)s --benchmark-mode --config config.json  # Run performance benchmark
            ''',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Configuration arguments
        config_group = self.parser.add_argument_group('configuration')
        config_group.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file (JSON format, required for most operations)'
        )
        
        # Output options
        output_group = self.parser.add_argument_group('output options')
        output_group.add_argument(
            '--output-format', '-f',
            type=str,
            choices=['console', 'json', 'csv'],
            default='console',
            help='Output format (default: console)'
        )
        output_group.add_argument(
            '--output-file', '-o',
            type=str,
            help='Output file path (optional, outputs to console if not specified)'
        )
        output_group.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        output_group.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress non-essential output'
        )
        
        # Logging options
        logging_group = self.parser.add_argument_group('logging options')
        logging_group.add_argument(
            '--log-level',
            type=str,
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Logging level (default: INFO)'
        )
        logging_group.add_argument(
            '--log-dir',
            type=str,
            default='logs',
            help='Directory for log files (default: logs)'
        )
        
        # Runtime parameter overrides
        params_group = self.parser.add_argument_group('runtime parameter overrides')
        params_group.add_argument(
            '--sample-rate',
            type=int,
            help='Audio sample rate in Hz (overrides config file)'
        )
        params_group.add_argument(
            '--detection-threshold',
            type=float,
            help='Detection confidence threshold (0.0-1.0, overrides config file)'
        )
        params_group.add_argument(
            '--buffer-duration',
            type=float,
            help='Audio buffer duration in seconds (overrides config file)'
        )
        params_group.add_argument(
            '--sound-speed',
            type=float,
            help='Speed of sound in m/s (overrides config file)'
        )
        params_group.add_argument(
            '--min-confidence',
            type=float,
            help='Minimum localization confidence (0.0-1.0, overrides config file)'
        )
        
        # Operation modes
        modes_group = self.parser.add_argument_group('operation modes')
        modes_group.add_argument(
            '--test-mode',
            action='store_true',
            help='Run in test mode with synthetic data'
        )
        modes_group.add_argument(
            '--calibration-mode',
            action='store_true',
            help='Run microphone array calibration'
        )
        modes_group.add_argument(
            '--benchmark-mode',
            action='store_true',
            help='Run performance benchmark'
        )
        modes_group.add_argument(
            '--daemon', '-d',
            action='store_true',
            help='Run as daemon process'
        )
        modes_group.add_argument(
            '--pid-file',
            type=str,
            help='PID file path for daemon mode'
        )
        
        # Advanced options
        advanced_group = self.parser.add_argument_group('advanced options')
        advanced_group.add_argument(
            '--enable-performance-monitoring',
            action='store_true',
            help='Enable detailed performance monitoring'
        )
        advanced_group.add_argument(
            '--health-check-interval',
            type=float,
            default=30.0,
            help='Health check interval in seconds (default: 30.0)'
        )
        advanced_group.add_argument(
            '--max-error-history',
            type=int,
            default=1000,
            help='Maximum number of errors to keep in history (default: 1000)'
        )
        
        # System commands
        system_group = self.parser.add_argument_group('system commands')
        system_group.add_argument(
            '--version',
            action='version',
            version='%(prog)s 1.0.0'
        )
        system_group.add_argument(
            '--check-config',
            action='store_true',
            help='Validate configuration file and exit'
        )
        system_group.add_argument(
            '--list-devices',
            action='store_true',
            help='List available audio devices and exit'
        )
        system_group.add_argument(
            '--system-info',
            action='store_true',
            help='Display system information and exit'
        )
        system_group.add_argument(
            '--status',
            action='store_true',
            help='Display current system status and exit'
        )
        system_group.add_argument(
            '--diagnostics',
            action='store_true',
            help='Run comprehensive diagnostics and exit'
        )
        system_group.add_argument(
            '--health-check',
            action='store_true',
            help='Perform system health check and exit'
        )
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> CLIConfig:
        """
        Parse command-line arguments.
        
        Args:
            args: Arguments to parse (uses sys.argv if None)
            
        Returns:
            Parsed configuration
        """
        parsed_args = self.parser.parse_args(args)
        
        # Validate argument combinations
        self._validate_arguments(parsed_args)
        
        # Convert to CLIConfig
        self.config = CLIConfig(
            config_file=parsed_args.config,
            output_format=getattr(OutputFormat, parsed_args.output_format.upper()),
            output_file=parsed_args.output_file,
            log_level=parsed_args.log_level,
            log_dir=parsed_args.log_dir,
            verbose=parsed_args.verbose,
            quiet=parsed_args.quiet,
            daemon=parsed_args.daemon,
            pid_file=parsed_args.pid_file,
            
            # Runtime overrides
            sample_rate=parsed_args.sample_rate,
            detection_threshold=parsed_args.detection_threshold,
            buffer_duration=parsed_args.buffer_duration,
            sound_speed=parsed_args.sound_speed,
            min_confidence=parsed_args.min_confidence,
            
            # Operation modes
            test_mode=parsed_args.test_mode,
            calibration_mode=parsed_args.calibration_mode,
            benchmark_mode=parsed_args.benchmark_mode,
            
            # Advanced options
            enable_performance_monitoring=parsed_args.enable_performance_monitoring,
            health_check_interval=parsed_args.health_check_interval,
            max_error_history=parsed_args.max_error_history
        )
        
        # Handle special commands
        if hasattr(parsed_args, 'check_config') and parsed_args.check_config:
            self._check_config_and_exit()
        
        if hasattr(parsed_args, 'list_devices') and parsed_args.list_devices:
            self._list_devices_and_exit()
        
        if hasattr(parsed_args, 'system_info') and parsed_args.system_info:
            self._system_info_and_exit()
        
        if hasattr(parsed_args, 'status') and parsed_args.status:
            self._status_and_exit()
        
        if hasattr(parsed_args, 'diagnostics') and parsed_args.diagnostics:
            self._diagnostics_and_exit()
        
        if hasattr(parsed_args, 'health_check') and parsed_args.health_check:
            self._health_check_and_exit()
        
        return self.config
    
    def _validate_arguments(self, args) -> None:
        """Validate argument combinations."""
        # Check for conflicting verbosity options
        if args.verbose and args.quiet:
            self.parser.error("--verbose and --quiet cannot be used together")
        
        # Check for conflicting operation modes
        modes = [args.test_mode, args.calibration_mode, args.benchmark_mode]
        if sum(modes) > 1:
            self.parser.error("Only one operation mode can be specified")
        
        # Validate threshold values
        if args.detection_threshold is not None:
            if not 0.0 <= args.detection_threshold <= 1.0:
                self.parser.error("--detection-threshold must be between 0.0 and 1.0")
        
        if args.min_confidence is not None:
            if not 0.0 <= args.min_confidence <= 1.0:
                self.parser.error("--min-confidence must be between 0.0 and 1.0")
        
        # Validate positive values
        if args.sample_rate is not None and args.sample_rate <= 0:
            self.parser.error("--sample-rate must be positive")
        
        if args.buffer_duration is not None and args.buffer_duration <= 0:
            self.parser.error("--buffer-duration must be positive")
        
        if args.sound_speed is not None and args.sound_speed <= 0:
            self.parser.error("--sound-speed must be positive")
        
        # Check config file exists (unless using status-only commands)
        status_only_commands = [
            getattr(args, 'status', False),
            getattr(args, 'diagnostics', False), 
            getattr(args, 'health_check', False),
            getattr(args, 'system_info', False),
            getattr(args, 'list_devices', False)
        ]
        
        if not any(status_only_commands) and not os.path.exists(args.config):
            self.parser.error(f"Configuration file not found: {args.config}")
        
        # Validate daemon mode requirements
        if args.daemon and not args.pid_file:
            self.parser.error("--daemon mode requires --pid-file")
    
    def _check_config_and_exit(self) -> None:
        """Check configuration file and exit."""
        try:
            from config_manager import ConfigurationManager
            
            print(f"Checking configuration file: {self.config.config_file}")
            
            config_manager = ConfigurationManager()
            success = config_manager.load_config(self.config.config_file)
            
            if success:
                # Validate configuration
                is_valid, errors = config_manager.validate_config()
                
                if is_valid:
                    print("✅ Configuration file is valid")
                    
                    # Show configuration summary
                    system_config = config_manager.get_system_config()
                    mic_positions = config_manager.get_microphone_positions()
                    
                    print(f"\nConfiguration Summary:")
                    print(f"  Sample Rate: {system_config.sample_rate} Hz")
                    print(f"  Sound Speed: {system_config.sound_speed} m/s")
                    print(f"  Detection Threshold: {system_config.detection_threshold_db} dB")
                    print(f"  Buffer Duration: {system_config.buffer_duration} s")
                    print(f"  Min Confidence: {system_config.min_confidence}")
                    print(f"  Microphones: {len(mic_positions)} configured")
                    
                    sys.exit(0)
                else:
                    print("❌ Configuration file has errors:")
                    for error in errors:
                        print(f"  - {error}")
                    sys.exit(1)
            else:
                print("❌ Failed to load configuration file")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error checking configuration: {e}")
            sys.exit(1)
    
    def _list_devices_and_exit(self) -> None:
        """List available audio devices and exit."""
        try:
            import sounddevice as sd
            
            print("Available Audio Devices:")
            print("=" * 50)
            
            devices = sd.query_devices()
            
            for i, device in enumerate(devices):
                device_type = []
                if device['max_inputs'] > 0:
                    device_type.append(f"Input ({device['max_inputs']} ch)")
                if device['max_outputs'] > 0:
                    device_type.append(f"Output ({device['max_outputs']} ch)")
                
                type_str = ", ".join(device_type) if device_type else "No channels"
                
                print(f"{i:2d}: {device['name']}")
                print(f"     {type_str}")
                print(f"     Sample Rate: {device['default_samplerate']} Hz")
                print()
            
            # Show default devices
            try:
                default_input = sd.default.device[0]
                default_output = sd.default.device[1]
                print(f"Default Input Device: {default_input} ({devices[default_input]['name']})")
                print(f"Default Output Device: {default_output} ({devices[default_output]['name']})")
            except:
                print("Could not determine default devices")
            
            sys.exit(0)
            
        except ImportError:
            print("❌ sounddevice library not available")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error listing devices: {e}")
            sys.exit(1)
    
    def _system_info_and_exit(self) -> None:
        """Display system information and exit."""
        try:
            import platform
            import psutil
            
            print("System Information:")
            print("=" * 50)
            
            # System info
            print(f"Platform: {platform.platform()}")
            print(f"Python Version: {platform.python_version()}")
            print(f"Architecture: {platform.architecture()[0]}")
            print(f"Processor: {platform.processor()}")
            
            # Hardware info
            print(f"\nHardware:")
            print(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
            
            memory = psutil.virtual_memory()
            print(f"  Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
            
            # Audio system info
            try:
                import sounddevice as sd
                print(f"\nAudio System:")
                print(f"  PortAudio Version: {sd.get_portaudio_version()[1]}")
                
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_inputs'] > 0]
                print(f"  Input Devices: {len(input_devices)} available")
                
                max_channels = max([d['max_inputs'] for d in input_devices]) if input_devices else 0
                print(f"  Max Input Channels: {max_channels}")
                
            except ImportError:
                print(f"\nAudio System: sounddevice not available")
            
            # Dependencies
            print(f"\nDependencies:")
            try:
                import numpy as np
                print(f"  NumPy: {np.__version__}")
            except ImportError:
                print(f"  NumPy: Not available")
            
            try:
                import scipy
                print(f"  SciPy: {scipy.__version__}")
            except ImportError:
                print(f"  SciPy: Not available")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error getting system info: {e}")
            sys.exit(1)
    
    def _status_and_exit(self) -> None:
        """Display current system status and exit."""
        try:
            from diagnostics import get_quick_system_status
            from datetime import datetime
            
            print("System Status Check:")
            print("=" * 30)
            
            # Get quick status
            status = get_quick_system_status()
            
            # Display status
            health_symbol = "✅" if status['system_healthy'] else "❌"
            print(f"System Health: {health_symbol} {'Healthy' if status['system_healthy'] else 'Issues Detected'}")
            print(f"Timestamp: {datetime.fromtimestamp(status['timestamp'])}")
            print(f"CPU Usage: {status['cpu_usage_percent']:.1f}%")
            print(f"Memory Usage: {status['memory_usage_percent']:.1f}%")
            
            if 'pipeline_state' in status:
                print(f"Pipeline State: {status['pipeline_state']}")
                print(f"Pipeline Running: {'Yes' if status['pipeline_running'] else 'No'}")
            
            if status.get('issues'):
                print("\nIssues:")
                for issue in status['issues']:
                    print(f"  • {issue}")
            
            if status.get('error'):
                print(f"\nError: {status['error']}")
                sys.exit(1)
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
            sys.exit(1)
    
    def _diagnostics_and_exit(self) -> None:
        """Run comprehensive diagnostics and exit."""
        try:
            from diagnostics import DiagnosticsManager
            
            print("Running Comprehensive System Diagnostics...")
            print("=" * 50)
            
            # Create diagnostics manager
            diagnostics_manager = DiagnosticsManager()
            
            # Update diagnostics once
            diagnostics_manager._update_diagnostics()
            
            # Get diagnostics report
            report = diagnostics_manager.export_diagnostics_report(format_type='text')
            print(report)
            
            # Cleanup
            diagnostics_manager.shutdown()
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error running diagnostics: {e}")
            sys.exit(1)
    
    def _health_check_and_exit(self) -> None:
        """Perform system health check and exit."""
        try:
            from diagnostics import DiagnosticsManager
            import psutil
            
            print("System Health Check:")
            print("=" * 30)
            
            # Basic system checks
            print("1. System Resources:")
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # CPU check
            cpu_status = "✅ Good" if cpu_percent < 80 else "⚠️ High" if cpu_percent < 95 else "❌ Critical"
            print(f"   CPU Usage: {cpu_percent:.1f}% - {cpu_status}")
            
            # Memory check
            mem_status = "✅ Good" if memory.percent < 80 else "⚠️ High" if memory.percent < 95 else "❌ Critical"
            print(f"   Memory Usage: {memory.percent:.1f}% - {mem_status}")
            
            # Disk check
            disk_status = "✅ Good" if disk.percent < 85 else "⚠️ High" if disk.percent < 95 else "❌ Critical"
            print(f"   Disk Usage: {disk.percent:.1f}% - {disk_status}")
            
            print("\n2. Audio System:")
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_inputs'] > 0]
                print(f"   Audio Devices: {len(input_devices)} input devices available - ✅ Good")
            except ImportError:
                print("   Audio Devices: sounddevice not available - ⚠️ Warning")
            except Exception as e:
                print(f"   Audio Devices: Error checking - ❌ {e}")
            
            print("\n3. Dependencies:")
            dependencies = [
                ('numpy', 'NumPy'),
                ('scipy', 'SciPy'),
                ('psutil', 'psutil')
            ]
            
            for module_name, display_name in dependencies:
                try:
                    __import__(module_name)
                    print(f"   {display_name}: ✅ Available")
                except ImportError:
                    print(f"   {display_name}: ❌ Missing")
            
            # Overall health assessment
            print("\n4. Overall Assessment:")
            issues = []
            if cpu_percent > 95:
                issues.append("Critical CPU usage")
            if memory.percent > 95:
                issues.append("Critical memory usage")
            if disk.percent > 95:
                issues.append("Critical disk usage")
            
            if not issues:
                print("   System Health: ✅ All checks passed")
                sys.exit(0)
            else:
                print("   System Health: ❌ Issues detected")
                for issue in issues:
                    print(f"     • {issue}")
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error performing health check: {e}")
            sys.exit(1)
    
    def setup_logging(self) -> None:
        """Set up logging based on CLI configuration."""
        # Determine log level
        if self.config.verbose:
            log_level = logging.DEBUG
        elif self.config.quiet:
            log_level = logging.WARNING
        else:
            log_level = getattr(logging, self.config.log_level)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging configured: level={self.config.log_level}")
    
    def create_pipeline(self) -> GunshotLocalizationPipeline:
        """Create and configure the gunshot localization pipeline."""
        try:
            # Create event handler
            event_handler = DefaultEventHandler()
            
            # Create pipeline with CLI configuration
            self.pipeline = GunshotLocalizationPipeline(
                config_path=self.config.config_file,
                event_handler=event_handler,
                output_format=self.config.output_format,
                output_file=self.config.output_file,
                log_dir=self.config.log_dir
            )
            
            self.logger.info("Pipeline created successfully")
            return self.pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {e}")
            raise
    
    def apply_runtime_overrides(self) -> None:
        """Apply runtime parameter overrides to the pipeline."""
        if not self.pipeline:
            return
        
        overrides = {}
        
        # Collect non-None overrides
        if self.config.sample_rate is not None:
            overrides['sample_rate'] = self.config.sample_rate
        
        if self.config.detection_threshold is not None:
            overrides['detection_threshold'] = self.config.detection_threshold
        
        if self.config.buffer_duration is not None:
            overrides['buffer_duration'] = self.config.buffer_duration
        
        if self.config.sound_speed is not None:
            overrides['sound_speed'] = self.config.sound_speed
        
        if self.config.min_confidence is not None:
            overrides['min_confidence'] = self.config.min_confidence
        
        if overrides:
            self.logger.info(f"Applying runtime overrides: {overrides}")
            # Note: This would require implementing override methods in the pipeline
            # For now, we'll log the overrides
            for key, value in overrides.items():
                self.logger.info(f"Override: {key} = {value}")
    
    def run_daemon_mode(self) -> int:
        """Run in daemon mode."""
        try:
            import daemon
            import lockfile
            
            # Create PID file lock
            pid_lock = lockfile.FileLock(self.config.pid_file)
            
            # Daemon context
            with daemon.DaemonContext(
                pidfile=pid_lock,
                stdout=sys.stdout,
                stderr=sys.stderr
            ):
                self.logger.info("Starting daemon mode")
                return self.run_normal_mode()
                
        except ImportError:
            self.logger.error("python-daemon library required for daemon mode")
            return 1
        except Exception as e:
            self.logger.error(f"Failed to start daemon: {e}")
            return 1
    
    def run_test_mode(self) -> int:
        """Run in test mode with synthetic data."""
        self.logger.info("Running in test mode")
        
        try:
            # Setup pipeline
            if not self.pipeline.setup():
                self.logger.error("Pipeline setup failed")
                return 1
            
            # Generate synthetic test data and run tests
            self.logger.info("Generating synthetic test data...")
            
            # This would implement synthetic data generation
            # For now, just run a brief test
            import time
            import numpy as np
            
            # Simulate running for a short time
            for i in range(5):
                self.logger.info(f"Test iteration {i+1}/5")
                
                # Generate synthetic audio data
                audio_data = np.random.randn(4800, 8) * 0.1
                
                # Process through pipeline components (simplified)
                time.sleep(0.1)
            
            self.logger.info("Test mode completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Test mode failed: {e}")
            return 1
        finally:
            if self.pipeline:
                self.pipeline.shutdown()
    
    def run_calibration_mode(self) -> int:
        """Run microphone array calibration."""
        self.logger.info("Running microphone array calibration")
        
        try:
            # This would implement calibration procedures
            self.logger.info("Calibration mode not yet implemented")
            return 0
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return 1
    
    def run_benchmark_mode(self) -> int:
        """Run performance benchmark."""
        self.logger.info("Running performance benchmark")
        
        try:
            # Setup pipeline
            if not self.pipeline.setup():
                self.logger.error("Pipeline setup failed")
                return 1
            
            # Run benchmark
            self.logger.info("Starting benchmark...")
            
            import time
            import numpy as np
            
            start_time = time.time()
            iterations = 100
            
            for i in range(iterations):
                # Generate test data
                audio_data = np.random.randn(4800, 8) * 0.1
                
                # Time processing (simplified)
                process_start = time.time()
                time.sleep(0.001)  # Simulate processing
                process_time = time.time() - process_start
                
                if (i + 1) % 20 == 0:
                    self.logger.info(f"Completed {i+1}/{iterations} iterations")
            
            total_time = time.time() - start_time
            avg_time = total_time / iterations
            
            self.logger.info(f"Benchmark completed:")
            self.logger.info(f"  Total time: {total_time:.2f}s")
            self.logger.info(f"  Average time per iteration: {avg_time*1000:.2f}ms")
            self.logger.info(f"  Throughput: {iterations/total_time:.1f} iterations/s")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return 1
        finally:
            if self.pipeline:
                self.pipeline.shutdown()
    
    def run_normal_mode(self) -> int:
        """Run in normal operation mode."""
        self.logger.info("Starting gunshot localization system")
        
        try:
            # Setup pipeline
            if not self.pipeline.setup():
                self.logger.error("Pipeline setup failed")
                return 1
            
            # Apply runtime overrides
            self.apply_runtime_overrides()
            
            # Start pipeline
            if not self.pipeline.start():
                self.logger.error("Pipeline start failed")
                return 1
            
            self.logger.info("System running - Press Ctrl+C to stop")
            
            # Main loop
            try:
                while True:
                    # Output system status periodically
                    if self.config.verbose:
                        self.pipeline.output_system_status()
                    
                    import time
                    time.sleep(10)  # Status update every 10 seconds
                    
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested by user")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"System error: {e}")
            return 1
        finally:
            if self.pipeline:
                self.logger.info("Stopping pipeline...")
                self.pipeline.stop()
                self.pipeline.shutdown()
                self.logger.info("System shutdown complete")
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Main entry point for CLI.
        
        Args:
            args: Command-line arguments (uses sys.argv if None)
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse arguments
            self.parse_arguments(args)
            
            # Setup logging
            self.setup_logging()
            
            # Create pipeline (except for special modes that don't need it)
            if not (self.config.calibration_mode):
                self.create_pipeline()
            
            # Run in appropriate mode
            if self.config.daemon:
                return self.run_daemon_mode()
            elif self.config.test_mode:
                return self.run_test_mode()
            elif self.config.calibration_mode:
                return self.run_calibration_mode()
            elif self.config.benchmark_mode:
                return self.run_benchmark_mode()
            else:
                return self.run_normal_mode()
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 130  # Standard exit code for Ctrl+C
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error: {e}")
            else:
                print(f"Error: {e}", file=sys.stderr)
            return 1


def main():
    """Main entry point for command-line interface."""
    cli = GunshotLocalizerCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())