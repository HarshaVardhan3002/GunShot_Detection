"""
Demo script for output formatting and structured logging systems.
"""
import time
import json
import tempfile
import os
from output_formatter import (
    RealTimeOutputManager, OutputFormat, 
    format_detection_console, format_localization_console
)
from structured_logger import StructuredLogger, PerformanceTracker


def demo_console_formatting():
    """Demo console output formatting."""
    print("=== Console Output Formatting Demo ===")
    
    # Demo detection formatting
    print("\n1. Detection Events:")
    print(format_detection_console(0.95, "AmplitudeBasedDetector", 12.5))
    print(format_detection_console(0.67, "FrequencyDomainDetector", 18.2))
    print(format_detection_console(0.42, "AdaptiveThresholdDetector", 25.1))
    
    # Demo localization formatting
    print("\n2. Localization Results:")
    print(format_localization_console(12.34, 56.78, 0.89, 0.15, 45.2))
    print(format_localization_console(-5.67, 23.45, 0.72, 0.28, 38.7))
    print(format_localization_console(0.12, -8.90, 0.34, 1.25, 67.3))


def demo_json_output():
    """Demo JSON output formatting."""
    print("\n=== JSON Output Demo ===")
    
    # Create temporary file for JSON output
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()
    
    try:
        # Create JSON output manager
        output_manager = RealTimeOutputManager(
            output_format=OutputFormat.JSON,
            enable_file_output=True,
            output_file=temp_file.name
        )
        
        print(f"JSON output will be written to: {temp_file.name}")
        
        # Generate sample events
        print("\nGenerating sample events...")
        
        # Detection events
        for i in range(3):
            confidence = 0.8 + i * 0.05
            method = f"Detector_{i+1}"
            processing_time = 0.01 + i * 0.005
            
            output_manager.output_detection(confidence, method, processing_time)
            time.sleep(0.1)
        
        # Localization events
        for i in range(3):
            x, y, z = i * 10.0, i * 5.0, 0.0
            confidence = 0.7 + i * 0.1
            error = 0.5 - i * 0.1
            processing_time = 0.02 + i * 0.005
            channels_used = 6 + i
            
            output_manager.output_localization(
                x, y, z, confidence, error, processing_time, channels_used
            )
            time.sleep(0.1)
        
        # System status
        status = {
            'pipeline_state': 'running',
            'processing_mode': 'event_driven',
            'is_running': True,
            'metrics': {
                'total_detections': 6,
                'successful_localizations': 3,
                'average_processing_time': 0.025,
                'localization_accuracy': 0.85
            }
        }
        output_manager.output_system_status(status)
        
        # Performance metrics
        metrics = {
            'average_latency': 0.045,
            'detection_rate': 2.5,
            'uptime': 120.0
        }
        output_manager.output_performance(metrics)
        
        # Close output manager
        output_manager.close()
        
        # Show file contents
        print(f"\nJSON output file contents:")
        with open(temp_file.name, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[-5:], 1):  # Show last 5 lines
                try:
                    data = json.loads(line.strip())
                    print(f"{i}. {data['event_type']}: {json.dumps(data, indent=2)}")
                except:
                    print(f"{i}. {line.strip()}")
        
        # Show output statistics
        print(f"\nOutput Statistics:")
        stats = {
            'total_outputs': len(lines),
            'output_format': 'json',
            'file_size_bytes': os.path.getsize(temp_file.name)
        }
        print(json.dumps(stats, indent=2))
        
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def demo_structured_logging():
    """Demo structured logging system."""
    print("\n=== Structured Logging Demo ===")
    
    # Create temporary log directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create structured logger
        logger = StructuredLogger(
            name="demo_logger",
            log_dir=temp_dir,
            enable_console=True,
            enable_file=True,
            enable_performance_log=True
        )
        
        print(f"Logs will be written to: {temp_dir}")
        
        # Log various events
        print("\nLogging sample events...")
        
        # Detection events
        for i in range(3):
            confidence = 0.8 + i * 0.05
            method = f"Detector_{i+1}"
            processing_time = 0.01 + i * 0.005
            channels = [1, 2, 3, 4, 5, 6, 7, 8]
            metadata = {"signal_strength": 0.9 - i * 0.1, "noise_level": 0.1 + i * 0.05}
            
            logger.log_detection(confidence, method, processing_time, channels, metadata)
            time.sleep(0.1)
        
        # Localization events
        for i in range(3):
            x, y, z = i * 10.0, i * 5.0, 0.0
            confidence = 0.7 + i * 0.1
            error = 0.5 - i * 0.1
            processing_time = 0.02 + i * 0.005
            channels_used = [1, 2, 3, 4, 5, 6]
            method = "cross_correlation_tdoa"
            metadata = {"convergence_iterations": 5 + i, "residual_error": error}
            
            logger.log_localization(
                x, y, z, confidence, error, processing_time,
                channels_used, method, metadata
            )
            time.sleep(0.1)
        
        # Performance metrics
        logger.log_performance(
            latency_ms=25.0,
            accuracy=0.85,
            throughput=10.0,
            resource_usage={"cpu_percent": 45.0, "memory_mb": 128.0, "threads": 4},
            component_times={"detection": 10.0, "localization": 15.0}
        )
        
        # Component status
        logger.log_component_status(
            component="audio_capture",
            status="running",
            metrics={"buffer_utilization": 0.75, "sample_rate": 48000}
        )
        
        # Error event
        logger.log_error(
            error="Microphone disconnected",
            context="audio_capture",
            component="audio_capture",
            additional_data={"microphone_id": 3, "last_signal_time": time.time()}
        )
        
        # Warning event
        logger.log_warning(
            warning="High background noise detected",
            component="gunshot_detector",
            additional_data={"noise_level_db": -15.0, "threshold_db": -20.0}
        )
        
        # Show log statistics
        print(f"\nLog Statistics:")
        stats = logger.get_log_statistics()
        print(json.dumps(stats, indent=2, default=str))
        
        # Show log files created
        print(f"\nLog files created:")
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  {file}: {file_size} bytes")
        
        # Show sample log entries
        main_log_file = None
        for file in os.listdir(temp_dir):
            if file.endswith('.jsonl'):
                main_log_file = os.path.join(temp_dir, file)
                break
        
        if main_log_file:
            print(f"\nSample log entries from {os.path.basename(main_log_file)}:")
            with open(main_log_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[-3:], 1):  # Show last 3 entries
                    try:
                        data = json.loads(line.strip())
                        print(f"{i}. {data['event_type']}: {data['message']}")
                        print(f"   Data: {json.dumps(data['data'], indent=4)}")
                    except:
                        print(f"{i}. {line.strip()}")
        
        # Close logger
        logger.close()
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_performance_tracking():
    """Demo performance tracking system."""
    print("\n=== Performance Tracking Demo ===")
    
    # Create temporary log directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create logger and performance tracker
        logger = StructuredLogger(
            name="perf_demo",
            log_dir=temp_dir,
            enable_console=False,  # Reduce console noise
            enable_performance_log=True
        )
        
        tracker = PerformanceTracker(logger)
        
        print("Simulating pipeline operations with performance tracking...")
        
        # Simulate multiple processing cycles
        for cycle in range(5):
            print(f"\nCycle {cycle + 1}:")
            
            # Detection phase
            tracker.start_timing("detection")
            time.sleep(0.01)  # Simulate detection processing
            detection_time = tracker.end_timing("detection")
            print(f"  Detection: {detection_time*1000:.1f}ms")
            
            # Channel selection phase
            tracker.start_timing("channel_selection")
            time.sleep(0.005)  # Simulate channel selection
            selection_time = tracker.end_timing("channel_selection")
            print(f"  Channel Selection: {selection_time*1000:.1f}ms")
            
            # Localization phase
            tracker.start_timing("localization")
            time.sleep(0.02)  # Simulate localization processing
            localization_time = tracker.end_timing("localization")
            print(f"  Localization: {localization_time*1000:.1f}ms")
            
            # Record additional metrics
            total_time = detection_time + selection_time + localization_time
            tracker.record_metric("total_processing_time", total_time)
            tracker.record_metric("accuracy", 0.8 + cycle * 0.02)
            
            # Log performance summary every few cycles
            if (cycle + 1) % 2 == 0:
                accuracy = 0.8 + cycle * 0.02
                throughput = 1.0 / total_time if total_time > 0 else 0
                tracker.log_performance_summary(accuracy, throughput)
                print(f"  Performance logged: {accuracy:.1%} accuracy, {throughput:.1f} ops/s")
        
        # Show final performance statistics
        print(f"\nFinal Performance Metrics:")
        print(f"  Component Times: {tracker.component_times}")
        print(f"  Recorded Metrics: {tracker.metrics}")
        
        # Show performance log file
        perf_log_file = None
        for file in os.listdir(temp_dir):
            if file.startswith('performance_'):
                perf_log_file = os.path.join(temp_dir, file)
                break
        
        if perf_log_file:
            print(f"\nPerformance log entries:")
            with open(perf_log_file, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, 1):
                    print(f"{i}. {line.strip()}")
        
        # Close logger
        logger.close()
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_output_format_comparison():
    """Demo different output formats side by side."""
    print("\n=== Output Format Comparison ===")
    
    # Sample data
    detection_data = {
        'confidence': 0.87,
        'method': 'AmplitudeBasedDetector',
        'processing_time': 0.0125
    }
    
    localization_data = {
        'x': 12.34, 'y': 56.78, 'z': 0.0,
        'confidence': 0.75,
        'error': 0.45,
        'processing_time': 0.0235,
        'channels_used': 6
    }
    
    # Console format
    print("\n1. Console Format:")
    console_detection = format_detection_console(
        detection_data['confidence'], 
        detection_data['method'], 
        detection_data['processing_time'] * 1000
    )
    console_localization = format_localization_console(
        localization_data['x'], localization_data['y'],
        localization_data['confidence'], localization_data['error'],
        localization_data['processing_time'] * 1000
    )
    print(console_detection)
    print(console_localization)
    
    # JSON format
    print("\n2. JSON Format:")
    from output_formatter import JSONFormatter
    json_formatter = JSONFormatter(pretty_print=True)
    
    json_detection = json_formatter.format_detection(
        "2023-01-01T12:34:56.789",
        detection_data['confidence'],
        detection_data['method'],
        detection_data['processing_time']
    )
    json_localization = json_formatter.format_localization(
        "2023-01-01T12:34:56.890",
        localization_data['x'], localization_data['y'], localization_data['z'],
        localization_data['confidence'], localization_data['error'],
        localization_data['processing_time'], localization_data['channels_used']
    )
    print(json_detection)
    print(json_localization)
    
    # CSV format
    print("\n3. CSV Format:")
    from output_formatter import CSVFormatter
    csv_formatter = CSVFormatter()
    
    print(csv_formatter.get_detection_header())
    csv_detection = csv_formatter.format_detection(
        "2023-01-01T12:34:56.789",
        detection_data['confidence'],
        detection_data['method'],
        detection_data['processing_time']
    )
    print(csv_detection)
    
    print(csv_formatter.get_localization_header())
    csv_localization = csv_formatter.format_localization(
        "2023-01-01T12:34:56.890",
        localization_data['x'], localization_data['y'], localization_data['z'],
        localization_data['confidence'], localization_data['error'],
        localization_data['processing_time'], localization_data['channels_used']
    )
    print(csv_localization)


def main():
    """Main demo function."""
    print("Gunshot Localization Output & Logging Demo")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_console_formatting()
        demo_json_output()
        demo_structured_logging()
        demo_performance_tracking()
        demo_output_format_comparison()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Console output with color coding")
        print("✓ JSON structured output")
        print("✓ CSV format for data analysis")
        print("✓ Structured logging with metadata")
        print("✓ Performance tracking and metrics")
        print("✓ File output capabilities")
        print("✓ Multiple log levels and event types")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()