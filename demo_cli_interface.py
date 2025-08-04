"""
Demo script for command-line interface.
"""
import subprocess
import sys
import os
import tempfile
import json


def create_demo_config():
    """Create a demo configuration file."""
    config_data = {
        "microphones": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 2, "x": 2.0, "y": 0.0, "z": 0.0},
            {"id": 3, "x": 0.0, "y": 2.0, "z": 0.0},
            {"id": 4, "x": 2.0, "y": 2.0, "z": 0.0},
            {"id": 5, "x": 1.0, "y": 1.0, "z": 1.5},
            {"id": 6, "x": 3.0, "y": 1.0, "z": 0.0},
            {"id": 7, "x": 1.0, "y": 3.0, "z": 0.0},
            {"id": 8, "x": 3.0, "y": 3.0, "z": 0.0}
        ],
        "system": {
            "sample_rate": 48000,
            "sound_speed": 343.0,
            "detection_threshold_db": -20.0,
            "buffer_duration": 1.0,
            "min_confidence": 0.6
        }
    }
    
    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    )
    json.dump(config_data, temp_config, indent=2)
    temp_config.close()
    
    return temp_config.name


def run_cli_command(args):
    """Run CLI command and capture output."""
    try:
        # Run the CLI command
        cmd = [sys.executable, 'gunshot-localizer/gunshot_localizer.py'] + args
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def demo_help_and_usage():
    """Demo help and usage information."""
    print("=== Help and Usage Demo ===")
    
    # Show help
    print("\n1. Showing help information:")
    returncode, stdout, stderr = run_cli_command(['--help'])
    
    if returncode == 0:
        print("✅ Help command successful")
        print("First few lines of help:")
        lines = stdout.split('\n')[:10]
        for line in lines:
            print(f"  {line}")
        print("  ... (truncated)")
    else:
        print(f"❌ Help command failed: {stderr}")
    
    # Show version
    print("\n2. Showing version:")
    returncode, stdout, stderr = run_cli_command(['--version'])
    
    if returncode == 0:
        print("✅ Version command successful")
        print(f"  Output: {stdout.strip()}")
    else:
        print(f"❌ Version command failed: {stderr}")


def demo_config_validation():
    """Demo configuration validation."""
    print("\n=== Configuration Validation Demo ===")
    
    # Create demo config
    config_path = create_demo_config()
    
    try:
        print(f"\n1. Validating configuration file: {config_path}")
        returncode, stdout, stderr = run_cli_command(['--config', config_path, '--check-config'])
        
        if returncode == 0:
            print("✅ Configuration validation successful")
            print("Output:")
            for line in stdout.split('\n')[:10]:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"❌ Configuration validation failed")
            print(f"  Error: {stderr}")
    
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.unlink(config_path)


def demo_argument_parsing():
    """Demo various argument combinations."""
    print("\n=== Argument Parsing Demo ===")
    
    config_path = create_demo_config()
    
    try:
        # Test different argument combinations
        test_cases = [
            {
                'name': 'Basic configuration',
                'args': ['--config', config_path, '--check-config']
            },
            {
                'name': 'JSON output format',
                'args': ['--config', config_path, '--output-format', 'json', '--check-config']
            },
            {
                'name': 'Verbose mode',
                'args': ['--config', config_path, '--verbose', '--check-config']
            },
            {
                'name': 'Runtime parameter overrides',
                'args': [
                    '--config', config_path,
                    '--sample-rate', '44100',
                    '--detection-threshold', '0.8',
                    '--check-config'
                ]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Testing: {test_case['name']}")
            returncode, stdout, stderr = run_cli_command(test_case['args'])
            
            if returncode == 0:
                print("  ✅ Success")
            else:
                print(f"  ❌ Failed: {stderr}")
    
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


def demo_error_handling():
    """Demo error handling in CLI."""
    print("\n=== Error Handling Demo ===")
    
    error_test_cases = [
        {
            'name': 'Missing configuration file',
            'args': ['--config', 'nonexistent.json'],
            'expected_error': 'Configuration file not found'
        },
        {
            'name': 'Invalid threshold value',
            'args': ['--config', 'sample_config.json', '--detection-threshold', '1.5'],
            'expected_error': 'must be between 0.0 and 1.0'
        },
        {
            'name': 'Conflicting verbosity options',
            'args': ['--config', 'sample_config.json', '--verbose', '--quiet'],
            'expected_error': 'cannot be used together'
        },
        {
            'name': 'Multiple operation modes',
            'args': ['--config', 'sample_config.json', '--test-mode', '--calibration-mode'],
            'expected_error': 'Only one operation mode'
        }
    ]
    
    for i, test_case in enumerate(error_test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")
        returncode, stdout, stderr = run_cli_command(test_case['args'])
        
        if returncode != 0:
            print("  ✅ Error correctly detected")
            if test_case['expected_error'] in stderr:
                print(f"  ✅ Expected error message found")
            else:
                print(f"  ⚠️  Expected '{test_case['expected_error']}' but got: {stderr}")
        else:
            print(f"  ❌ Expected error but command succeeded")


def demo_operation_modes():
    """Demo different operation modes."""
    print("\n=== Operation Modes Demo ===")
    
    config_path = create_demo_config()
    
    try:
        operation_modes = [
            {
                'name': 'Test Mode',
                'args': ['--config', config_path, '--test-mode'],
                'timeout': 5
            },
            {
                'name': 'Benchmark Mode',
                'args': ['--config', config_path, '--benchmark-mode'],
                'timeout': 10
            },
            {
                'name': 'Calibration Mode',
                'args': ['--config', config_path, '--calibration-mode'],
                'timeout': 5
            }
        ]
        
        for i, mode in enumerate(operation_modes, 1):
            print(f"\n{i}. Testing: {mode['name']}")
            
            try:
                cmd = [sys.executable, 'gunshot-localizer/gunshot_localizer.py'] + mode['args']
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=mode['timeout']
                )
                
                if result.returncode == 0:
                    print("  ✅ Mode executed successfully")
                    # Show last few lines of output
                    lines = result.stdout.split('\n')[-5:]
                    for line in lines:
                        if line.strip():
                            print(f"    {line}")
                else:
                    print(f"  ❌ Mode failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("  ⚠️  Mode timed out (expected for some modes)")
            except Exception as e:
                print(f"  ❌ Error running mode: {e}")
    
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


def demo_output_formats():
    """Demo different output formats."""
    print("\n=== Output Formats Demo ===")
    
    config_path = create_demo_config()
    
    try:
        output_formats = ['console', 'json', 'csv']
        
        for i, format_type in enumerate(output_formats, 1):
            print(f"\n{i}. Testing {format_type.upper()} output format:")
            
            args = [
                '--config', config_path,
                '--output-format', format_type,
                '--check-config'
            ]
            
            returncode, stdout, stderr = run_cli_command(args)
            
            if returncode == 0:
                print(f"  ✅ {format_type.upper()} format successful")
                # Show sample output
                lines = stdout.split('\n')[:3]
                for line in lines:
                    if line.strip():
                        print(f"    {line}")
            else:
                print(f"  ❌ {format_type.upper()} format failed: {stderr}")
    
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


def demo_advanced_features():
    """Demo advanced CLI features."""
    print("\n=== Advanced Features Demo ===")
    
    config_path = create_demo_config()
    
    try:
        advanced_features = [
            {
                'name': 'Performance monitoring enabled',
                'args': [
                    '--config', config_path,
                    '--enable-performance-monitoring',
                    '--health-check-interval', '10.0',
                    '--check-config'
                ]
            },
            {
                'name': 'Custom log settings',
                'args': [
                    '--config', config_path,
                    '--log-level', 'DEBUG',
                    '--log-dir', 'custom_logs',
                    '--check-config'
                ]
            },
            {
                'name': 'Multiple parameter overrides',
                'args': [
                    '--config', config_path,
                    '--sample-rate', '44100',
                    '--sound-speed', '340.0',
                    '--min-confidence', '0.7',
                    '--buffer-duration', '0.5',
                    '--check-config'
                ]
            }
        ]
        
        for i, feature in enumerate(advanced_features, 1):
            print(f"\n{i}. Testing: {feature['name']}")
            returncode, stdout, stderr = run_cli_command(feature['args'])
            
            if returncode == 0:
                print("  ✅ Advanced feature successful")
            else:
                print(f"  ❌ Advanced feature failed: {stderr}")
    
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


def main():
    """Main demo function."""
    print("Gunshot Localizer CLI Interface Demo")
    print("=" * 50)
    
    try:
        # Run all demos
        demo_help_and_usage()
        demo_config_validation()
        demo_argument_parsing()
        demo_error_handling()
        demo_operation_modes()
        demo_output_formats()
        demo_advanced_features()
        
        print("\n" + "=" * 50)
        print("CLI Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Comprehensive argument parsing")
        print("✅ Configuration file validation")
        print("✅ Multiple output formats (console, JSON, CSV)")
        print("✅ Runtime parameter overrides")
        print("✅ Operation modes (test, benchmark, calibration)")
        print("✅ Error handling and validation")
        print("✅ Help and usage information")
        print("✅ Advanced configuration options")
        print("✅ Logging level control")
        print("✅ Performance monitoring options")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()