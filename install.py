#!/usr/bin/env python3
"""
Installation and Setup Script for Gunshot Localization System.
This script handles dependency installation, system configuration, and initial setup.
"""
import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional


class GunshotLocalizerInstaller:
    """Installer for the gunshot localization system."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        
    def _get_system_info(self) -> Dict:
        """Get system information for installation."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable
        }
    
    def print_header(self):
        """Print installation header."""
        print("=" * 70)
        print("Gunshot Localization System - Installation & Setup")
        print("=" * 70)
        print(f"Platform: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"Python: {self.system_info['python_version']}")
        print(f"Project Root: {self.project_root}")
        print()
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print("Checking Python version...")
        
        major, minor = sys.version_info[:2]
        required_major, required_minor = 3, 8
        
        if major < required_major or (major == required_major and minor < required_minor):
            print(f"❌ Python {required_major}.{required_minor}+ required, found {major}.{minor}")
            return False
        
        print(f"✅ Python {major}.{minor} is compatible")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required Python packages."""
        print("\nInstalling Python dependencies...")
        
        # Read requirements from requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            # Install requirements
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print(f"❌ Failed to install dependencies:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
    
    def check_audio_system(self) -> bool:
        """Check if audio system is available."""
        print("\nChecking audio system...")
        
        try:
            import sounddevice as sd
            
            # Get available audio devices
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] >= 8]
            
            print(f"Found {len(devices)} audio devices")
            print(f"Found {len(input_devices)} devices with 8+ input channels")
            
            if len(input_devices) == 0:
                print("⚠️  No audio devices with 8+ input channels found")
                print("   You may need to connect a multi-channel audio interface")
                return False
            
            print("✅ Audio system is available")
            
            # List suitable devices
            print("\nSuitable audio devices:")
            for i, device in enumerate(input_devices):
                print(f"  {device['index']}: {device['name']} "
                      f"({device['max_input_channels']} channels)")
            
            return True
            
        except ImportError:
            print("❌ sounddevice not available - install dependencies first")
            return False
        except Exception as e:
            print(f"❌ Error checking audio system: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        print("\nCreating directories...")
        
        directories = [
            self.config_dir,
            self.logs_dir,
            self.project_root / "data",
            self.project_root / "calibration"
        ]
        
        try:
            for directory in directories:
                directory.mkdir(exist_ok=True)
                print(f"✅ Created/verified: {directory}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating directories: {e}")
            return False
    
    def create_default_config(self) -> bool:
        """Create default configuration files."""
        print("\nCreating default configuration...")
        
        try:
            # Create default system configuration
            default_config = {
                "microphones": [
                    {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0, "name": "Mic 1 - Front Left"},
                    {"id": 2, "x": 4.0, "y": 0.0, "z": 0.0, "name": "Mic 2 - Front Right"},
                    {"id": 3, "x": 0.0, "y": 4.0, "z": 0.0, "name": "Mic 3 - Rear Left"},
                    {"id": 4, "x": 4.0, "y": 4.0, "z": 0.0, "name": "Mic 4 - Rear Right"},
                    {"id": 5, "x": 2.0, "y": 2.0, "z": 2.0, "name": "Mic 5 - Center Elevated"},
                    {"id": 6, "x": 6.0, "y": 2.0, "z": 0.0, "name": "Mic 6 - Extended Right"},
                    {"id": 7, "x": 2.0, "y": 6.0, "z": 0.0, "name": "Mic 7 - Extended Rear"},
                    {"id": 8, "x": 6.0, "y": 6.0, "z": 0.0, "name": "Mic 8 - Far Corner"}
                ],
                "system": {
                    "sample_rate": 48000,
                    "sound_speed": 343.0,
                    "detection_threshold_db": -25.0,
                    "buffer_duration": 1.0,
                    "min_confidence": 0.6,
                    "max_processing_latency_ms": 500,
                    "log_level": "INFO",
                    "enable_diagnostics": True
                },
                "audio": {
                    "device_index": None,
                    "channels": 8,
                    "chunk_size": 1024,
                    "enable_monitoring": True
                },
                "detection": {
                    "method": "amplitude_frequency",
                    "amplitude_threshold_db": -25.0,
                    "frequency_bands": [
                        {"min_hz": 100, "max_hz": 500, "weight": 0.3},
                        {"min_hz": 500, "max_hz": 2000, "weight": 0.5},
                        {"min_hz": 2000, "max_hz": 8000, "weight": 0.2}
                    ],
                    "min_duration_ms": 10,
                    "max_duration_ms": 500
                },
                "localization": {
                    "method": "cross_correlation",
                    "max_distance_m": 100.0,
                    "min_microphones": 4,
                    "correlation_window_ms": 50,
                    "interpolation_factor": 4
                },
                "output": {
                    "format": "console",
                    "enable_file_output": True,
                    "output_file": "detections.json",
                    "coordinate_precision": 3
                }
            }
            
            config_file = self.config_dir / "default_config.json"
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"✅ Created default configuration: {config_file}")
            
            # Create sample configurations for different scenarios
            self._create_sample_configs()
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating configuration: {e}")
            return False
    
    def _create_sample_configs(self):
        """Create sample configurations for different deployment scenarios."""
        
        # Indoor small room configuration
        indoor_config = {
            "name": "Indoor Small Room (4m x 4m)",
            "description": "Configuration for indoor deployment in a small room",
            "microphones": [
                {"id": 1, "x": 0.5, "y": 0.5, "z": 1.5, "name": "Corner 1"},
                {"id": 2, "x": 3.5, "y": 0.5, "z": 1.5, "name": "Corner 2"},
                {"id": 3, "x": 0.5, "y": 3.5, "z": 1.5, "name": "Corner 3"},
                {"id": 4, "x": 3.5, "y": 3.5, "z": 1.5, "name": "Corner 4"},
                {"id": 5, "x": 2.0, "y": 2.0, "z": 2.5, "name": "Center Ceiling"},
                {"id": 6, "x": 2.0, "y": 0.5, "z": 1.5, "name": "Front Center"},
                {"id": 7, "x": 2.0, "y": 3.5, "z": 1.5, "name": "Rear Center"},
                {"id": 8, "x": 0.5, "y": 2.0, "z": 1.5, "name": "Left Center"}
            ],
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -30.0,
                "min_confidence": 0.7
            }
        }
        
        # Outdoor perimeter configuration
        outdoor_config = {
            "name": "Outdoor Perimeter (20m x 20m)",
            "description": "Configuration for outdoor perimeter monitoring",
            "microphones": [
                {"id": 1, "x": 0.0, "y": 0.0, "z": 3.0, "name": "Corner 1"},
                {"id": 2, "x": 20.0, "y": 0.0, "z": 3.0, "name": "Corner 2"},
                {"id": 3, "x": 0.0, "y": 20.0, "z": 3.0, "name": "Corner 3"},
                {"id": 4, "x": 20.0, "y": 20.0, "z": 3.0, "name": "Corner 4"},
                {"id": 5, "x": 10.0, "y": 10.0, "z": 4.0, "name": "Center Elevated"},
                {"id": 6, "x": 10.0, "y": 0.0, "z": 3.0, "name": "Front Center"},
                {"id": 7, "x": 10.0, "y": 20.0, "z": 3.0, "name": "Rear Center"},
                {"id": 8, "x": 0.0, "y": 10.0, "z": 3.0, "name": "Left Center"}
            ],
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -20.0,
                "min_confidence": 0.5
            }
        }
        
        # Large venue configuration
        venue_config = {
            "name": "Large Venue (50m x 30m)",
            "description": "Configuration for large venue or facility monitoring",
            "microphones": [
                {"id": 1, "x": 5.0, "y": 5.0, "z": 4.0, "name": "Zone 1"},
                {"id": 2, "x": 45.0, "y": 5.0, "z": 4.0, "name": "Zone 2"},
                {"id": 3, "x": 5.0, "y": 25.0, "z": 4.0, "name": "Zone 3"},
                {"id": 4, "x": 45.0, "y": 25.0, "z": 4.0, "name": "Zone 4"},
                {"id": 5, "x": 25.0, "y": 15.0, "z": 6.0, "name": "Center High"},
                {"id": 6, "x": 25.0, "y": 5.0, "z": 4.0, "name": "Front Center"},
                {"id": 7, "x": 25.0, "y": 25.0, "z": 4.0, "name": "Rear Center"},
                {"id": 8, "x": 15.0, "y": 15.0, "z": 4.0, "name": "Center Low"}
            ],
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -15.0,
                "min_confidence": 0.4
            }
        }
        
        # Save sample configurations
        configs = [
            ("indoor_small_room.json", indoor_config),
            ("outdoor_perimeter.json", outdoor_config),
            ("large_venue.json", venue_config)
        ]
        
        for filename, config in configs:
            config_file = self.config_dir / filename
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Created sample configuration: {filename}")
    
    def create_startup_scripts(self) -> bool:
        """Create startup and service scripts."""
        print("\nCreating startup scripts...")
        
        try:
            # Create main startup script
            startup_script = f"""#!/bin/bash
# Gunshot Localization System Startup Script

# Set working directory
cd "{self.project_root}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the system with default configuration
python main.py --config config/default_config.json --verbose

# Keep script running
while true; do
    echo "System stopped. Restarting in 5 seconds..."
    sleep 5
    python main.py --config config/default_config.json --verbose
done
"""
            
            startup_file = self.project_root / "start_system.sh"
            with open(startup_file, 'w') as f:
                f.write(startup_script)
            
            # Make executable on Unix systems
            if self.system_info['platform'] != 'Windows':
                os.chmod(startup_file, 0o755)
            
            print(f"✅ Created startup script: {startup_file}")
            
            # Create Windows batch file
            if self.system_info['platform'] == 'Windows':
                batch_script = f"""@echo off
REM Gunshot Localization System Startup Script for Windows

cd /d "{self.project_root}"

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
)

REM Run the system with default configuration
python main.py --config config\\default_config.json --verbose

pause
"""
                batch_file = self.project_root / "start_system.bat"
                with open(batch_file, 'w') as f:
                    f.write(batch_script)
                
                print(f"✅ Created Windows batch file: {batch_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating startup scripts: {e}")
            return False
    
    def create_calibration_tools(self) -> bool:
        """Create calibration and validation tools."""
        print("\nCreating calibration tools...")
        
        try:
            calibration_script = '''#!/usr/bin/env python3
"""
Microphone Array Calibration Tool.
This tool helps validate microphone positioning and system calibration.
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config_manager import ConfigurationManager
from audio_capture import AudioCaptureEngine
from tdoa_localizer import CrossCorrelationTDoALocalizer


def validate_microphone_positions(config_file):
    """Validate microphone positions for triangulation feasibility."""
    print("Validating microphone positions...")
    
    config_manager = ConfigurationManager()
    if not config_manager.load_config(config_file):
        print("❌ Failed to load configuration")
        return False
    
    is_valid, errors = config_manager.validate_config()
    
    if is_valid:
        print("✅ Microphone configuration is valid")
        
        # Check geometric properties
        positions = config_manager.get_microphone_positions()
        coords = np.array([[p.x, p.y, p.z] for p in positions])
        
        # Calculate array dimensions
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
        z_range = np.max(coords[:, 2]) - np.min(coords[:, 2])
        
        print(f"Array dimensions: {x_range:.1f}m x {y_range:.1f}m x {z_range:.1f}m")
        
        # Calculate average spacing
        distances = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                distances.append(dist)
        
        avg_spacing = np.mean(distances)
        min_spacing = np.min(distances)
        
        print(f"Average microphone spacing: {avg_spacing:.2f}m")
        print(f"Minimum microphone spacing: {min_spacing:.2f}m")
        
        if min_spacing < 0.5:
            print("⚠️  Warning: Some microphones are very close together")
        
        return True
    else:
        print("Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False


def test_audio_capture():
    """Test audio capture functionality."""
    print("\\nTesting audio capture...")
    
    try:
        audio_capture = AudioCaptureEngine(
            sample_rate=48000,
            channels=8,
            buffer_duration=1.0
        )
        
        print("✅ Audio capture engine initialized")
        
        # Test device availability
        if hasattr(audio_capture, 'get_device_info'):
            device_info = audio_capture.get_device_info()
            print(f"Audio device: {device_info.get('name', 'Unknown')}")
            print(f"Channels: {device_info.get('channels', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"Audio capture test failed: {e}")
        return False


def run_system_check(config_file):
    """Run comprehensive system check."""
    print("Running system check...")
    print("=" * 50)
    
    checks = [
        ("Microphone Positions", lambda: validate_microphone_positions(config_file)),
        ("Audio Capture", test_audio_capture),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\\n{check_name}:")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            results.append((check_name, False))
    
    print("\\n" + "=" * 50)
    print("System Check Summary:")
    print("=" * 50)
    
    passed = 0
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{check_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\nOverall: {passed}/{len(results)} checks passed")
    return passed == len(results)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Microphone array calibration tool')
    parser.add_argument('--config', default='config/default_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    print("Gunshot Localization System - Calibration Tool")
    print("=" * 60)
    
    success = run_system_check(args.config)
    sys.exit(0 if success else 1)
'''
            
            calibration_file = self.project_root / "calibrate_system.py"
            with open(calibration_file, 'w') as f:
                f.write(calibration_script)
            
            print(f"✅ Created calibration tool: {calibration_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating calibration tools: {e}")
            return False
    
    def run_installation(self) -> bool:
        """Run the complete installation process."""
        self.print_header()
        
        steps = [
            ("Python Version Check", self.check_python_version),
            ("Install Dependencies", self.install_dependencies),
            ("Check Audio System", self.check_audio_system),
            ("Create Directories", self.create_directories),
            ("Create Configuration", self.create_default_config),
            ("Create Startup Scripts", self.create_startup_scripts),
            ("Create Calibration Tools", self.create_calibration_tools),
        ]
        
        results = []
        for step_name, step_func in steps:
            try:
                result = step_func()
                results.append((step_name, result))
                if not result:
                    print(f"⚠️  {step_name} failed, but continuing...")
            except Exception as e:
                print(f"❌ {step_name} error: {e}")
                results.append((step_name, False))
        
        # Print summary
        print("\n" + "=" * 70)
        print("INSTALLATION SUMMARY")
        print("=" * 70)
        
        passed = 0
        for step_name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"{step_name}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{len(results)} steps completed successfully")
        
        if passed >= len(results) - 1:  # Allow one failure
            print("\nInstallation completed successfully!")
            print("\nNext steps:")
            print("1. Review and customize config/default_config.json")
            print("2. Run calibration: python calibrate_system.py")
            print("3. Start system: python main.py --config config/default_config.json")
            return True
        else:
            print("\nInstallation completed with issues.")
            print("Please review the failed steps above.")
            return False


def main():
    """Main installation function."""
    installer = GunshotLocalizerInstaller()
    success = installer.run_installation()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())