#!/usr/bin/env python3
"""
Quick Setup Script for Gunshot Localization System.
This script provides guided setup for common deployment scenarios.
"""
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


class QuickSetup:
    """Quick setup wizard for common deployment scenarios."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        
    def print_header(self):
        """Print setup header."""
        print("=" * 60)
        print("Gunshot Localization System - Quick Setup")
        print("=" * 60)
        print("This wizard will help you configure the system for common scenarios.")
        print()
    
    def get_user_choice(self, prompt: str, choices: List[str]) -> str:
        """Get user choice from a list of options."""
        print(prompt)
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")
        
        while True:
            try:
                choice_num = int(input("\nEnter your choice (number): "))
                if 1 <= choice_num <= len(choices):
                    return choices[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")
    
    def get_user_input(self, prompt: str, default: str = None) -> str:
        """Get user input with optional default."""
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        else:
            return input(f"{prompt}: ").strip()
    
    def get_float_input(self, prompt: str, default: float = None) -> float:
        """Get float input from user."""
        while True:
            try:
                if default is not None:
                    user_input = input(f"{prompt} [{default}]: ").strip()
                    return float(user_input) if user_input else default
                else:
                    return float(input(f"{prompt}: "))
            except ValueError:
                print("Please enter a valid number")
    
    def select_deployment_scenario(self) -> str:
        """Select deployment scenario."""
        scenarios = [
            "Indoor Small Room (4m x 4m)",
            "Indoor Large Room (10m x 10m)", 
            "Outdoor Perimeter (20m x 20m)",
            "Large Venue/Facility (50m x 30m)",
            "Custom Configuration"
        ]
        
        return self.get_user_choice(
            "Select your deployment scenario:",
            scenarios
        )
    
    def configure_indoor_small(self) -> Dict:
        """Configure for indoor small room."""
        print("\nConfiguring for Indoor Small Room...")
        
        # Get room dimensions
        length = self.get_float_input("Room length (meters)", 4.0)
        width = self.get_float_input("Room width (meters)", 4.0)
        height = self.get_float_input("Microphone height (meters)", 2.0)
        
        # Calculate microphone positions
        margin = 0.5  # Margin from walls
        center_x = length / 2
        center_y = width / 2
        
        microphones = [
            {"id": 1, "x": margin, "y": margin, "z": height, "name": "Front Left Corner"},
            {"id": 2, "x": length - margin, "y": margin, "z": height, "name": "Front Right Corner"},
            {"id": 3, "x": margin, "y": width - margin, "z": height, "name": "Rear Left Corner"},
            {"id": 4, "x": length - margin, "y": width - margin, "z": height, "name": "Rear Right Corner"},
            {"id": 5, "x": center_x, "y": center_y, "z": height + 0.5, "name": "Center Elevated"},
            {"id": 6, "x": center_x, "y": margin, "z": height, "name": "Front Center"},
            {"id": 7, "x": center_x, "y": width - margin, "z": height, "name": "Rear Center"},
            {"id": 8, "x": margin, "y": center_y, "z": height, "name": "Left Center"}
        ]
        
        return {
            "microphones": microphones,
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -30.0,
                "min_confidence": 0.7,
                "max_processing_latency_ms": 500
            },
            "detection": {
                "amplitude_threshold_db": -30.0,
                "frequency_bands": [
                    {"min_hz": 200, "max_hz": 800, "weight": 0.4},
                    {"min_hz": 800, "max_hz": 3000, "weight": 0.6}
                ]
            }
        }
    
    def configure_indoor_large(self) -> Dict:
        """Configure for indoor large room."""
        print("\nConfiguring for Indoor Large Room...")
        
        length = self.get_float_input("Room length (meters)", 10.0)
        width = self.get_float_input("Room width (meters)", 10.0)
        height = self.get_float_input("Microphone height (meters)", 3.0)
        
        margin = 1.0
        center_x = length / 2
        center_y = width / 2
        
        microphones = [
            {"id": 1, "x": margin, "y": margin, "z": height, "name": "Front Left"},
            {"id": 2, "x": length - margin, "y": margin, "z": height, "name": "Front Right"},
            {"id": 3, "x": margin, "y": width - margin, "z": height, "name": "Rear Left"},
            {"id": 4, "x": length - margin, "y": width - margin, "z": height, "name": "Rear Right"},
            {"id": 5, "x": center_x, "y": center_y, "z": height + 1.0, "name": "Center High"},
            {"id": 6, "x": center_x, "y": margin, "z": height, "name": "Front Center"},
            {"id": 7, "x": center_x, "y": width - margin, "z": height, "name": "Rear Center"},
            {"id": 8, "x": margin, "y": center_y, "z": height, "name": "Left Center"}
        ]
        
        return {
            "microphones": microphones,
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -25.0,
                "min_confidence": 0.6,
                "max_processing_latency_ms": 500
            },
            "detection": {
                "amplitude_threshold_db": -25.0,
                "frequency_bands": [
                    {"min_hz": 150, "max_hz": 600, "weight": 0.3},
                    {"min_hz": 600, "max_hz": 2500, "weight": 0.5},
                    {"min_hz": 2500, "max_hz": 6000, "weight": 0.2}
                ]
            }
        }
    
    def configure_outdoor_perimeter(self) -> Dict:
        """Configure for outdoor perimeter."""
        print("\nConfiguring for Outdoor Perimeter...")
        
        length = self.get_float_input("Perimeter length (meters)", 20.0)
        width = self.get_float_input("Perimeter width (meters)", 20.0)
        height = self.get_float_input("Microphone height (meters)", 3.0)
        
        center_x = length / 2
        center_y = width / 2
        
        microphones = [
            {"id": 1, "x": 0.0, "y": 0.0, "z": height, "name": "Corner 1"},
            {"id": 2, "x": length, "y": 0.0, "z": height, "name": "Corner 2"},
            {"id": 3, "x": 0.0, "y": width, "z": height, "name": "Corner 3"},
            {"id": 4, "x": length, "y": width, "z": height, "name": "Corner 4"},
            {"id": 5, "x": center_x, "y": center_y, "z": height + 1.0, "name": "Center Elevated"},
            {"id": 6, "x": center_x, "y": 0.0, "z": height, "name": "Front Center"},
            {"id": 7, "x": center_x, "y": width, "z": height, "name": "Rear Center"},
            {"id": 8, "x": 0.0, "y": center_y, "z": height, "name": "Left Center"}
        ]
        
        return {
            "microphones": microphones,
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -20.0,
                "min_confidence": 0.5,
                "max_processing_latency_ms": 500
            },
            "detection": {
                "amplitude_threshold_db": -20.0,
                "frequency_bands": [
                    {"min_hz": 100, "max_hz": 500, "weight": 0.3},
                    {"min_hz": 500, "max_hz": 2000, "weight": 0.5},
                    {"min_hz": 2000, "max_hz": 8000, "weight": 0.2}
                ]
            }
        }
    
    def configure_large_venue(self) -> Dict:
        """Configure for large venue."""
        print("\nConfiguring for Large Venue...")
        
        length = self.get_float_input("Venue length (meters)", 50.0)
        width = self.get_float_input("Venue width (meters)", 30.0)
        height = self.get_float_input("Microphone height (meters)", 4.0)
        
        margin = 5.0
        center_x = length / 2
        center_y = width / 2
        
        microphones = [
            {"id": 1, "x": margin, "y": margin, "z": height, "name": "Zone 1"},
            {"id": 2, "x": length - margin, "y": margin, "z": height, "name": "Zone 2"},
            {"id": 3, "x": margin, "y": width - margin, "z": height, "name": "Zone 3"},
            {"id": 4, "x": length - margin, "y": width - margin, "z": height, "name": "Zone 4"},
            {"id": 5, "x": center_x, "y": center_y, "z": height + 2.0, "name": "Center High"},
            {"id": 6, "x": center_x, "y": margin, "z": height, "name": "Front Center"},
            {"id": 7, "x": center_x, "y": width - margin, "z": height, "name": "Rear Center"},
            {"id": 8, "x": center_x - 10, "y": center_y, "z": height, "name": "Center Left"}
        ]
        
        return {
            "microphones": microphones,
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": -15.0,
                "min_confidence": 0.4,
                "max_processing_latency_ms": 500
            },
            "detection": {
                "amplitude_threshold_db": -15.0,
                "frequency_bands": [
                    {"min_hz": 80, "max_hz": 400, "weight": 0.2},
                    {"min_hz": 400, "max_hz": 1500, "weight": 0.5},
                    {"min_hz": 1500, "max_hz": 6000, "weight": 0.3}
                ]
            }
        }
    
    def configure_custom(self) -> Dict:
        """Configure custom setup."""
        print("\nConfiguring Custom Setup...")
        print("You will need to manually enter microphone positions.")
        
        microphones = []
        for i in range(1, 9):
            print(f"\nMicrophone {i}:")
            x = self.get_float_input(f"  X position (meters)")
            y = self.get_float_input(f"  Y position (meters)")
            z = self.get_float_input(f"  Z position (meters)")
            name = self.get_user_input(f"  Name", f"Microphone {i}")
            
            microphones.append({
                "id": i,
                "x": x,
                "y": y,
                "z": z,
                "name": name
            })
        
        # Get system parameters
        print("\nSystem Configuration:")
        threshold = self.get_float_input("Detection threshold (dB)", -25.0)
        confidence = self.get_float_input("Minimum confidence", 0.6)
        
        return {
            "microphones": microphones,
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0,
                "detection_threshold_db": threshold,
                "min_confidence": confidence,
                "max_processing_latency_ms": 500
            },
            "detection": {
                "amplitude_threshold_db": threshold,
                "frequency_bands": [
                    {"min_hz": 100, "max_hz": 500, "weight": 0.3},
                    {"min_hz": 500, "max_hz": 2000, "weight": 0.5},
                    {"min_hz": 2000, "max_hz": 8000, "weight": 0.2}
                ]
            }
        }
    
    def add_common_settings(self, config: Dict) -> Dict:
        """Add common settings to configuration."""
        config.update({
            "audio": {
                "device_index": None,
                "channels": 8,
                "chunk_size": 1024,
                "enable_monitoring": True
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
            },
            "logging": {
                "level": "INFO",
                "enable_file_logging": True,
                "log_directory": "logs",
                "max_log_size_mb": 100,
                "backup_count": 10
            }
        })
        
        return config
    
    def save_configuration(self, config: Dict, filename: str):
        """Save configuration to file."""
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        config_file = self.config_dir / filename
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to: {config_file}")
        return config_file
    
    def display_configuration_summary(self, config: Dict):
        """Display configuration summary."""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"Microphones: {len(config['microphones'])}")
        print(f"Detection Threshold: {config['system']['detection_threshold_db']} dB")
        print(f"Minimum Confidence: {config['system']['min_confidence']}")
        print(f"Sample Rate: {config['system']['sample_rate']} Hz")
        
        print("\nMicrophone Positions:")
        for mic in config['microphones']:
            print(f"  {mic['id']}: ({mic['x']:.1f}, {mic['y']:.1f}, {mic['z']:.1f}) - {mic['name']}")
        
        # Calculate coverage area
        x_coords = [mic['x'] for mic in config['microphones']]
        y_coords = [mic['y'] for mic in config['microphones']]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        print(f"\nCoverage Area: {x_range:.1f}m x {y_range:.1f}m")
    
    def run_setup(self):
        """Run the complete setup process."""
        self.print_header()
        
        # Select scenario
        scenario = self.select_deployment_scenario()
        
        # Configure based on scenario
        if "Indoor Small Room" in scenario:
            config = self.configure_indoor_small()
        elif "Indoor Large Room" in scenario:
            config = self.configure_indoor_large()
        elif "Outdoor Perimeter" in scenario:
            config = self.configure_outdoor_perimeter()
        elif "Large Venue" in scenario:
            config = self.configure_large_venue()
        else:  # Custom
            config = self.configure_custom()
        
        # Add common settings
        config = self.add_common_settings(config)
        
        # Display summary
        self.display_configuration_summary(config)
        
        # Confirm and save
        print("\n" + "=" * 60)
        confirm = input("Save this configuration? (y/n): ").lower().strip()
        
        if confirm == 'y':
            filename = self.get_user_input("Configuration filename", "quick_setup_config.json")
            if not filename.endswith('.json'):
                filename += '.json'
            
            config_file = self.save_configuration(config, filename)
            
            print("\n🎉 Setup completed successfully!")
            print("\nNext steps:")
            print(f"1. Review configuration: {config_file}")
            print("2. Run system calibration: python calibrate_system.py --config " + str(config_file))
            print("3. Start system: python main.py --config " + str(config_file))
            
            return True
        else:
            print("Setup cancelled.")
            return False


def main():
    """Main setup function."""
    try:
        setup = QuickSetup()
        success = setup.run_setup()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        return 1
    except Exception as e:
        print(f"\nError during setup: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())