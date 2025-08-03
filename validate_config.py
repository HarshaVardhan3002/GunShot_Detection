#!/usr/bin/env python3
"""
Configuration validation utility for gunshot localization system.
"""
import argparse
import sys
from pathlib import Path

from config_manager import ConfigurationManager


def main():
    """Main validation utility."""
    parser = argparse.ArgumentParser(
        description="Validate gunshot localization system configuration"
    )
    
    parser.add_argument(
        "config_file",
        help="Path to configuration file to validate"
    )
    
    parser.add_argument(
        "--generate-template",
        type=str,
        choices=["default", "square", "circular", "linear"],
        help="Generate a configuration template of specified type"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for generated template"
    )
    
    args = parser.parse_args()
    
    config_manager = ConfigurationManager()
    
    # Generate template if requested
    if args.generate_template:
        output_path = args.output or f"config_template_{args.generate_template}.json"
        success = config_manager.generate_config_template(output_path, args.generate_template)
        if success:
            print(f"✓ Template generated: {output_path}")
        else:
            print(f"✗ Failed to generate template")
            sys.exit(1)
        return
    
    # Validate configuration file
    config_path = Path(args.config_file)
    
    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        sys.exit(1)
    
    print(f"Validating configuration: {config_path}")
    print("-" * 50)
    
    # Load configuration
    load_success = config_manager.load_config(str(config_path))
    if not load_success:
        print("✗ Failed to load configuration file")
        sys.exit(1)
    
    print("✓ Configuration file loaded successfully")
    
    # Validate configuration
    is_valid, errors = config_manager.validate_config()
    
    if is_valid:
        print("✓ Configuration is valid")
        
        # Display configuration summary
        positions = config_manager.get_microphone_positions()
        system_config = config_manager.get_system_config()
        
        print("\nConfiguration Summary:")
        print(f"  Microphones: {len(positions)}")
        print(f"  Sample Rate: {system_config.sample_rate} Hz")
        print(f"  Sound Speed: {system_config.sound_speed} m/s")
        print(f"  Detection Threshold: {system_config.detection_threshold_db} dB")
        print(f"  Buffer Duration: {system_config.buffer_duration} s")
        print(f"  Min Confidence: {system_config.min_confidence}")
        
        print("\nMicrophone Positions:")
        for pos in positions:
            print(f"  Mic {pos.id}: ({pos.x:6.1f}, {pos.y:6.1f}, {pos.z:6.1f})")
        
    else:
        print("✗ Configuration validation failed")
        print("\nErrors found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()