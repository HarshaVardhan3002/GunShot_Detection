"""
Main entry point for gunshot localization system.
"""
import argparse
import sys
from pathlib import Path

from utils import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-assisted gunshot localization system"
    )
    
    parser.add_argument(
        "--mic_config",
        type=str,
        default="config/mic_positions.json",
        help="Path to microphone configuration file"
    )
    
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=48000,
        help="Audio sampling rate in Hz"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        help="Optional log file path"
    )
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting gunshot localization system")
    
    # Validate configuration file exists
    config_path = Path(args.mic_config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Using configuration: {config_path}")
    logger.info(f"Sampling rate: {args.sampling_rate} Hz")
    
    # TODO: Initialize and start system components
    logger.info("System initialization complete")
    
    try:
        # TODO: Main processing loop
        logger.info("Starting main processing loop")
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info("Gunshot localization system stopped")


if __name__ == "__main__":
    main()