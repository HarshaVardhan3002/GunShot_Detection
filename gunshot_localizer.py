#!/usr/bin/env python3
"""
Main entry point for the gunshot localization system.

This script provides the command-line interface for running the gunshot
localization system with various options and modes.
"""
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli_interface import main

if __name__ == '__main__':
    sys.exit(main())