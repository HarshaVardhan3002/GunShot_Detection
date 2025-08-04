# Installation and Setup Guide

## Gunshot Localization System

This document provides comprehensive instructions for installing and setting up the gunshot localization system.

## Quick Start

### Option 1: Automated Installation

```bash
# Run the automated installation script
python install.py

# Follow the prompts and verify each step completes successfully
```

### Option 2: Quick Setup Wizard

```bash
# Run the quick setup wizard for common scenarios
python quick_setup.py

# Select your deployment scenario and follow the guided setup
```

### Option 3: Manual Installation

Follow the detailed guides for custom installations:

- [Hardware Setup Guide](HARDWARE_SETUP.md) - Hardware requirements and microphone positioning
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Complete deployment procedures

## Installation Scripts Overview

### 1. `install.py` - Main Installation Script

**Purpose**: Automated installation and system setup

**Features**:

- Python version compatibility check
- Dependency installation from requirements.txt
- Audio system verification
- Directory structure creation
- Default configuration generation
- Startup script creation
- Calibration tool setup

**Usage**:

```bash
python install.py
```

**What it does**:

1. ✅ Checks Python 3.8+ compatibility
2. ✅ Installs required Python packages
3. ⚠️ Verifies audio hardware (warns if no 8-channel interface found)
4. ✅ Creates necessary directories (config/, logs/, data/, calibration/)
5. ✅ Generates default configuration files
6. ✅ Creates startup scripts (start_system.sh, start_system.bat)
7. ✅ Sets up calibration tools

### 2. `quick_setup.py` - Guided Configuration Wizard

**Purpose**: Interactive setup for common deployment scenarios

**Supported Scenarios**:

- Indoor Small Room (4m x 4m)
- Indoor Large Room (10m x 10m)
- Outdoor Perimeter (20m x 20m)
- Large Venue/Facility (50m x 30m)
- Custom Configuration

**Usage**:

```bash
python quick_setup.py
```

**Features**:

- Interactive microphone positioning
- Scenario-specific optimization
- Automatic parameter tuning
- Configuration validation
- Summary and preview

### 3. `calibrate_system.py` - System Validation Tool

**Purpose**: Validate installation and calibrate system

**Usage**:

```bash
python calibrate_system.py --config config/your_config.json
```

**Validation Checks**:

- Microphone position validation
- Audio capture functionality
- System configuration verification
- Performance benchmarking

## Configuration Files

### Default Configurations

The installation creates several configuration templates:

#### `config/default_config.json`

- General-purpose configuration
- 8-microphone square array setup
- Balanced detection parameters

#### `config/indoor_small_room.json`

- Optimized for small indoor spaces
- Higher sensitivity settings
- Reduced false positive filtering

#### `config/outdoor_perimeter.json`

- Optimized for outdoor monitoring
- Weather-resistant parameters
- Long-range detection settings

#### `config/large_venue.json`

- Optimized for large facilities
- Wide-area coverage
- High-performance settings

### Configuration Structure

```json
{
  "microphones": [
    {
      "id": 1,
      "x": 0.0,
      "y": 0.0,
      "z": 1.5,
      "name": "Microphone 1"
    }
    // ... 8 microphones total
  ],
  "system": {
    "sample_rate": 48000,
    "sound_speed": 343.0,
    "detection_threshold_db": -25.0,
    "min_confidence": 0.6
  },
  "audio": {
    "device_index": null,
    "channels": 8,
    "chunk_size": 1024
  },
  "detection": {
    "amplitude_threshold_db": -25.0,
    "frequency_bands": [
      { "min_hz": 100, "max_hz": 500, "weight": 0.3 },
      { "min_hz": 500, "max_hz": 2000, "weight": 0.5 },
      { "min_hz": 2000, "max_hz": 8000, "weight": 0.2 }
    ]
  }
}
```

## System Requirements

### Hardware Requirements

- **Computer**: Intel i5/AMD Ryzen 5 or better
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Audio Interface**: 8-channel audio interface with phantom power
- **Microphones**: 8 microphones (condenser recommended)

### Software Requirements

- **Operating System**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 or higher
- **Dependencies**: Listed in requirements.txt

### Audio Hardware

**Recommended Audio Interfaces**:

- Focusrite Scarlett 18i20 (3rd Gen) - ~$500
- PreSonus Studio 1824c - ~$400
- MOTU 8M - ~$800
- Behringer U-PHORIA UMC1820 - ~$200

**Recommended Microphones**:

- Audio-Technica AT2020 - ~$100 each
- Rode PodMic - ~$200 each
- Shure SM57 - ~$100 each

## Installation Process

### Step 1: System Preparation

```bash
# Update system packages (Ubuntu/Debian)
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install python3 python3-pip python3-venv build-essential portaudio19-dev

# Create project directory
mkdir -p /opt/gunshot-localizer
cd /opt/gunshot-localizer
```

### Step 2: Python Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install System

```bash
# Copy system files to project directory
# (Copy all files from the gunshot-localizer directory)

# Run installation
python install.py
```

### Step 4: Configure System

```bash
# Option A: Use quick setup wizard
python quick_setup.py

# Option B: Manually edit configuration
nano config/default_config.json
```

### Step 5: Validate Installation

```bash
# Run system validation
python calibrate_system.py --config config/default_config.json

# Run performance tests
python run_performance_tests.py
```

### Step 6: Start System

```bash
# Start system manually
python main.py --config config/default_config.json

# Or use startup script
./start_system.sh        # Linux/macOS
start_system.bat         # Windows
```

## Troubleshooting

### Common Installation Issues

#### Python Version Error

**Problem**: "Python 3.8+ required"
**Solution**:

```bash
# Install Python 3.8+
sudo apt install python3.8 python3.8-venv python3.8-dev
# or download from python.org
```

#### Audio System Not Found

**Problem**: "No audio devices with 8+ input channels found"
**Solution**:

- Connect multi-channel audio interface
- Install audio interface drivers
- Verify device recognition:

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

#### Permission Errors

**Problem**: Permission denied during installation
**Solution**:

```bash
# Fix directory permissions
sudo chown -R $USER:$USER /opt/gunshot-localizer

# Or install in user directory
mkdir ~/gunshot-localizer
cd ~/gunshot-localizer
```

#### Dependency Installation Fails

**Problem**: pip install errors
**Solution**:

```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install system dependencies first
sudo apt install build-essential python3-dev portaudio19-dev

# Try installing dependencies individually
pip install numpy scipy sounddevice
```

### Configuration Issues

#### Invalid Microphone Positions

**Problem**: "Microphone configuration validation failed"
**Solution**:

- Verify all 8 microphones are defined
- Check coordinate format (x, y, z as numbers)
- Ensure minimum spacing between microphones
- Run validation:

```bash
python calibrate_system.py --config config/your_config.json
```

#### Audio Device Not Detected

**Problem**: "Audio capture test failed"
**Solution**:

- Check audio interface connection
- Verify drivers are installed
- Test device manually:

```bash
python test_audio_capture.py
```

## Post-Installation

### System Service Setup (Linux)

```bash
# Create systemd service
sudo nano /etc/systemd/system/gunshot-localizer.service

# Enable and start service
sudo systemctl enable gunshot-localizer
sudo systemctl start gunshot-localizer
```

### Monitoring Setup

```bash
# Set up log rotation
sudo nano /etc/logrotate.d/gunshot-localizer

# Create monitoring cron job
crontab -e
# Add: */5 * * * * /opt/gunshot-localizer/monitor_system.sh
```

### Performance Optimization

```bash
# Run performance benchmark
python run_performance_tests.py --benchmark

# Optimize system settings
# - Adjust audio buffer size
# - Configure CPU affinity
# - Set process priority
```

## Support and Documentation

### Additional Resources

- [Hardware Setup Guide](HARDWARE_SETUP.md) - Detailed hardware installation
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment procedures
- [API Documentation](API_REFERENCE.md) - Programming interface
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions

### Configuration Examples

- `config/indoor_small_room.json` - Small room setup
- `config/outdoor_perimeter.json` - Perimeter monitoring
- `config/large_venue.json` - Large facility monitoring

### Validation Tools

- `calibrate_system.py` - System validation and calibration
- `run_performance_tests.py` - Performance benchmarking
- `test_audio_capture.py` - Audio system testing

### Maintenance Scripts

- `monitor_system.sh` - System health monitoring
- `start_system.sh` / `start_system.bat` - System startup
- `backup_config.py` - Configuration backup utility

## Getting Help

### Self-Diagnosis

1. Check system logs: `tail -f logs/system.log`
2. Run validation: `python calibrate_system.py`
3. Test performance: `python run_performance_tests.py --quick`
4. Verify configuration: `python -c "import json; json.load(open('config/default_config.json'))"`

### Support Channels

- System logs and error messages
- Configuration validation output
- Performance test results
- Hardware compatibility information

---

**Note**: This installation guide covers standard deployment scenarios. For specialized requirements or custom installations, consult the detailed hardware and deployment guides, or contact technical support.
