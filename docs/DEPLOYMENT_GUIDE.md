# Deployment Guide

## Gunshot Localization System

This guide provides step-by-step instructions for deploying the gunshot localization system in various environments.

## Table of Contents

1. [Pre-Deployment Planning](#pre-deployment-planning)
2. [System Installation](#system-installation)
3. [Configuration](#configuration)
4. [Testing and Validation](#testing-and-validation)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)

## Pre-Deployment Planning

### Site Assessment

Before deployment, conduct a thorough site assessment:

#### Physical Environment

- **Coverage Area**: Define the area to be monitored
- **Obstacles**: Identify walls, buildings, trees, or other obstructions
- **Acoustic Properties**: Assess reverberation, echo, and ambient noise
- **Access Points**: Plan cable routes and equipment placement
- **Power Availability**: Ensure adequate power supply locations
- **Network Infrastructure**: Plan data connectivity requirements

#### Security Requirements

- **Detection Range**: Specify required detection distance
- **Accuracy Requirements**: Define acceptable location error margins
- **Response Time**: Specify maximum acceptable latency
- **Integration Needs**: Plan integration with existing security systems
- **Alert Mechanisms**: Define how alerts will be handled

#### Environmental Factors

- **Weather Conditions**: Consider temperature, humidity, precipitation
- **Ambient Noise**: Measure background noise levels
- **Interference Sources**: Identify potential RF or acoustic interference
- **Seasonal Variations**: Account for changing conditions

### Resource Planning

#### Personnel Requirements

- **Installation Team**: 2-4 technicians for hardware installation
- **System Administrator**: 1 person for software configuration
- **Security Personnel**: For coordination and testing
- **Project Manager**: For overall coordination

#### Equipment Checklist

- [ ] 8 microphones (with specifications from hardware guide)
- [ ] Multi-channel audio interface
- [ ] Computer system (meeting minimum requirements)
- [ ] Cables (XLR, USB, power, network)
- [ ] Mounting hardware (poles, brackets, enclosures)
- [ ] Tools (drill, cable tester, multimeter, laptop)
- [ ] Safety equipment (hard hats, safety vests, first aid)

#### Timeline Planning

- **Week 1**: Site preparation and equipment procurement
- **Week 2**: Hardware installation and cable routing
- **Week 3**: Software installation and configuration
- **Week 4**: Testing, calibration, and validation
- **Week 5**: Training and handover

## System Installation

### Step 1: Prepare Installation Environment

#### Software Prerequisites

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
# or
sudo yum update -y  # CentOS/RHEL

# Install Python 3.8+
sudo apt install python3 python3-pip python3-venv
# or
sudo yum install python3 python3-pip

# Install system dependencies
sudo apt install build-essential portaudio19-dev
# or
sudo yum install gcc portaudio-devel
```

#### Create Project Directory

```bash
# Create installation directory
sudo mkdir -p /opt/gunshot-localizer
sudo chown $USER:$USER /opt/gunshot-localizer
cd /opt/gunshot-localizer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Clone or copy system files
# (Copy all system files to this directory)
```

### Step 2: Run Installation Script

```bash
# Make installation script executable
chmod +x install.py

# Run installation
python install.py

# Follow prompts and verify each step completes successfully
```

### Step 3: Hardware Installation

#### Microphone Placement

1. **Survey Installation Points**

   - Use GPS or measuring tools to determine exact positions
   - Mark installation points according to chosen configuration
   - Verify line-of-sight between microphones

2. **Install Mounting Hardware**

   - Install poles, brackets, or ceiling mounts
   - Ensure all mounts are secure and level
   - Test mount stability under expected loads

3. **Install Microphones**
   - Mount microphones according to configuration
   - Orient microphones correctly (if directional)
   - Install windscreens for outdoor installations
   - Apply weatherproofing as needed

#### Cable Installation

1. **Route Cables**

   - Run XLR cables from each microphone to central location
   - Use cable conduits or raceways for protection
   - Maintain separation from power cables
   - Leave service loops at both ends

2. **Test Connections**
   - Test each cable with cable tester
   - Verify continuity and proper wiring
   - Check for shorts or interference
   - Label all cables clearly

#### Audio Interface Setup

1. **Connect Hardware**

   - Connect all microphone cables to audio interface
   - Connect audio interface to computer via USB/Thunderbolt
   - Connect power and network cables
   - Verify all connections are secure

2. **Install Drivers**
   - Install manufacturer's audio drivers
   - Configure sample rate to 48kHz
   - Set buffer size for optimal latency
   - Test audio input levels

## Configuration

### Step 1: System Configuration

#### Update Configuration File

```bash
# Edit the main configuration file
nano config/default_config.json
```

Key configuration parameters:

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
    // ... add all 8 microphones with exact measured positions
  ],
  "system": {
    "sample_rate": 48000,
    "sound_speed": 343.0,
    "detection_threshold_db": -25.0,
    "min_confidence": 0.6
  },
  "audio": {
    "device_index": null, // Will auto-detect
    "channels": 8,
    "chunk_size": 1024
  }
}
```

#### Environment-Specific Settings

**Indoor Configuration:**

```json
{
  "detection": {
    "amplitude_threshold_db": -30.0,
    "frequency_bands": [
      { "min_hz": 200, "max_hz": 800, "weight": 0.4 },
      { "min_hz": 800, "max_hz": 3000, "weight": 0.6 }
    ]
  },
  "system": {
    "sound_speed": 343.0,
    "min_confidence": 0.7
  }
}
```

**Outdoor Configuration:**

```json
{
  "detection": {
    "amplitude_threshold_db": -20.0,
    "frequency_bands": [
      { "min_hz": 100, "max_hz": 500, "weight": 0.3 },
      { "min_hz": 500, "max_hz": 2000, "weight": 0.5 },
      { "min_hz": 2000, "max_hz": 8000, "weight": 0.2 }
    ]
  },
  "system": {
    "sound_speed": 343.0, // Adjust for temperature/humidity
    "min_confidence": 0.5
  }
}
```

### Step 2: Audio System Configuration

#### Test Audio Input

```bash
# Test audio capture
python -c "
import sounddevice as sd
print('Available audio devices:')
print(sd.query_devices())
"

# Test microphone input levels
python test_audio_capture.py --config config/default_config.json
```

#### Calibrate Input Levels

1. **Set Gain Levels**

   - Adjust audio interface gain for each channel
   - Target: -20dB to -10dB for normal ambient noise
   - Ensure no clipping during loud events
   - Balance levels across all channels

2. **Test Dynamic Range**
   - Test with quiet and loud sounds
   - Verify adequate headroom for gunshots
   - Check for distortion or clipping
   - Document optimal gain settings

### Step 3: Network Configuration

#### Local Network Setup

```bash
# Configure static IP (if required)
sudo nano /etc/netplan/01-network-manager-all.yaml

# Example static IP configuration:
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses: [192.168.1.100/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]

# Apply network configuration
sudo netplan apply
```

#### Firewall Configuration

```bash
# Allow necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (if web interface)
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

## Testing and Validation

### Step 1: System Validation

#### Run Calibration Tool

```bash
# Run system calibration
python calibrate_system.py --config config/default_config.json

# Expected output:
# ✅ Microphone configuration is valid
# ✅ Audio capture engine initialized
# ✅ All system checks passed
```

#### Basic Functionality Test

```bash
# Start system in test mode
python main.py --config config/default_config.json --test-mode

# Monitor output for:
# - Audio input levels
# - Detection events
# - System performance metrics
```

### Step 2: Detection Testing

#### Controlled Sound Tests

1. **Clap Test**

   - Perform hand claps at known positions
   - Verify detection and location accuracy
   - Test at various distances and angles
   - Document results

2. **Recorded Gunshot Test** (if available)

   - Use recorded gunshot audio played through speakers
   - Test at multiple positions within coverage area
   - Verify detection sensitivity and accuracy
   - Measure response time

3. **False Positive Testing**
   - Test with various noise sources:
     - Vehicle backfires
     - Construction noise
     - Fireworks
     - Thunder
   - Verify system correctly rejects false positives

#### Performance Validation

```bash
# Run performance tests
python run_performance_tests.py

# Expected results:
# - Detection latency < 50ms
# - Localization latency < 500ms
# - Memory usage stable
# - No system errors
```

### Step 3: Accuracy Testing

#### Position Accuracy Test

1. **Known Position Tests**

   - Generate test sounds at surveyed positions
   - Compare detected location to actual position
   - Calculate position error statistics
   - Target: <2m error for 90% of detections

2. **Coverage Area Mapping**
   - Test detection at grid points across coverage area
   - Create detection probability map
   - Identify any coverage gaps
   - Adjust configuration if needed

#### Environmental Testing

1. **Weather Conditions** (outdoor installations)

   - Test in various weather conditions
   - Monitor performance degradation
   - Adjust thresholds if necessary

2. **Time of Day Variations**
   - Test during different times of day
   - Account for changing ambient noise
   - Document any performance variations

## Production Deployment

### Step 1: Service Configuration

#### Create System Service

```bash
# Create systemd service file
sudo nano /etc/systemd/system/gunshot-localizer.service
```

Service file content:

```ini
[Unit]
Description=Gunshot Localization System
After=network.target sound.target

[Service]
Type=simple
User=gunshot-localizer
Group=gunshot-localizer
WorkingDirectory=/opt/gunshot-localizer
Environment=PATH=/opt/gunshot-localizer/venv/bin
ExecStart=/opt/gunshot-localizer/venv/bin/python main.py --config config/default_config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable gunshot-localizer

# Start service
sudo systemctl start gunshot-localizer

# Check service status
sudo systemctl status gunshot-localizer
```

### Step 2: Monitoring Setup

#### Log Configuration

```bash
# Configure log rotation
sudo nano /etc/logrotate.d/gunshot-localizer
```

Log rotation configuration:

```
/opt/gunshot-localizer/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 gunshot-localizer gunshot-localizer
    postrotate
        systemctl reload gunshot-localizer
    endscript
}
```

#### Health Monitoring

```bash
# Create monitoring script
nano /opt/gunshot-localizer/monitor_system.sh
```

Monitoring script:

```bash
#!/bin/bash
# System health monitoring script

LOG_FILE="/opt/gunshot-localizer/logs/health.log"
CONFIG_FILE="/opt/gunshot-localizer/config/default_config.json"

# Check if service is running
if ! systemctl is-active --quiet gunshot-localizer; then
    echo "$(date): Service not running - attempting restart" >> $LOG_FILE
    systemctl restart gunshot-localizer
fi

# Check system resources
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')

echo "$(date): CPU: ${CPU_USAGE}%, Memory: ${MEMORY_USAGE}%" >> $LOG_FILE

# Check disk space
DISK_USAGE=$(df /opt/gunshot-localizer | tail -1 | awk '{print $5}' | cut -d'%' -f1)
if [ $DISK_USAGE -gt 80 ]; then
    echo "$(date): WARNING - Disk usage high: ${DISK_USAGE}%" >> $LOG_FILE
fi
```

#### Set Up Cron Job

```bash
# Add monitoring to crontab
crontab -e

# Add line:
*/5 * * * * /opt/gunshot-localizer/monitor_system.sh
```

### Step 3: Integration Setup

#### Alert Integration

Configure alert mechanisms based on requirements:

**Email Alerts:**

```python
# Add to configuration
"alerts": {
    "email": {
        "enabled": true,
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "username": "alerts@example.com",
        "password": "password",
        "recipients": ["security@example.com"]
    }
}
```

**Webhook Integration:**

```python
# Add to configuration
"alerts": {
    "webhook": {
        "enabled": true,
        "url": "https://api.example.com/alerts",
        "headers": {
            "Authorization": "Bearer token123"
        }
    }
}
```

## Monitoring and Maintenance

### Daily Operations

#### System Health Checks

```bash
# Check system status
systemctl status gunshot-localizer

# Check recent logs
tail -f /opt/gunshot-localizer/logs/system.log

# Check detection statistics
python -c "
from structured_logger import StructuredLogger
logger = StructuredLogger('system', '/opt/gunshot-localizer/logs')
stats = logger.get_log_statistics()
print(f'Detections today: {stats.get(\"detections_today\", 0)}')
"
```

#### Performance Monitoring

```bash
# Run quick performance check
python run_performance_tests.py --quick

# Check system resources
htop
df -h
```

### Weekly Maintenance

#### System Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
source /opt/gunshot-localizer/venv/bin/activate
pip install --upgrade -r requirements.txt
```

#### Log Review

```bash
# Review error logs
grep -i error /opt/gunshot-localizer/logs/*.log

# Review detection patterns
python analyze_detections.py --period week
```

### Monthly Maintenance

#### Calibration Verification

```bash
# Run full calibration check
python calibrate_system.py --config config/default_config.json --full-check

# Performance benchmark
python run_performance_tests.py --benchmark
```

#### Hardware Inspection

- Check microphone connections
- Inspect cables for damage
- Clean equipment as needed
- Verify mounting hardware integrity

## Troubleshooting

### Common Issues and Solutions

#### System Won't Start

**Symptoms:** Service fails to start
**Diagnosis:**

```bash
# Check service logs
journalctl -u gunshot-localizer -f

# Check configuration
python -c "import json; json.load(open('config/default_config.json'))"

# Test audio system
python test_audio_capture.py
```

**Solutions:**

- Verify configuration file syntax
- Check audio device availability
- Ensure proper permissions
- Restart audio services

#### Poor Detection Performance

**Symptoms:** Missing detections or false positives
**Diagnosis:**

```bash
# Check detection thresholds
grep threshold config/default_config.json

# Monitor input levels
python monitor_audio_levels.py

# Review recent detections
tail -100 logs/detections.log
```

**Solutions:**

- Adjust detection thresholds
- Recalibrate microphone positions
- Check for environmental changes
- Update noise floor estimates

#### High System Latency

**Symptoms:** Slow response times
**Diagnosis:**

```bash
# Check system performance
python run_performance_tests.py --latency-only

# Monitor system resources
top -p $(pgrep -f gunshot-localizer)
```

**Solutions:**

- Reduce audio buffer size
- Optimize system configuration
- Check for background processes
- Consider hardware upgrade

### Emergency Procedures

#### System Recovery

```bash
# Stop service
sudo systemctl stop gunshot-localizer

# Backup current configuration
cp -r config config.backup.$(date +%Y%m%d)

# Restore from known good configuration
cp config/default_config.json.backup config/default_config.json

# Restart service
sudo systemctl start gunshot-localizer
```

#### Data Recovery

```bash
# Backup logs before maintenance
tar -czf logs.backup.$(date +%Y%m%d).tar.gz logs/

# Recover from backup if needed
tar -xzf logs.backup.YYYYMMDD.tar.gz
```

### Support Contacts

#### Technical Support

- System Administrator: [contact information]
- Hardware Vendor: [contact information]
- Software Support: [contact information]

#### Emergency Contacts

- Security Team: [contact information]
- IT Support: [contact information]
- Management: [contact information]

---

**Note**: This deployment guide provides general procedures. Specific installations may require customization based on local requirements, regulations, and security policies. Always follow organizational procedures and consult with appropriate personnel before deployment.
