# Hardware Setup Guide

## Gunshot Localization System

This guide provides detailed instructions for setting up the hardware components of the gunshot localization system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Audio Hardware](#audio-hardware)
3. [Microphone Selection](#microphone-selection)
4. [Microphone Positioning](#microphone-positioning)
5. [Cable Management](#cable-management)
6. [Environmental Considerations](#environmental-considerations)
7. [Installation Scenarios](#installation-scenarios)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Hardware Requirements

- **Computer**: Intel i5 or AMD Ryzen 5 (or equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for system and logs
- **USB Ports**: 1 available USB port for audio interface
- **Network**: Ethernet connection recommended for remote monitoring

### Recommended Hardware

- **Computer**: Intel i7 or AMD Ryzen 7 with dedicated GPU
- **RAM**: 32GB for high-performance applications
- **Storage**: SSD with 50GB+ free space
- **Network**: Gigabit Ethernet for real-time data streaming

## Audio Hardware

### Multi-Channel Audio Interface

The system requires an 8-channel audio interface capable of simultaneous recording.

#### Recommended Audio Interfaces

1. **Focusrite Scarlett 18i20 (3rd Gen)**

   - 8 XLR/TRS inputs with phantom power
   - USB 2.0 connectivity
   - Sample rates up to 192kHz
   - Price: ~$500

2. **PreSonus Studio 1824c**

   - 8 XLR inputs with phantom power
   - USB-C connectivity
   - Low latency monitoring
   - Price: ~$400

3. **MOTU 8M**

   - 8 XLR inputs with phantom power
   - Thunderbolt/USB connectivity
   - Professional grade converters
   - Price: ~$800

4. **Behringer U-PHORIA UMC1820**
   - 8 XLR inputs with phantom power
   - USB 2.0 connectivity
   - Budget-friendly option
   - Price: ~$200

### Audio Interface Setup

1. **Driver Installation**

   - Install manufacturer's drivers before connecting
   - Use ASIO drivers on Windows for low latency
   - Verify driver compatibility with your OS

2. **Sample Rate Configuration**

   - Set interface to 48kHz sample rate
   - Use 24-bit depth for best quality
   - Buffer size: 256-512 samples for low latency

3. **Phantom Power**
   - Enable +48V phantom power for condenser microphones
   - Verify all channels have phantom power enabled
   - Check power consumption doesn't exceed interface limits

## Microphone Selection

### Microphone Types

#### Recommended: Condenser Microphones

**Advantages:**

- High sensitivity for distant sound detection
- Wide frequency response
- Low self-noise

**Recommended Models:**

1. **Audio-Technica AT2020** (~$100 each)

   - Cardioid pattern
   - 20Hz-20kHz frequency response
   - 144dB maximum SPL

2. **Rode PodMic** (~$200 each)

   - Dynamic microphone (no phantom power needed)
   - Broadcast quality
   - High SPL handling

3. **Shure SM57** (~$100 each)
   - Dynamic microphone
   - Extremely durable
   - Industry standard

#### Alternative: Omnidirectional Microphones

**Use Cases:**

- Indoor installations
- When precise directionality isn't critical
- Smaller coverage areas

### Microphone Specifications

- **Frequency Response**: 20Hz - 20kHz minimum
- **Sensitivity**: -40dBV/Pa or higher
- **Maximum SPL**: 130dB+ for gunshot detection
- **Self-Noise**: <20dBA
- **Pattern**: Cardioid or omnidirectional

## Microphone Positioning

### General Principles

#### 1. Array Geometry

- **Minimum Spacing**: 0.5m between microphones
- **Maximum Spacing**: 50m (depends on application)
- **Height Variation**: Use different heights for 3D localization
- **Coverage Area**: Ensure overlapping coverage zones

#### 2. Positioning Guidelines

- **Line of Sight**: Minimize obstructions between microphones
- **Reflective Surfaces**: Avoid positioning near large reflective surfaces
- **Wind Protection**: Use windscreens for outdoor installations
- **Vibration Isolation**: Mount on stable, vibration-free surfaces

### Standard Array Configurations

#### Configuration 1: Square Array (Small Area)

```
Coverage: 4m x 4m indoor space
Height: 1.5-2.5m above ground

Mic Positions:
1. (0.5, 0.5, 1.5) - Corner 1
2. (3.5, 0.5, 1.5) - Corner 2
3. (0.5, 3.5, 1.5) - Corner 3
4. (3.5, 3.5, 1.5) - Corner 4
5. (2.0, 2.0, 2.5) - Center Elevated
6. (2.0, 0.5, 1.5) - Front Center
7. (2.0, 3.5, 1.5) - Rear Center
8. (0.5, 2.0, 1.5) - Left Center
```

#### Configuration 2: Perimeter Array (Medium Area)

```
Coverage: 20m x 20m outdoor perimeter
Height: 3-4m above ground

Mic Positions:
1. (0, 0, 3)     - Corner 1
2. (20, 0, 3)    - Corner 2
3. (0, 20, 3)    - Corner 3
4. (20, 20, 3)   - Corner 4
5. (10, 10, 4)   - Center Elevated
6. (10, 0, 3)    - Front Center
7. (10, 20, 3)   - Rear Center
8. (0, 10, 3)    - Left Center
```

#### Configuration 3: Large Venue Array

```
Coverage: 50m x 30m facility
Height: 4-6m above ground

Mic Positions:
1. (5, 5, 4)     - Zone 1
2. (45, 5, 4)    - Zone 2
3. (5, 25, 4)    - Zone 3
4. (45, 25, 4)   - Zone 4
5. (25, 15, 6)   - Center High
6. (25, 5, 4)    - Front Center
7. (25, 25, 4)   - Rear Center
8. (15, 15, 4)   - Center Low
```

### Mounting Hardware

#### Indoor Mounting

- **Ceiling Mounts**: Use adjustable ceiling mounts
- **Wall Mounts**: Articulating wall mounts for positioning
- **Stands**: Heavy-duty microphone stands with boom arms
- **Shock Mounts**: Reduce vibration transmission

#### Outdoor Mounting

- **Pole Mounts**: Weather-resistant pole mounting systems
- **Building Mounts**: Secure mounting to building structures
- **Ground Stakes**: For temporary installations
- **Weatherproofing**: IP65+ rated enclosures

## Cable Management

### Cable Requirements

- **Type**: Balanced XLR cables (male to female)
- **Length**: Minimize length, maximum 100m per run
- **Quality**: Use high-quality, low-noise cables
- **Shielding**: Quad-star or similar high-rejection shielding

### Cable Routing

1. **Separation**: Keep audio cables away from power cables
2. **Protection**: Use cable conduits or raceways
3. **Strain Relief**: Proper strain relief at all connections
4. **Labeling**: Label all cables for easy identification
5. **Testing**: Test all cables before final installation

### Power Considerations

- **Phantom Power**: Ensure audio interface can supply 48V to all channels
- **Power Consumption**: Calculate total power requirements
- **UPS Backup**: Consider uninterruptible power supply for critical applications
- **Grounding**: Proper electrical grounding to prevent noise

## Environmental Considerations

### Indoor Installations

- **Acoustics**: Consider room acoustics and reverberation
- **HVAC**: Account for air conditioning noise
- **Lighting**: Avoid fluorescent lights (electrical interference)
- **Temperature**: Maintain stable temperature (10-35°C)
- **Humidity**: Control humidity (30-70% RH)

### Outdoor Installations

- **Weather Protection**: IP65+ rated equipment
- **Temperature Range**: -20°C to +60°C operating range
- **Wind**: Use windscreens and secure mounting
- **Precipitation**: Ensure water drainage from equipment
- **UV Protection**: UV-resistant materials and coatings

### Interference Sources

- **Electrical**: Power lines, transformers, motors
- **Radio Frequency**: Cell towers, WiFi, Bluetooth
- **Mechanical**: Fans, pumps, traffic
- **Environmental**: Wind, rain, wildlife

## Installation Scenarios

### Scenario 1: School Security System

**Requirements:**

- Indoor coverage of hallways and common areas
- Integration with existing security systems
- Minimal visual impact

**Recommended Setup:**

- Ceiling-mounted omnidirectional microphones
- Centralized audio interface in security office
- Network integration for alerts

### Scenario 2: Perimeter Security

**Requirements:**

- Outdoor perimeter monitoring
- Weather resistance
- Long-range detection

**Recommended Setup:**

- Pole-mounted directional microphones
- Weatherproof equipment enclosures
- Solar power with battery backup

### Scenario 3: Event Venue Monitoring

**Requirements:**

- Large area coverage
- Temporary installation capability
- Real-time location reporting

**Recommended Setup:**

- Portable microphone stands
- Wireless audio transmission
- Mobile command center

## Calibration and Testing

### Initial Setup Testing

1. **Connectivity Test**: Verify all microphones are connected
2. **Level Check**: Confirm proper input levels
3. **Synchronization**: Test timing synchronization
4. **Coverage Test**: Verify detection coverage area

### Calibration Procedure

1. **Position Measurement**: Accurately measure microphone positions
2. **Configuration Update**: Update system configuration file
3. **Test Signals**: Use known test signals for calibration
4. **Validation**: Test with actual gunshot recordings (if available)

### Performance Verification

- **Detection Range**: Test detection at various distances
- **Accuracy**: Verify location accuracy with known positions
- **False Positive Rate**: Test with various noise sources
- **Response Time**: Measure system response latency

## Troubleshooting

### Common Issues

#### No Audio Input

**Symptoms:** System shows no audio input
**Solutions:**

- Check audio interface drivers
- Verify phantom power is enabled
- Test cables and connections
- Check microphone functionality

#### Poor Detection Performance

**Symptoms:** Missing detections or false positives
**Solutions:**

- Adjust detection thresholds
- Check microphone positioning
- Verify environmental noise levels
- Calibrate system timing

#### Inaccurate Localization

**Symptoms:** Location estimates are incorrect
**Solutions:**

- Verify microphone position measurements
- Check system configuration
- Test with known source positions
- Recalibrate timing synchronization

#### High System Latency

**Symptoms:** Slow response times
**Solutions:**

- Reduce audio buffer size
- Check CPU usage
- Optimize system configuration
- Upgrade hardware if necessary

### Maintenance Schedule

#### Daily Checks

- Verify system is running
- Check log files for errors
- Monitor detection statistics

#### Weekly Checks

- Test microphone functionality
- Check cable connections
- Review system performance metrics

#### Monthly Checks

- Clean microphones and equipment
- Verify calibration accuracy
- Update system software
- Review and archive logs

#### Annual Checks

- Complete system recalibration
- Replace consumable components
- Hardware inspection and maintenance
- Performance benchmark testing

## Support and Documentation

### Additional Resources

- System configuration examples in `config/` directory
- Calibration tools: `calibrate_system.py`
- Installation script: `install.py`
- User manual and API documentation

### Technical Support

For technical support and advanced configuration assistance:

- Check system logs in `logs/` directory
- Run diagnostic tools
- Consult troubleshooting section
- Contact system administrator

---

**Note**: This guide provides general recommendations. Specific installations may require customization based on local conditions, regulations, and requirements. Always consult with acoustic engineers and security professionals for critical applications.
