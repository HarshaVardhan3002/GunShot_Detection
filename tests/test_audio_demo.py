#!/usr/bin/env python3
"""
Demo script for AudioCaptureEngine functionality.
"""
import time
import numpy as np
from audio_capture import AudioCaptureEngine


def main():
    """Demo the audio capture engine."""
    print("AudioCaptureEngine Demo")
    print("=" * 40)
    
    # Create audio capture engine
    engine = AudioCaptureEngine(
        sample_rate=48000,
        channels=8,
        buffer_duration=2.0
    )
    
    print(f"Initialized: {engine.sample_rate}Hz, {engine.channels} channels")
    
    # List available audio devices
    print("\nAvailable compatible audio devices:")
    devices = engine.list_audio_devices()
    if devices:
        for device in devices:
            print(f"  ID {device['id']}: {device['name']} ({device['channels']} channels)")
    else:
        print("  No compatible devices found")
    
    # Test device compatibility (if devices available)
    if devices:
        device_id = devices[0]['id']
        compatible = engine.test_device_compatibility(device_id)
        print(f"\nDevice {device_id} compatibility: {'✓' if compatible else '✗'}")
    
    print("\nNote: This demo requires actual audio hardware to capture real audio.")
    print("For testing without hardware, use the unit tests instead.")
    
    # Demonstrate mock capture (without actual hardware)
    print("\nDemonstrating buffer operations...")
    
    # Simulate audio callback with synthetic data
    engine._capturing = True
    
    # Generate synthetic audio data
    frames = 1024
    synthetic_audio = np.random.random((frames, 8)).astype(np.float32) * 0.1
    
    # Simulate audio callback
    engine._audio_callback(synthetic_audio, frames, None, None)
    
    # Get buffer
    buffer = engine.get_audio_buffer()
    if buffer:
        print(f"Buffer captured: {buffer.data.shape} samples")
        print(f"Buffer duration: {buffer.duration}s")
        print(f"Sample rate: {buffer.sample_rate}Hz")
        print(f"Timestamp: {buffer.timestamp}")
    
    # Show channel status
    status = engine.get_channel_status()
    print(f"\nChannel status:")
    for channel_id, active in status.items():
        print(f"  Channel {channel_id}: {'Active' if active else 'Inactive'}")
    
    # Show statistics
    stats = engine.get_capture_statistics()
    print(f"\nCapture statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    engine._capturing = False
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()