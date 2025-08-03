#!/usr/bin/env python3
"""
Simple test to verify gunshot detection is working.
"""
import numpy as np
from gunshot_detector import AmplitudeBasedDetector


def main():
    detector = AmplitudeBasedDetector(sample_rate=8000, channels=4, threshold_db=-20.0)
    
    print(f"Threshold linear: {detector.threshold_linear}")
    print(f"Threshold dB: {detector.threshold_db}")
    
    # Create simple test signal
    samples = 1000
    audio_data = np.zeros((samples, 4), dtype=np.float32)
    
    # Add strong impulse that should definitely trigger
    impulse_amplitude = 0.5  # Much higher than threshold of 0.1
    impulse_start = 400
    impulse_duration = 50
    
    print(f"Adding impulse: amplitude={impulse_amplitude}, duration={impulse_duration} samples")
    
    # Add to all channels
    for ch in range(4):
        audio_data[impulse_start:impulse_start + impulse_duration, ch] = impulse_amplitude
    
    # Test detection
    detected, confidence, metadata = detector.detect_gunshot(audio_data)
    
    print(f"Detection result:")
    print(f"  Detected: {detected}")
    print(f"  Confidence: {confidence:.3f}")
    print(f"  Metadata: {metadata}")
    
    # Check individual channel analysis
    for ch in range(4):
        result = detector._analyze_channel_amplitude(audio_data[:, ch], ch)
        print(f"  Channel {ch}: peak={result['peak_amplitude']:.3f}, trigger={result['amplitude_trigger']}, conf={result['confidence']:.3f}")


if __name__ == "__main__":
    main()