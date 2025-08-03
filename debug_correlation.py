#!/usr/bin/env python3
"""
Debug cross-correlation calculation.
"""
import numpy as np
from scipy import signal

def test_correlation():
    # Create simple test signals
    samples = 100
    signal1 = np.sin(2 * np.pi * 10 * np.arange(samples) / 100)
    
    # Create delayed version
    delay_samples = 5
    signal2 = np.zeros(samples)
    signal2[delay_samples:] = signal1[:-delay_samples]
    
    print(f"Original delay: {delay_samples} samples")
    
    # Test correlation
    correlation = signal.correlate(signal1, signal2, mode='full')
    peak_index = np.argmax(np.abs(correlation))
    
    print(f"Correlation length: {len(correlation)}")
    print(f"Peak index: {peak_index}")
    print(f"Signal1 length: {len(signal1)}")
    print(f"Signal2 length: {len(signal2)}")
    
    # Calculate delay
    zero_lag_index = len(signal2) - 1
    delay_calculated = peak_index - zero_lag_index
    
    print(f"Zero lag index: {zero_lag_index}")
    print(f"Calculated delay: {delay_calculated} samples")
    print(f"Expected delay: {-delay_samples} samples (negative because signal2 is delayed)")

if __name__ == "__main__":
    test_correlation()