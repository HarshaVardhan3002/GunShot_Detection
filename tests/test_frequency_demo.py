#!/usr/bin/env python3
"""
Demo script for frequency domain analysis in gunshot detection.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from gunshot_detector import AmplitudeBasedDetector


def generate_frequency_test_signals(samples, sample_rate):
    """Generate test signals with different frequency characteristics."""
    t = np.arange(samples) / sample_rate
    signals = {}
    
    # 1. Typical gunshot (dominant mid-frequencies)
    gunshot = (0.4 * np.sin(2 * np.pi * 1200 * t) +    # 1.2kHz
               0.3 * np.sin(2 * np.pi * 2800 * t) +     # 2.8kHz
               0.2 * np.sin(2 * np.pi * 800 * t) +      # 800Hz
               0.1 * np.sin(2 * np.pi * 4500 * t))      # 4.5kHz
    
    # Apply gunshot-like envelope (fast attack, exponential decay)
    envelope = np.exp(-np.arange(samples) / (samples / 8))
    signals['gunshot'] = gunshot * envelope
    
    # 2. Thunder (more low-frequency content)
    thunder = (0.5 * np.sin(2 * np.pi * 200 * t) +     # 200Hz
               0.3 * np.sin(2 * np.pi * 400 * t) +      # 400Hz
               0.2 * np.sin(2 * np.pi * 1000 * t))      # 1kHz
    thunder_envelope = np.exp(-np.arange(samples) / (samples / 15))  # Longer decay
    signals['thunder'] = thunder * thunder_envelope
    
    # 3. Car backfire (similar to gunshot but different spectral shape)
    backfire = (0.3 * np.sin(2 * np.pi * 600 * t) +    # 600Hz
                0.4 * np.sin(2 * np.pi * 1800 * t) +    # 1.8kHz
                0.2 * np.sin(2 * np.pi * 3500 * t) +    # 3.5kHz
                0.1 * np.sin(2 * np.pi * 7000 * t))     # 7kHz
    backfire_envelope = np.exp(-np.arange(samples) / (samples / 6))
    signals['car_backfire'] = backfire * backfire_envelope
    
    # 4. Fireworks (broader frequency content)
    fireworks = (0.2 * np.sin(2 * np.pi * 500 * t) +   # 500Hz
                 0.3 * np.sin(2 * np.pi * 1500 * t) +   # 1.5kHz
                 0.3 * np.sin(2 * np.pi * 3000 * t) +   # 3kHz
                 0.2 * np.sin(2 * np.pi * 6000 * t))    # 6kHz
    fireworks_envelope = np.exp(-np.arange(samples) / (samples / 10))
    signals['fireworks'] = fireworks * fireworks_envelope
    
    # 5. Door slam (very low frequency dominant)
    door_slam = (0.6 * np.sin(2 * np.pi * 100 * t) +   # 100Hz
                 0.3 * np.sin(2 * np.pi * 300 * t) +    # 300Hz
                 0.1 * np.sin(2 * np.pi * 800 * t))     # 800Hz
    door_envelope = np.exp(-np.arange(samples) / (samples / 12))
    signals['door_slam'] = door_slam * door_envelope
    
    # 6. High-frequency noise (electronic interference)
    hf_noise = (0.4 * np.sin(2 * np.pi * 8000 * t) +   # 8kHz
                0.3 * np.sin(2 * np.pi * 12000 * t) +   # 12kHz
                0.3 * np.sin(2 * np.pi * 15000 * t))    # 15kHz
    hf_envelope = np.exp(-np.arange(samples) / (samples / 4))  # Very fast decay
    signals['hf_noise'] = hf_noise * hf_envelope
    
    return signals


def test_frequency_analysis_comparison():
    """Compare detection with and without frequency analysis."""
    print("=== Frequency Analysis Comparison Test ===")
    
    detector = AmplitudeBasedDetector(
        sample_rate=48000,
        channels=8,
        threshold_db=-25.0
    )
    
    samples = 2048
    test_signals = generate_frequency_test_signals(samples, 48000)
    
    results = {}
    
    for signal_name, signal in test_signals.items():
        print(f"\nTesting: {signal_name}")
        
        # Create multi-channel version
        multi_channel_signal = np.zeros((samples, 8), dtype=np.float32)
        for ch in range(6):  # Use 6 channels
            # Add slight variations between channels
            variation = 1.0 + (ch - 3) * 0.05  # ±15% variation
            multi_channel_signal[:, ch] = signal * variation * 0.3  # Scale amplitude
        
        # Test with frequency analysis enabled
        detector.set_frequency_analysis_enabled(True)
        detected_freq, conf_freq, meta_freq = detector.detect_gunshot(multi_channel_signal)
        
        # Wait for cooldown
        time.sleep(0.6)
        
        # Test with frequency analysis disabled
        detector.set_frequency_analysis_enabled(False)
        detected_amp, conf_amp, meta_amp = detector.detect_gunshot(multi_channel_signal)
        
        results[signal_name] = {
            'freq_detected': detected_freq,
            'freq_confidence': conf_freq,
            'amp_detected': detected_amp,
            'amp_confidence': conf_amp,
            'freq_metadata': meta_freq,
            'amp_metadata': meta_amp
        }
        
        print(f"  Amplitude only: Detected={detected_amp}, Confidence={conf_amp:.3f}")
        print(f"  With frequency: Detected={detected_freq}, Confidence={conf_freq:.3f}")
        
        if detected_freq and 'frequency_profile' in meta_freq:
            freq_profile = meta_freq['frequency_profile']
            print(f"  Frequency bands: Low={freq_profile.get('low', 0):.2f}, "
                  f"Mid={freq_profile.get('mid', 0):.2f}, High={freq_profile.get('high', 0):.2f}")
            print(f"  Gunshot similarity: {meta_freq.get('gunshot_similarity', 0):.3f}")
        
        # Wait for cooldown
        time.sleep(0.6)
    
    return results


def test_frequency_band_analysis():
    """Test detailed frequency band analysis."""
    print("\n=== Frequency Band Analysis Test ===")
    
    detector = AmplitudeBasedDetector(sample_rate=48000, channels=8)
    detector.set_frequency_analysis_enabled(True)
    
    samples = 2048
    test_signals = generate_frequency_test_signals(samples, 48000)
    
    print(f"Frequency bands: {detector.frequency_bands}")
    print(f"Gunshot signature: {detector.gunshot_signature}")
    
    for signal_name, signal in test_signals.items():
        print(f"\nAnalyzing: {signal_name}")
        
        # Create multi-channel signal
        multi_channel = np.zeros((samples, 8), dtype=np.float32)
        multi_channel[:, 0] = signal * 0.4  # Use first channel
        
        # Analyze frequency profile
        freq_analysis = detector.analyze_frequency_profile(multi_channel, channel=0)
        
        if 'frequency_profile' in freq_analysis:
            profile = freq_analysis['frequency_profile']
            print(f"  Frequency distribution:")
            for band, energy in profile.items():
                print(f"    {band}: {energy:.3f}")
            
            print(f"  Spectral centroid: {freq_analysis.get('spectral_centroid', 0):.1f} Hz")
            print(f"  Spectral rolloff: {freq_analysis.get('spectral_rolloff', 0):.1f} Hz")
            print(f"  Gunshot similarity: {freq_analysis.get('gunshot_similarity', 0):.3f}")
            print(f"  Frequency confidence: {freq_analysis.get('frequency_confidence', 0):.3f}")
    
    return detector


def test_custom_frequency_bands():
    """Test custom frequency band configuration."""
    print("\n=== Custom Frequency Bands Test ===")
    
    detector = AmplitudeBasedDetector(sample_rate=48000, channels=8)
    
    # Define custom frequency bands for specific application
    custom_bands = {
        'sub_low': (20, 200),      # Very low frequencies
        'low': (200, 800),         # Low frequencies
        'mid_low': (800, 2000),    # Mid-low frequencies
        'mid_high': (2000, 5000),  # Mid-high frequencies
        'high': (5000, 12000),     # High frequencies
        'ultra_high': (12000, 24000)  # Ultra-high frequencies
    }
    
    detector.configure_frequency_bands(custom_bands)
    print(f"Custom frequency bands configured: {detector.frequency_bands}")
    
    # Define custom gunshot signature for these bands
    custom_signature = {
        'sub_low': 0.05,
        'low': 0.15,
        'mid_low': 0.35,
        'mid_high': 0.30,
        'high': 0.12,
        'ultra_high': 0.03
    }
    
    detector.set_gunshot_signature(custom_signature)
    print(f"Custom gunshot signature: {detector.gunshot_signature}")
    
    # Test with a signal
    samples = 2048
    t = np.arange(samples) / 48000
    test_signal = (0.3 * np.sin(2 * np.pi * 1500 * t) +  # Mid-low
                   0.4 * np.sin(2 * np.pi * 3000 * t))    # Mid-high
    
    multi_channel = np.zeros((samples, 8), dtype=np.float32)
    multi_channel[:, 0] = test_signal * 0.4
    
    detector.set_frequency_analysis_enabled(True)
    detected, confidence, metadata = detector.detect_gunshot(multi_channel)
    
    print(f"\nTest signal detection:")
    print(f"  Detected: {detected}, Confidence: {confidence:.3f}")
    
    if 'frequency_profile' in metadata:
        profile = metadata['frequency_profile']
        print(f"  Custom band energies:")
        for band, energy in profile.items():
            print(f"    {band}: {energy:.3f}")
    
    return detector


def test_spectral_features():
    """Test spectral feature extraction."""
    print("\n=== Spectral Features Test ===")
    
    detector = AmplitudeBasedDetector(sample_rate=48000, channels=8)
    
    # Create signals with known spectral characteristics
    samples = 2048
    t = np.arange(samples) / 48000
    
    test_cases = {
        'narrow_band': 0.5 * np.sin(2 * np.pi * 2000 * t),  # Single frequency
        'broad_band': np.random.random(samples) * 0.3,       # White noise
        'low_pass': 0.4 * np.sin(2 * np.pi * 500 * t),      # Low frequency
        'high_pass': 0.4 * np.sin(2 * np.pi * 8000 * t),    # High frequency
    }
    
    for case_name, signal in test_cases.items():
        print(f"\nSpectral features for {case_name}:")
        
        # Analyze spectral features directly
        freqs = np.fft.fftfreq(len(signal), 1/48000)[:len(signal)//2]
        magnitude_spectrum = np.abs(np.fft.fft(signal))[:len(signal)//2]
        
        features = detector._calculate_spectral_features(magnitude_spectrum, freqs)
        
        print(f"  Spectral centroid: {features['centroid']:.1f} Hz")
        print(f"  Spectral rolloff: {features['rolloff']:.1f} Hz")
        print(f"  Spectral flatness: {features['flatness']:.3f}")
        print(f"  Dominant frequency: {features['dominant_freq']:.1f} Hz")


def visualize_frequency_analysis(detector, test_signals):
    """Create visualizations of frequency analysis results."""
    print("\n=== Creating Frequency Analysis Visualizations ===")
    
    try:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Frequency Domain Analysis of Different Signals')
        
        signal_names = list(test_signals.keys())[:6]  # First 6 signals
        
        for i, signal_name in enumerate(signal_names):
            row = i // 2
            col = i % 2
            
            signal = test_signals[signal_name]
            
            # Create multi-channel version
            multi_channel = np.zeros((len(signal), 8), dtype=np.float32)
            multi_channel[:, 0] = signal * 0.4
            
            # Analyze frequency
            freq_analysis = detector.analyze_frequency_profile(multi_channel, channel=0)
            
            if 'frequency_profile' in freq_analysis:
                # Plot frequency band energies
                bands = list(freq_analysis['frequency_profile'].keys())
                energies = list(freq_analysis['frequency_profile'].values())
                
                bars = axes[row, col].bar(bands, energies)
                axes[row, col].set_title(f'{signal_name.replace("_", " ").title()}\n'
                                       f'Similarity: {freq_analysis.get("gunshot_similarity", 0):.2f}')
                axes[row, col].set_ylabel('Energy Ratio')
                axes[row, col].tick_params(axis='x', rotation=45)
                
                # Color bars based on gunshot signature
                gunshot_sig = detector.gunshot_signature
                for bar, band in zip(bars, bands):
                    expected_energy = gunshot_sig.get(band, 0)
                    actual_energy = freq_analysis['frequency_profile'][band]
                    
                    # Color based on how close to expected
                    if abs(actual_energy - expected_energy) < 0.1:
                        bar.set_color('green')
                    elif abs(actual_energy - expected_energy) < 0.2:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'frequency_analysis_{int(time.time())}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Frequency analysis plot saved as: {plot_filename}")
        
        plt.close()
        
        # Create spectral comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Spectral Characteristics Comparison')
        
        # Collect data for comparison
        centroids = []
        rolloffs = []
        similarities = []
        names = []
        
        for signal_name, signal in list(test_signals.items())[:4]:
            multi_channel = np.zeros((len(signal), 8), dtype=np.float32)
            multi_channel[:, 0] = signal * 0.4
            
            freq_analysis = detector.analyze_frequency_profile(multi_channel, channel=0)
            
            centroids.append(freq_analysis.get('spectral_centroid', 0))
            rolloffs.append(freq_analysis.get('spectral_rolloff', 0))
            similarities.append(freq_analysis.get('gunshot_similarity', 0))
            names.append(signal_name.replace('_', ' ').title())
        
        # Plot comparisons
        axes[0, 0].bar(names, centroids)
        axes[0, 0].set_title('Spectral Centroid')
        axes[0, 0].set_ylabel('Frequency (Hz)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(names, rolloffs)
        axes[0, 1].set_title('Spectral Rolloff')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        bars = axes[1, 0].bar(names, similarities)
        axes[1, 0].set_title('Gunshot Similarity')
        axes[1, 0].set_ylabel('Similarity Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Color similarity bars
        for bar, sim in zip(bars, similarities):
            if sim > 0.7:
                bar.set_color('green')
            elif sim > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Plot gunshot signature reference
        sig_bands = list(detector.gunshot_signature.keys())
        sig_energies = list(detector.gunshot_signature.values())
        axes[1, 1].bar(sig_bands, sig_energies, color='blue', alpha=0.7)
        axes[1, 1].set_title('Reference Gunshot Signature')
        axes[1, 1].set_ylabel('Expected Energy Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save spectral comparison plot
        spectral_filename = f'spectral_comparison_{int(time.time())}.png'
        plt.savefig(spectral_filename, dpi=150, bbox_inches='tight')
        print(f"Spectral comparison plot saved as: {spectral_filename}")
        
        plt.close()
        
    except Exception as e:
        print(f"Visualization error (matplotlib may not be available): {e}")


def main():
    """Main demo function."""
    print("Frequency Domain Analysis Demo")
    print("=" * 50)
    
    try:
        # Test 1: Comparison of detection with/without frequency analysis
        comparison_results = test_frequency_analysis_comparison()
        
        # Test 2: Detailed frequency band analysis
        freq_detector = test_frequency_band_analysis()
        
        # Test 3: Custom frequency bands
        custom_detector = test_custom_frequency_bands()
        
        # Test 4: Spectral features
        test_spectral_features()
        
        # Test 5: Visualizations
        samples = 2048
        test_signals = generate_frequency_test_signals(samples, 48000)
        visualize_frequency_analysis(freq_detector, test_signals)
        
        # Summary
        print("\n=== Demo Summary ===")
        print("✓ Frequency analysis comparison completed")
        print("✓ Frequency band analysis tested")
        print("✓ Custom frequency bands configured")
        print("✓ Spectral features extracted")
        print("✓ Visualizations generated")
        
        # Analysis of results
        print(f"\nFrequency Analysis Benefits:")
        gunshot_detected = comparison_results.get('gunshot', {})
        if gunshot_detected:
            freq_conf = gunshot_detected.get('freq_confidence', 0)
            amp_conf = gunshot_detected.get('amp_confidence', 0)
            print(f"  Gunshot detection confidence: Amplitude={amp_conf:.3f}, Combined={freq_conf:.3f}")
        
        # Count improved detections
        improved_count = 0
        for signal_name, result in comparison_results.items():
            if result['freq_confidence'] > result['amp_confidence']:
                improved_count += 1
        
        print(f"  Signals with improved confidence: {improved_count}/{len(comparison_results)}")
        
        # Final statistics
        final_stats = freq_detector.get_frequency_statistics()
        print(f"\nFinal Frequency Statistics:")
        print(f"  Samples analyzed: {final_stats['samples_analyzed']}")
        print(f"  Average gunshot similarity: {final_stats['avg_gunshot_similarity']:.3f}")
        print(f"  Average spectral centroid: {final_stats['avg_spectral_centroid']:.1f} Hz")
        
        print(f"\n✓ Frequency domain analysis demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()