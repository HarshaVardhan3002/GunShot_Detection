#!/usr/bin/env python3
"""
Demo script for amplitude-based gunshot detection.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from gunshot_detector import AmplitudeBasedDetector, DetectionEvent


def generate_gunshot_signal(samples, sample_rate, amplitude=0.5, duration_ms=50):
    """Generate synthetic gunshot signal."""
    duration_samples = int(sample_rate * duration_ms / 1000)
    
    # Create impulse envelope (fast rise, exponential decay)
    t = np.arange(duration_samples) / sample_rate
    envelope = amplitude * np.exp(-t * 50)  # Exponential decay
    
    # Add frequency content typical of gunshots
    freq1 = 1000  # 1kHz component
    freq2 = 3000  # 3kHz component
    freq3 = 500   # Low frequency component
    
    signal = (envelope * (
        0.5 * np.sin(2 * np.pi * freq1 * t) +
        0.3 * np.sin(2 * np.pi * freq2 * t) +
        0.2 * np.sin(2 * np.pi * freq3 * t)
    ))
    
    # Embed in larger signal with noise
    full_signal = np.random.random(samples) * 0.001  # Background noise
    start_idx = samples // 3  # Place gunshot 1/3 into signal
    end_idx = start_idx + len(signal)
    
    if end_idx <= samples:
        full_signal[start_idx:end_idx] += signal
    
    return full_signal, start_idx


def generate_false_positive_signals(samples, sample_rate):
    """Generate signals that might cause false positives."""
    signals = {}
    
    # 1. Continuous loud noise
    signals['continuous_noise'] = np.random.random(samples) * 0.2
    
    # 2. Music/speech-like signal
    t = np.arange(samples) / sample_rate
    music_signal = (0.1 * np.sin(2 * np.pi * 440 * t) +  # A4 note
                   0.08 * np.sin(2 * np.pi * 880 * t) +   # A5 note
                   0.06 * np.sin(2 * np.pi * 220 * t))    # A3 note
    signals['music'] = music_signal + np.random.random(samples) * 0.01
    
    # 3. Very short spike (electrical noise)
    short_spike = np.random.random(samples) * 0.001
    spike_duration = 5  # Very short
    spike_start = samples // 2
    short_spike[spike_start:spike_start + spike_duration] = 0.4
    signals['short_spike'] = short_spike
    
    # 4. Very long event (machinery)
    long_event = np.random.random(samples) * 0.001
    event_duration = samples // 2  # Very long
    event_start = samples // 4
    long_event[event_start:event_start + event_duration] += 0.15
    signals['long_event'] = long_event
    
    # 5. Gradual increase (not impulsive)
    gradual = np.random.random(samples) * 0.001
    ramp_duration = samples // 3
    ramp_start = samples // 3
    for i in range(ramp_duration):
        gradual[ramp_start + i] += 0.2 * (i / ramp_duration)
    signals['gradual_increase'] = gradual
    
    return signals


def test_basic_detection():
    """Test basic gunshot detection functionality."""
    print("=== Basic Gunshot Detection Test ===")
    
    detector = AmplitudeBasedDetector(
        sample_rate=48000,
        channels=8,
        threshold_db=-20.0
    )
    
    print(f"Detector initialized: {detector.sample_rate}Hz, {detector.channels} channels")
    print(f"Detection threshold: {detector.threshold_db}dB")
    
    # Test 1: No signal (should not detect)
    print("\nTest 1: Background noise only")
    samples = 2000
    noise_only = np.random.random((samples, 8)).astype(np.float32) * 0.001
    
    detected, confidence, metadata = detector.detect_gunshot(noise_only)
    print(f"  Detected: {detected}, Confidence: {confidence:.3f}")
    
    # Test 2: Strong gunshot signal
    print("\nTest 2: Strong gunshot signal")
    gunshot_data = np.random.random((samples, 8)).astype(np.float32) * 0.001
    
    # Add gunshot to multiple channels with slight variations
    for ch in range(6):  # 6 out of 8 channels
        gunshot_signal, start_idx = generate_gunshot_signal(samples, 48000, 
                                                          amplitude=0.4 + ch*0.02)
        gunshot_data[:, ch] = gunshot_signal
    
    detected, confidence, metadata = detector.detect_gunshot(gunshot_data)
    print(f"  Detected: {detected}, Confidence: {confidence:.3f}")
    print(f"  Triggered channels: {metadata.get('triggered_channels', [])}")
    print(f"  Peak amplitude: {metadata.get('peak_amplitude', 0):.3f}")
    print(f"  Duration: {metadata.get('duration_ms', 0):.1f} ms")
    print(f"  SNR: {metadata.get('snr', 0):.1f} dB")
    
    return detector


def test_false_positive_rejection():
    """Test rejection of false positive signals."""
    print("\n=== False Positive Rejection Test ===")
    
    detector = AmplitudeBasedDetector(
        sample_rate=48000,
        channels=8,
        threshold_db=-20.0
    )
    
    samples = 2000
    false_positive_signals = generate_false_positive_signals(samples, 48000)
    
    results = {}
    
    for signal_type, signal in false_positive_signals.items():
        print(f"\nTesting: {signal_type}")
        
        # Create multi-channel version
        multi_channel_signal = np.zeros((samples, 8), dtype=np.float32)
        for ch in range(8):
            multi_channel_signal[:, ch] = signal + np.random.random(samples) * 0.0005
        
        detected, confidence, metadata = detector.detect_gunshot(multi_channel_signal)
        results[signal_type] = (detected, confidence)
        
        print(f"  Detected: {detected}, Confidence: {confidence:.3f}")
        if detected:
            print(f"  Duration: {metadata.get('duration_ms', 0):.1f} ms")
            print(f"  Rise time: {metadata.get('rise_time_ms', 0):.1f} ms")
    
    # Count false positives
    false_positives = sum(1 for detected, _ in results.values() if detected)
    print(f"\nFalse positive summary:")
    print(f"  Total tests: {len(results)}")
    print(f"  False positives: {false_positives}")
    print(f"  False positive rate: {false_positives/len(results)*100:.1f}%")
    
    return results


def test_adaptive_threshold():
    """Test adaptive threshold functionality."""
    print("\n=== Adaptive Threshold Test ===")
    
    detector = AmplitudeBasedDetector(
        sample_rate=48000,
        channels=8,
        threshold_db=-25.0
    )
    
    print(f"Initial threshold: {detector.threshold_db:.1f} dB")
    print(f"Initial noise floor: {detector.noise_floor:.6f}")
    
    # Simulate different noise environments
    noise_levels = [0.0001, 0.001, 0.005, 0.01, 0.02]
    
    for noise_level in noise_levels:
        print(f"\nSimulating noise level: {noise_level:.4f}")
        
        # Feed noisy signal to update noise floor
        for _ in range(20):  # Multiple updates to build history
            noisy_signal = np.random.random((1000, 8)).astype(np.float32) * noise_level
            detector._update_noise_floor(noisy_signal)
        
        # Set adaptive threshold
        detector.set_adaptive_threshold(detector.noise_floor)
        
        print(f"  Updated noise floor: {detector.noise_floor:.6f}")
        print(f"  Updated threshold: {detector.threshold_db:.1f} dB")
        print(f"  Threshold linear: {detector.threshold_linear:.6f}")
    
    return detector


def test_detection_parameters():
    """Test different detection parameter configurations."""
    print("\n=== Detection Parameter Test ===")
    
    # Test different configurations
    configs = [
        {'name': 'Sensitive', 'threshold_db': -30.0, 'min_duration_ms': 5, 'detection_cooldown': 0.2},
        {'name': 'Standard', 'threshold_db': -20.0, 'min_duration_ms': 10, 'detection_cooldown': 0.5},
        {'name': 'Conservative', 'threshold_db': -15.0, 'min_duration_ms': 20, 'detection_cooldown': 1.0}
    ]
    
    samples = 1500
    # Create test signal with moderate gunshot
    test_signal = np.random.random((samples, 8)).astype(np.float32) * 0.001
    gunshot_signal, _ = generate_gunshot_signal(samples, 48000, amplitude=0.15, duration_ms=30)
    
    for ch in range(4):  # 4 channels triggered
        test_signal[:, ch] = gunshot_signal
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration:")
        
        detector = AmplitudeBasedDetector(sample_rate=48000, channels=8)
        detector.configure_detection_parameters(**{k: v for k, v in config.items() if k != 'name'})
        
        detected, confidence, metadata = detector.detect_gunshot(test_signal)
        results[config['name']] = (detected, confidence, metadata)
        
        print(f"  Threshold: {detector.threshold_db:.1f} dB")
        print(f"  Detected: {detected}, Confidence: {confidence:.3f}")
        if detected:
            print(f"  Triggered channels: {metadata.get('triggered_channels', [])}")
    
    return results


def test_multi_channel_scenarios():
    """Test various multi-channel detection scenarios."""
    print("\n=== Multi-Channel Scenario Test ===")
    
    detector = AmplitudeBasedDetector(sample_rate=48000, channels=8, threshold_db=-20.0)
    
    samples = 1800
    scenarios = {
        'All channels': list(range(8)),
        'Half channels': [0, 1, 2, 3],
        'Few channels': [0, 2, 5],
        'Single channel': [3],
        'Adjacent channels': [2, 3, 4],
        'Scattered channels': [0, 2, 4, 6]
    }
    
    results = {}
    
    for scenario_name, active_channels in scenarios.items():
        print(f"\nScenario: {scenario_name} (channels {[ch+1 for ch in active_channels]})")
        
        # Create test signal
        test_signal = np.random.random((samples, 8)).astype(np.float32) * 0.001
        
        # Add gunshot to specified channels
        for ch in active_channels:
            gunshot_signal, _ = generate_gunshot_signal(samples, 48000, 
                                                      amplitude=0.25 + np.random.random()*0.1)
            test_signal[:, ch] = gunshot_signal
        
        detected, confidence, metadata = detector.detect_gunshot(test_signal)
        results[scenario_name] = (detected, confidence, len(active_channels))
        
        print(f"  Detected: {detected}, Confidence: {confidence:.3f}")
        if detected:
            print(f"  Expected channels: {len(active_channels)}, Detected: {len(metadata.get('triggered_channels', []))}")
        
        # Wait for cooldown
        time.sleep(0.6)
    
    return results


def visualize_detection_results(detector, test_signal, detection_result):
    """Create visualizations of detection results."""
    print("\n=== Creating Detection Visualizations ===")
    
    try:
        detected, confidence, metadata = detection_result
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Gunshot Detection Analysis (Detected: {detected}, Confidence: {confidence:.2f})')
        
        # Plot 1: Multi-channel waveforms
        sample_range = slice(0, min(2000, test_signal.shape[0]))
        for ch in range(min(4, test_signal.shape[1])):
            axes[0, 0].plot(test_signal[sample_range, ch], label=f'Ch {ch+1}', alpha=0.7)
        
        axes[0, 0].set_title('Multi-Channel Waveforms')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Detection statistics
        stats = detector.get_detection_statistics()
        stat_names = ['Avg Confidence', 'Avg SNR (dB)', 'Noise Floor']
        stat_values = [
            stats['avg_confidence'],
            stats['avg_snr_db'] / 50,  # Normalize for display
            stats['current_noise_floor'] * 1000  # Scale for visibility
        ]
        
        bars = axes[0, 1].bar(stat_names, stat_values)
        axes[0, 1].set_title('Detection Statistics')
        axes[0, 1].set_ylabel('Normalized Value')
        
        # Color bars based on values
        colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in stat_values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Plot 3: Signal envelope (first channel)
        if test_signal.shape[1] > 0:
            envelope = detector._calculate_envelope(test_signal[:, 0])
            axes[1, 0].plot(envelope, 'b-', label='Envelope')
            axes[1, 0].axhline(y=detector.threshold_linear, color='r', linestyle='--', label='Threshold')
            axes[1, 0].set_title('Signal Envelope (Channel 1)')
            axes[1, 0].set_xlabel('Sample')
            axes[1, 0].set_ylabel('Amplitude')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Detection metadata
        if detected and metadata:
            meta_names = ['Duration (ms)', 'Rise Time (ms)', 'Peak Amp', 'SNR (dB)']
            meta_values = [
                metadata.get('duration_ms', 0),
                metadata.get('rise_time_ms', 0),
                metadata.get('peak_amplitude', 0) * 100,  # Scale for visibility
                metadata.get('snr', 0)
            ]
            
            axes[1, 1].bar(meta_names, meta_values)
            axes[1, 1].set_title('Detection Metadata')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Detection', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=16)
            axes[1, 1].set_title('Detection Metadata')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'detection_analysis_{int(time.time())}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Detection analysis plot saved as: {plot_filename}")
        
        plt.close()
        
    except Exception as e:
        print(f"Visualization error (matplotlib may not be available): {e}")


def main():
    """Main demo function."""
    print("Amplitude-Based Gunshot Detection Demo")
    print("=" * 50)
    
    try:
        # Test 1: Basic detection
        detector = test_basic_detection()
        
        # Test 2: False positive rejection
        fp_results = test_false_positive_rejection()
        
        # Test 3: Adaptive threshold
        adaptive_detector = test_adaptive_threshold()
        
        # Test 4: Parameter configurations
        param_results = test_detection_parameters()
        
        # Test 5: Multi-channel scenarios
        mc_results = test_multi_channel_scenarios()
        
        # Test 6: Create visualization
        print("\n=== Creating Test Visualization ===")
        samples = 2000
        vis_signal = np.random.random((samples, 8)).astype(np.float32) * 0.001
        
        # Add clear gunshot to multiple channels
        for ch in range(5):
            gunshot_signal, _ = generate_gunshot_signal(samples, 48000, amplitude=0.3)
            vis_signal[:, ch] = gunshot_signal
        
        vis_result = detector.detect_gunshot(vis_signal)
        visualize_detection_results(detector, vis_signal, vis_result)
        
        # Summary
        print("\n=== Demo Summary ===")
        print("✓ Basic detection functionality tested")
        print(f"✓ False positive rejection: {sum(1 for d, _ in fp_results.values() if d)}/{len(fp_results)} false positives")
        print("✓ Adaptive threshold functionality verified")
        print(f"✓ Parameter configurations tested: {len(param_results)} configs")
        print(f"✓ Multi-channel scenarios: {len(mc_results)} scenarios tested")
        print("✓ Visualization generated")
        
        # Final statistics
        final_stats = detector.get_detection_statistics()
        print(f"\nFinal Detector Statistics:")
        print(f"  Total detections: {final_stats['total_detections']}")
        print(f"  Average confidence: {final_stats['avg_confidence']:.3f}")
        print(f"  Current threshold: {final_stats['current_threshold_db']:.1f} dB")
        print(f"  Noise floor: {final_stats['current_noise_floor']:.6f}")
        
        print(f"\n✓ Amplitude-based gunshot detection demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()