#!/usr/bin/env python3
"""
Demo script for audio stream synchronization features.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from audio_capture import AudioCaptureEngine, AudioStreamSynchronizer


def generate_test_signals(samples, sample_rate, channels=8):
    """Generate test signals with known synchronization characteristics."""
    t = np.arange(samples) / sample_rate
    signals = np.zeros((samples, channels), dtype=np.float32)
    
    # Base signal: mix of frequencies typical for gunshots
    base_freq1 = 1000  # 1kHz
    base_freq2 = 3000  # 3kHz
    base_signal = (np.sin(2 * np.pi * base_freq1 * t) * 0.5 + 
                   np.sin(2 * np.pi * base_freq2 * t) * 0.3)
    
    # Channel 0: Reference signal
    signals[:, 0] = base_signal
    
    # Channel 1: Perfect copy (should have perfect sync)
    signals[:, 1] = base_signal
    
    # Channel 2: Delayed version (simulates microphone spacing)
    delay_samples = int(0.001 * sample_rate)  # 1ms delay
    signals[delay_samples:, 2] = base_signal[:-delay_samples] * 0.9
    
    # Channel 3: Phase-shifted version
    signals[:, 3] = np.sin(2 * np.pi * base_freq1 * t + np.pi/4) * 0.5
    
    # Channel 4: Attenuated version (distant microphone)
    signals[:, 4] = base_signal * 0.3 + np.random.random(samples) * 0.05
    
    # Channel 5: Noisy version
    signals[:, 5] = base_signal * 0.7 + np.random.random(samples) * 0.2
    
    # Channel 6: Different frequency content
    signals[:, 6] = np.sin(2 * np.pi * 2000 * t) * 0.4
    
    # Channel 7: Mostly noise (failed microphone)
    signals[:, 7] = np.random.random(samples) * 0.1
    
    return signals


def test_synchronization_analysis():
    """Test synchronization analysis with known signals."""
    print("=== Synchronization Analysis Test ===")
    
    # Create synchronizer
    synchronizer = AudioStreamSynchronizer(channels=8, sample_rate=48000)
    
    # Generate test signals
    samples = 2048
    test_signals = generate_test_signals(samples, 48000, 8)
    
    print(f"Generated test signals: {test_signals.shape}")
    print(f"Signal RMS levels: {np.sqrt(np.mean(test_signals**2, axis=0))}")
    
    # Analyze synchronization
    metrics = synchronizer.analyze_channel_synchronization(test_signals, time.time())
    
    print(f"\nSynchronization Analysis Results:")
    print(f"  Overall sync quality: {metrics.sync_quality_score:.3f}")
    print(f"  Phase coherence: {metrics.phase_coherence:.3f}")
    print(f"  Clock drift: {metrics.clock_drift_ms:.3f} ms")
    print(f"  Reference channel: {metrics.reference_channel}")
    
    print(f"\nChannel Alignment Scores:")
    for ch, score in metrics.channel_alignment.items():
        print(f"  Channel {ch}: {score:.3f}")
    
    # Test delay compensation
    print(f"\nTesting delay compensation...")
    compensated_signals = synchronizer.compensate_channel_delays(test_signals)
    
    print(f"Original shape: {test_signals.shape}")
    print(f"Compensated shape: {compensated_signals.shape}")
    
    # Show delay values
    diagnostics = synchronizer.get_synchronization_diagnostics()
    print(f"Channel delays (ms): {diagnostics['channel_delays_ms']}")
    
    return metrics, compensated_signals


def test_reference_channel_optimization():
    """Test finding optimal reference channel."""
    print("\n=== Reference Channel Optimization ===")
    
    synchronizer = AudioStreamSynchronizer(channels=8, sample_rate=48000)
    test_signals = generate_test_signals(1024, 48000, 8)
    
    # Test different reference channels
    results = {}
    
    for ref_ch in range(8):
        synchronizer.set_reference_channel(ref_ch)
        metrics = synchronizer.analyze_channel_synchronization(test_signals, time.time())
        
        # Calculate average alignment (excluding reference channel)
        alignment_scores = [score for ch, score in metrics.channel_alignment.items() if ch != ref_ch]
        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0
        
        results[ref_ch] = {
            'sync_quality': metrics.sync_quality_score,
            'avg_alignment': avg_alignment,
            'phase_coherence': metrics.phase_coherence
        }
        
        print(f"Reference Channel {ref_ch}:")
        print(f"  Sync Quality: {metrics.sync_quality_score:.3f}")
        print(f"  Avg Alignment: {avg_alignment:.3f}")
        print(f"  Phase Coherence: {metrics.phase_coherence:.3f}")
    
    # Find best reference channel
    best_ref = max(results.keys(), key=lambda ch: results[ch]['sync_quality'])
    print(f"\nOptimal reference channel: {best_ref}")
    print(f"Best sync quality: {results[best_ref]['sync_quality']:.3f}")
    
    return best_ref, results


def test_engine_synchronization():
    """Test synchronization integration with AudioCaptureEngine."""
    print("\n=== AudioCaptureEngine Synchronization Test ===")
    
    # Create engine
    engine = AudioCaptureEngine(
        sample_rate=48000,
        channels=8,
        buffer_duration=1.0
    )
    
    print(f"Engine initialized with sync enabled: {engine._sync_enabled}")
    
    # Enable capturing for testing
    engine._capturing = True
    
    # Test synchronization controls
    print("\nTesting synchronization controls:")
    
    # Test reference channel setting
    result = engine.set_sync_reference_channel(3)
    print(f"Set reference channel 3: {result}")
    
    # Test invalid reference channel
    result = engine.set_sync_reference_channel(10)
    print(f"Set invalid reference channel 10: {result}")
    
    # Get synchronization status
    status = engine.get_synchronization_status()
    print(f"\nSynchronization Status:")
    print(f"  Enabled: {status['sync_enabled']}")
    print(f"  Reference Channel: {status['reference_channel']}")
    print(f"  Sync Quality: {status['sync_quality']:.3f}")
    
    # Simulate audio processing with synchronization
    print(f"\nSimulating synchronized audio processing...")
    
    for i in range(5):
        # Generate test frame
        frames = 1024
        test_frame = generate_test_signals(frames, 48000, 8)
        
        # Process through synchronization
        result = engine._apply_synchronization(test_frame, time.time())
        
        print(f"Frame {i+1}: Input {test_frame.shape} -> Output {result.shape}")
        
        # Small delay between frames
        time.sleep(0.02)
    
    # Get final metrics
    final_metrics = engine.get_synchronization_metrics()
    if final_metrics:
        print(f"\nFinal Synchronization Metrics:")
        print(f"  Sync Quality: {final_metrics.sync_quality_score:.3f}")
        print(f"  Phase Coherence: {final_metrics.phase_coherence:.3f}")
        print(f"  Clock Drift: {final_metrics.clock_drift_ms:.3f} ms")
    
    engine._capturing = False
    return engine


def test_calibration_simulation():
    """Simulate synchronization calibration process."""
    print("\n=== Synchronization Calibration Simulation ===")
    
    engine = AudioCaptureEngine(
        sample_rate=48000,
        channels=8,
        buffer_duration=2.0
    )
    
    # Enable capturing
    engine._capturing = True
    
    # Simulate calibration data collection
    print("Simulating calibration data collection...")
    
    calibration_frames = []
    for i in range(20):  # 20 frames of calibration data
        frames = 1024
        # Simulate ambient noise with some correlation
        ambient_noise = np.random.random((frames, 8)) * 0.05
        
        # Add some correlated signal to simulate real environment
        if i % 5 == 0:  # Occasional correlated events
            event_signal = generate_test_signals(frames, 48000, 8) * 0.1
            ambient_noise += event_signal
        
        calibration_frames.append(ambient_noise)
        
        # Process through engine
        engine._apply_synchronization(ambient_noise, time.time())
        time.sleep(0.01)
    
    # Get calibration results
    if engine._sync_metrics_history:
        metrics_history = list(engine._sync_metrics_history)
        
        sync_scores = [m.sync_quality_score for m in metrics_history]
        coherence_scores = [m.phase_coherence for m in metrics_history]
        drift_values = [m.clock_drift_ms for m in metrics_history]
        
        print(f"\nCalibration Results:")
        print(f"  Frames analyzed: {len(metrics_history)}")
        print(f"  Average sync quality: {np.mean(sync_scores):.3f}")
        print(f"  Sync quality std: {np.std(sync_scores):.3f}")
        print(f"  Average phase coherence: {np.mean(coherence_scores):.3f}")
        print(f"  Average clock drift: {np.mean(drift_values):.3f} ms")
        print(f"  Drift stability: {1.0 - np.std(drift_values):.3f}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if np.mean(sync_scores) < 0.8:
            print("  • Consider checking microphone connections")
        if np.std(sync_scores) > 0.1:
            print("  • Synchronization quality is unstable")
        if abs(np.mean(drift_values)) > 1.0:
            print("  • Significant clock drift detected")
        if np.mean(coherence_scores) < 0.7:
            print("  • Poor phase coherence - check microphone placement")
    
    engine._capturing = False
    return engine


def visualize_synchronization_results(metrics, signals):
    """Create visualizations of synchronization results."""
    print("\n=== Creating Synchronization Visualizations ===")
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Audio Stream Synchronization Analysis')
        
        # Plot 1: Channel alignment scores
        channels = list(metrics.channel_alignment.keys())
        alignment_scores = list(metrics.channel_alignment.values())
        
        axes[0, 0].bar(channels, alignment_scores)
        axes[0, 0].set_title('Channel Alignment Scores')
        axes[0, 0].set_xlabel('Channel')
        axes[0, 0].set_ylabel('Alignment Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Signal waveforms (first 1000 samples)
        sample_range = slice(0, min(1000, signals.shape[0]))
        for ch in range(min(4, signals.shape[1])):  # Show first 4 channels
            axes[0, 1].plot(signals[sample_range, ch], label=f'Ch {ch}', alpha=0.7)
        
        axes[0, 1].set_title('Signal Waveforms')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Sync quality metrics
        metric_names = ['Sync Quality', 'Phase Coherence']
        metric_values = [metrics.sync_quality_score, metrics.phase_coherence]
        
        bars = axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Synchronization Quality Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        
        # Color bars based on quality
        for bar, value in zip(bars, metric_values):
            if value > 0.8:
                bar.set_color('green')
            elif value > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Plot 4: Cross-correlation example (channels 0 and 1)
        if signals.shape[1] >= 2:
            correlation = np.correlate(signals[:, 0], signals[:, 1], mode='full')
            correlation_range = np.arange(-len(signals[:, 1]) + 1, len(signals[:, 0]))
            
            axes[1, 1].plot(correlation_range, correlation)
            axes[1, 1].set_title('Cross-Correlation (Ch0 vs Ch1)')
            axes[1, 1].set_xlabel('Lag (samples)')
            axes[1, 1].set_ylabel('Correlation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'sync_analysis_{int(time.time())}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Synchronization analysis plot saved as: {plot_filename}")
        
        # Don't show plot in automated testing
        # plt.show()
        plt.close()
        
    except Exception as e:
        print(f"Visualization error (matplotlib may not be available): {e}")


def main():
    """Main demo function."""
    print("Audio Stream Synchronization Demo")
    print("=" * 50)
    
    try:
        # Test 1: Basic synchronization analysis
        metrics, compensated_signals = test_synchronization_analysis()
        
        # Test 2: Reference channel optimization
        best_ref, ref_results = test_reference_channel_optimization()
        
        # Test 3: Engine integration
        engine = test_engine_synchronization()
        
        # Test 4: Calibration simulation
        calibrated_engine = test_calibration_simulation()
        
        # Test 5: Export synchronization data
        print("\n=== Testing Data Export ===")
        export_file = f"sync_data_{int(time.time())}.json"
        success = engine.export_synchronization_data(export_file)
        print(f"Data export successful: {success}")
        if success:
            print(f"Synchronization data exported to: {export_file}")
        
        # Test 6: Visualizations
        visualize_synchronization_results(metrics, compensated_signals)
        
        print("\n=== Demo Summary ===")
        print(f"✓ Synchronization analysis completed")
        print(f"✓ Optimal reference channel: {best_ref}")
        print(f"✓ Engine integration tested")
        print(f"✓ Calibration simulation completed")
        print(f"✓ Data export tested")
        print(f"✓ Visualizations generated")
        
        print(f"\nFinal Recommendations:")
        print(f"• Use channel {best_ref} as reference for best synchronization")
        print(f"• Sync quality score: {metrics.sync_quality_score:.3f}")
        if metrics.sync_quality_score < 0.8:
            print(f"• Consider improving microphone setup for better synchronization")
        
        print(f"\n✓ Audio stream synchronization demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()