"""
Demo script for TDoA calculation functionality.
"""
import numpy as np
import matplotlib.pyplot as plt
from tdoa_localizer import CrossCorrelationTDoALocalizer, MicrophonePosition
import time


def create_test_microphone_array():
    """Create a test microphone array in a square configuration."""
    positions = []
    # Create a 2x2 square array with 1m spacing
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        positions.append(MicrophonePosition(i, x, y, 0.0))
    return positions


def simulate_gunshot_signal(sample_rate=48000, duration=0.1):
    """Create a synthetic gunshot-like signal."""
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    
    # Sharp attack followed by exponential decay
    attack_samples = 50
    signal = np.zeros(samples)
    
    # Sharp impulse
    signal[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Exponential decay with some high-frequency content
    decay = np.exp(-t[attack_samples:] * 30)
    noise = 0.3 * np.random.normal(0, 1, len(decay))
    signal[attack_samples:] = decay + noise
    
    return signal


def simulate_source_at_position(source_x, source_y, mic_positions, base_signal, sample_rate, sound_speed=343.0):
    """Simulate audio received at microphones from a source at given position."""
    num_mics = len(mic_positions)
    signal_length = len(base_signal)
    audio_channels = np.zeros((signal_length, num_mics))
    
    print(f"Simulating source at ({source_x:.1f}, {source_y:.1f})")
    
    for i, mic_pos in enumerate(mic_positions):
        # Calculate distance from source to microphone
        distance = np.sqrt((source_x - mic_pos.x)**2 + (source_y - mic_pos.y)**2)
        
        # Calculate delay in samples
        delay_seconds = distance / sound_speed
        delay_samples = int(delay_seconds * sample_rate)
        
        print(f"  Mic {i} at ({mic_pos.x:.1f}, {mic_pos.y:.1f}): distance={distance:.2f}m, delay={delay_seconds*1000:.1f}ms")
        
        # Create delayed version
        if delay_samples < signal_length:
            delayed_signal = np.zeros(signal_length)
            delayed_signal[delay_samples:] = base_signal[:signal_length - delay_samples]
            
            # Add some noise
            noise = 0.1 * np.random.normal(0, 1, signal_length)
            audio_channels[:, i] = delayed_signal + noise
        else:
            # Signal arrives after our window
            audio_channels[:, i] = 0.1 * np.random.normal(0, 1, signal_length)
    
    return audio_channels


def demo_tdoa_calculation():
    """Demonstrate TDoA calculation with synthetic data."""
    print("=== TDoA Calculation Demo ===\n")
    
    # Create microphone array
    mic_positions = create_test_microphone_array()
    print("Microphone Array Configuration:")
    for i, pos in enumerate(mic_positions):
        print(f"  Mic {i}: ({pos.x:.1f}, {pos.y:.1f})")
    print()
    
    # Initialize TDoA localizer
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=48000,
        sound_speed=343.0
    )
    
    # Configure for better performance with synthetic signals
    localizer.configure_correlation_parameters(
        enable_preprocessing=True,
        min_correlation_threshold=0.2,
        interpolation_factor=4
    )
    
    # Create test signal
    base_signal = simulate_gunshot_signal()
    print(f"Generated gunshot signal: {len(base_signal)} samples, {len(base_signal)/48000*1000:.1f}ms duration\n")
    
    # Test different source positions
    test_positions = [
        (0.5, 0.5),  # Center of array
        (2.0, 0.5),  # East of array
        (0.5, 2.0),  # North of array
        (-1.0, -1.0) # Southwest of array
    ]
    
    for source_x, source_y in test_positions:
        print(f"--- Testing Source Position: ({source_x:.1f}, {source_y:.1f}) ---")
        
        # Simulate audio channels
        audio_channels = simulate_source_at_position(
            source_x, source_y, mic_positions, base_signal, 48000
        )
        
        # Calculate TDoA
        start_time = time.time()
        tdoa_matrix = localizer.calculate_tdoa(audio_channels)
        calculation_time = time.time() - start_time
        
        print(f"\nTDoA Calculation Results (completed in {calculation_time*1000:.1f}ms):")
        print("TDoA Matrix (ms):")
        print("     ", end="")
        for j in range(len(mic_positions)):
            print(f"Mic{j:1d}  ", end="")
        print()
        
        for i in range(len(mic_positions)):
            print(f"Mic{i}: ", end="")
            for j in range(len(mic_positions)):
                tdoa_ms = tdoa_matrix[i, j] * 1000
                print(f"{tdoa_ms:5.1f} ", end="")
            print()
        
        # Get correlation statistics
        stats = localizer.get_correlation_statistics()
        print(f"\nCorrelation Statistics:")
        print(f"  Average correlation: {stats['avg_correlation']:.3f}")
        print(f"  Min correlation: {stats['min_correlation']:.3f}")
        print(f"  Max correlation: {stats['max_correlation']:.3f}")
        
        # Get TDoA statistics
        tdoa_stats = localizer.get_tdoa_statistics()
        print(f"\nTDoA Statistics:")
        print(f"  Average TDoA magnitude: {tdoa_stats['avg_tdoa_magnitude']*1000:.1f}ms")
        print(f"  Max TDoA magnitude: {tdoa_stats['max_tdoa_magnitude']*1000:.1f}ms")
        print(f"  TDoA consistency: {tdoa_stats['tdoa_consistency']:.3f}")
        
        # Analyze signal quality
        quality = localizer.analyze_signal_quality(audio_channels)
        print(f"\nSignal Quality Analysis:")
        print(f"  Overall quality: {quality['overall_quality']:.3f}")
        print(f"  Average SNR: {np.mean(quality['channel_snr']):.1f} dB")
        print(f"  Average cross-correlation: {np.mean(quality['cross_channel_correlation']):.3f}")
        
        print("\n" + "="*60 + "\n")
    
    # Performance test
    print("--- Performance Test ---")
    num_iterations = 100
    audio_channels = simulate_source_at_position(
        1.0, 1.0, mic_positions, base_signal, 48000
    )
    
    start_time = time.time()
    for _ in range(num_iterations):
        tdoa_matrix = localizer.calculate_tdoa(audio_channels)
    total_time = time.time() - start_time
    
    avg_time = total_time / num_iterations
    print(f"Average calculation time over {num_iterations} iterations: {avg_time*1000:.2f}ms")
    print(f"Processing rate: {1/avg_time:.1f} calculations/second")
    
    # Check if we meet real-time requirements (should be much faster than audio buffer)
    buffer_duration = len(base_signal) / 48000  # Duration of audio buffer
    real_time_ratio = avg_time / buffer_duration
    print(f"Real-time performance ratio: {real_time_ratio:.3f} (lower is better)")
    
    if real_time_ratio < 0.1:
        print("✓ Excellent real-time performance")
    elif real_time_ratio < 0.5:
        print("✓ Good real-time performance")
    else:
        print("⚠ May struggle with real-time processing")


def demo_correlation_methods():
    """Compare different correlation methods."""
    print("\n=== Correlation Method Comparison ===\n")
    
    # Create simple test setup
    mic_positions = create_test_microphone_array()
    base_signal = simulate_gunshot_signal(duration=0.05)  # Shorter for speed
    
    audio_channels = simulate_source_at_position(
        1.5, 0.5, mic_positions, base_signal, 48000
    )
    
    methods = ['fft', 'direct']
    
    for method in methods:
        print(f"--- Testing {method.upper()} Correlation ---")
        
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=mic_positions,
            sample_rate=48000
        )
        
        localizer.configure_correlation_parameters(
            correlation_method=method,
            enable_preprocessing=True
        )
        
        # Time the calculation
        start_time = time.time()
        tdoa_matrix = localizer.calculate_tdoa(audio_channels)
        calculation_time = time.time() - start_time
        
        stats = localizer.get_correlation_statistics()
        
        print(f"  Calculation time: {calculation_time*1000:.2f}ms")
        print(f"  Average correlation: {stats['avg_correlation']:.3f}")
        print(f"  Max TDoA magnitude: {np.max(np.abs(tdoa_matrix))*1000:.1f}ms")
        print()


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        demo_tdoa_calculation()
        demo_correlation_methods()
        
        print("Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()