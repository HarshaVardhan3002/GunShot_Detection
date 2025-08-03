"""
Demo script for intensity filter functionality.
"""
import numpy as np
import matplotlib.pyplot as plt
from intensity_filter import RMSIntensityFilter
import time


def create_test_signal(amplitude: float, frequency: float, duration: float, 
                      noise_level: float = 0.0, sample_rate: int = 48000) -> np.ndarray:
    """Create a test signal with specified parameters."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Create sine wave
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, samples)
        signal += noise
    
    return signal


def create_gunshot_like_signal(amplitude: float, noise_level: float = 0.0, 
                              sample_rate: int = 48000) -> np.ndarray:
    """Create a gunshot-like signal with realistic characteristics."""
    duration = 0.1  # 100ms
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    
    # Gunshot characteristics: sharp attack, exponential decay, broadband
    attack_samples = 50  # Very sharp attack
    signal = np.zeros(samples)
    
    # Sharp impulse
    signal[:attack_samples] = amplitude * np.linspace(0, 1, attack_samples)
    
    # Exponential decay with multiple frequency components
    decay = np.exp(-t[attack_samples:] * 30)
    
    # Add multiple frequency components typical of gunshots
    freq_components = [500, 1000, 2000, 4000]  # Hz
    for freq in freq_components:
        component = 0.25 * np.sin(2 * np.pi * freq * t[attack_samples:])
        signal[attack_samples:] += amplitude * component * decay
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, samples)
        signal += noise
    
    return signal


def demo_basic_weighting():
    """Demonstrate basic channel weighting functionality."""
    print("=== Basic Channel Weighting Demo ===\n")
    
    # Initialize filter
    filter_obj = RMSIntensityFilter(sample_rate=48000)
    
    # Create test scenarios
    scenarios = [
        {
            'name': 'Uniform Quality',
            'channels': [
                (1.0, 1000, 0.1),  # Same amplitude, frequency, noise
                (1.0, 1000, 0.1),
                (1.0, 1000, 0.1),
                (1.0, 1000, 0.1)
            ]
        },
        {
            'name': 'Varying Amplitudes',
            'channels': [
                (2.0, 1000, 0.1),  # High amplitude
                (1.0, 1000, 0.1),  # Medium amplitude
                (0.5, 1000, 0.1),  # Low amplitude
                (0.2, 1000, 0.1)   # Very low amplitude
            ]
        },
        {
            'name': 'Varying SNR',
            'channels': [
                (1.0, 1000, 0.05),  # Low noise (high SNR)
                (1.0, 1000, 0.1),   # Medium noise
                (1.0, 1000, 0.2),   # High noise
                (1.0, 1000, 0.4)    # Very high noise (low SNR)
            ]
        },
        {
            'name': 'Mixed Quality',
            'channels': [
                (1.5, 1000, 0.05),  # Excellent quality
                (0.8, 1000, 0.1),   # Good quality
                (0.4, 1000, 0.2),   # Poor quality
                (0.1, 1000, 0.3)    # Very poor quality
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"--- {scenario['name']} ---")
        
        # Create multi-channel audio
        duration = 0.1  # 100ms
        samples = int(duration * 48000)
        num_channels = len(scenario['channels'])
        audio_channels = np.zeros((samples, num_channels))
        
        for ch, (amplitude, frequency, noise_level) in enumerate(scenario['channels']):
            signal = create_test_signal(amplitude, frequency, duration, noise_level)
            audio_channels[:, ch] = signal
        
        # Calculate weights
        start_time = time.time()
        weights = filter_obj.calculate_weights(audio_channels)
        calc_time = time.time() - start_time
        
        # Display results
        print("Channel | Amplitude | Noise Level | Weight  | Normalized")
        print("-" * 55)
        
        max_weight = np.max(weights)
        for ch, (amplitude, frequency, noise_level) in enumerate(scenario['channels']):
            normalized_weight = weights[ch] / max_weight if max_weight > 0 else 0
            print(f"   {ch}    |   {amplitude:.1f}     |    {noise_level:.2f}     | {weights[ch]:.3f}  |   {normalized_weight:.3f}")
        
        print(f"Calculation time: {calc_time*1000:.2f}ms")
        print(f"Weight range: {np.max(weights) - np.min(weights):.3f}")
        print()


def demo_noise_floor_estimation():
    """Demonstrate noise floor estimation methods."""
    print("=== Noise Floor Estimation Demo ===\n")
    
    # Test different noise floor estimation methods
    methods = ['percentile', 'minimum', 'adaptive']
    
    # Create test signals with known noise characteristics
    test_cases = [
        {'name': 'Clean Signal', 'signal_amp': 1.0, 'noise_level': 0.01},
        {'name': 'Moderate Noise', 'signal_amp': 1.0, 'noise_level': 0.1},
        {'name': 'High Noise', 'signal_amp': 1.0, 'noise_level': 0.3},
        {'name': 'Very High Noise', 'signal_amp': 1.0, 'noise_level': 0.5}
    ]
    
    print("Test Case        | Method     | Estimated Noise Floor | Actual Noise")
    print("-" * 65)
    
    for test_case in test_cases:
        # Create test signal
        clean_signal = create_test_signal(test_case['signal_amp'], 1000, 0.1, 0.0)
        noise = np.random.normal(0, test_case['noise_level'], len(clean_signal))
        noisy_signal = clean_signal + noise
        
        for method in methods:
            filter_obj = RMSIntensityFilter(noise_estimation_method=method)
            estimated_noise = filter_obj.estimate_noise_floor(noisy_signal)
            
            print(f"{test_case['name']:15} | {method:10} | {estimated_noise:18.4f} | {test_case['noise_level']:11.2f}")
    
    print()


def demo_channel_filtering():
    """Demonstrate channel filtering based on quality."""
    print("=== Channel Filtering Demo ===\n")
    
    filter_obj = RMSIntensityFilter()
    
    # Create 8-channel array with varying quality
    channel_configs = [
        (1.5, 1000, 0.05),  # Excellent
        (1.2, 1000, 0.08),  # Very good
        (1.0, 1000, 0.1),   # Good
        (0.8, 1000, 0.15),  # Fair
        (0.5, 1000, 0.2),   # Poor
        (0.3, 1000, 0.3),   # Very poor
        (0.1, 1000, 0.4),   # Extremely poor
        (0.05, 1000, 0.5)   # Unusable
    ]
    
    # Create audio channels
    duration = 0.1
    samples = int(duration * 48000)
    audio_channels = np.zeros((samples, len(channel_configs)))
    
    for ch, (amplitude, frequency, noise_level) in enumerate(channel_configs):
        signal = create_test_signal(amplitude, frequency, duration, noise_level)
        audio_channels[:, ch] = signal
    
    # Calculate weights
    weights = filter_obj.calculate_weights(audio_channels)
    
    # Test different filtering thresholds
    thresholds = [0.2, 0.4, 0.6, 0.8]
    
    print("Channel Quality Overview:")
    print("Channel | Amplitude | Noise | Weight | Quality")
    print("-" * 45)
    
    quality_labels = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor', 'Very Poor', 'Extremely Poor', 'Unusable']
    for ch, (amplitude, frequency, noise_level) in enumerate(channel_configs):
        print(f"   {ch}    |   {amplitude:.2f}    | {noise_level:.2f}  | {weights[ch]:.3f} | {quality_labels[ch]}")
    
    print(f"\nFiltering Results:")
    print("Threshold | Valid Channels | Channel Indices")
    print("-" * 45)
    
    for threshold in thresholds:
        valid_channels = filter_obj.filter_low_snr_channels(weights, threshold)
        print(f"  {threshold:.1f}     |      {len(valid_channels)}        | {valid_channels}")
    
    print()


def demo_adaptive_behavior():
    """Demonstrate adaptive behavior over time."""
    print("=== Adaptive Behavior Demo ===\n")
    
    # Enable adaptive adjustments
    filter_obj = RMSIntensityFilter()
    filter_obj.enable_adaptive_thresholds = True
    
    # Simulate changing conditions over time
    print("Simulating changing microphone conditions over time...")
    
    # Initial conditions
    base_configs = [
        (1.0, 1000, 0.1),  # Channel 0: stable
        (0.8, 1000, 0.15), # Channel 1: degrading
        (0.6, 1000, 0.2),  # Channel 2: improving
        (0.4, 1000, 0.25)  # Channel 3: unstable
    ]
    
    time_steps = 10
    all_weights = []
    
    for step in range(time_steps):
        # Modify conditions over time
        current_configs = []
        for ch, (amp, freq, noise) in enumerate(base_configs):
            if ch == 1:  # Degrading channel
                noise += step * 0.02
                amp -= step * 0.05
            elif ch == 2:  # Improving channel
                noise -= step * 0.01
                amp += step * 0.03
            elif ch == 3:  # Unstable channel
                noise += 0.05 * np.sin(step)
                amp += 0.1 * np.cos(step)
            
            # Ensure reasonable bounds
            noise = max(0.05, min(0.5, noise))
            amp = max(0.1, min(2.0, amp))
            
            current_configs.append((amp, freq, noise))
        
        # Create audio for this time step
        duration = 0.1
        samples = int(duration * 48000)
        audio_channels = np.zeros((samples, len(current_configs)))
        
        for ch, (amplitude, frequency, noise_level) in enumerate(current_configs):
            signal = create_test_signal(amplitude, frequency, duration, noise_level)
            audio_channels[:, ch] = signal
        
        # Calculate weights
        weights = filter_obj.calculate_weights(audio_channels)
        all_weights.append(weights)
        
        print(f"Step {step:2d}: Weights = [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}, {weights[3]:.3f}]")
    
    # Show adaptation summary
    print(f"\nAdaptation Summary:")
    print("Channel | Initial Weight | Final Weight | Change")
    print("-" * 45)
    
    for ch in range(len(base_configs)):
        initial_weight = all_weights[0][ch]
        final_weight = all_weights[-1][ch]
        change = final_weight - initial_weight
        print(f"   {ch}    |     {initial_weight:.3f}     |    {final_weight:.3f}    | {change:+.3f}")
    
    print()


def demo_performance_analysis():
    """Demonstrate performance analysis and benchmarking."""
    print("=== Performance Analysis Demo ===\n")
    
    # Test with different array sizes
    array_sizes = [4, 6, 8, 12, 16]
    
    print("Array Size | Avg Time (ms) | Processing Rate (Hz) | Memory Usage")
    print("-" * 60)
    
    for num_channels in array_sizes:
        filter_obj = RMSIntensityFilter()
        
        # Create test data
        duration = 0.1
        samples = int(duration * 48000)
        audio_channels = np.random.normal(0, 0.1, (samples, num_channels))
        
        # Add some signal to each channel
        for ch in range(num_channels):
            signal = create_test_signal(1.0, 1000, duration, 0.1)
            audio_channels[:, ch] += signal
        
        # Benchmark performance
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            weights = filter_obj.calculate_weights(audio_channels)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        processing_rate = 1.0 / avg_time
        
        # Estimate memory usage (rough)
        memory_mb = (samples * num_channels * 8) / (1024 * 1024)  # 8 bytes per float64
        
        print(f"    {num_channels:2d}     |    {avg_time*1000:6.2f}    |      {processing_rate:7.0f}      |   {memory_mb:.2f} MB")
    
    print()
    
    # Real-time performance assessment
    print("Real-time Performance Assessment:")
    buffer_duration = 0.1  # 100ms buffer
    max_acceptable_time = buffer_duration * 0.1  # 10% of buffer time
    
    filter_obj = RMSIntensityFilter()
    test_audio = np.random.normal(0, 0.1, (int(0.1 * 48000), 8))
    
    # Single calculation timing
    start_time = time.time()
    weights = filter_obj.calculate_weights(test_audio)
    single_time = time.time() - start_time
    
    print(f"Single calculation time: {single_time*1000:.2f}ms")
    print(f"Buffer duration: {buffer_duration*1000:.0f}ms")
    print(f"Processing overhead: {single_time/buffer_duration*100:.1f}%")
    
    if single_time < max_acceptable_time:
        print("✓ Excellent real-time performance")
    elif single_time < buffer_duration * 0.2:
        print("✓ Good real-time performance")
    elif single_time < buffer_duration * 0.5:
        print("⚠ Marginal real-time performance")
    else:
        print("✗ Poor real-time performance")


def demo_quality_reporting():
    """Demonstrate quality reporting functionality."""
    print("=== Quality Reporting Demo ===\n")
    
    filter_obj = RMSIntensityFilter()
    
    # Create scenarios with different overall quality
    scenarios = [
        {
            'name': 'Excellent Setup',
            'channels': [(1.5, 1000, 0.02), (1.3, 1000, 0.03), (1.4, 1000, 0.025), (1.2, 1000, 0.04)]
        },
        {
            'name': 'Good Setup',
            'channels': [(1.0, 1000, 0.08), (0.9, 1000, 0.1), (1.1, 1000, 0.07), (0.8, 1000, 0.12)]
        },
        {
            'name': 'Poor Setup',
            'channels': [(0.3, 1000, 0.2), (0.4, 1000, 0.25), (0.2, 1000, 0.3), (0.1, 1000, 0.4)]
        },
        {
            'name': 'Mixed Setup',
            'channels': [(1.5, 1000, 0.05), (0.8, 1000, 0.1), (0.3, 1000, 0.3), (0.05, 1000, 0.5)]
        }
    ]
    
    for scenario in scenarios:
        print(f"--- {scenario['name']} ---")
        
        # Create audio channels
        duration = 0.1
        samples = int(duration * 48000)
        audio_channels = np.zeros((samples, len(scenario['channels'])))
        
        for ch, (amplitude, frequency, noise_level) in enumerate(scenario['channels']):
            signal = create_test_signal(amplitude, frequency, duration, noise_level)
            audio_channels[:, ch] = signal
        
        # Calculate weights and get report
        weights = filter_obj.calculate_weights(audio_channels)
        report = filter_obj.get_channel_quality_report()
        
        # Display report
        print(f"Overall Status: {report['status'].upper()}")
        print(f"Recommendation: {report['recommendation']}")
        
        summary = report['summary']
        print(f"Valid Channels: {summary['valid_channels']}/{summary['total_channels']}")
        print(f"Average SNR: {summary['avg_snr_db']:.1f} dB")
        print(f"Average Weight: {summary['avg_weight']:.3f}")
        
        print("Channel Details:")
        for ch_info in report['channels']:
            status = "✓" if ch_info['is_valid'] else "✗"
            print(f"  Ch{ch_info['channel']}: {status} SNR={ch_info['snr_db']:.1f}dB, Weight={ch_info['weight']:.3f}")
        
        print()


def visualize_filter_behavior():
    """Create visualizations of filter behavior."""
    print("=== Filter Behavior Visualization ===\n")
    
    try:
        filter_obj = RMSIntensityFilter()
        
        # Create test data with varying quality
        num_channels = 6
        channel_configs = [
            (1.5, 1000, 0.05),  # Excellent
            (1.0, 1000, 0.1),   # Good
            (0.7, 1000, 0.15),  # Fair
            (0.4, 1000, 0.25),  # Poor
            (0.2, 1000, 0.35),  # Very poor
            (0.1, 1000, 0.45)   # Extremely poor
        ]
        
        # Create audio channels
        duration = 0.1
        samples = int(duration * 48000)
        audio_channels = np.zeros((samples, num_channels))
        
        for ch, (amplitude, frequency, noise_level) in enumerate(channel_configs):
            signal = create_test_signal(amplitude, frequency, duration, noise_level)
            audio_channels[:, ch] = signal
        
        # Calculate weights
        weights = filter_obj.calculate_weights(audio_channels)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Signal amplitudes and noise levels
        channels = range(num_channels)
        amplitudes = [config[0] for config in channel_configs]
        noise_levels = [config[2] for config in channel_configs]
        
        ax1.bar([ch - 0.2 for ch in channels], amplitudes, 0.4, label='Signal Amplitude', alpha=0.7)
        ax1.bar([ch + 0.2 for ch in channels], noise_levels, 0.4, label='Noise Level', alpha=0.7)
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Signal Amplitude vs Noise Level by Channel')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Calculated weights
        ax2.bar(channels, weights, alpha=0.7, color='green')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('Weight')
        ax2.set_title('Calculated Channel Weights')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: SNR comparison
        # Get quality metrics for SNR
        filter_obj.calculate_weights(audio_channels)  # Ensure metrics are calculated
        if filter_obj.quality_history:
            latest_metrics = filter_obj.quality_history[-1]
            snr_values = [m.snr_db for m in latest_metrics]
            
            ax3.bar(channels, snr_values, alpha=0.7, color='orange')
            ax3.set_xlabel('Channel')
            ax3.set_ylabel('SNR (dB)')
            ax3.set_title('Signal-to-Noise Ratio by Channel')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=6, color='red', linestyle='--', label='Minimum SNR Threshold')
            ax3.legend()
        
        plt.tight_layout()
        plt.savefig('intensity_filter_analysis.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'intensity_filter_analysis.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        demo_basic_weighting()
        demo_noise_floor_estimation()
        demo_channel_filtering()
        demo_adaptive_behavior()
        demo_performance_analysis()
        demo_quality_reporting()
        visualize_filter_behavior()
        
        print("Intensity filter demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()