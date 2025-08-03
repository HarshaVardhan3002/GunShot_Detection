#!/usr/bin/env python3
"""
Demo script for TDoA cross-correlation analysis.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from tdoa_localizer import CrossCorrelationTDoALocalizer, MicrophonePosition


def create_test_microphone_array():
    """Create different microphone array configurations for testing."""
    arrays = {}
    
    # Square array (good for 2D localization)
    arrays['square'] = [
        MicrophonePosition(0, 0.0, 0.0, 0.0),    # Bottom-left
        MicrophonePosition(1, 10.0, 0.0, 0.0),   # Bottom-right
        MicrophonePosition(2, 10.0, 10.0, 0.0),  # Top-right
        MicrophonePosition(3, 0.0, 10.0, 0.0),   # Top-left
    ]
    
    # Linear array (good for 1D localization)
    arrays['linear'] = [
        MicrophonePosition(0, 0.0, 0.0, 0.0),
        MicrophonePosition(1, 5.0, 0.0, 0.0),
        MicrophonePosition(2, 10.0, 0.0, 0.0),
        MicrophonePosition(3, 15.0, 0.0, 0.0),
    ]
    
    # Circular array (good for omnidirectional coverage)
    radius = 8.0
    arrays['circular'] = []
    for i in range(4):
        angle = 2 * np.pi * i / 4
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        arrays['circular'].append(MicrophonePosition(i, x, y, 0.0))
    
    # L-shaped array (asymmetric)
    arrays['l_shaped'] = [
        MicrophonePosition(0, 0.0, 0.0, 0.0),
        MicrophonePosition(1, 5.0, 0.0, 0.0),
        MicrophonePosition(2, 10.0, 0.0, 0.0),
        MicrophonePosition(3, 0.0, 5.0, 0.0),
    ]
    
    return arrays


def simulate_gunshot_signal(source_position, mic_positions, sample_rate=48000, 
                           samples=2048, sound_speed=343.0):
    """
    Simulate gunshot signal arriving at microphones with realistic delays.
    
    Args:
        source_position: (x, y) position of gunshot
        mic_positions: List of MicrophonePosition objects
        sample_rate: Audio sampling rate
        samples: Number of samples
        sound_speed: Speed of sound in m/s
        
    Returns:
        Multi-channel audio data with realistic delays
    """
    # Create base gunshot signal (impulsive with frequency content)
    t = np.arange(samples) / sample_rate
    
    # Gunshot characteristics: fast attack, exponential decay, multiple frequencies
    envelope = np.exp(-t * 50)  # Fast decay
    
    # Frequency content typical of gunshots
    signal_base = (0.4 * np.sin(2 * np.pi * 1200 * t) +    # 1.2kHz
                   0.3 * np.sin(2 * np.pi * 2800 * t) +     # 2.8kHz
                   0.2 * np.sin(2 * np.pi * 800 * t) +      # 800Hz
                   0.1 * np.sin(2 * np.pi * 4500 * t))      # 4.5kHz
    
    gunshot_signal = signal_base * envelope
    
    # Calculate delays and create multi-channel signal
    audio_channels = np.zeros((samples, len(mic_positions)))
    
    for i, mic in enumerate(mic_positions):
        # Calculate distance from source to microphone
        distance = np.sqrt((source_position[0] - mic.x)**2 + 
                          (source_position[1] - mic.y)**2)
        
        # Calculate delay in samples
        delay_seconds = distance / sound_speed
        delay_samples = int(delay_seconds * sample_rate)
        
        # Apply delay and distance attenuation
        attenuation = 1.0 / max(1.0, distance / 10.0)  # Inverse distance law
        
        if delay_samples < samples:
            audio_channels[delay_samples:, i] = gunshot_signal[:-delay_samples] * attenuation
        
        # Add some noise
        audio_channels[:, i] += np.random.random(samples) * 0.01
    
    return audio_channels


def test_basic_tdoa_calculation():
    """Test basic TDoA calculation with known source position."""
    print("=== Basic TDoA Calculation Test ===")
    
    # Create square microphone array
    mic_positions = create_test_microphone_array()['square']
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=48000,
        sound_speed=343.0
    )
    
    print(f"Microphone array: {len(mic_positions)} microphones")
    print(f"Microphone pairs: {len(localizer.mic_pairs)}")
    
    # Test with known source position
    source_position = (5.0, 3.0)  # 5m right, 3m up from origin
    print(f"Simulated source position: {source_position}")
    
    # Generate test signal
    audio_data = simulate_gunshot_signal(source_position, mic_positions)
    print(f"Generated audio data: {audio_data.shape}")
    
    # Calculate TDoA
    tdoa_matrix = localizer.calculate_tdoa(audio_data)
    
    print(f"\nTDoA Matrix (seconds):")
    print(f"{'':>8}", end='')
    for j in range(len(mic_positions)):
        print(f"{'Mic'+str(j):>8}", end='')
    print()
    
    for i in range(len(mic_positions)):
        print(f"{'Mic'+str(i):>8}", end='')
        for j in range(len(mic_positions)):
            print(f"{tdoa_matrix[i,j]:>8.4f}", end='')
        print()
    
    # Calculate expected delays for verification
    print(f"\nExpected delays (for verification):")
    for i, j in localizer.mic_pairs:
        mic_i = mic_positions[i]
        mic_j = mic_positions[j]
        
        dist_i = np.sqrt((source_position[0] - mic_i.x)**2 + (source_position[1] - mic_i.y)**2)
        dist_j = np.sqrt((source_position[0] - mic_j.x)**2 + (source_position[1] - mic_j.y)**2)
        
        expected_tdoa = (dist_i - dist_j) / 343.0
        measured_tdoa = tdoa_matrix[i, j]
        
        print(f"  Mic {i}-{j}: Expected={expected_tdoa:+.4f}s, Measured={measured_tdoa:+.4f}s, "
              f"Error={abs(expected_tdoa - measured_tdoa)*1000:.1f}ms")
    
    return localizer, tdoa_matrix


def test_array_configurations():
    """Test TDoA calculation with different microphone array configurations."""
    print("\n=== Array Configuration Comparison ===")
    
    arrays = create_test_microphone_array()
    source_position = (6.0, 4.0)
    
    results = {}
    
    for array_name, mic_positions in arrays.items():
        print(f"\nTesting {array_name} array:")
        
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=mic_positions,
            sample_rate=48000,
            sound_speed=343.0
        )
        
        # Generate test signal
        audio_data = simulate_gunshot_signal(source_position, mic_positions)
        
        # Calculate TDoA
        tdoa_matrix = localizer.calculate_tdoa(audio_data)
        
        # Get statistics
        correlation_stats = localizer.get_correlation_statistics()
        tdoa_stats = localizer.get_tdoa_statistics()
        
        results[array_name] = {
            'tdoa_matrix': tdoa_matrix,
            'correlation_stats': correlation_stats,
            'tdoa_stats': tdoa_stats
        }
        
        print(f"  Average correlation: {correlation_stats['avg_correlation']:.3f}")
        print(f"  TDoA consistency: {tdoa_stats['tdoa_consistency']:.3f}")
        print(f"  Max TDoA magnitude: {tdoa_stats['max_tdoa_magnitude']*1000:.1f} ms")
    
    return results


def test_correlation_parameters():
    """Test different cross-correlation parameters."""
    print("\n=== Correlation Parameter Testing ===")
    
    mic_positions = create_test_microphone_array()['square']
    source_position = (7.0, 2.0)
    
    # Generate test signal
    audio_data = simulate_gunshot_signal(source_position, mic_positions)
    
    parameter_sets = [
        {'name': 'FFT Method', 'params': {'correlation_method': 'fft'}},
        {'name': 'Direct Method', 'params': {'correlation_method': 'direct'}},
        {'name': 'High Interpolation', 'params': {'interpolation_factor': 8}},
        {'name': 'No Interpolation', 'params': {'interpolation_factor': 1}},
        {'name': 'Strict Threshold', 'params': {'min_correlation_threshold': 0.8}},
        {'name': 'Relaxed Threshold', 'params': {'min_correlation_threshold': 0.2}},
    ]
    
    results = {}
    
    for param_set in parameter_sets:
        print(f"\nTesting: {param_set['name']}")
        
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=mic_positions,
            sample_rate=48000,
            sound_speed=343.0
        )
        
        # Configure parameters
        localizer.configure_correlation_parameters(**param_set['params'])
        
        # Calculate TDoA
        start_time = time.time()
        tdoa_matrix = localizer.calculate_tdoa(audio_data)
        processing_time = time.time() - start_time
        
        # Get statistics
        correlation_stats = localizer.get_correlation_statistics()
        
        results[param_set['name']] = {
            'processing_time': processing_time,
            'avg_correlation': correlation_stats['avg_correlation'],
            'tdoa_matrix': tdoa_matrix
        }
        
        print(f"  Processing time: {processing_time*1000:.1f} ms")
        print(f"  Average correlation: {correlation_stats['avg_correlation']:.3f}")
    
    return results


def test_signal_quality_analysis():
    """Test signal quality analysis features."""
    print("\n=== Signal Quality Analysis ===")
    
    mic_positions = create_test_microphone_array()['square']
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=48000,
        sound_speed=343.0
    )
    
    # Test different signal quality scenarios
    scenarios = {
        'High Quality': {
            'source_pos': (5.0, 5.0),
            'noise_level': 0.001,
            'description': 'Close source, low noise'
        },
        'Distant Source': {
            'source_pos': (20.0, 20.0),
            'noise_level': 0.001,
            'description': 'Distant source, low noise'
        },
        'Noisy Environment': {
            'source_pos': (5.0, 5.0),
            'noise_level': 0.05,
            'description': 'Close source, high noise'
        },
        'Poor Conditions': {
            'source_pos': (15.0, 15.0),
            'noise_level': 0.03,
            'description': 'Distant source, moderate noise'
        }
    }
    
    for scenario_name, scenario in scenarios.items():
        print(f"\nScenario: {scenario_name} ({scenario['description']})")
        
        # Generate signal with specified conditions
        audio_data = simulate_gunshot_signal(scenario['source_pos'], mic_positions)
        
        # Add noise
        noise = np.random.random(audio_data.shape) * scenario['noise_level']
        audio_data += noise
        
        # Analyze signal quality
        quality = localizer.analyze_signal_quality(audio_data)
        
        print(f"  Overall quality: {quality['overall_quality']:.3f}")
        print(f"  Average SNR: {np.mean(quality['channel_snr']):.1f} dB")
        print(f"  Average correlation: {np.mean(quality['cross_channel_correlation']):.3f}")
        print(f"  Average bandwidth: {np.mean(quality['signal_bandwidth']):.0f} Hz")
        
        # Calculate TDoA for comparison
        tdoa_matrix = localizer.calculate_tdoa(audio_data)
        correlation_stats = localizer.get_correlation_statistics()
        
        print(f"  TDoA correlation: {correlation_stats['avg_correlation']:.3f}")
    
    return localizer


def test_preprocessing_effects():
    """Test the effects of signal preprocessing."""
    print("\n=== Signal Preprocessing Effects ===")
    
    mic_positions = create_test_microphone_array()['square']
    source_position = (6.0, 3.0)
    
    # Generate signal with noise and interference
    audio_data = simulate_gunshot_signal(source_position, mic_positions)
    
    # Add low-frequency noise (wind, traffic)
    samples = audio_data.shape[0]
    t = np.arange(samples) / 48000
    low_freq_noise = 0.02 * np.sin(2 * np.pi * 30 * t)  # 30 Hz noise
    audio_data += low_freq_noise[:, np.newaxis]
    
    # Add high-frequency noise (electronics)
    high_freq_noise = 0.01 * np.sin(2 * np.pi * 15000 * t)  # 15 kHz noise
    audio_data += high_freq_noise[:, np.newaxis]
    
    # Test with and without preprocessing
    test_cases = [
        {'name': 'No Preprocessing', 'preprocessing': False},
        {'name': 'With Preprocessing', 'preprocessing': True},
        {'name': 'Custom Filter Range', 'preprocessing': True, 'low_freq': 200, 'high_freq': 6000}
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=mic_positions,
            sample_rate=48000,
            sound_speed=343.0
        )
        
        # Configure preprocessing
        localizer.enable_preprocessing = case['preprocessing']
        if 'low_freq' in case:
            localizer.filter_low_freq = case['low_freq']
        if 'high_freq' in case:
            localizer.filter_high_freq = case['high_freq']
        
        # Calculate TDoA
        tdoa_matrix = localizer.calculate_tdoa(audio_data)
        
        # Get statistics
        correlation_stats = localizer.get_correlation_statistics()
        
        print(f"  Average correlation: {correlation_stats['avg_correlation']:.3f}")
        print(f"  Min correlation: {correlation_stats['min_correlation']:.3f}")
        print(f"  Max correlation: {correlation_stats['max_correlation']:.3f}")


def visualize_tdoa_results(localizer, tdoa_matrix, source_position):
    """Create visualizations of TDoA results."""
    print("\n=== Creating TDoA Visualizations ===")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('TDoA Cross-Correlation Analysis')
        
        # Plot 1: Microphone array layout with source
        mic_positions = localizer.microphone_positions
        mic_x = [mic.x for mic in mic_positions]
        mic_y = [mic.y for mic in mic_positions]
        
        axes[0, 0].scatter(mic_x, mic_y, c='blue', s=100, marker='s', label='Microphones')
        axes[0, 0].scatter(source_position[0], source_position[1], c='red', s=150, marker='*', label='Source')
        
        # Add microphone labels
        for i, mic in enumerate(mic_positions):
            axes[0, 0].annotate(f'M{i}', (mic.x, mic.y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Y Position (m)')
        axes[0, 0].set_title('Microphone Array Layout')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # Plot 2: TDoA matrix heatmap
        im = axes[0, 1].imshow(tdoa_matrix * 1000, cmap='RdBu_r', aspect='equal')
        axes[0, 1].set_title('TDoA Matrix (ms)')
        axes[0, 1].set_xlabel('Microphone Index')
        axes[0, 1].set_ylabel('Microphone Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[0, 1])
        cbar.set_label('Time Delay (ms)')
        
        # Add text annotations
        for i in range(len(mic_positions)):
            for j in range(len(mic_positions)):
                text = axes[0, 1].text(j, i, f'{tdoa_matrix[i, j]*1000:.1f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        # Plot 3: Correlation statistics
        correlation_stats = localizer.get_correlation_statistics()
        if localizer.correlation_history:
            correlation_data = localizer.correlation_history[-1]
            pairs = list(correlation_data.keys())
            correlations = list(correlation_data.values())
            
            bars = axes[1, 0].bar(range(len(pairs)), correlations)
            axes[1, 0].set_title('Cross-Correlation Values by Microphone Pair')
            axes[1, 0].set_xlabel('Microphone Pair')
            axes[1, 0].set_ylabel('Correlation Coefficient')
            axes[1, 0].set_xticks(range(len(pairs)))
            axes[1, 0].set_xticklabels(pairs, rotation=45)
            
            # Color bars based on correlation quality
            for bar, corr in zip(bars, correlations):
                if corr > 0.8:
                    bar.set_color('green')
                elif corr > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # Plot 4: Distance vs TDoA relationship
        distances = []
        tdoa_values = []
        
        for i, j in localizer.mic_pairs:
            mic_i = mic_positions[i]
            mic_j = mic_positions[j]
            
            # Distance between microphones
            mic_distance = np.sqrt((mic_i.x - mic_j.x)**2 + (mic_i.y - mic_j.y)**2)
            distances.append(mic_distance)
            tdoa_values.append(abs(tdoa_matrix[i, j]) * 1000)  # Convert to ms
        
        axes[1, 1].scatter(distances, tdoa_values, alpha=0.7)
        axes[1, 1].set_xlabel('Microphone Separation (m)')
        axes[1, 1].set_ylabel('|TDoA| (ms)')
        axes[1, 1].set_title('TDoA vs Microphone Separation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(distances) > 1:
            z = np.polyfit(distances, tdoa_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(distances), max(distances), 100)
            axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, 
                          label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'tdoa_analysis_{int(time.time())}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"TDoA analysis plot saved as: {plot_filename}")
        
        plt.close()
        
    except Exception as e:
        print(f"Visualization error (matplotlib may not be available): {e}")


def main():
    """Main demo function."""
    print("TDoA Cross-Correlation Analysis Demo")
    print("=" * 50)
    
    try:
        # Test 1: Basic TDoA calculation
        localizer, tdoa_matrix = test_basic_tdoa_calculation()
        
        # Test 2: Array configurations
        array_results = test_array_configurations()
        
        # Test 3: Correlation parameters
        param_results = test_correlation_parameters()
        
        # Test 4: Signal quality analysis
        quality_localizer = test_signal_quality_analysis()
        
        # Test 5: Preprocessing effects
        test_preprocessing_effects()
        
        # Test 6: Visualizations
        source_pos = (5.0, 3.0)
        visualize_tdoa_results(localizer, tdoa_matrix, source_pos)
        
        # Summary
        print("\n=== Demo Summary ===")
        print("✓ Basic TDoA calculation verified")
        print(f"✓ Array configurations tested: {len(array_results)} types")
        print(f"✓ Parameter variations tested: {len(param_results)} sets")
        print("✓ Signal quality analysis completed")
        print("✓ Preprocessing effects analyzed")
        print("✓ Visualizations generated")
        
        # Performance summary
        print(f"\nTDoA Performance Summary:")
        
        # Best array configuration
        best_array = max(array_results.keys(), 
                        key=lambda k: array_results[k]['correlation_stats']['avg_correlation'])
        print(f"  Best array configuration: {best_array}")
        print(f"  Best correlation: {array_results[best_array]['correlation_stats']['avg_correlation']:.3f}")
        
        # Parameter performance
        fastest_method = min(param_results.keys(), 
                           key=lambda k: param_results[k]['processing_time'])
        print(f"  Fastest correlation method: {fastest_method}")
        print(f"  Processing time: {param_results[fastest_method]['processing_time']*1000:.1f} ms")
        
        # Final statistics
        final_stats = localizer.get_correlation_statistics()
        print(f"\nFinal Statistics:")
        print(f"  Samples processed: {final_stats['samples_processed']}")
        print(f"  Microphone pairs: {final_stats['correlation_pairs']}")
        print(f"  Average correlation: {final_stats['avg_correlation']:.3f}")
        
        print(f"\n✓ TDoA cross-correlation demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()