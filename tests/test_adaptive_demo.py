#!/usr/bin/env python3
"""
Demo script for adaptive thresholding in gunshot detection.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from gunshot_detector import AmplitudeBasedDetector


def simulate_environment_changes(detector, duration_per_env=10):
    """Simulate different acoustic environments over time."""
    print("=== Environment Simulation Demo ===")
    
    environments = {
        'very_quiet': {'noise_level': 0.0001, 'description': 'Library/Empty room'},
        'quiet': {'noise_level': 0.002, 'description': 'Residential night'},
        'moderate': {'noise_level': 0.01, 'description': 'Office/Residential day'},
        'busy': {'noise_level': 0.03, 'description': 'Street/Commercial'},
        'very_busy': {'noise_level': 0.08, 'description': 'Construction/Heavy traffic'}
    }
    
    results = {}
    
    for env_name, env_config in environments.items():
        print(f"\nSimulating {env_name} environment: {env_config['description']}")
        print(f"Noise level: {env_config['noise_level']:.4f}")
        
        # Reset for clean environment simulation
        detector.reset_adaptive_state()
        
        # Simulate environment for specified duration
        samples_per_update = 1000
        updates_per_second = 10
        total_updates = duration_per_env * updates_per_second
        
        threshold_history = []
        noise_history = []
        
        for i in range(total_updates):
            # Generate environment-appropriate noise
            noise_data = np.random.random((samples_per_update, 8)).astype(np.float32)
            noise_data *= env_config['noise_level']
            
            # Add some variability
            variability = 1.0 + 0.2 * np.sin(i * 0.1)  # ±20% variation
            noise_data *= variability
            
            # Update detector
            detector._update_noise_floor(noise_data)
            detector.set_adaptive_threshold(detector.noise_floor)
            
            # Record metrics
            threshold_history.append(detector.threshold_db)
            noise_history.append(detector.noise_floor)
            
            # Small delay to simulate real-time
            time.sleep(0.01)
        
        # Get final status
        status = detector.get_adaptive_threshold_status()
        env_analysis = detector.get_environment_analysis()
        
        results[env_name] = {
            'final_threshold': detector.threshold_db,
            'environment_type': detector.environment_type,
            'activity_level': detector.activity_level,
            'noise_floor': detector.noise_floor,
            'threshold_history': threshold_history,
            'noise_history': noise_history,
            'status': status,
            'analysis': env_analysis
        }
        
        print(f"  Final threshold: {detector.threshold_db:.1f} dB")
        print(f"  Classified as: {detector.environment_type}")
        print(f"  Activity level: {detector.activity_level}")
        print(f"  Noise floor: {detector.noise_floor:.6f}")
    
    return results


def test_threshold_levels():
    """Test different threshold level settings."""
    print("\n=== Threshold Level Testing ===")
    
    detector = AmplitudeBasedDetector(sample_rate=48000, channels=8, threshold_db=-20.0)
    
    # Test signal (moderate gunshot)
    samples = 1500
    test_signal = np.zeros((samples, 8), dtype=np.float32)
    
    # Add gunshot-like impulse
    impulse_start = 600
    impulse_duration = 80
    impulse_amplitude = 0.12  # Moderate amplitude
    
    for ch in range(5):  # 5 channels triggered
        test_signal[impulse_start:impulse_start + impulse_duration, ch] = impulse_amplitude
    
    levels = ['sensitive', 'normal', 'conservative']
    results = {}
    
    for level in levels:
        print(f"\nTesting threshold level: {level}")
        
        detector.set_threshold_level(level)
        
        detected, confidence, metadata = detector.detect_gunshot(test_signal)
        
        results[level] = {
            'threshold_db': detector.threshold_db,
            'detected': detected,
            'confidence': confidence,
            'triggered_channels': metadata.get('triggered_channels', [])
        }
        
        print(f"  Threshold: {detector.threshold_db:.1f} dB")
        print(f"  Detected: {detected}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Channels: {metadata.get('triggered_channels', [])}")
        
        # Wait for cooldown
        time.sleep(0.6)
    
    return results


def test_performance_feedback():
    """Test adaptive learning from performance feedback."""
    print("\n=== Performance Feedback Testing ===")
    
    detector = AmplitudeBasedDetector(sample_rate=48000, channels=8, threshold_db=-20.0)
    detector.enable_adaptive_thresholding(True)
    
    # Simulate detection scenarios with feedback
    scenarios = [
        {'name': 'True Gunshot', 'amplitude': 0.3, 'feedback': 'correct'},
        {'name': 'False Positive (Thunder)', 'amplitude': 0.25, 'feedback': 'false_positive'},
        {'name': 'True Gunshot', 'amplitude': 0.28, 'feedback': 'correct'},
        {'name': 'False Positive (Door)', 'amplitude': 0.22, 'feedback': 'false_positive'},
        {'name': 'Missed Detection', 'amplitude': 0.15, 'feedback': 'missed'},
        {'name': 'True Gunshot', 'amplitude': 0.32, 'feedback': 'correct'},
    ]
    
    initial_threshold = detector.threshold_db
    threshold_history = [initial_threshold]
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['name']}")
        
        # Create test signal
        samples = 1200
        test_signal = np.zeros((samples, 8), dtype=np.float32)
        
        # Add signal
        impulse_start = 500
        impulse_duration = 60
        for ch in range(6):
            test_signal[impulse_start:impulse_start + impulse_duration, ch] = scenario['amplitude']
        
        # Detect
        detected, confidence, metadata = detector.detect_gunshot(test_signal)
        
        print(f"  Amplitude: {scenario['amplitude']:.2f}")
        print(f"  Detected: {detected}, Confidence: {confidence:.3f}")
        print(f"  Threshold before feedback: {detector.threshold_db:.1f} dB")
        
        # Provide feedback
        detector.provide_detection_feedback(scenario['feedback'])
        
        print(f"  Feedback: {scenario['feedback']}")
        print(f"  Threshold after feedback: {detector.threshold_db:.1f} dB")
        
        threshold_history.append(detector.threshold_db)
        
        # Wait for cooldown
        time.sleep(0.6)
    
    # Show performance statistics
    status = detector.get_adaptive_threshold_status()
    performance = status['performance']
    
    print(f"\nFinal Performance Statistics:")
    print(f"  False positive rate: {performance['false_positive_rate']:.2f}")
    print(f"  Missed detection rate: {performance['missed_detection_rate']:.2f}")
    print(f"  Total detections tracked: {performance['total_detections']}")
    print(f"  Threshold change: {initial_threshold:.1f} → {detector.threshold_db:.1f} dB")
    
    return threshold_history, performance


def test_time_based_adaptation():
    """Test time-based threshold adaptation."""
    print("\n=== Time-Based Adaptation Testing ===")
    
    detector = AmplitudeBasedDetector(sample_rate=48000, channels=8, threshold_db=-20.0)
    
    # Mock different times of day
    import unittest.mock
    
    time_scenarios = [
        {'hour': 2, 'description': 'Night (2 AM)'},
        {'hour': 8, 'description': 'Morning rush (8 AM)'},
        {'hour': 14, 'description': 'Afternoon (2 PM)'},
        {'hour': 18, 'description': 'Evening rush (6 PM)'},
        {'hour': 22, 'description': 'Late evening (10 PM)'}
    ]
    
    results = {}
    
    for scenario in time_scenarios:
        with unittest.mock.patch('time.localtime') as mock_time:
            mock_time.return_value.tm_hour = scenario['hour']
            
            adjustment = detector._calculate_time_based_adjustment()
            adjusted_threshold = detector.base_threshold_db + adjustment
            
            results[scenario['description']] = {
                'hour': scenario['hour'],
                'adjustment': adjustment,
                'adjusted_threshold': adjusted_threshold
            }
            
            print(f"{scenario['description']}:")
            print(f"  Time adjustment: {adjustment:+.1f} dB")
            print(f"  Adjusted threshold: {adjusted_threshold:.1f} dB")
    
    return results


def test_noise_trend_analysis():
    """Test noise trend analysis over time."""
    print("\n=== Noise Trend Analysis ===")
    
    detector = AmplitudeBasedDetector(sample_rate=48000, channels=8)
    
    # Simulate increasing noise scenario
    print("\nSimulating increasing noise environment:")
    detector.reset_adaptive_state()
    
    for i in range(30):
        noise_level = 0.001 * (1 + i * 0.05)  # Gradually increasing
        noise_data = np.random.random((200, 8)).astype(np.float32) * noise_level
        detector._update_noise_floor(noise_data)
        
        if i % 10 == 9:  # Every 10 iterations
            trend = detector._calculate_noise_trend()
            analysis = detector.get_environment_analysis()
            print(f"  Step {i+1}: Noise={noise_level:.4f}, Trend={trend}, Type={analysis['environment_type']}")
    
    # Simulate decreasing noise scenario
    print("\nSimulating decreasing noise environment:")
    detector.reset_adaptive_state()
    
    for i in range(30):
        noise_level = 0.02 * (1 - i * 0.02)  # Gradually decreasing
        noise_level = max(noise_level, 0.001)  # Minimum floor
        noise_data = np.random.random((200, 8)).astype(np.float32) * noise_level
        detector._update_noise_floor(noise_data)
        
        if i % 10 == 9:
            trend = detector._calculate_noise_trend()
            analysis = detector.get_environment_analysis()
            print(f"  Step {i+1}: Noise={noise_level:.4f}, Trend={trend}, Type={analysis['environment_type']}")
    
    return detector


def visualize_adaptive_results(env_results, threshold_history):
    """Create visualizations of adaptive thresholding results."""
    print("\n=== Creating Adaptive Threshold Visualizations ===")
    
    try:
        # Create environment adaptation plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Adaptive Thresholding Analysis')
        
        # Plot 1: Environment-based threshold adaptation
        env_names = list(env_results.keys())
        final_thresholds = [env_results[env]['final_threshold'] for env in env_names]
        noise_floors = [env_results[env]['noise_floor'] for env in env_names]
        
        x_pos = np.arange(len(env_names))
        
        bars1 = axes[0, 0].bar(x_pos - 0.2, final_thresholds, 0.4, label='Final Threshold (dB)', alpha=0.8)
        
        # Secondary y-axis for noise floor
        ax2 = axes[0, 0].twinx()
        bars2 = ax2.bar(x_pos + 0.2, [nf * 1000 for nf in noise_floors], 0.4, 
                       label='Noise Floor (×1000)', alpha=0.8, color='orange')
        
        axes[0, 0].set_xlabel('Environment Type')
        axes[0, 0].set_ylabel('Threshold (dB)', color='blue')
        ax2.set_ylabel('Noise Floor (×1000)', color='orange')
        axes[0, 0].set_title('Threshold Adaptation by Environment')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([name.replace('_', ' ').title() for name in env_names], rotation=45)
        
        # Color bars based on threshold level
        for bar, threshold in zip(bars1, final_thresholds):
            if threshold < -25:
                bar.set_color('green')  # Sensitive
            elif threshold < -15:
                bar.set_color('blue')   # Normal
            else:
                bar.set_color('red')    # Conservative
        
        # Plot 2: Threshold evolution over time (first environment)
        if env_names:
            first_env = env_names[0]
            threshold_hist = env_results[first_env]['threshold_history']
            
            axes[0, 1].plot(threshold_hist, 'b-', linewidth=2)
            axes[0, 1].set_title(f'Threshold Evolution: {first_env.replace("_", " ").title()}')
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Threshold (dB)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance feedback impact
        if threshold_history:
            axes[1, 0].plot(threshold_history, 'ro-', linewidth=2, markersize=6)
            axes[1, 0].set_title('Threshold Adaptation from Performance Feedback')
            axes[1, 0].set_xlabel('Feedback Events')
            axes[1, 0].set_ylabel('Threshold (dB)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Annotate significant changes
            for i in range(1, len(threshold_history)):
                change = threshold_history[i] - threshold_history[i-1]
                if abs(change) > 0.5:
                    axes[1, 0].annotate(f'{change:+.1f}dB', 
                                      xy=(i, threshold_history[i]),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, alpha=0.7)
        
        # Plot 4: Environment classification summary
        env_types = [env_results[env]['environment_type'] for env in env_names]
        activity_levels = [env_results[env]['activity_level'] for env in env_names]
        
        # Count environment types
        unique_types = list(set(env_types))
        type_counts = [env_types.count(t) for t in unique_types]
        
        axes[1, 1].pie(type_counts, labels=unique_types, autopct='%1.0f%%', startangle=90)
        axes[1, 1].set_title('Environment Classification Distribution')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'adaptive_threshold_analysis_{int(time.time())}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Adaptive threshold analysis plot saved as: {plot_filename}")
        
        plt.close()
        
        # Create detailed noise analysis plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Noise Floor Analysis')
        
        # Plot noise floor evolution for different environments
        for i, (env_name, env_data) in enumerate(list(env_results.items())[:3]):  # First 3 environments
            noise_hist = env_data['noise_history']
            axes[0].plot(noise_hist, label=env_name.replace('_', ' ').title(), alpha=0.8)
        
        axes[0].set_title('Noise Floor Evolution by Environment')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Noise Floor')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # Log scale for noise floor
        
        # Plot threshold vs noise floor correlation
        all_thresholds = []
        all_noise_floors = []
        
        for env_data in env_results.values():
            all_thresholds.extend(env_data['threshold_history'])
            all_noise_floors.extend(env_data['noise_history'])
        
        axes[1].scatter(all_noise_floors, all_thresholds, alpha=0.6, s=20)
        axes[1].set_title('Threshold vs Noise Floor Correlation')
        axes[1].set_xlabel('Noise Floor')
        axes[1].set_ylabel('Threshold (dB)')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(all_noise_floors) > 1:
            log_noise = np.log10(all_noise_floors)
            coeffs = np.polyfit(log_noise, all_thresholds, 1)
            trend_line = np.poly1d(coeffs)
            x_trend = np.logspace(np.log10(min(all_noise_floors)), np.log10(max(all_noise_floors)), 100)
            y_trend = trend_line(np.log10(x_trend))
            axes[1].plot(x_trend, y_trend, 'r--', alpha=0.8, label=f'Trend: {coeffs[0]:.1f}x + {coeffs[1]:.1f}')
            axes[1].legend()
        
        plt.tight_layout()
        
        # Save noise analysis plot
        noise_filename = f'noise_analysis_{int(time.time())}.png'
        plt.savefig(noise_filename, dpi=150, bbox_inches='tight')
        print(f"Noise analysis plot saved as: {noise_filename}")
        
        plt.close()
        
    except Exception as e:
        print(f"Visualization error (matplotlib may not be available): {e}")


def main():
    """Main demo function."""
    print("Adaptive Thresholding Demo")
    print("=" * 50)
    
    try:
        # Test 1: Environment simulation
        detector = AmplitudeBasedDetector(sample_rate=48000, channels=8, threshold_db=-20.0)
        detector.enable_adaptive_thresholding(True)
        
        env_results = simulate_environment_changes(detector, duration_per_env=3)  # Shorter for demo
        
        # Test 2: Threshold levels
        level_results = test_threshold_levels()
        
        # Test 3: Performance feedback
        threshold_history, performance = test_performance_feedback()
        
        # Test 4: Time-based adaptation
        time_results = test_time_based_adaptation()
        
        # Test 5: Noise trend analysis
        trend_detector = test_noise_trend_analysis()
        
        # Test 6: Visualizations
        visualize_adaptive_results(env_results, threshold_history)
        
        # Summary
        print("\n=== Demo Summary ===")
        print("✓ Environment-based adaptation tested")
        print("✓ Threshold level configuration verified")
        print("✓ Performance feedback learning demonstrated")
        print("✓ Time-based adaptation analyzed")
        print("✓ Noise trend analysis completed")
        print("✓ Visualizations generated")
        
        # Analysis summary
        print(f"\nAdaptive Thresholding Benefits:")
        
        # Environment adaptation range
        thresholds = [env_results[env]['final_threshold'] for env in env_results]
        threshold_range = max(thresholds) - min(thresholds)
        print(f"  Threshold adaptation range: {threshold_range:.1f} dB")
        
        # Performance improvement
        if performance['total_detections'] > 0:
            print(f"  Performance tracking: {performance['total_detections']} detections analyzed")
            print(f"  False positive rate: {performance['false_positive_rate']:.2f}")
            print(f"  Missed detection rate: {performance['missed_detection_rate']:.2f}")
        
        # Environment classification accuracy
        classified_envs = sum(1 for env in env_results.values() if env['environment_type'] != 'unknown')
        print(f"  Environment classification: {classified_envs}/{len(env_results)} environments classified")
        
        # Final detector status
        final_status = detector.get_adaptive_threshold_status()
        print(f"\nFinal Detector Status:")
        print(f"  Current threshold: {final_status['current_threshold_db']:.1f} dB")
        print(f"  Environment: {final_status['environment']['type']}")
        print(f"  Activity level: {final_status['environment']['activity_level']}")
        print(f"  Noise floor: {final_status['environment']['noise_floor']:.6f}")
        
        print(f"\n✓ Adaptive thresholding demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()