"""
Demo script for adaptive channel selector functionality.
"""
import numpy as np
import matplotlib.pyplot as plt
from adaptive_channel_selector import AdaptiveChannelSelector, SelectionStrategy
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


def create_multi_channel_test_data(channel_configs: list, sample_rate: int = 48000) -> np.ndarray:
    """Create multi-channel test data."""
    duration = 0.1  # 100ms
    samples = int(duration * sample_rate)
    num_channels = len(channel_configs)
    
    audio_channels = np.zeros((samples, num_channels))
    
    for ch, (amplitude, frequency, noise_level) in enumerate(channel_configs):
        signal = create_test_signal(amplitude, frequency, duration, noise_level, sample_rate)
        audio_channels[:, ch] = signal
    
    return audio_channels


def demo_basic_selection_strategies():
    """Demonstrate different selection strategies."""
    print("=== Selection Strategies Demo ===\n")
    
    # Create 8-channel array with varying quality
    channel_configs = [
        (1.5, 1000, 0.05),  # Excellent
        (1.2, 1000, 0.08),  # Very good
        (1.0, 1000, 0.1),   # Good
        (0.8, 1000, 0.15),  # Fair
        (0.5, 1000, 0.2),   # Poor
        (0.3, 1000, 0.3),   # Very poor
        (1.1, 1000, 0.09),  # Good
        (0.9, 1000, 0.12)   # Fair
    ]
    
    audio_channels = create_multi_channel_test_data(channel_configs)
    
    # Test different strategies
    strategies = [
        SelectionStrategy.QUALITY_BASED,
        SelectionStrategy.CONFIDENCE_WEIGHTED,
        SelectionStrategy.ADAPTIVE_THRESHOLD,
        SelectionStrategy.GEOMETRIC_OPTIMAL
    ]
    
    selector = AdaptiveChannelSelector(num_channels=8)
    
    print("Channel Quality Overview:")
    print("Ch | Amplitude | Noise | Quality")
    print("-" * 32)
    quality_labels = ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor', 'Very Poor', 'Good', 'Fair']
    for ch, (amplitude, frequency, noise_level) in enumerate(channel_configs):
        print(f" {ch} |   {amplitude:.1f}     | {noise_level:.2f}  | {quality_labels[ch]}")
    
    print(f"\nStrategy Comparison (selecting 4 channels):")
    print("Strategy              | Selected Channels | Confidence | Fallback")
    print("-" * 65)
    
    for strategy in strategies:
        selector.primary_strategy = strategy
        result = selector.select_channels(
            audio_channels, 
            detection_confidence=0.7,
            required_channels=4
        )
        
        selected_str = str(result.selected_channels)
        fallback_str = "Yes" if result.fallback_applied else "No"
        
        print(f"{strategy.value:20} | {selected_str:16} | {result.selection_confidence:8.3f}   | {fallback_str}")
    
    print()


def demo_confidence_based_adaptation():
    """Demonstrate adaptation based on detection confidence."""
    print("=== Confidence-Based Adaptation Demo ===\n")
    
    # Create uniform quality channels
    channel_configs = [(1.0, 1000, 0.1)] * 8
    audio_channels = create_multi_channel_test_data(channel_configs)
    
    selector = AdaptiveChannelSelector(num_channels=8)
    selector.primary_strategy = SelectionStrategy.CONFIDENCE_WEIGHTED
    
    # Test with different confidence levels
    confidence_levels = [0.9, 0.7, 0.5, 0.3, 0.1]
    
    print("Detection Confidence | Channels Selected | Strategy Used")
    print("-" * 55)
    
    for confidence in confidence_levels:
        result = selector.select_channels(
            audio_channels,
            detection_confidence=confidence
        )
        
        print(f"       {confidence:.1f}           |        {len(result.selected_channels)}         | {result.strategy_used.value}")
    
    print(f"\nAdaptation Logic:")
    print("- High confidence (>0.8): Use minimum channels (more selective)")
    print("- Medium confidence (0.5-0.8): Use preferred number of channels")
    print("- Low confidence (<0.5): Use more channels for robustness")
    print()


def demo_performance_feedback_learning():
    """Demonstrate learning from performance feedback."""
    print("=== Performance Feedback Learning Demo ===\n")
    
    # Create array with some consistently poor channels
    channel_configs = [
        (1.2, 1000, 0.08),  # Good channels
        (1.1, 1000, 0.09),
        (1.3, 1000, 0.07),
        (1.0, 1000, 0.1),
        (0.2, 1000, 0.4),   # Poor channels (4, 5)
        (0.1, 1000, 0.5),
        (1.15, 1000, 0.085), # Good channels
        (1.25, 1000, 0.075)
    ]
    
    audio_channels = create_multi_channel_test_data(channel_configs)
    selector = AdaptiveChannelSelector(num_channels=8)
    
    print("Simulating learning over 20 selections...")
    print("Poor channels: 4, 5 (will cause poor triangulation performance)")
    print()
    
    # Track channel usage over time
    channel_usage_history = []
    performance_history = []
    
    for iteration in range(20):
        result = selector.select_channels(audio_channels, detection_confidence=0.7)
        
        # Simulate triangulation performance
        # Poor performance if channels 4 or 5 are used
        poor_channels_used = any(ch in [4, 5] for ch in result.selected_channels)
        
        if poor_channels_used:
            # Poor performance when bad channels are used
            triangulation_confidence = 0.3 + np.random.normal(0, 0.1)
            triangulation_error = 0.8 + np.random.normal(0, 0.2)
        else:
            # Good performance when bad channels are avoided
            triangulation_confidence = 0.8 + np.random.normal(0, 0.1)
            triangulation_error = 0.1 + np.random.normal(0, 0.05)
        
        # Clip to reasonable ranges
        triangulation_confidence = np.clip(triangulation_confidence, 0.1, 0.95)
        triangulation_error = np.clip(triangulation_error, 0.01, 2.0)
        
        # Provide feedback
        selector.update_performance_feedback(
            selected_channels=result.selected_channels,
            triangulation_confidence=triangulation_confidence,
            triangulation_error=triangulation_error
        )
        
        # Track usage
        channel_usage = [0] * 8
        for ch in result.selected_channels:
            channel_usage[ch] = 1
        channel_usage_history.append(channel_usage)
        performance_history.append(triangulation_confidence)
        
        if iteration % 5 == 4:  # Print every 5 iterations
            poor_usage = sum(1 for ch in result.selected_channels if ch in [4, 5])
            print(f"Iteration {iteration+1:2d}: Selected {result.selected_channels}, "
                  f"Poor channels used: {poor_usage}, Performance: {triangulation_confidence:.2f}")
    
    # Analyze learning
    early_usage = np.mean(channel_usage_history[:5], axis=0)
    late_usage = np.mean(channel_usage_history[-5:], axis=0)
    
    print(f"\nLearning Analysis:")
    print("Channel | Early Usage | Late Usage | Change")
    print("-" * 42)
    
    for ch in range(8):
        change = late_usage[ch] - early_usage[ch]
        change_str = f"{change:+.2f}"
        print(f"   {ch}    |    {early_usage[ch]:.2f}     |   {late_usage[ch]:.2f}    | {change_str}")
    
    print(f"\nPerformance Improvement:")
    early_perf = np.mean(performance_history[:5])
    late_perf = np.mean(performance_history[-5:])
    improvement = late_perf - early_perf
    print(f"Early average performance: {early_perf:.3f}")
    print(f"Late average performance:  {late_perf:.3f}")
    print(f"Improvement: {improvement:+.3f}")
    print()


def demo_adaptive_thresholding():
    """Demonstrate adaptive thresholding behavior."""
    print("=== Adaptive Thresholding Demo ===\n")
    
    selector = AdaptiveChannelSelector(num_channels=8)
    selector.adaptation_enabled = True
    
    # Simulate different detection confidence patterns
    scenarios = [
        ("High Confidence Period", [0.9] * 15),
        ("Low Confidence Period", [0.2] * 15),
        ("Mixed Confidence Period", [0.9, 0.8, 0.3, 0.7, 0.5] * 3)
    ]
    
    print("Scenario                | Initial Threshold | Final Threshold | Change")
    print("-" * 70)
    
    for scenario_name, confidence_pattern in scenarios:
        # Reset adaptation state
        selector.reset_adaptation_state()
        initial_threshold = selector.base_quality_threshold
        
        # Simulate the confidence pattern
        for confidence in confidence_pattern:
            selector._update_adaptation_state(confidence)
        
        # Get final threshold
        final_threshold = getattr(selector, 'current_quality_threshold', selector.base_quality_threshold)
        change = final_threshold - initial_threshold
        
        print(f"{scenario_name:22} |      {initial_threshold:.3f}      |     {final_threshold:.3f}     | {change:+.3f}")
    
    print(f"\nAdaptation Logic:")
    print("- High confidence detections → Higher threshold (more selective)")
    print("- Low confidence detections → Lower threshold (less selective)")
    print("- Environmental noise also affects thresholds")
    print()


def demo_fallback_strategies():
    """Demonstrate fallback strategy usage."""
    print("=== Fallback Strategies Demo ===\n")
    
    # Create challenging scenario that might trigger fallbacks
    channel_configs = [
        (0.1, 1000, 0.5),   # Very poor quality channels
        (0.05, 1000, 0.6),
        (0.08, 1000, 0.55),
        (0.12, 1000, 0.45),
        (0.15, 1000, 0.4),
        (0.09, 1000, 0.52),
        (0.11, 1000, 0.48),
        (0.07, 1000, 0.58)
    ]
    
    audio_channels = create_multi_channel_test_data(channel_configs)
    
    selector = AdaptiveChannelSelector(num_channels=8)
    
    # Set high threshold to force fallbacks
    selector.base_quality_threshold = 1.0  # Very high threshold
    
    print("Testing with very high quality threshold (1.0) to trigger fallbacks...")
    print("All channels have poor quality (0.05-0.15 amplitude, 0.4-0.6 noise)")
    print()
    
    # Test multiple selections
    fallback_count = 0
    strategy_usage = {}
    
    for i in range(10):
        result = selector.select_channels(
            audio_channels,
            detection_confidence=0.6,
            required_channels=4
        )
        
        if result.fallback_applied:
            fallback_count += 1
        
        strategy = result.strategy_used.value
        strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        print(f"Selection {i+1}: Strategy={strategy:20} Fallback={str(result.fallback_applied):5} "
              f"Channels={len(result.selected_channels)} Confidence={result.selection_confidence:.3f}")
    
    print(f"\nFallback Summary:")
    print(f"Fallback usage: {fallback_count}/10 selections ({fallback_count*10}%)")
    print(f"Strategy usage: {strategy_usage}")
    print()


def demo_channel_recommendations():
    """Demonstrate channel recommendation system."""
    print("=== Channel Recommendations Demo ===\n")
    
    # Test different array conditions
    scenarios = [
        {
            'name': 'Healthy Array',
            'configs': [(1.2, 1000, 0.08)] * 8,
            'performance_pattern': [0.8] * 20
        },
        {
            'name': 'Degraded Array',
            'configs': [
                (1.2, 1000, 0.08), (0.1, 1000, 0.5), (1.1, 1000, 0.09), (0.05, 1000, 0.6),
                (1.3, 1000, 0.07), (1.0, 1000, 0.1), (0.08, 1000, 0.55), (1.15, 1000, 0.085)
            ],
            'performance_pattern': [0.3, 0.8, 0.4, 0.7] * 5  # Mixed performance
        },
        {
            'name': 'Noisy Environment',
            'configs': [(0.8, 1000, 0.3)] * 8,
            'performance_pattern': [0.4] * 20
        }
    ]
    
    for scenario in scenarios:
        print(f"--- {scenario['name']} ---")
        
        audio_channels = create_multi_channel_test_data(scenario['configs'])
        selector = AdaptiveChannelSelector(num_channels=8)
        
        # Simulate usage pattern
        for i, performance in enumerate(scenario['performance_pattern']):
            result = selector.select_channels(audio_channels)
            
            # Simulate triangulation performance
            error = 0.1 + (1 - performance) * 0.5  # Convert performance to error
            selector.update_performance_feedback(
                selected_channels=result.selected_channels,
                triangulation_confidence=performance,
                triangulation_error=error
            )
        
        # Get recommendations
        recommendations = selector.get_channel_recommendations()
        
        print(f"Status: {recommendations['status'].upper()}")
        
        if recommendations['issues']:
            print("Issues identified:")
            for issue in recommendations['issues']:
                print(f"  - {issue}")
        
        if recommendations['suggestions']:
            print("Suggestions:")
            for suggestion in recommendations['suggestions']:
                print(f"  - {suggestion}")
        
        print()


def demo_performance_analysis():
    """Demonstrate performance analysis and statistics."""
    print("=== Performance Analysis Demo ===\n")
    
    selector = AdaptiveChannelSelector(num_channels=8)
    
    # Create test data
    channel_configs = [(1.0, 1000, 0.1)] * 8
    audio_channels = create_multi_channel_test_data(channel_configs)
    
    # Perform multiple selections with different strategies
    strategies = list(SelectionStrategy)
    
    print("Performance Comparison Across Strategies:")
    print("Strategy              | Avg Time (ms) | Avg Channels | Success Rate")
    print("-" * 68)
    
    for strategy in strategies:
        selector.primary_strategy = strategy
        
        # Benchmark performance
        times = []
        channel_counts = []
        successes = 0
        
        for _ in range(50):
            start_time = time.time()
            result = selector.select_channels(audio_channels)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            channel_counts.append(len(result.selected_channels))
            
            if len(result.selected_channels) >= selector.min_channels:
                successes += 1
        
        avg_time = np.mean(times)
        avg_channels = np.mean(channel_counts)
        success_rate = successes / 50 * 100
        
        print(f"{strategy.value:20} | {avg_time:9.2f}   | {avg_channels:10.1f}   | {success_rate:9.0f}%")
    
    # Overall statistics
    stats = selector.get_selection_statistics()
    
    print(f"\nOverall Statistics:")
    print(f"Total selections: {stats['total_selections']}")
    print(f"Fallback usage: {stats['fallback_count']}")
    print(f"Average channels selected: {stats['average_channels_selected']:.1f}")
    print(f"Adaptation events: {stats['adaptation_events']}")
    
    if 'recent_performance' in stats:
        perf = stats['recent_performance']
        print(f"Recent performance:")
        print(f"  Average confidence: {perf['avg_confidence']:.3f}")
        print(f"  Average error: {perf['avg_error']:.3f}m")
        print(f"  Average performance: {perf['avg_performance']:.3f}")
    
    print()


def visualize_adaptation_behavior():
    """Create visualization of adaptation behavior."""
    print("=== Adaptation Behavior Visualization ===\n")
    
    try:
        # Simulate adaptation over time with changing conditions
        selector = AdaptiveChannelSelector(num_channels=8)
        selector.adaptation_enabled = True
        
        # Create different phases
        phases = [
            ("Good Conditions", 20, 0.8),
            ("Degrading Conditions", 15, 0.4),
            ("Poor Conditions", 10, 0.2),
            ("Recovery", 15, 0.7)
        ]
        
        # Track metrics over time
        time_points = []
        thresholds = []
        performances = []
        channel_usage = {i: [] for i in range(8)}
        
        time_point = 0
        
        # Create test audio
        channel_configs = [
            (1.2, 1000, 0.08), (1.0, 1000, 0.1), (0.8, 1000, 0.15), (1.1, 1000, 0.09),
            (0.3, 1000, 0.3), (0.2, 1000, 0.4), (1.3, 1000, 0.07), (0.9, 1000, 0.12)
        ]
        audio_channels = create_multi_channel_test_data(channel_configs)
        
        for phase_name, duration, base_performance in phases:
            for _ in range(duration):
                # Add some noise to performance
                performance = base_performance + np.random.normal(0, 0.1)
                performance = np.clip(performance, 0.1, 0.95)
                
                # Update adaptation state
                selector._update_adaptation_state(performance)
                
                # Perform selection
                result = selector.select_channels(audio_channels)
                
                # Provide feedback
                error = 0.1 + (1 - performance) * 0.5
                selector.update_performance_feedback(
                    selected_channels=result.selected_channels,
                    triangulation_confidence=performance,
                    triangulation_error=error
                )
                
                # Record metrics
                time_points.append(time_point)
                current_threshold = getattr(selector, 'current_quality_threshold', selector.base_quality_threshold)
                thresholds.append(current_threshold)
                performances.append(performance)
                
                # Record channel usage
                for ch in range(8):
                    usage = 1 if ch in result.selected_channels else 0
                    channel_usage[ch].append(usage)
                
                time_point += 1
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Performance and threshold over time
        ax1.plot(time_points, performances, label='Detection Performance', alpha=0.7)
        ax1.plot(time_points, thresholds, label='Adaptive Threshold', alpha=0.7)
        ax1.set_ylabel('Value')
        ax1.set_title('Performance and Adaptive Threshold Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add phase boundaries
        phase_boundaries = [0, 20, 35, 45, 60]
        phase_names = ["Good", "Degrading", "Poor", "Recovery"]
        for i, boundary in enumerate(phase_boundaries[1:]):
            ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
            if i < len(phase_names):
                ax1.text(phase_boundaries[i] + 5, 0.9, phase_names[i], rotation=0, alpha=0.7)
        
        # Plot 2: Channel usage over time
        colors = plt.cm.tab10(np.linspace(0, 1, 8))
        for ch in range(8):
            # Smooth the usage data for better visualization
            smoothed_usage = np.convolve(channel_usage[ch], np.ones(5)/5, mode='same')
            ax2.plot(time_points, smoothed_usage, label=f'Ch {ch}', color=colors[ch], alpha=0.7)
        
        ax2.set_ylabel('Usage Probability')
        ax2.set_title('Channel Usage Over Time (Smoothed)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Channel usage heatmap
        usage_matrix = np.array([channel_usage[ch] for ch in range(8)])
        im = ax3.imshow(usage_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax3.set_ylabel('Channel')
        ax3.set_xlabel('Time')
        ax3.set_title('Channel Usage Heatmap')
        ax3.set_yticks(range(8))
        ax3.set_yticklabels([f'Ch {i}' for i in range(8)])
        
        # Add colorbar
        plt.colorbar(im, ax=ax3, label='Usage (0=Not Used, 1=Used)')
        
        plt.tight_layout()
        plt.savefig('adaptive_channel_selector_analysis.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'adaptive_channel_selector_analysis.png'")
        
        # Print summary
        print(f"\nAdaptation Summary:")
        print(f"Initial threshold: {selector.base_quality_threshold:.3f}")
        print(f"Final threshold: {thresholds[-1]:.3f}")
        print(f"Threshold range: {min(thresholds):.3f} - {max(thresholds):.3f}")
        
        # Channel usage summary
        print(f"\nChannel Usage Summary:")
        for ch in range(8):
            avg_usage = np.mean(channel_usage[ch])
            print(f"Channel {ch}: {avg_usage:.1%} usage")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        demo_basic_selection_strategies()
        demo_confidence_based_adaptation()
        demo_performance_feedback_learning()
        demo_adaptive_thresholding()
        demo_fallback_strategies()
        demo_channel_recommendations()
        demo_performance_analysis()
        visualize_adaptation_behavior()
        
        print("Adaptive channel selector demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()