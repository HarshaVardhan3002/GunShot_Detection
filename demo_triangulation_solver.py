"""
Demo script for triangulation solver functionality.
"""
import numpy as np
import matplotlib.pyplot as plt
from tdoa_localizer import CrossCorrelationTDoALocalizer, MicrophonePosition
import time


def create_microphone_array(array_type="square", size=2.0):
    """Create different microphone array configurations."""
    positions = []
    
    if array_type == "square":
        # 4-microphone square array
        coords = [(0, 0), (size, 0), (size, size), (0, size)]
        for i, (x, y) in enumerate(coords):
            positions.append(MicrophonePosition(i, x, y, 0.0))
    
    elif array_type == "circular":
        # 8-microphone circular array
        radius = size / 2
        center_x, center_y = size / 2, size / 2
        for i in range(8):
            angle = 2 * np.pi * i / 8
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            positions.append(MicrophonePosition(i, x, y, 0.0))
    
    elif array_type == "linear":
        # 6-microphone linear array
        spacing = size / 5
        for i in range(6):
            x = i * spacing
            y = 0.0
            positions.append(MicrophonePosition(i, x, y, 0.0))
    
    return positions


def create_synthetic_tdoa_matrix(source_pos, mic_positions, sound_speed=343.0):
    """Create synthetic TDoA matrix for known source and microphone positions."""
    num_mics = len(mic_positions)
    tdoa_matrix = np.zeros((num_mics, num_mics))
    
    # Calculate distances from source to each microphone
    distances = []
    for mic_pos in mic_positions:
        distance = np.sqrt((source_pos[0] - mic_pos.x)**2 + 
                          (source_pos[1] - mic_pos.y)**2 + 
                          (source_pos[2] if len(source_pos) > 2 else 0.0 - mic_pos.z)**2)
        distances.append(distance)
    
    # Calculate TDoA matrix
    for i in range(num_mics):
        for j in range(num_mics):
            if i != j:
                tdoa_matrix[i, j] = (distances[j] - distances[i]) / sound_speed
    
    return tdoa_matrix


def demo_triangulation_accuracy():
    """Demonstrate triangulation accuracy with known source positions."""
    print("=== Triangulation Accuracy Demo ===\n")
    
    # Test different array configurations
    array_configs = [
        ("square", 2.0),
        ("circular", 3.0),
        ("linear", 4.0)
    ]
    
    for array_type, size in array_configs:
        print(f"--- {array_type.title()} Array (size: {size}m) ---")
        
        # Create microphone array
        mic_positions = create_microphone_array(array_type, size)
        print(f"Microphones: {len(mic_positions)}")
        for i, pos in enumerate(mic_positions):
            print(f"  Mic {i}: ({pos.x:.1f}, {pos.y:.1f})")
        
        # Initialize localizer
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=mic_positions,
            sample_rate=48000,
            sound_speed=343.0
        )
        
        # Test source positions
        if array_type == "square":
            test_sources = [
                (1.0, 1.0),    # Center
                (0.5, 0.5),    # Southwest
                (1.5, 1.5),    # Northeast
                (3.0, 1.0),    # East of array
            ]
        elif array_type == "circular":
            center = size / 2
            test_sources = [
                (center, center),           # Center
                (center + 0.5, center),     # East of center
                (center, center + 0.5),     # North of center
                (center + 2.0, center),     # Outside array
            ]
        else:  # linear
            test_sources = [
                (size/2, 0.0),     # Center of array
                (size/2, 1.0),     # North of center
                (size/2, -1.0),    # South of center
                (size + 1.0, 0.0), # East of array
            ]
        
        print(f"\nTriangulation Results:")
        print("Source Position -> Estimated Position (Error)")
        print("-" * 50)
        
        total_error = 0
        successful_triangulations = 0
        
        for source_x, source_y in test_sources:
            # Create synthetic TDoA matrix
            source_pos = (source_x, source_y, 0.0)
            tdoa_matrix = create_synthetic_tdoa_matrix(source_pos, mic_positions)
            
            # Perform triangulation
            start_time = time.time()
            result = localizer.triangulate_source(tdoa_matrix)
            calc_time = time.time() - start_time
            
            # Calculate error
            if result.confidence > 0.1:
                position_error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
                total_error += position_error
                successful_triangulations += 1
                
                print(f"({source_x:4.1f}, {source_y:4.1f}) -> ({result.x:4.1f}, {result.y:4.1f}) "
                      f"(error: {position_error:.2f}m, conf: {result.confidence:.2f}, "
                      f"time: {calc_time*1000:.1f}ms)")
            else:
                print(f"({source_x:4.1f}, {source_y:4.1f}) -> FAILED "
                      f"(conf: {result.confidence:.2f}, time: {calc_time*1000:.1f}ms)")
        
        if successful_triangulations > 0:
            avg_error = total_error / successful_triangulations
            print(f"\nAverage Error: {avg_error:.3f}m")
            print(f"Success Rate: {successful_triangulations}/{len(test_sources)} "
                  f"({100*successful_triangulations/len(test_sources):.0f}%)")
        
        print("\n" + "="*60 + "\n")


def demo_noise_robustness():
    """Demonstrate triangulation robustness to measurement noise."""
    print("=== Noise Robustness Demo ===\n")
    
    # Create square array
    mic_positions = create_microphone_array("square", 2.0)
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=48000
    )
    
    # Test source position
    source_pos = (1.0, 1.0, 0.0)
    clean_tdoa_matrix = create_synthetic_tdoa_matrix(source_pos, mic_positions)
    
    # Test different noise levels
    noise_levels = [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005]  # 0 to 5ms noise
    
    print("Noise Level (ms) | Position Error (m) | Confidence | Status")
    print("-" * 60)
    
    for noise_level in noise_levels:
        # Add noise to TDoA measurements
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, clean_tdoa_matrix.shape)
            # Maintain antisymmetry and zero diagonal
            noise = (noise - noise.T) / 2
            np.fill_diagonal(noise, 0)
            noisy_tdoa_matrix = clean_tdoa_matrix + noise
        else:
            noisy_tdoa_matrix = clean_tdoa_matrix
        
        # Perform triangulation
        result = localizer.triangulate_source(noisy_tdoa_matrix)
        
        # Calculate results
        if result.confidence > 0.05:
            position_error = np.sqrt((result.x - source_pos[0])**2 + 
                                   (result.y - source_pos[1])**2)
            status = "SUCCESS"
        else:
            position_error = float('inf')
            status = "FAILED"
        
        print(f"{noise_level*1000:8.1f}     | {position_error:13.3f}   | "
              f"{result.confidence:8.3f}   | {status}")
    
    print("\n" + "="*60 + "\n")


def demo_geometric_constraints():
    """Demonstrate geometric constraint validation."""
    print("=== Geometric Constraints Demo ===\n")
    
    # Create square array
    mic_positions = create_microphone_array("square", 2.0)
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=48000
    )
    
    # Test cases with different constraint violations
    test_cases = [
        {
            'name': 'Valid Source (Center)',
            'tdoa_matrix': create_synthetic_tdoa_matrix((1.0, 1.0, 0.0), mic_positions),
            'expected': 'PASS'
        },
        {
            'name': 'Valid Source (Offset)',
            'tdoa_matrix': create_synthetic_tdoa_matrix((3.0, 1.0, 0.0), mic_positions),
            'expected': 'PASS'
        },
        {
            'name': 'Unreasonable TDoAs (100ms)',
            'tdoa_matrix': np.array([
                [0.0,  0.1,  0.1,  0.1],
                [-0.1, 0.0,  0.1,  0.1],
                [-0.1, -0.1, 0.0,  0.1],
                [-0.1, -0.1, -0.1, 0.0]
            ]),
            'expected': 'FAIL'
        },
        {
            'name': 'Inconsistent TDoAs',
            'tdoa_matrix': np.array([
                [0.0,   0.01,  0.02,  0.03],
                [-0.01, 0.0,   0.05, -0.02],
                [-0.02, -0.05, 0.0,   0.04],
                [-0.03, 0.02, -0.04,  0.0]
            ]),
            'expected': 'FAIL'
        }
    ]
    
    print("Test Case                    | Result     | Confidence | Residual Error")
    print("-" * 70)
    
    for test_case in test_cases:
        result = localizer.triangulate_source(test_case['tdoa_matrix'])
        
        if result.confidence > 0.1:
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"{test_case['name']:28} | {status:10} | {result.confidence:8.3f}   | "
              f"{result.residual_error:10.6f}")
    
    print("\n" + "="*60 + "\n")


def demo_microphone_selection():
    """Demonstrate microphone selection for triangulation."""
    print("=== Microphone Selection Demo ===\n")
    
    # Create 8-microphone circular array
    mic_positions = create_microphone_array("circular", 4.0)
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=48000
    )
    
    # Create TDoA matrix with some poor quality measurements
    source_pos = (2.0, 2.0, 0.0)
    clean_tdoa_matrix = create_synthetic_tdoa_matrix(source_pos, mic_positions)
    
    # Simulate poor correlation for some microphone pairs
    poor_quality_tdoa = clean_tdoa_matrix.copy()
    
    # Add large errors to some pairs (simulating poor correlation)
    poor_quality_tdoa[0, 1] = 0.05  # 50ms error
    poor_quality_tdoa[1, 0] = -0.05
    poor_quality_tdoa[2, 3] = 0.08  # 80ms error
    poor_quality_tdoa[3, 2] = -0.08
    
    # Mock correlation history to simulate quality differences
    mock_correlation = {}
    for i in range(len(mic_positions)):
        for j in range(i + 1, len(mic_positions)):
            if (i == 0 and j == 1) or (i == 2 and j == 3):
                mock_correlation[f"{i}-{j}"] = 0.2  # Poor correlation
            else:
                mock_correlation[f"{i}-{j}"] = 0.8  # Good correlation
    
    localizer.correlation_history.append(mock_correlation)
    
    # Perform triangulation
    result = localizer.triangulate_source(poor_quality_tdoa)
    
    print(f"Source Position: ({source_pos[0]:.1f}, {source_pos[1]:.1f})")
    print(f"Estimated Position: ({result.x:.1f}, {result.y:.1f})")
    print(f"Position Error: {np.sqrt((result.x - source_pos[0])**2 + (result.y - source_pos[1])**2):.3f}m")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Microphones Used: {result.microphones_used}")
    print(f"Total Available: {len(mic_positions)}")
    
    # Show which microphones were excluded
    excluded_mics = [i for i in range(len(mic_positions)) if i not in result.microphones_used]
    if excluded_mics:
        print(f"Excluded Microphones: {excluded_mics} (due to poor quality)")
    else:
        print("All microphones used")
    
    print("\n" + "="*60 + "\n")


def demo_performance_analysis():
    """Demonstrate performance analysis of triangulation solver."""
    print("=== Performance Analysis ===\n")
    
    # Test different array sizes
    array_sizes = [4, 6, 8]
    
    for num_mics in array_sizes:
        print(f"--- {num_mics}-Microphone Array ---")
        
        # Create circular array
        mic_positions = []
        radius = 2.0
        for i in range(num_mics):
            angle = 2 * np.pi * i / num_mics
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            mic_positions.append(MicrophonePosition(i, x, y, 0.0))
        
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=mic_positions,
            sample_rate=48000
        )
        
        # Create test TDoA matrix
        source_pos = (0.5, 0.5, 0.0)
        tdoa_matrix = create_synthetic_tdoa_matrix(source_pos, mic_positions)
        
        # Benchmark performance
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            result = localizer.triangulate_source(tdoa_matrix)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        
        # Test final result
        final_result = localizer.triangulate_source(tdoa_matrix)
        position_error = np.sqrt((final_result.x - source_pos[0])**2 + 
                               (final_result.y - source_pos[1])**2)
        
        print(f"  Average Time: {avg_time*1000:.2f}ms")
        print(f"  Processing Rate: {1/avg_time:.0f} triangulations/second")
        print(f"  Position Error: {position_error:.3f}m")
        print(f"  Confidence: {final_result.confidence:.3f}")
        print(f"  Microphones Used: {len(final_result.microphones_used)}/{num_mics}")
        
        # Real-time performance assessment
        if avg_time < 0.01:
            performance = "Excellent"
        elif avg_time < 0.02:
            performance = "Good"
        elif avg_time < 0.05:
            performance = "Acceptable"
        else:
            performance = "Poor"
        
        print(f"  Real-time Performance: {performance}")
        print()


def visualize_triangulation_results():
    """Create a simple visualization of triangulation results."""
    print("=== Triangulation Visualization ===\n")
    
    try:
        # Create square array
        mic_positions = create_microphone_array("square", 2.0)
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=mic_positions,
            sample_rate=48000
        )
        
        # Test multiple source positions
        test_sources = [
            (1.0, 1.0),   # Center
            (0.5, 0.5),   # Southwest
            (1.5, 0.5),   # Southeast
            (1.5, 1.5),   # Northeast
            (0.5, 1.5),   # Northwest
            (2.5, 1.0),   # East
            (1.0, 2.5),   # North
        ]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot microphones
        mic_x = [pos.x for pos in mic_positions]
        mic_y = [pos.y for pos in mic_positions]
        plt.scatter(mic_x, mic_y, c='blue', s=100, marker='s', label='Microphones')
        
        # Annotate microphones
        for i, pos in enumerate(mic_positions):
            plt.annotate(f'M{i}', (pos.x, pos.y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # Test triangulation for each source
        for i, (source_x, source_y) in enumerate(test_sources):
            # Create TDoA matrix
            source_pos = (source_x, source_y, 0.0)
            tdoa_matrix = create_synthetic_tdoa_matrix(source_pos, mic_positions)
            
            # Triangulate
            result = localizer.triangulate_source(tdoa_matrix)
            
            # Plot results
            if result.confidence > 0.1:
                # Plot true source
                plt.scatter(source_x, source_y, c='red', s=80, marker='*', alpha=0.7)
                
                # Plot estimated source
                plt.scatter(result.x, result.y, c='green', s=60, marker='o', alpha=0.7)
                
                # Draw error line
                plt.plot([source_x, result.x], [source_y, result.y], 
                        'k--', alpha=0.5, linewidth=1)
                
                # Annotate
                error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
                plt.annotate(f'S{i}\nErr:{error:.2f}m', 
                           (source_x, source_y), xytext=(10, 10),
                           textcoords='offset points', fontsize=7)
        
        # Formatting
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Triangulation Results\n(Red=True, Green=Estimated, Blue=Microphones)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(['Error Lines', 'Microphones', 'True Sources', 'Estimated Sources'])
        
        # Save plot
        plt.savefig('triangulation_results.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'triangulation_results.png'")
        
        # Show statistics
        print("\nVisualization shows:")
        print("- Blue squares: Microphone positions")
        print("- Red stars: True source positions")
        print("- Green circles: Estimated source positions")
        print("- Dashed lines: Position errors")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        demo_triangulation_accuracy()
        demo_noise_robustness()
        demo_geometric_constraints()
        demo_microphone_selection()
        demo_performance_analysis()
        visualize_triangulation_results()
        
        print("Triangulation solver demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()