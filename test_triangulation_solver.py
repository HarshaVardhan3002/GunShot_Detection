"""
Unit tests for triangulation solver functionality.
"""
import unittest
import numpy as np
import time
from unittest.mock import Mock, patch
from tdoa_localizer import (
    CrossCorrelationTDoALocalizer, 
    MicrophonePosition, 
    LocationResult
)


class TestTriangulationSolver(unittest.TestCase):
    """Test cases for triangulation solver using multilateration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a square microphone array for testing
        self.mic_positions = [
            MicrophonePosition(0, 0.0, 0.0, 0.0),    # Bottom-left
            MicrophonePosition(1, 2.0, 0.0, 0.0),    # Bottom-right
            MicrophonePosition(2, 2.0, 2.0, 0.0),    # Top-right
            MicrophonePosition(3, 0.0, 2.0, 0.0),    # Top-left
        ]
        
        self.sample_rate = 48000
        self.sound_speed = 343.0
        
        self.localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=self.mic_positions,
            sample_rate=self.sample_rate,
            sound_speed=self.sound_speed
        )
    
    def create_synthetic_tdoa_matrix(self, source_x, source_y, source_z=0.0):
        """Create a synthetic TDoA matrix for a known source position."""
        num_mics = len(self.mic_positions)
        tdoa_matrix = np.zeros((num_mics, num_mics))
        
        # Calculate distances from source to each microphone
        distances = []
        for mic_pos in self.mic_positions:
            distance = np.sqrt((source_x - mic_pos.x)**2 + 
                             (source_y - mic_pos.y)**2 + 
                             (source_z - mic_pos.z)**2)
            distances.append(distance)
        
        # Calculate TDoA matrix
        for i in range(num_mics):
            for j in range(num_mics):
                if i != j:
                    # Time difference = (distance_j - distance_i) / sound_speed
                    tdoa_matrix[i, j] = (distances[j] - distances[i]) / self.sound_speed
        
        return tdoa_matrix
    
    def test_triangulation_center_source(self):
        """Test triangulation with source at center of microphone array."""
        # Source at center of 2x2 array
        source_x, source_y = 1.0, 1.0
        
        # Create synthetic TDoA matrix
        tdoa_matrix = self.create_synthetic_tdoa_matrix(source_x, source_y)
        
        # Perform triangulation
        result = self.localizer.triangulate_source(tdoa_matrix)
        
        # Verify result
        self.assertIsInstance(result, LocationResult)
        self.assertGreater(result.confidence, 0.5)  # Should have good confidence
        self.assertLess(result.residual_error, 0.01)  # Low error expected
        
        # Check position accuracy (within 10cm)
        position_error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
        self.assertLess(position_error, 0.1, f"Position error {position_error:.3f}m too large")
        
        # Verify metadata
        self.assertGreater(len(result.microphones_used), 2)
        self.assertIsNotNone(result.tdoa_matrix)
        self.assertGreater(result.timestamp, 0)
    
    def test_triangulation_offset_source(self):
        """Test triangulation with source offset from array center."""
        # Source east of array
        source_x, source_y = 3.0, 1.0
        
        # Create synthetic TDoA matrix
        tdoa_matrix = self.create_synthetic_tdoa_matrix(source_x, source_y)
        
        # Perform triangulation
        result = self.localizer.triangulate_source(tdoa_matrix)
        
        # Verify result
        self.assertGreater(result.confidence, 0.3)  # Reasonable confidence
        self.assertLess(result.residual_error, 0.02)  # Acceptable error
        
        # Check position accuracy (within 20cm for offset source)
        position_error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
        self.assertLess(position_error, 0.2, f"Position error {position_error:.3f}m too large")
    
    def test_triangulation_multiple_positions(self):
        """Test triangulation with multiple known source positions."""
        test_positions = [
            (1.0, 1.0),   # Center
            (0.5, 0.5),   # Southwest quadrant
            (1.5, 0.5),   # Southeast quadrant
            (1.5, 1.5),   # Northeast quadrant
            (0.5, 1.5),   # Northwest quadrant
            (2.5, 1.0),   # East of array
            (1.0, 2.5),   # North of array
        ]
        
        for source_x, source_y in test_positions:
            with self.subTest(position=(source_x, source_y)):
                # Create synthetic TDoA matrix
                tdoa_matrix = self.create_synthetic_tdoa_matrix(source_x, source_y)
                
                # Perform triangulation
                result = self.localizer.triangulate_source(tdoa_matrix)
                
                # Verify basic result properties
                self.assertIsInstance(result, LocationResult)
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
                self.assertGreaterEqual(result.residual_error, 0.0)
                
                # Check position accuracy (more lenient for edge cases)
                position_error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
                self.assertLess(position_error, 0.5, 
                               f"Position error {position_error:.3f}m too large for source at ({source_x}, {source_y})")
    
    def test_triangulation_with_noise(self):
        """Test triangulation robustness to TDoA measurement noise."""
        source_x, source_y = 1.0, 1.0
        
        # Create clean TDoA matrix
        clean_tdoa_matrix = self.create_synthetic_tdoa_matrix(source_x, source_y)
        
        # Test different noise levels
        noise_levels = [0.0001, 0.0005, 0.001]  # 0.1ms, 0.5ms, 1ms noise
        
        for noise_level in noise_levels:
            with self.subTest(noise_level=noise_level):
                # Add noise to TDoA measurements
                noise = np.random.normal(0, noise_level, clean_tdoa_matrix.shape)
                # Keep diagonal zero and maintain antisymmetry
                noise = (noise - noise.T) / 2
                np.fill_diagonal(noise, 0)
                
                noisy_tdoa_matrix = clean_tdoa_matrix + noise
                
                # Perform triangulation
                result = self.localizer.triangulate_source(noisy_tdoa_matrix)
                
                # Verify result quality degrades gracefully with noise
                self.assertIsInstance(result, LocationResult)
                self.assertGreaterEqual(result.confidence, 0.0)
                
                # For low noise, we expect reasonable results
                if noise_level <= 0.0001 and result.confidence > 0.1:
                    position_error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
                    self.assertLess(position_error, 0.5)  # Relaxed tolerance
    
    def test_microphone_selection(self):
        """Test microphone selection for triangulation."""
        # Create TDoA matrix with some poor quality measurements
        source_x, source_y = 1.0, 1.0
        tdoa_matrix = self.create_synthetic_tdoa_matrix(source_x, source_y)
        
        # Simulate poor correlation for some microphone pairs
        # by adding large TDoA values that exceed reasonable limits
        tdoa_matrix[0, 1] = 0.1  # 100ms - unreasonably large
        tdoa_matrix[1, 0] = -0.1
        
        # Mock correlation history to simulate poor correlation quality
        mock_correlation = {
            '0-1': 0.1,  # Poor correlation
            '0-2': 0.8,  # Good correlation
            '0-3': 0.7,  # Good correlation
            '1-2': 0.6,  # Moderate correlation
            '1-3': 0.5,  # Moderate correlation
            '2-3': 0.9   # Excellent correlation
        }
        self.localizer.correlation_history.append(mock_correlation)
        
        # Perform triangulation
        result = self.localizer.triangulate_source(tdoa_matrix)
        
        # Should still get a reasonable result by excluding poor microphones
        self.assertIsInstance(result, LocationResult)
        self.assertGreater(len(result.microphones_used), 2)
        
        # Microphone 0 might be excluded due to poor correlation with mic 1
        # But we should still get a valid triangulation
        if result.confidence > 0.1:  # If triangulation succeeded
            position_error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
            self.assertLess(position_error, 1.0)  # Relaxed tolerance
    
    def test_geometric_constraint_validation(self):
        """Test geometric constraint validation."""
        # Create TDoA matrix that would result in unreasonable solution
        tdoa_matrix = np.array([
            [0.0,  0.1,  0.2,  0.3],   # Very large TDoAs
            [-0.1, 0.0,  0.1,  0.2],
            [-0.2, -0.1, 0.0,  0.1],
            [-0.3, -0.2, -0.1, 0.0]
        ])
        
        # This should fail geometric constraints
        result = self.localizer.triangulate_source(tdoa_matrix)
        
        # Should return failed result
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.residual_error, float('inf'))
        self.assertEqual(len(result.microphones_used), 0)
    
    def test_insufficient_microphones(self):
        """Test handling of insufficient microphones."""
        # Test with minimum 3 microphones but poor quality
        insufficient_mics = self.mic_positions[:3]
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=insufficient_mics,
            sample_rate=self.sample_rate,
            sound_speed=self.sound_speed
        )
        
        # Create 3x3 TDoA matrix with unreasonable values
        tdoa_matrix = np.array([
            [0.0,  0.1,  0.2],   # Very large TDoAs
            [-0.1, 0.0,  0.1],
            [-0.2, -0.1, 0.0]
        ])
        
        # Should fail due to poor quality measurements
        result = localizer.triangulate_source(tdoa_matrix)
        
        # Should return failed result or very low confidence
        self.assertLessEqual(result.confidence, 0.1)
    
    def test_optimization_convergence(self):
        """Test optimization convergence with different initial conditions."""
        source_x, source_y = 1.5, 0.5
        tdoa_matrix = self.create_synthetic_tdoa_matrix(source_x, source_y)
        
        # Run triangulation multiple times to test consistency
        results = []
        for _ in range(5):
            result = self.localizer.triangulate_source(tdoa_matrix)
            results.append(result)
        
        # All results should be similar (convergent)
        positions = [(r.x, r.y) for r in results if r.confidence > 0.1]
        
        if len(positions) > 1:
            # Calculate standard deviation of positions
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
            
            x_std = np.std(x_positions)
            y_std = np.std(y_positions)
            
            # Should be consistent (low standard deviation)
            self.assertLess(x_std, 0.1, "X positions not consistent across runs")
            self.assertLess(y_std, 0.1, "Y positions not consistent across runs")
    
    def test_confidence_calculation(self):
        """Test confidence calculation for different scenarios."""
        # High quality scenario
        source_x, source_y = 1.0, 1.0
        clean_tdoa_matrix = self.create_synthetic_tdoa_matrix(source_x, source_y)
        
        # Mock high-quality correlation history
        high_quality_correlation = {
            '0-1': 0.9, '0-2': 0.9, '0-3': 0.9,
            '1-2': 0.9, '1-3': 0.9, '2-3': 0.9
        }
        self.localizer.correlation_history.append(high_quality_correlation)
        
        result_high = self.localizer.triangulate_source(clean_tdoa_matrix)
        
        # Low quality scenario
        noisy_tdoa_matrix = clean_tdoa_matrix + np.random.normal(0, 0.002, clean_tdoa_matrix.shape)
        np.fill_diagonal(noisy_tdoa_matrix, 0)
        
        # Mock low-quality correlation history
        low_quality_correlation = {
            '0-1': 0.3, '0-2': 0.3, '0-3': 0.3,
            '1-2': 0.3, '1-3': 0.3, '2-3': 0.3
        }
        self.localizer.correlation_history.append(low_quality_correlation)
        
        result_low = self.localizer.triangulate_source(noisy_tdoa_matrix)
        
        # High quality should have higher confidence
        if result_high.confidence > 0 and result_low.confidence > 0:
            self.assertGreater(result_high.confidence, result_low.confidence)
    
    def test_residual_error_calculation(self):
        """Test residual error calculation accuracy."""
        source_x, source_y = 1.0, 1.0
        
        # Perfect TDoA matrix (no noise)
        perfect_tdoa_matrix = self.create_synthetic_tdoa_matrix(source_x, source_y)
        result_perfect = self.localizer.triangulate_source(perfect_tdoa_matrix)
        
        # Noisy TDoA matrix
        noise = np.random.normal(0, 0.001, perfect_tdoa_matrix.shape)
        noise = (noise - noise.T) / 2  # Maintain antisymmetry
        np.fill_diagonal(noise, 0)
        noisy_tdoa_matrix = perfect_tdoa_matrix + noise
        result_noisy = self.localizer.triangulate_source(noisy_tdoa_matrix)
        
        # Perfect case should have lower residual error
        if result_perfect.confidence > 0 and result_noisy.confidence > 0:
            self.assertLess(result_perfect.residual_error, result_noisy.residual_error)
            
        # Both should have reasonable residual errors
        if result_perfect.confidence > 0:
            self.assertLess(result_perfect.residual_error, 0.005)  # 5ms equivalent
        if result_noisy.confidence > 0:
            self.assertLess(result_noisy.residual_error, 0.02)   # 20ms equivalent
    
    def test_invalid_input_handling(self):
        """Test handling of invalid input matrices."""
        # Wrong size matrix
        wrong_size_matrix = np.random.random((3, 3))
        result = self.localizer.triangulate_source(wrong_size_matrix)
        self.assertEqual(result.confidence, 0.0)
        
        # Matrix with NaN values
        nan_matrix = self.create_synthetic_tdoa_matrix(1.0, 1.0)
        nan_matrix[0, 1] = np.nan
        nan_matrix[1, 0] = -np.nan  # Maintain antisymmetry
        result = self.localizer.triangulate_source(nan_matrix)
        # NaN handling might still produce a result, so just check it doesn't crash
        self.assertIsInstance(result, LocationResult)
        
        # Matrix with infinite values
        inf_matrix = self.create_synthetic_tdoa_matrix(1.0, 1.0)
        inf_matrix[0, 1] = np.inf
        inf_matrix[1, 0] = -np.inf  # Maintain antisymmetry
        result = self.localizer.triangulate_source(inf_matrix)
        # Infinite values should be filtered out by quality checks
        self.assertIsInstance(result, LocationResult)


class TestTriangulationIntegration(unittest.TestCase):
    """Integration tests for triangulation with full TDoA pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create 8-microphone circular array
        self.mic_positions = []
        radius = 2.0
        for i in range(8):
            angle = 2 * np.pi * i / 8
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            self.mic_positions.append(MicrophonePosition(i, x, y, 0.0))
        
        self.localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=self.mic_positions,
            sample_rate=48000
        )
    
    def simulate_audio_from_source(self, source_x, source_y, signal_length=2048):
        """Simulate audio received from a source at given position."""
        # Create impulse signal
        base_signal = np.zeros(signal_length)
        base_signal[100] = 1.0  # Impulse at sample 100
        
        # Add some noise
        base_signal += 0.1 * np.random.normal(0, 1, signal_length)
        
        # Calculate delays and create multi-channel audio
        audio_channels = np.zeros((signal_length, len(self.mic_positions)))
        
        for i, mic_pos in enumerate(self.mic_positions):
            distance = np.sqrt((source_x - mic_pos.x)**2 + (source_y - mic_pos.y)**2)
            delay_samples = int(distance / 343.0 * 48000)
            
            if delay_samples < signal_length - 100:
                delayed_signal = np.zeros(signal_length)
                delayed_signal[100 + delay_samples:] = base_signal[:signal_length - 100 - delay_samples]
                audio_channels[:, i] = delayed_signal
            else:
                audio_channels[:, i] = 0.1 * np.random.normal(0, 1, signal_length)
        
        return audio_channels
    
    def test_end_to_end_triangulation(self):
        """Test complete pipeline from audio to triangulated position."""
        # Test source positions
        test_positions = [
            (0.0, 0.0),   # Center
            (1.0, 0.0),   # East
            (0.0, 1.0),   # North
            (-1.0, 0.0),  # West
            (0.0, -1.0),  # South
        ]
        
        for source_x, source_y in test_positions:
            with self.subTest(position=(source_x, source_y)):
                # Simulate audio
                audio_channels = self.simulate_audio_from_source(source_x, source_y)
                
                # Calculate TDoA
                tdoa_matrix = self.localizer.calculate_tdoa(audio_channels)
                
                # Triangulate
                result = self.localizer.triangulate_source(tdoa_matrix)
                
                # Verify result
                self.assertIsInstance(result, LocationResult)
                
                # Check position accuracy (relaxed for integration test)
                if result.confidence > 0.1:
                    position_error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
                    self.assertLess(position_error, 1.0, 
                                   f"Position error {position_error:.3f}m too large for source at ({source_x}, {source_y})")
    
    def test_performance_benchmarks(self):
        """Test triangulation performance benchmarks."""
        # Create test data
        source_x, source_y = 1.0, 1.0
        audio_channels = self.simulate_audio_from_source(source_x, source_y)
        tdoa_matrix = self.localizer.calculate_tdoa(audio_channels)
        
        # Benchmark triangulation speed
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            result = self.localizer.triangulate_source(tdoa_matrix)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        
        # Should be fast enough for real-time processing (relaxed tolerance)
        self.assertLess(avg_time, 0.02, f"Triangulation too slow: {avg_time*1000:.2f}ms")
        
        print(f"Triangulation performance: {avg_time*1000:.2f}ms average")
    
    def test_accuracy_vs_array_size(self):
        """Test triangulation accuracy with different array sizes."""
        source_x, source_y = 0.5, 0.5
        
        # Test with different numbers of microphones
        for num_mics in [4, 6, 8]:
            with self.subTest(num_mics=num_mics):
                # Use subset of microphones
                subset_mics = self.mic_positions[:num_mics]
                localizer = CrossCorrelationTDoALocalizer(
                    microphone_positions=subset_mics,
                    sample_rate=48000
                )
                
                # Simulate audio
                audio_channels = self.simulate_audio_from_source(source_x, source_y, 1024)
                audio_channels = audio_channels[:, :num_mics]  # Use only subset
                
                # Calculate TDoA and triangulate
                tdoa_matrix = localizer.calculate_tdoa(audio_channels)
                result = localizer.triangulate_source(tdoa_matrix)
                
                # More microphones should generally give better results
                self.assertIsInstance(result, LocationResult)
                
                if result.confidence > 0.1:
                    position_error = np.sqrt((result.x - source_x)**2 + (result.y - source_y)**2)
                    # Accuracy should improve with more microphones
                    if num_mics >= 6:
                        self.assertLess(position_error, 1.5)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)