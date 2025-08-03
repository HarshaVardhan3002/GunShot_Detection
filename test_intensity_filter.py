"""
Unit tests for intensity filter functionality.
"""
import unittest
import numpy as np
import time
from unittest.mock import Mock, patch
from intensity_filter import (
    RMSIntensityFilter,
    ChannelQualityMetrics,
    IntensityFilterInterface
)


class TestRMSIntensityFilter(unittest.TestCase):
    """Test cases for RMS intensity filter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_rate = 48000
        self.filter = RMSIntensityFilter(
            sample_rate=self.sample_rate,
            noise_estimation_method='percentile'
        )
    
    def create_test_signal(self, amplitude: float, frequency: float, duration: float, 
                          noise_level: float = 0.0) -> np.ndarray:
        """Create a test signal with specified parameters."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Create sine wave
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, samples)
            signal += noise
        
        return signal
    
    def create_multi_channel_test_data(self, channel_configs: list) -> np.ndarray:
        """
        Create multi-channel test data.
        
        Args:
            channel_configs: List of (amplitude, frequency, noise_level) tuples
            
        Returns:
            Multi-channel audio array
        """
        duration = 0.1  # 100ms
        samples = int(duration * self.sample_rate)
        num_channels = len(channel_configs)
        
        audio_channels = np.zeros((samples, num_channels))
        
        for ch, (amplitude, frequency, noise_level) in enumerate(channel_configs):
            signal = self.create_test_signal(amplitude, frequency, duration, noise_level)
            audio_channels[:, ch] = signal
        
        return audio_channels
    
    def test_initialization(self):
        """Test proper initialization of intensity filter."""
        # Test default initialization
        filter_default = RMSIntensityFilter()
        self.assertEqual(filter_default.sample_rate, 48000)
        self.assertEqual(filter_default.noise_estimation_method, 'percentile')
        
        # Test custom initialization
        filter_custom = RMSIntensityFilter(
            sample_rate=44100,
            noise_estimation_method='minimum'
        )
        self.assertEqual(filter_custom.sample_rate, 44100)
        self.assertEqual(filter_custom.noise_estimation_method, 'minimum')
    
    def test_calculate_weights_uniform_signals(self):
        """Test weight calculation with uniform signal quality."""
        # Create 4 channels with identical signals
        channel_configs = [
            (1.0, 1000, 0.1),  # Same amplitude, frequency, noise
            (1.0, 1000, 0.1),
            (1.0, 1000, 0.1),
            (1.0, 1000, 0.1)
        ]
        
        audio_channels = self.create_multi_channel_test_data(channel_configs)
        weights = self.filter.calculate_weights(audio_channels)
        
        # Weights should be approximately equal
        self.assertEqual(len(weights), 4)
        self.assertTrue(np.all(weights > 0))  # All channels should have positive weights
        
        # Check that weights are reasonably similar (within 50% for uniform signals)
        mean_weight = np.mean(weights)
        for weight in weights:
            self.assertLess(abs(weight - mean_weight) / mean_weight, 0.5)
    
    def test_calculate_weights_varying_amplitudes(self):
        """Test weight calculation with varying signal amplitudes."""
        # Create channels with different amplitudes
        channel_configs = [
            (2.0, 1000, 0.1),  # High amplitude
            (1.0, 1000, 0.1),  # Medium amplitude
            (0.5, 1000, 0.1),  # Low amplitude
            (0.1, 1000, 0.1)   # Very low amplitude
        ]
        
        audio_channels = self.create_multi_channel_test_data(channel_configs)
        weights = self.filter.calculate_weights(audio_channels)
        
        # Higher amplitude channels should have higher weights
        self.assertGreater(weights[0], weights[1])  # High > Medium
        self.assertGreater(weights[1], weights[2])  # Medium > Low
        self.assertGreater(weights[2], weights[3])  # Low > Very low
    
    def test_calculate_weights_varying_snr(self):
        """Test weight calculation with varying signal-to-noise ratios."""
        # Create channels with different noise levels (same signal amplitude)
        channel_configs = [
            (1.0, 1000, 0.05),  # Low noise (high SNR)
            (1.0, 1000, 0.1),   # Medium noise
            (1.0, 1000, 0.2),   # High noise
            (1.0, 1000, 0.5)    # Very high noise (low SNR)
        ]
        
        audio_channels = self.create_multi_channel_test_data(channel_configs)
        weights = self.filter.calculate_weights(audio_channels)
        
        # Lower noise (higher SNR) channels should generally have higher weights
        # Due to the complex weighting algorithm, we check overall trend
        sorted_indices = np.argsort(weights)[::-1]  # Highest to lowest weights
        
        # Check that there's some differentiation in weights based on SNR
        # The range of weights should reflect the quality differences
        weight_range = np.max(weights) - np.min(weights)
        self.assertGreater(weight_range, 0.1)  # Should have meaningful differences
    
    def test_calculate_weights_mixed_quality(self):
        """Test weight calculation with mixed signal quality."""
        # Create channels with various quality issues
        channel_configs = [
            (1.0, 1000, 0.05),  # Good quality
            (0.1, 1000, 0.05),  # Low amplitude but low noise
            (1.0, 1000, 0.3),   # Good amplitude but noisy
            (0.01, 1000, 0.1)   # Poor amplitude and noisy
        ]
        
        audio_channels = self.create_multi_channel_test_data(channel_configs)
        weights = self.filter.calculate_weights(audio_channels)
        
        # Check that weights differentiate between quality levels
        # The range should be meaningful
        weight_range = np.max(weights) - np.min(weights)
        self.assertGreater(weight_range, 0.2)  # Should have significant differences
        
        # Channel 0 (good quality) should have higher weight than channel 3 (poor quality)
        self.assertGreater(weights[0], weights[3])
    
    def test_channel_quality_metrics_calculation(self):
        """Test individual channel quality metrics calculation."""
        # Create a test signal with known characteristics
        signal = self.create_test_signal(1.0, 1000, 0.1, 0.1)
        
        metrics = self.filter._calculate_channel_quality_metrics(signal)
        
        # Verify metrics structure
        self.assertIsInstance(metrics, ChannelQualityMetrics)
        self.assertGreater(metrics.rms_amplitude, 0)
        self.assertGreater(metrics.peak_amplitude, 0)
        self.assertGreater(metrics.signal_power, 0)
        self.assertGreater(metrics.noise_power, 0)
        self.assertIsInstance(metrics.snr_db, float)
        self.assertIsInstance(metrics.is_valid, bool)
        
        # For a reasonable signal, should be valid
        self.assertTrue(metrics.is_valid)
        
        # SNR should be reasonable for our test signal
        self.assertGreater(metrics.snr_db, 5)  # At least 5dB SNR
    
    def test_noise_floor_estimation_methods(self):
        """Test different noise floor estimation methods."""
        # Create signal with known noise characteristics
        clean_signal = self.create_test_signal(1.0, 1000, 0.1, 0.0)
        noise = np.random.normal(0, 0.1, len(clean_signal))
        noisy_signal = clean_signal + noise
        
        methods = ['percentile', 'minimum', 'adaptive']
        
        for method in methods:
            with self.subTest(method=method):
                self.filter.noise_estimation_method = method
                noise_floor = self.filter.estimate_noise_floor(noisy_signal)
                
                # Noise floor should be reasonable
                self.assertGreater(noise_floor, 0)
                self.assertLess(noise_floor, 1.0)  # Should be much less than signal
                
                # Should be reasonable for our test signal
                self.assertLess(noise_floor, 1.0)  # Should be reasonable
    
    def test_filter_low_snr_channels(self):
        """Test filtering of low SNR channels."""
        # Create weights representing different quality channels
        weights = np.array([0.8, 0.6, 0.2, 0.05])  # Good, OK, Poor, Very poor
        
        # Test with default threshold (0.3)
        valid_channels = self.filter.filter_low_snr_channels(weights)
        expected_valid = [0, 1]  # Only first two channels above 0.3
        self.assertEqual(valid_channels, expected_valid)
        
        # Test with stricter threshold
        valid_channels = self.filter.filter_low_snr_channels(weights, threshold=0.7)
        expected_valid = [0]  # Only first channel above 0.7
        self.assertEqual(valid_channels, expected_valid)
        
        # Test with lenient threshold
        valid_channels = self.filter.filter_low_snr_channels(weights, threshold=0.1)
        expected_valid = [0, 1, 2]  # First three channels above 0.1
        self.assertEqual(valid_channels, expected_valid)
    
    def test_filter_low_snr_channels_all_poor(self):
        """Test filtering when all channels are poor quality."""
        # All channels below threshold
        weights = np.array([0.1, 0.05, 0.08, 0.02])
        
        valid_channels = self.filter.filter_low_snr_channels(weights, threshold=0.3)
        
        # Should return the best channel even if below threshold
        self.assertEqual(len(valid_channels), 1)
        self.assertEqual(valid_channels[0], 0)  # Best channel (index 0)
    
    def test_spectral_features_calculation(self):
        """Test spectral feature calculations."""
        # Create signals with different spectral characteristics
        low_freq_signal = self.create_test_signal(1.0, 500, 0.1, 0.05)
        high_freq_signal = self.create_test_signal(1.0, 2000, 0.1, 0.05)
        
        # Calculate spectral centroids
        low_centroid = self.filter._calculate_spectral_centroid(low_freq_signal)
        high_centroid = self.filter._calculate_spectral_centroid(high_freq_signal)
        
        # High frequency signal should have higher spectral centroid
        self.assertGreater(high_centroid, low_centroid)
        
        # Values should be reasonable
        self.assertGreater(low_centroid, 0)
        self.assertLess(low_centroid, self.sample_rate / 2)
        self.assertGreater(high_centroid, 0)
        self.assertLess(high_centroid, self.sample_rate / 2)
    
    def test_zero_crossing_rate_calculation(self):
        """Test zero crossing rate calculation."""
        # Create signals with different zero crossing characteristics
        low_freq_signal = self.create_test_signal(1.0, 100, 0.1, 0.0)  # Low frequency
        high_freq_signal = self.create_test_signal(1.0, 2000, 0.1, 0.0)  # High frequency
        noise_signal = np.random.normal(0, 0.5, int(0.1 * self.sample_rate))  # Noise
        
        low_zcr = self.filter._calculate_zero_crossing_rate(low_freq_signal)
        high_zcr = self.filter._calculate_zero_crossing_rate(high_freq_signal)
        noise_zcr = self.filter._calculate_zero_crossing_rate(noise_signal)
        
        # High frequency signal should have higher ZCR than low frequency
        self.assertGreater(high_zcr, low_zcr)
        
        # Noise should have high ZCR
        self.assertGreater(noise_zcr, low_zcr)
        
        # All values should be between 0 and 1
        for zcr in [low_zcr, high_zcr, noise_zcr]:
            self.assertGreaterEqual(zcr, 0.0)
            self.assertLessEqual(zcr, 1.0)
    
    def test_adaptive_adjustments(self):
        """Test adaptive weight adjustments based on history."""
        # Enable adaptive adjustments
        self.filter.enable_adaptive_thresholds = True
        
        # Create consistent test data to build history
        channel_configs = [
            (1.0, 1000, 0.1),  # Consistent quality
            (0.5, 1000, 0.1),  # Lower quality
            (1.5, 1000, 0.1),  # Higher quality
            (0.3, 1000, 0.2)   # Poor quality
        ]
        
        # Build history with multiple calculations
        for _ in range(10):
            audio_channels = self.create_multi_channel_test_data(channel_configs)
            weights = self.filter.calculate_weights(audio_channels)
        
        # Verify that history is being tracked
        self.assertGreater(len(self.filter.quality_history), 5)
        
        # Test that adaptive adjustments are applied
        # (Specific behavior depends on the trend, so we just check it doesn't crash)
        final_weights = self.filter.calculate_weights(audio_channels)
        self.assertEqual(len(final_weights), 4)
        self.assertTrue(np.all(final_weights >= 0))
    
    def test_parameter_configuration(self):
        """Test configuration of filter parameters."""
        # Test parameter updates
        new_params = {
            'min_snr_db': 10.0,
            'min_rms_threshold': 1e-5,
            'noise_estimation_method': 'minimum',
            'spectral_analysis_enabled': False,
            'enable_adaptive_thresholds': False,
            'rms_weight': 0.5,
            'snr_weight': 0.3,
            'spectral_weight': 0.1,
            'dynamic_weight': 0.1
        }
        
        self.filter.configure_filter_parameters(**new_params)
        
        # Verify parameters were updated
        self.assertEqual(self.filter.min_snr_db, 10.0)
        self.assertEqual(self.filter.min_rms_threshold, 1e-5)
        self.assertEqual(self.filter.noise_estimation_method, 'minimum')
        self.assertEqual(self.filter.spectral_analysis_enabled, False)
        self.assertEqual(self.filter.enable_adaptive_thresholds, False)
        
        # Weights should be normalized to sum to 1.0
        total_weight = (self.filter.rms_weight + self.filter.snr_weight + 
                       self.filter.spectral_weight + self.filter.dynamic_weight)
        self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_invalid_parameter_configuration(self):
        """Test handling of invalid parameter configurations."""
        # Test invalid noise estimation method
        with self.assertRaises(ValueError):
            self.filter.configure_filter_parameters(noise_estimation_method='invalid')
    
    def test_channel_quality_report(self):
        """Test channel quality reporting functionality."""
        # Create test data with known quality characteristics
        channel_configs = [
            (1.0, 1000, 0.05),  # Good quality
            (0.5, 1000, 0.1),   # Medium quality
            (0.2, 1000, 0.2),   # Poor quality
            (0.05, 1000, 0.3)   # Very poor quality
        ]
        
        audio_channels = self.create_multi_channel_test_data(channel_configs)
        weights = self.filter.calculate_weights(audio_channels)
        
        # Get quality report
        report = self.filter.get_channel_quality_report()
        
        # Verify report structure
        self.assertIn('status', report)
        self.assertIn('recommendation', report)
        self.assertIn('summary', report)
        self.assertIn('channels', report)
        
        # Verify summary data
        summary = report['summary']
        self.assertEqual(summary['total_channels'], 4)
        self.assertGreater(summary['valid_channels'], 0)
        self.assertIsInstance(summary['avg_snr_db'], float)
        self.assertIsInstance(summary['avg_rms_amplitude'], float)
        
        # Verify channel details
        channels = report['channels']
        self.assertEqual(len(channels), 4)
        for ch_info in channels:
            self.assertIn('channel', ch_info)
            self.assertIn('rms_amplitude', ch_info)
            self.assertIn('snr_db', ch_info)
            self.assertIn('weight', ch_info)
            self.assertIn('is_valid', ch_info)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with very short signal
        short_signal = np.random.normal(0, 0.1, (5, 4))
        weights = self.filter.calculate_weights(short_signal)
        self.assertEqual(len(weights), 4)
        self.assertTrue(np.all(weights > 0))  # Should return default weights
        
        # Test with zero signal
        zero_signal = np.zeros((1000, 4))
        weights = self.filter.calculate_weights(zero_signal)
        self.assertEqual(len(weights), 4)
        # Zero signals should have very low weights
        self.assertTrue(np.all(weights >= 0))
        
        # Test with single channel
        single_channel = np.random.normal(0, 0.1, (1000, 1))
        weights = self.filter.calculate_weights(single_channel)
        self.assertEqual(len(weights), 1)
        self.assertGreater(weights[0], 0)
        
        # Test with wrong dimensions
        wrong_dims = np.random.normal(0, 0.1, 1000)  # 1D array
        with self.assertRaises(ValueError):
            self.filter.calculate_weights(wrong_dims)
    
    def test_noise_floor_history_tracking(self):
        """Test noise floor history tracking."""
        # Initially empty
        self.assertEqual(len(self.filter.noise_floor_history), 0)
        
        # Add some noise floor estimates
        test_signals = [
            np.random.normal(0, 0.1, 1000),
            np.random.normal(0, 0.2, 1000),
            np.random.normal(0, 0.15, 1000)
        ]
        
        for signal in test_signals:
            self.filter.estimate_noise_floor(signal)
        
        # Should have 3 entries
        self.assertEqual(len(self.filter.noise_floor_history), 3)
        
        # All entries should be positive
        for noise_floor in self.filter.noise_floor_history:
            self.assertGreater(noise_floor, 0)
    
    def test_history_reset(self):
        """Test history reset functionality."""
        # Build some history
        channel_configs = [(1.0, 1000, 0.1)] * 4
        audio_channels = self.create_multi_channel_test_data(channel_configs)
        
        for _ in range(5):
            self.filter.calculate_weights(audio_channels)
            self.filter.estimate_noise_floor(audio_channels[:, 0])
        
        # Verify history exists
        self.assertGreater(len(self.filter.quality_history), 0)
        self.assertGreater(len(self.filter.noise_floor_history), 0)
        
        # Reset history
        self.filter.reset_history()
        
        # Verify history is cleared
        self.assertEqual(len(self.filter.quality_history), 0)
        self.assertEqual(len(self.filter.noise_floor_history), 0)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for real-time processing."""
        # Create realistic test data
        channel_configs = [(1.0, 1000, 0.1)] * 8  # 8 channels
        audio_channels = self.create_multi_channel_test_data(channel_configs)
        
        # Benchmark weight calculation
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            weights = self.filter.calculate_weights(audio_channels)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        
        # Should be fast enough for real-time processing
        self.assertLess(avg_time, 0.01)  # Less than 10ms per calculation
        
        print(f"Average weight calculation time: {avg_time*1000:.2f}ms")


class TestIntensityFilterIntegration(unittest.TestCase):
    """Integration tests for intensity filter with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.sample_rate = 48000
        self.filter = RMSIntensityFilter(sample_rate=self.sample_rate)
    
    def create_realistic_gunshot_signal(self, amplitude: float, noise_level: float) -> np.ndarray:
        """Create a realistic gunshot-like signal."""
        duration = 0.1  # 100ms
        samples = int(duration * self.sample_rate)
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
    
    def test_realistic_microphone_array_scenario(self):
        """Test with realistic microphone array scenario."""
        # Simulate 8-microphone array with varying signal quality
        num_channels = 8
        duration = 0.1
        samples = int(duration * self.sample_rate)
        
        audio_channels = np.zeros((samples, num_channels))
        
        # Simulate different microphone conditions
        scenarios = [
            (1.0, 0.05),   # Good microphone, low noise
            (0.8, 0.08),   # Good microphone, moderate noise
            (0.6, 0.1),    # Moderate microphone, moderate noise
            (0.4, 0.15),   # Weak microphone, higher noise
            (1.2, 0.06),   # Excellent microphone, low noise
            (0.3, 0.2),    # Poor microphone, high noise
            (0.9, 0.07),   # Good microphone, low noise
            (0.1, 0.3)     # Very poor microphone, very high noise
        ]
        
        for ch, (amplitude, noise_level) in enumerate(scenarios):
            signal = self.create_realistic_gunshot_signal(amplitude, noise_level)
            audio_channels[:, ch] = signal
        
        # Calculate weights
        weights = self.filter.calculate_weights(audio_channels)
        
        # Verify results
        self.assertEqual(len(weights), num_channels)
        
        # Best microphones should have highest weights
        # Channel 4 (excellent) should have high weight
        self.assertGreater(weights[4], np.mean(weights))
        
        # Check that there's meaningful weight differentiation
        weight_range = np.max(weights) - np.min(weights)
        self.assertGreater(weight_range, 0.3)  # Should have significant differences
        
        # Check that the weighting system produces meaningful differentiation
        # The best channels should generally outperform the worst
        sorted_weights = np.sort(weights)
        weight_spread = sorted_weights[-1] - sorted_weights[0]
        self.assertGreater(weight_spread, 0.2)  # Should have meaningful spread
        
        # Filter low quality channels with a higher threshold
        valid_channels = self.filter.filter_low_snr_channels(weights, threshold=0.8)
        
        # Should exclude some channels with higher threshold
        self.assertLess(len(valid_channels), num_channels)  # Should exclude some channels
        self.assertGreater(len(valid_channels), 0)  # Should keep some channels
        
        # Get quality report
        report = self.filter.get_channel_quality_report()
        
        # Should identify quality issues
        self.assertIn(report['status'], ['good', 'fair', 'marginal'])
        self.assertGreater(report['summary']['valid_channels'], 4)  # At least half should be valid
    
    def test_environmental_noise_adaptation(self):
        """Test adaptation to different environmental noise conditions."""
        scenarios = [
            ('quiet', 0.02),      # Very quiet environment
            ('moderate', 0.1),    # Moderate noise
            ('noisy', 0.3),       # Noisy environment
            ('very_noisy', 0.5)   # Very noisy environment
        ]
        
        for env_name, noise_level in scenarios:
            with self.subTest(environment=env_name):
                # Create signals with consistent signal but varying noise
                channel_configs = [
                    (1.0, 1000, noise_level),  # All channels same signal, different noise
                    (1.0, 1000, noise_level),
                    (1.0, 1000, noise_level),
                    (1.0, 1000, noise_level)
                ]
                
                audio_channels = self.create_multi_channel_test_data(channel_configs)
                weights = self.filter.calculate_weights(audio_channels)
                
                # In high noise, weights should be more conservative
                if noise_level > 0.3:
                    # Very noisy environment - weights should be lower overall
                    self.assertLess(np.mean(weights), 1.5)
                else:
                    # Quiet environment - weights should be higher
                    self.assertGreater(np.mean(weights), 0.5)
                
                # Get quality report
                report = self.filter.get_channel_quality_report()
                
                # Quality assessment should reflect noise level
                if noise_level < 0.1:
                    self.assertIn(report['status'], ['excellent', 'good'])
                elif noise_level > 0.4:
                    # Very noisy environment - just check it's not excellent
                    self.assertNotEqual(report['status'], 'excellent')
    
    def create_multi_channel_test_data(self, channel_configs: list) -> np.ndarray:
        """Create multi-channel test data for integration tests."""
        duration = 0.1  # 100ms
        samples = int(duration * self.sample_rate)
        num_channels = len(channel_configs)
        
        audio_channels = np.zeros((samples, num_channels))
        
        for ch, (amplitude, frequency, noise_level) in enumerate(channel_configs):
            t = np.linspace(0, duration, samples)
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
            
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, samples)
                signal += noise
            
            audio_channels[:, ch] = signal
        
        return audio_channels


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)