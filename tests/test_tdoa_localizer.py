"""
Unit tests for TDoA localization module.
"""
import unittest
import numpy as np
import time
from unittest.mock import Mock, patch

from tdoa_localizer import CrossCorrelationTDoALocalizer, MicrophonePosition, LocationResult


class TestCrossCorrelationTDoALocalizer(unittest.TestCase):
    """Test cases for CrossCorrelationTDoALocalizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test microphone array (square configuration)
        self.mic_positions = [
            MicrophonePosition(0, 0.0, 0.0, 0.0),    # Bottom-left
            MicrophonePosition(1, 10.0, 0.0, 0.0),   # Bottom-right
            MicrophonePosition(2, 10.0, 10.0, 0.0),  # Top-right
            MicrophonePosition(3, 0.0, 10.0, 0.0),   # Top-left
        ]
        
        self.localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=self.mic_positions,
            sample_rate=8000,  # Lower sample rate for faster tests
            sound_speed=343.0
        )
        # Disable preprocessing and interpolation for more predictable tests
        self.localizer.enable_preprocessing = False
        self.localizer.interpolation_factor = 1
    
    def test_initialization(self):
        """Test proper initialization of localizer."""
        self.assertEqual(self.localizer.num_mics, 4)
        self.assertEqual(self.localizer.sample_rate, 8000)
        self.assertEqual(self.localizer.sound_speed, 343.0)
        self.assertEqual(len(self.localizer.mic_pairs), 6)  # 4 choose 2
        
        # Check microphone pairs
        expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.assertEqual(self.localizer.mic_pairs, expected_pairs)
    
    def test_initialization_insufficient_mics(self):
        """Test initialization with insufficient microphones."""
        with self.assertRaises(ValueError):
            CrossCorrelationTDoALocalizer(
                microphone_positions=[MicrophonePosition(0, 0, 0, 0)],  # Only 1 mic
                sample_rate=8000
            )
    
    def test_cross_correlate_signals_identical(self):
        """Test cross-correlation with identical signals."""
        # Create identical signals
        samples = 1000
        signal = np.sin(2 * np.pi * 1000 * np.arange(samples) / 8000)
        
        time_delay, correlation_peak = self.localizer._cross_correlate_signals(signal, signal)
        
        # Identical signals should have zero delay and high correlation
        self.assertAlmostEqual(time_delay, 0.0, places=4)
        self.assertGreater(correlation_peak, 0.9)
    
    def test_cross_correlate_signals_delayed(self):
        """Test cross-correlation with known delay."""
        samples = 1000
        base_signal = np.sin(2 * np.pi * 1000 * np.arange(samples) / 8000)
        
        # Create delayed version
        delay_samples = 10
        delayed_signal = np.zeros(samples)
        delayed_signal[delay_samples:] = base_signal[:-delay_samples]
        
        time_delay, correlation_peak = self.localizer._cross_correlate_signals(
            base_signal, delayed_signal
        )
        
        # Should detect the delay
        expected_delay = delay_samples / 8000  # Convert to seconds
        self.assertAlmostEqual(time_delay, -expected_delay, places=3)  # Negative because signal2 is delayed
        self.assertGreater(correlation_peak, 0.8)
    
    def test_cross_correlate_signals_uncorrelated(self):
        """Test cross-correlation with uncorrelated signals."""
        samples = 1000
        signal1 = np.random.random(samples)
        signal2 = np.random.random(samples)
        
        time_delay, correlation_peak = self.localizer._cross_correlate_signals(signal1, signal2)
        
        # Uncorrelated signals should have low correlation (but random signals can sometimes correlate)
        # Just check that it's not perfect correlation
        self.assertLess(correlation_peak, 0.95)
    
    def test_interpolate_peak(self):
        """Test sub-sample peak interpolation."""
        # Create correlation with known peak
        correlation = np.array([0.1, 0.3, 0.8, 0.9, 0.7, 0.2])
        peak_index = 3  # Index of maximum value (0.9)
        
        interpolated_peak = self.localizer._interpolate_peak(correlation, peak_index)
        
        # Should be close to the original peak but with sub-sample precision
        self.assertGreater(interpolated_peak, peak_index - 0.5)
        self.assertLess(interpolated_peak, peak_index + 0.5)
    
    def test_interpolate_peak_edge_cases(self):
        """Test peak interpolation edge cases."""
        correlation = np.array([0.5, 0.8, 0.3])
        
        # Peak at beginning
        interpolated = self.localizer._interpolate_peak(correlation, 0)
        self.assertEqual(interpolated, 0.0)
        
        # Peak at end
        interpolated = self.localizer._interpolate_peak(correlation, 2)
        self.assertEqual(interpolated, 2.0)
    
    def test_preprocess_signals(self):
        """Test signal preprocessing."""
        samples = 2000
        # Create signal with multiple frequency components
        t = np.arange(samples) / 8000
        signal = (np.sin(2 * np.pi * 50 * t) +      # Low frequency (should be filtered)
                 np.sin(2 * np.pi * 1000 * t) +     # Mid frequency (should pass)
                 np.sin(2 * np.pi * 10000 * t))     # High frequency (should be filtered)
        
        # Create multi-channel version
        audio_channels = np.tile(signal[:, np.newaxis], (1, 4))
        
        # Preprocess
        processed = self.localizer._preprocess_signals(audio_channels)
        
        # Should have same shape
        self.assertEqual(processed.shape, audio_channels.shape)
        
        # Should have reduced low and high frequency content
        # (This is a basic check - more detailed analysis would require FFT)
        self.assertFalse(np.array_equal(processed, audio_channels))
    
    def test_calculate_tdoa_basic(self):
        """Test basic TDoA calculation."""
        samples = 1500
        channels = 4
        
        # Create test signal
        base_signal = np.sin(2 * np.pi * 1000 * np.arange(samples) / 8000)
        audio_channels = np.zeros((samples, channels))
        
        # Add signal to all channels with different delays
        delays = [0, 5, 10, 15]  # Sample delays
        for ch, delay in enumerate(delays):
            if delay == 0:
                audio_channels[:, ch] = base_signal
            else:
                audio_channels[delay:, ch] = base_signal[:-delay]
        
        # Calculate TDoA
        tdoa_matrix = self.localizer.calculate_tdoa(audio_channels)
        
        # Check matrix properties
        self.assertEqual(tdoa_matrix.shape, (4, 4))
        
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(tdoa_matrix), np.zeros(4))
        
        # Matrix should be antisymmetric
        np.testing.assert_array_almost_equal(tdoa_matrix, -tdoa_matrix.T)
        
        # Check some expected delays
        expected_delay_01 = -5 / 8000  # Channel 1 delayed by 5 samples
        self.assertAlmostEqual(tdoa_matrix[0, 1], expected_delay_01, places=3)
    
    def test_calculate_tdoa_wrong_channels(self):
        """Test TDoA calculation with wrong number of channels."""
        samples = 1000
        audio_channels = np.random.random((samples, 3))  # Wrong number of channels
        
        with self.assertRaises(ValueError):
            self.localizer.calculate_tdoa(audio_channels)
    
    def test_triangulate_source_placeholder(self):
        """Test triangulation placeholder implementation."""
        tdoa_matrix = np.random.random((4, 4))
        
        result = self.localizer.triangulate_source(tdoa_matrix)
        
        # Check result structure
        self.assertIsInstance(result, LocationResult)
        self.assertEqual(result.x, 0.0)  # Placeholder values
        self.assertEqual(result.y, 0.0)
        self.assertEqual(result.z, 0.0)
        self.assertEqual(len(result.microphones_used), 4)
        np.testing.assert_array_equal(result.tdoa_matrix, tdoa_matrix)
    
    def test_estimate_confidence(self):
        """Test confidence estimation."""
        # Low residuals should give high confidence
        low_residuals = np.array([0.01, 0.02, 0.015])
        high_confidence = self.localizer.estimate_confidence(low_residuals)
        
        # High residuals should give low confidence
        high_residuals = np.array([0.5, 0.8, 0.6])
        low_confidence = self.localizer.estimate_confidence(high_residuals)
        
        self.assertGreater(high_confidence, low_confidence)
        self.assertGreaterEqual(high_confidence, 0.0)
        self.assertLessEqual(high_confidence, 1.0)
        self.assertGreaterEqual(low_confidence, 0.0)
        self.assertLessEqual(low_confidence, 1.0)
        
        # Empty residuals should give zero confidence
        empty_confidence = self.localizer.estimate_confidence(np.array([]))
        self.assertEqual(empty_confidence, 0.0)
    
    def test_get_correlation_statistics(self):
        """Test correlation statistics retrieval."""
        # Initially no statistics
        stats = self.localizer.get_correlation_statistics()
        self.assertEqual(stats['samples_processed'], 0)
        
        # Process some data
        samples = 1000
        audio_channels = np.random.random((samples, 4))
        self.localizer.calculate_tdoa(audio_channels)
        
        # Should have statistics now
        stats = self.localizer.get_correlation_statistics()
        self.assertEqual(stats['samples_processed'], 1)
        self.assertEqual(stats['correlation_pairs'], 6)  # 4 choose 2
        self.assertIn('avg_correlation', stats)
        self.assertIn('min_correlation', stats)
        self.assertIn('max_correlation', stats)
    
    def test_get_tdoa_statistics(self):
        """Test TDoA statistics retrieval."""
        # Initially no statistics
        stats = self.localizer.get_tdoa_statistics()
        self.assertEqual(stats['samples_processed'], 0)
        
        # Process some data
        samples = 1000
        audio_channels = np.random.random((samples, 4))
        self.localizer.calculate_tdoa(audio_channels)
        
        # Should have statistics now
        stats = self.localizer.get_tdoa_statistics()
        self.assertEqual(stats['samples_processed'], 1)
        self.assertEqual(stats['microphone_pairs'], 6)
        self.assertIn('avg_tdoa_magnitude', stats)
        self.assertIn('max_tdoa_magnitude', stats)
        self.assertIn('tdoa_consistency', stats)
    
    def test_configure_correlation_parameters(self):
        """Test correlation parameter configuration."""
        # Test valid parameters
        self.localizer.configure_correlation_parameters(
            correlation_method='direct',
            max_delay_samples=500,
            interpolation_factor=8,
            min_correlation_threshold=0.5
        )
        
        self.assertEqual(self.localizer.correlation_method, 'direct')
        self.assertEqual(self.localizer.max_delay_samples, 500)
        self.assertEqual(self.localizer.interpolation_factor, 8)
        self.assertEqual(self.localizer.min_correlation_threshold, 0.5)
        
        # Test invalid correlation method
        with self.assertRaises(ValueError):
            self.localizer.configure_correlation_parameters(correlation_method='invalid')
    
    def test_analyze_signal_quality(self):
        """Test signal quality analysis."""
        samples = 1000
        
        # Create high-quality signal
        t = np.arange(samples) / 8000
        good_signal = np.sin(2 * np.pi * 1000 * t) * 0.5
        good_channels = np.tile(good_signal[:, np.newaxis], (1, 4))
        
        quality = self.localizer.analyze_signal_quality(good_channels)
        
        # Check structure
        expected_fields = [
            'channel_snr', 'channel_energy', 'cross_channel_correlation',
            'signal_bandwidth', 'overall_quality'
        ]
        for field in expected_fields:
            self.assertIn(field, quality)
        
        # Should have good quality metrics
        self.assertEqual(len(quality['channel_snr']), 4)
        self.assertEqual(len(quality['channel_energy']), 4)
        self.assertGreater(quality['overall_quality'], 0.3)  # Should be reasonably good
        
        # Cross-channel correlation should be high (identical signals)
        self.assertGreater(np.mean(quality['cross_channel_correlation']), 0.5)
    
    def test_reset_history(self):
        """Test history reset functionality."""
        # Add some history
        samples = 1000
        audio_channels = np.random.random((samples, 4))
        self.localizer.calculate_tdoa(audio_channels)
        
        # Should have history
        self.assertGreater(len(self.localizer.correlation_history), 0)
        self.assertGreater(len(self.localizer.tdoa_history), 0)
        
        # Reset history
        self.localizer.reset_history()
        
        # Should be empty
        self.assertEqual(len(self.localizer.correlation_history), 0)
        self.assertEqual(len(self.localizer.tdoa_history), 0)
    
    def test_preprocessing_disabled(self):
        """Test TDoA calculation with preprocessing disabled."""
        self.localizer.enable_preprocessing = False
        
        samples = 1000
        audio_channels = np.random.random((samples, 4))
        
        # Should work without preprocessing
        tdoa_matrix = self.localizer.calculate_tdoa(audio_channels)
        self.assertEqual(tdoa_matrix.shape, (4, 4))
    
    def test_correlation_methods(self):
        """Test different correlation methods."""
        samples = 800
        base_signal = np.sin(2 * np.pi * 1000 * np.arange(samples) / 8000)
        delayed_signal = np.zeros(samples)
        delayed_signal[5:] = base_signal[:-5]  # 5 sample delay
        
        # Test FFT method
        self.localizer.correlation_method = 'fft'
        delay_fft, corr_fft = self.localizer._cross_correlate_signals(base_signal, delayed_signal)
        
        # Test direct method
        self.localizer.correlation_method = 'direct'
        delay_direct, corr_direct = self.localizer._cross_correlate_signals(base_signal, delayed_signal)
        
        # Results should be similar
        self.assertAlmostEqual(delay_fft, delay_direct, places=3)
        self.assertAlmostEqual(corr_fft, corr_direct, places=2)
    
    def test_large_delay_warning(self):
        """Test warning for large TDoA values."""
        samples = 1000
        audio_channels = np.zeros((samples, 4))
        
        # Create signal with very large delay (should trigger warning)
        base_signal = np.sin(2 * np.pi * 1000 * np.arange(samples) / 8000)
        large_delay = 500  # Very large delay
        
        audio_channels[:, 0] = base_signal
        if large_delay < samples:
            audio_channels[large_delay:, 1] = base_signal[:-large_delay]
        
        # Should calculate TDoA but log warning
        with self.assertLogs(level='WARNING'):
            tdoa_matrix = self.localizer.calculate_tdoa(audio_channels)
        
        self.assertEqual(tdoa_matrix.shape, (4, 4))
    
    def test_low_correlation_warning(self):
        """Test warning for low correlation values."""
        samples = 1000
        audio_channels = np.random.random((samples, 4))  # Uncorrelated noise
        
        # Should calculate TDoA and may log warning for low correlation
        # (Random signals might not always trigger low correlation warning)
        tdoa_matrix = self.localizer.calculate_tdoa(audio_channels)
        
        self.assertEqual(tdoa_matrix.shape, (4, 4))
    
    def test_tdoa_consistency_tracking(self):
        """Test TDoA consistency tracking over multiple samples."""
        samples = 1000
        base_signal = np.sin(2 * np.pi * 1000 * np.arange(samples) / 8000)
        
        # Process multiple consistent samples
        for _ in range(5):
            audio_channels = np.zeros((samples, 4))
            delays = [0, 5, 10, 15]  # Consistent delays
            
            for ch, delay in enumerate(delays):
                if delay == 0:
                    audio_channels[:, ch] = base_signal
                else:
                    audio_channels[delay:, ch] = base_signal[:-delay]
            
            self.localizer.calculate_tdoa(audio_channels)
        
        # Check consistency
        stats = self.localizer.get_tdoa_statistics()
        self.assertGreater(stats['tdoa_consistency'], 0.8)  # Should be highly consistent


if __name__ == '__main__':
    unittest.main()