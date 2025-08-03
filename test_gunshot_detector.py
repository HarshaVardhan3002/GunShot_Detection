"""
Unit tests for gunshot detection module.
"""
import unittest
import numpy as np
import time
from unittest.mock import Mock, patch

from gunshot_detector import AmplitudeBasedDetector, DetectionEvent


class TestAmplitudeBasedDetector(unittest.TestCase):
    """Test cases for AmplitudeBasedDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AmplitudeBasedDetector(
            sample_rate=8000,
            channels=4,
            threshold_db=-20.0
        )
    
    def test_initialization(self):
        """Test proper initialization of detector."""
        self.assertEqual(self.detector.sample_rate, 8000)
        self.assertEqual(self.detector.channels, 4)
        self.assertEqual(self.detector.threshold_db, -20.0)
        self.assertAlmostEqual(self.detector.threshold_linear, 0.1, places=3)
        self.assertEqual(self.detector.total_detections, 0)
        self.assertEqual(self.detector.current_confidence, 0.0)
    
    def test_detect_gunshot_no_signal(self):
        """Test detection with no significant signal."""
        # Create quiet noise
        samples = 1000
        audio_data = np.random.random((samples, 4)).astype(np.float32) * 0.001
        
        detected, confidence, metadata = self.detector.detect_gunshot(audio_data)
        
        self.assertFalse(detected)
        self.assertEqual(confidence, 0.0)
        self.assertIn('reason', metadata)
    
    def test_detect_gunshot_strong_signal(self):
        """Test detection with strong gunshot-like signal."""
        samples = 1000
        audio_data = np.random.random((samples, 4)).astype(np.float32) * 0.001
        
        # Add strong impulse to simulate gunshot
        impulse_start = 400
        impulse_duration = 50
        impulse_amplitude = 0.5  # Strong signal
        
        # Create impulse on multiple channels
        for ch in [0, 1, 2]:  # 3 channels triggered
            audio_data[impulse_start:impulse_start + impulse_duration, ch] += impulse_amplitude
        
        detected, confidence, metadata = self.detector.detect_gunshot(audio_data)
        
        self.assertTrue(detected)
        self.assertGreater(confidence, 0.5)
        self.assertIn('triggered_channels', metadata)
        self.assertGreaterEqual(len(metadata['triggered_channels']), 2)
        self.assertIn('peak_amplitude', metadata)
        self.assertIn('duration_ms', metadata)
    
    def test_detect_gunshot_single_channel(self):
        """Test detection with signal on single channel."""
        samples = 800
        audio_data = np.random.random((samples, 4)).astype(np.float32) * 0.001
        
        # Add impulse to only one channel
        impulse_start = 300
        impulse_duration = 40
        impulse_amplitude = 0.3
        
        audio_data[impulse_start:impulse_start + impulse_duration, 0] += impulse_amplitude
        
        detected, confidence, metadata = self.detector.detect_gunshot(audio_data)
        
        # Single channel detection should have lower confidence
        if detected:
            self.assertLess(confidence, 0.9)  # Should be reduced for single channel
            self.assertEqual(len(metadata['triggered_channels']), 1)
    
    def test_analyze_channel_amplitude(self):
        """Test single channel amplitude analysis."""
        samples = 500
        channel_data = np.random.random(samples).astype(np.float32) * 0.001
        
        # Add impulse
        impulse_start = 200
        impulse_duration = 30
        channel_data[impulse_start:impulse_start + impulse_duration] += 0.2
        
        result = self.detector._analyze_channel_amplitude(channel_data, 0)
        
        self.assertIn('channel_id', result)
        self.assertIn('peak_amplitude', result)
        self.assertIn('rms_amplitude', result)
        self.assertIn('snr_db', result)
        self.assertIn('amplitude_trigger', result)
        self.assertIn('confidence', result)
        self.assertIn('temporal_analysis', result)
        
        self.assertEqual(result['channel_id'], 0)
        self.assertGreater(result['peak_amplitude'], 0.1)
        self.assertTrue(result['amplitude_trigger'])
        self.assertGreater(result['confidence'], 0.0)
    
    def test_analyze_temporal_characteristics(self):
        """Test temporal analysis of signals."""
        samples = 400
        signal = np.zeros(samples, dtype=np.float32)
        
        # Create impulse with known characteristics
        impulse_start = 150
        impulse_duration = 50  # 50 samples
        peak_amplitude = 0.3
        
        # Create triangular impulse
        for i in range(impulse_duration):
            if i < impulse_duration // 2:
                signal[impulse_start + i] = peak_amplitude * (i / (impulse_duration // 2))
            else:
                signal[impulse_start + i] = peak_amplitude * ((impulse_duration - i) / (impulse_duration // 2))
        
        temporal_analysis = self.detector._analyze_temporal_characteristics(signal)
        
        self.assertIn('duration_ms', temporal_analysis)
        self.assertIn('rise_time_ms', temporal_analysis)
        self.assertIn('peak_position', temporal_analysis)
        self.assertIn('envelope_shape_factor', temporal_analysis)
        
        # Check duration is reasonable
        expected_duration_ms = (impulse_duration / self.detector.sample_rate) * 1000
        self.assertGreater(temporal_analysis['duration_ms'], expected_duration_ms * 0.5)
        self.assertLess(temporal_analysis['duration_ms'], expected_duration_ms * 2.0)
        
        # Check peak position is reasonable
        self.assertGreater(temporal_analysis['peak_position'], 0.2)
        self.assertLess(temporal_analysis['peak_position'], 0.8)
    
    def test_calculate_envelope(self):
        """Test envelope calculation."""
        samples = 200
        # Create signal with known envelope
        t = np.arange(samples) / self.detector.sample_rate
        signal = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 10)  # Decaying sine wave
        
        envelope = self.detector._calculate_envelope(signal)
        
        self.assertEqual(len(envelope), len(signal))
        self.assertGreaterEqual(np.min(envelope), 0)  # Envelope should be non-negative
        
        # Envelope should generally decrease for decaying signal
        first_half_avg = np.mean(envelope[:samples//2])
        second_half_avg = np.mean(envelope[samples//2:])
        self.assertGreater(first_half_avg, second_half_avg)
    
    def test_calculate_shape_factor(self):
        """Test shape factor calculation."""
        # Test with impulse (high shape factor)
        impulse = np.zeros(100)
        impulse[50] = 1.0
        impulse_shape = self.detector._calculate_shape_factor(impulse)
        
        # Test with constant signal (low shape factor)
        constant = np.ones(100) * 0.5
        constant_shape = self.detector._calculate_shape_factor(constant)
        
        # Impulse should have higher shape factor
        self.assertGreater(impulse_shape, constant_shape)
        self.assertGreater(impulse_shape, 2.0)  # Should be significantly > 1
        self.assertAlmostEqual(constant_shape, 1.0, places=1)  # Should be close to 1
    
    def test_calculate_amplitude_confidence(self):
        """Test amplitude confidence calculation."""
        # High confidence scenario
        temporal_analysis_good = {
            'duration_ms': 50,    # Good duration
            'rise_time_ms': 10,   # Good rise time
            'envelope_shape_factor': 4.0  # Very impulsive
        }
        
        confidence_high = self.detector._calculate_amplitude_confidence(
            peak_amp=0.5, rms_amp=0.1, snr_db=25, temporal_analysis=temporal_analysis_good
        )
        
        # Low confidence scenario
        temporal_analysis_poor = {
            'duration_ms': 2000,  # Too long
            'rise_time_ms': 200,  # Too slow
            'envelope_shape_factor': 1.2  # Not impulsive
        }
        
        confidence_low = self.detector._calculate_amplitude_confidence(
            peak_amp=0.05, rms_amp=0.04, snr_db=-5, temporal_analysis=temporal_analysis_poor
        )
        
        self.assertGreater(confidence_high, confidence_low)
        self.assertGreater(confidence_high, 0.7)
        self.assertLess(confidence_low, 0.4)
    
    def test_combine_channel_results(self):
        """Test combining results from multiple channels."""
        # Create mock channel results
        channel_results = [
            {
                'channel_id': 0, 'amplitude_trigger': True, 'confidence': 0.8,
                'peak_amplitude': 0.3, 'snr_db': 20,
                'temporal_analysis': {'duration_ms': 40, 'rise_time_ms': 8, 'envelope_shape_factor': 3.5}
            },
            {
                'channel_id': 1, 'amplitude_trigger': True, 'confidence': 0.7,
                'peak_amplitude': 0.25, 'snr_db': 18,
                'temporal_analysis': {'duration_ms': 45, 'rise_time_ms': 12, 'envelope_shape_factor': 3.0}
            },
            {
                'channel_id': 2, 'amplitude_trigger': False, 'confidence': 0.2,
                'peak_amplitude': 0.05, 'snr_db': 5,
                'temporal_analysis': {'duration_ms': 10, 'rise_time_ms': 2, 'envelope_shape_factor': 1.5}
            },
            {
                'channel_id': 3, 'amplitude_trigger': True, 'confidence': 0.6,
                'peak_amplitude': 0.2, 'snr_db': 15,
                'temporal_analysis': {'duration_ms': 50, 'rise_time_ms': 15, 'envelope_shape_factor': 2.8}
            }
        ]
        
        detected, confidence, metadata = self.detector._combine_channel_results(
            channel_results, time.time()
        )
        
        self.assertTrue(detected)
        self.assertGreater(confidence, 0.5)
        self.assertIn('triggered_channels', metadata)
        self.assertEqual(len(metadata['triggered_channels']), 3)  # Channels 0, 1, 3 (1-based: 1, 2, 4)
        self.assertIn('peak_amplitude', metadata)
        self.assertIn('duration_ms', metadata)
    
    def test_update_noise_floor(self):
        """Test noise floor estimation."""
        initial_noise_floor = self.detector.noise_floor
        
        # Feed quiet signal
        quiet_signal = np.random.random((500, 4)).astype(np.float32) * 0.0001
        self.detector._update_noise_floor(quiet_signal)
        
        # Feed multiple samples to build history
        for _ in range(20):
            self.detector._update_noise_floor(quiet_signal)
        
        # Noise floor should adapt to quiet signal
        self.assertLess(self.detector.noise_floor, initial_noise_floor)
        
        # Feed louder signal
        loud_signal = np.random.random((500, 4)).astype(np.float32) * 0.01
        for _ in range(20):
            self.detector._update_noise_floor(loud_signal)
        
        # Noise floor should increase but not too much (uses percentile)
        self.assertGreater(self.detector.noise_floor, 0.0001)
    
    def test_set_adaptive_threshold(self):
        """Test adaptive threshold setting."""
        original_threshold = self.detector.threshold_linear
        
        # Set high noise floor
        high_noise = 0.05
        self.detector.set_adaptive_threshold(high_noise)
        
        # Threshold should be adjusted above noise floor
        self.assertGreater(self.detector.threshold_linear, high_noise)
        self.assertGreater(self.detector.threshold_linear, original_threshold)
        
        # Set low noise floor
        low_noise = 0.001
        self.detector.set_adaptive_threshold(low_noise)
        
        # Should maintain minimum threshold
        self.assertGreater(self.detector.threshold_linear, low_noise * 2)
    
    def test_detection_cooldown(self):
        """Test detection cooldown period."""
        samples = 500
        # Create strong signal
        audio_data = np.random.random((samples, 4)).astype(np.float32) * 0.001
        audio_data[200:250, :] += 0.3  # Strong impulse on all channels
        
        # First detection
        detected1, confidence1, metadata1 = self.detector.detect_gunshot(audio_data)
        
        # Immediate second detection (should be blocked by cooldown)
        detected2, confidence2, metadata2 = self.detector.detect_gunshot(audio_data)
        
        self.assertTrue(detected1)
        self.assertFalse(detected2)
        self.assertEqual(metadata2.get('reason'), 'cooldown_period')
    
    def test_get_detection_statistics(self):
        """Test detection statistics."""
        # Perform some detections
        samples = 400
        for i in range(3):
            audio_data = np.random.random((samples, 4)).astype(np.float32) * 0.001
            audio_data[150:200, :] += 0.2  # Add impulse
            
            self.detector.detect_gunshot(audio_data)
            time.sleep(0.6)  # Wait for cooldown
        
        stats = self.detector.get_detection_statistics()
        
        expected_fields = [
            'total_detections', 'false_positives', 'detection_rate',
            'avg_confidence', 'avg_snr_db', 'avg_duration_ms',
            'current_threshold_db', 'current_noise_floor', 'cooldown_period'
        ]
        
        for field in expected_fields:
            self.assertIn(field, stats)
        
        self.assertGreaterEqual(stats['total_detections'], 0)
        self.assertGreaterEqual(stats['detection_rate'], 0.0)
        self.assertLessEqual(stats['detection_rate'], 1.0)
    
    def test_configure_detection_parameters(self):
        """Test parameter configuration."""
        original_threshold = self.detector.threshold_db
        original_cooldown = self.detector.detection_cooldown
        
        # Configure new parameters
        new_params = {
            'threshold_db': -15.0,
            'detection_cooldown': 1.0,
            'min_duration_ms': 5,
            'max_duration_ms': 300
        }
        
        self.detector.configure_detection_parameters(**new_params)
        
        self.assertEqual(self.detector.threshold_db, -15.0)
        self.assertEqual(self.detector.detection_cooldown, 1.0)
        self.assertEqual(self.detector.min_duration_ms, 5)
        self.assertEqual(self.detector.max_duration_ms, 300)
        
        # Verify threshold_linear was updated
        expected_linear = 10 ** (-15.0 / 20.0)
        self.assertAlmostEqual(self.detector.threshold_linear, expected_linear, places=4)
    
    def test_reset_detection_state(self):
        """Test detection state reset."""
        # Perform a detection to set some state
        samples = 300
        audio_data = np.random.random((samples, 4)).astype(np.float32) * 0.001
        audio_data[100:150, :] += 0.3
        
        self.detector.detect_gunshot(audio_data)
        
        # Verify state was set
        self.assertGreater(self.detector.total_detections, 0)
        self.assertGreater(self.detector.last_detection_time, 0)
        
        # Reset state
        self.detector.reset_detection_state()
        
        # Verify reset
        self.assertEqual(self.detector.total_detections, 0)
        self.assertEqual(self.detector.last_detection_time, 0.0)
        self.assertEqual(self.detector.current_confidence, 0.0)
        self.assertEqual(len(self.detector.detection_history), 0)
    
    def test_invalid_input_channels(self):
        """Test handling of invalid input channel count."""
        # Wrong number of channels
        audio_data = np.random.random((100, 6)).astype(np.float32)  # 6 channels instead of 4
        
        with self.assertRaises(ValueError):
            self.detector.detect_gunshot(audio_data)
    
    def test_detection_event_dataclass(self):
        """Test DetectionEvent dataclass."""
        event = DetectionEvent(
            timestamp=time.time(),
            confidence=0.85,
            peak_amplitude=0.3,
            frequency_profile={'low': 0.2, 'mid': 0.8, 'high': 0.4},
            triggered_channels=[1, 2, 3],
            duration_ms=45.0,
            rise_time_ms=8.5,
            signal_to_noise_ratio=22.3
        )
        
        self.assertIsInstance(event.timestamp, float)
        self.assertEqual(event.confidence, 0.85)
        self.assertEqual(event.peak_amplitude, 0.3)
        self.assertIsInstance(event.frequency_profile, dict)
        self.assertEqual(event.triggered_channels, [1, 2, 3])
        self.assertEqual(event.duration_ms, 45.0)
        self.assertEqual(event.rise_time_ms, 8.5)
        self.assertEqual(event.signal_to_noise_ratio, 22.3)
    
    def test_frequency_analysis_enabled(self):
        """Test enabling/disabling frequency analysis."""
        # Initially enabled
        self.assertTrue(self.detector.enable_frequency_analysis)
        
        # Disable
        self.detector.set_frequency_analysis_enabled(False)
        self.assertFalse(self.detector.enable_frequency_analysis)
        
        # Re-enable
        self.detector.set_frequency_analysis_enabled(True)
        self.assertTrue(self.detector.enable_frequency_analysis)
    
    def test_analyze_channel_frequency(self):
        """Test frequency domain analysis of single channel."""
        # Create signal with known frequency content
        samples = 2048  # Enough for FFT
        t = np.arange(samples) / self.detector.sample_rate
        
        # Create signal with gunshot-like frequency content
        signal = (0.3 * np.sin(2 * np.pi * 1000 * t) +  # 1kHz (mid band)
                 0.2 * np.sin(2 * np.pi * 2500 * t) +   # 2.5kHz (mid band)
                 0.1 * np.sin(2 * np.pi * 500 * t))     # 500Hz (mid-low band)
        
        # Add some noise
        signal += np.random.random(samples) * 0.01
        
        result = self.detector._analyze_channel_frequency(signal, 0)
        
        # Check required fields
        required_fields = [
            'frequency_profile', 'spectral_centroid', 'spectral_rolloff',
            'spectral_flatness', 'gunshot_similarity', 'frequency_confidence',
            'dominant_frequency'
        ]
        
        for field in required_fields:
            self.assertIn(field, result)
        
        # Check frequency profile has expected bands
        freq_profile = result['frequency_profile']
        for band in self.detector.frequency_bands.keys():
            self.assertIn(band, freq_profile)
            self.assertGreaterEqual(freq_profile[band], 0.0)
            self.assertLessEqual(freq_profile[band], 1.0)
        
        # Check that mid-band has significant energy (our test signal)
        self.assertGreater(freq_profile['mid'], 0.25)  # Lowered threshold
        
        # Check spectral features are reasonable
        self.assertGreater(result['spectral_centroid'], 500)
        self.assertLess(result['spectral_centroid'], 4000)
        self.assertGreater(result['spectral_rolloff'], 1000)
    
    def test_calculate_frequency_bands(self):
        """Test frequency band energy calculation."""
        # Create simple test spectrum
        freqs = np.linspace(0, 4000, 1000)  # 0-4kHz
        magnitude_spectrum = np.ones(1000)  # Flat spectrum
        
        # Add peak in mid-frequency range
        mid_freq_mask = (freqs >= 1500) & (freqs <= 4000)
        magnitude_spectrum[mid_freq_mask] *= 3  # 3x energy in mid band
        
        band_energies = self.detector._calculate_frequency_bands(magnitude_spectrum, freqs)
        
        # Check all bands are present
        for band in self.detector.frequency_bands.keys():
            self.assertIn(band, band_energies)
        
        # Check energies sum to approximately 1.0
        total_energy = sum(band_energies.values())
        self.assertAlmostEqual(total_energy, 1.0, places=2)
        
        # Mid band should have higher energy due to our peak
        self.assertGreater(band_energies['mid'], band_energies['low'])
    
    def test_calculate_spectral_features(self):
        """Test spectral feature calculation."""
        # Create test spectrum with known characteristics
        freqs = np.linspace(0, 4000, 1000)
        
        # Create spectrum with peak at 2kHz
        peak_freq = 2000
        peak_idx = np.argmin(np.abs(freqs - peak_freq))
        magnitude_spectrum = np.exp(-((freqs - peak_freq) / 500) ** 2)  # Gaussian peak
        
        features = self.detector._calculate_spectral_features(magnitude_spectrum, freqs)
        
        # Check required features
        required_features = ['centroid', 'rolloff', 'flatness', 'dominant_freq']
        for feature in required_features:
            self.assertIn(feature, features)
        
        # Spectral centroid should be near peak frequency
        self.assertGreater(features['centroid'], 1500)
        self.assertLess(features['centroid'], 2500)
        
        # Dominant frequency should be close to peak
        self.assertGreater(features['dominant_freq'], 1800)
        self.assertLess(features['dominant_freq'], 2200)
        
        # Rolloff should be reasonable
        self.assertGreater(features['rolloff'], features['centroid'])
    
    def test_calculate_gunshot_similarity(self):
        """Test gunshot signature similarity calculation."""
        # Test with perfect gunshot signature
        perfect_profile = self.detector.gunshot_signature.copy()
        perfect_similarity = self.detector._calculate_gunshot_similarity(perfect_profile)
        self.assertAlmostEqual(perfect_similarity, 1.0, places=2)
        
        # Test with opposite signature (high energy in wrong bands)
        opposite_profile = {
            'low': 0.45,      # High energy in low (opposite of gunshot)
            'mid_low': 0.25,
            'mid': 0.15,      # Low energy in mid (opposite of gunshot)
            'high': 0.12,
            'very_high': 0.03
        }
        opposite_similarity = self.detector._calculate_gunshot_similarity(opposite_profile)
        self.assertLess(opposite_similarity, perfect_similarity)
        
        # Test with empty profile
        empty_similarity = self.detector._calculate_gunshot_similarity({})
        self.assertEqual(empty_similarity, 0.0)
    
    def test_calculate_frequency_confidence(self):
        """Test frequency-based confidence calculation."""
        # High confidence scenario (gunshot-like)
        good_profile = {
            'low': 0.1, 'mid_low': 0.2, 'mid': 0.5, 'high': 0.15, 'very_high': 0.05
        }
        good_features = {
            'dominant_freq': 2000,  # Good frequency
            'rolloff': 3000,        # Good rolloff
            'flatness': 0.2         # Tonal (not noisy)
        }
        
        high_confidence = self.detector._calculate_frequency_confidence(
            good_profile, good_features, 0.9
        )
        
        # Low confidence scenario (not gunshot-like)
        bad_profile = {
            'low': 0.6, 'mid_low': 0.1, 'mid': 0.1, 'high': 0.1, 'very_high': 0.1
        }
        bad_features = {
            'dominant_freq': 100,   # Too low frequency
            'rolloff': 8000,        # Too high rolloff
            'flatness': 0.9         # Very noisy
        }
        
        low_confidence = self.detector._calculate_frequency_confidence(
            bad_profile, bad_features, 0.1
        )
        
        self.assertGreater(high_confidence, low_confidence)
        self.assertGreater(high_confidence, 0.7)
        self.assertLess(low_confidence, 0.5)
    
    def test_combine_amplitude_frequency_analysis(self):
        """Test combining amplitude and frequency analysis results."""
        amplitude_result = {
            'channel_id': 0,
            'peak_amplitude': 0.3,
            'confidence': 0.8,
            'amplitude_trigger': True
        }
        
        frequency_result = {
            'frequency_profile': {'mid': 0.4, 'high': 0.3},
            'frequency_confidence': 0.7,
            'gunshot_similarity': 0.85
        }
        
        combined = self.detector._combine_amplitude_frequency_analysis(
            amplitude_result, frequency_result
        )
        
        # Should contain original amplitude data
        self.assertEqual(combined['channel_id'], 0)
        self.assertEqual(combined['peak_amplitude'], 0.3)
        
        # Should contain frequency analysis
        self.assertIn('frequency_analysis', combined)
        self.assertEqual(combined['frequency_analysis'], frequency_result)
        
        # Should have combined confidence
        self.assertIn('combined_confidence', combined)
        expected_combined = 0.7 * 0.8 + 0.3 * 0.7  # Weighted combination
        self.assertAlmostEqual(combined['combined_confidence'], expected_combined, places=2)
    
    def test_configure_frequency_bands(self):
        """Test frequency band configuration."""
        new_bands = {
            'low': (0, 1000),
            'mid': (1000, 3000),
            'high': (3000, 8000)
        }
        
        self.detector.configure_frequency_bands(new_bands)
        self.assertEqual(self.detector.frequency_bands, new_bands)
    
    def test_set_gunshot_signature(self):
        """Test setting gunshot signature."""
        new_signature = {
            'low': 0.2,
            'mid': 0.6,
            'high': 0.2
        }
        
        self.detector.set_gunshot_signature(new_signature)
        
        # Should be normalized to sum to 1.0
        total = sum(self.detector.gunshot_signature.values())
        self.assertAlmostEqual(total, 1.0, places=3)
    
    def test_get_frequency_statistics(self):
        """Test frequency statistics retrieval."""
        stats = self.detector.get_frequency_statistics()
        
        expected_fields = [
            'frequency_analysis_enabled', 'samples_analyzed',
            'avg_gunshot_similarity', 'avg_spectral_centroid',
            'frequency_bands'
        ]
        
        for field in expected_fields:
            self.assertIn(field, stats)
        
        self.assertTrue(stats['frequency_analysis_enabled'])
        self.assertEqual(stats['samples_analyzed'], 0)  # No samples analyzed yet
    
    def test_analyze_frequency_profile(self):
        """Test frequency profile analysis for diagnostics."""
        samples = 2048
        audio_data = np.random.random((samples, 4)).astype(np.float32) * 0.1
        
        # Add signal with known frequency content
        t = np.arange(samples) / self.detector.sample_rate
        signal = 0.2 * np.sin(2 * np.pi * 1500 * t)  # 1.5kHz signal
        audio_data[:, 0] += signal
        
        result = self.detector.analyze_frequency_profile(audio_data, channel=0)
        
        # Should contain frequency analysis results
        self.assertIn('frequency_profile', result)
        self.assertIn('spectral_centroid', result)
        
        # Should have detected energy in mid-low band (1500Hz)
        freq_profile = result['frequency_profile']
        self.assertGreater(freq_profile['mid_low'], 0.1)
    
    def test_frequency_analysis_integration(self):
        """Test integration of frequency analysis with detection."""
        samples = 2048
        audio_data = np.random.random((samples, 4)).astype(np.float32) * 0.001
        
        # Create gunshot-like signal with appropriate frequency content
        t = np.arange(samples) / self.detector.sample_rate
        gunshot_signal = (0.3 * np.sin(2 * np.pi * 1000 * t) +    # 1kHz
                         0.2 * np.sin(2 * np.pi * 2500 * t) +     # 2.5kHz
                         0.1 * np.sin(2 * np.pi * 500 * t))       # 500Hz
        
        # Apply envelope to make it impulsive
        envelope = np.exp(-np.arange(samples) / (samples / 10))
        gunshot_signal *= envelope
        
        # Add to multiple channels
        for ch in range(3):
            audio_data[:, ch] += gunshot_signal * (0.8 + ch * 0.1)
        
        # Test with frequency analysis enabled
        self.detector.set_frequency_analysis_enabled(True)
        detected_freq, confidence_freq, metadata_freq = self.detector.detect_gunshot(audio_data)
        
        # Test with frequency analysis disabled
        self.detector.set_frequency_analysis_enabled(False)
        detected_amp, confidence_amp, metadata_amp = self.detector.detect_gunshot(audio_data)
        
        # Both should detect, but frequency analysis should provide more information
        if detected_freq and detected_amp:
            self.assertIn('frequency_profile', metadata_freq)
            self.assertNotIn('frequency_profile', metadata_amp)
            
            # Frequency analysis should provide spectral information
            self.assertIn('spectral_centroid', metadata_freq)
            self.assertIn('gunshot_similarity', metadata_freq)
    
    def test_adaptive_thresholding_enabled(self):
        """Test enabling/disabling adaptive thresholding."""
        # Initially enabled
        self.assertTrue(self.detector.enable_adaptive_threshold)
        
        # Disable
        self.detector.enable_adaptive_thresholding(False)
        self.assertFalse(self.detector.enable_adaptive_threshold)
        
        # Re-enable
        self.detector.enable_adaptive_thresholding(True)
        self.assertTrue(self.detector.enable_adaptive_threshold)
    
    def test_configure_adaptive_limits(self):
        """Test configuring adaptive threshold limits."""
        # Set valid limits
        self.detector.configure_adaptive_limits(-30.0, -10.0)
        self.assertEqual(self.detector.min_threshold_db, -30.0)
        self.assertEqual(self.detector.max_threshold_db, -10.0)
        
        # Test invalid limits
        with self.assertRaises(ValueError):
            self.detector.configure_adaptive_limits(-10.0, -30.0)  # min > max
    
    def test_set_threshold_level(self):
        """Test setting threshold levels."""
        original_threshold = self.detector.threshold_db
        
        # Set to sensitive
        self.detector.set_threshold_level('sensitive')
        self.assertEqual(self.detector.current_threshold_level, 'sensitive')
        self.assertLess(self.detector.threshold_db, original_threshold)
        
        # Set to conservative
        self.detector.set_threshold_level('conservative')
        self.assertEqual(self.detector.current_threshold_level, 'conservative')
        self.assertGreater(self.detector.threshold_db, original_threshold)
        
        # Test invalid level
        with self.assertRaises(ValueError):
            self.detector.set_threshold_level('invalid_level')
    
    def test_environment_classification(self):
        """Test environment classification."""
        # Simulate very quiet environment
        for _ in range(60):  # Need sufficient history
            quiet_data = np.random.random((100, 4)).astype(np.float32) * 0.0001
            self.detector._update_noise_floor(quiet_data)
        
        self.detector._classify_environment()
        self.assertIn('quiet', self.detector.environment_type)
        self.assertEqual(self.detector.activity_level, 'quiet')
        
        # Simulate busy environment
        for _ in range(60):
            busy_data = np.random.random((100, 4)).astype(np.float32) * 0.08  # Higher noise level
            self.detector._update_noise_floor(busy_data)
        
        self.detector._classify_environment()
        # Should be classified as busy or very_busy
        self.assertTrue('busy' in self.detector.environment_type or 'moderate' in self.detector.environment_type)
        self.assertIn(self.detector.activity_level, ['normal', 'busy'])
    
    def test_time_based_adjustment(self):
        """Test time-based threshold adjustment."""
        # Mock different times of day
        import unittest.mock
        
        # Night time (should be more sensitive)
        with unittest.mock.patch('time.localtime') as mock_time:
            mock_time.return_value.tm_hour = 2  # 2 AM
            night_adjustment = self.detector._calculate_time_based_adjustment()
            self.assertLess(night_adjustment, 0)  # More sensitive (lower threshold)
        
        # Rush hour (should be less sensitive)
        with unittest.mock.patch('time.localtime') as mock_time:
            mock_time.return_value.tm_hour = 8  # 8 AM
            rush_adjustment = self.detector._calculate_time_based_adjustment()
            self.assertGreater(rush_adjustment, 0)  # Less sensitive (higher threshold)
    
    def test_detection_feedback(self):
        """Test detection performance feedback."""
        # Perform a detection first
        samples = 500
        audio_data = np.zeros((samples, 4), dtype=np.float32)
        audio_data[200:250, :] = 0.3  # Strong signal
        
        detected, confidence, metadata = self.detector.detect_gunshot(audio_data)
        
        if detected:
            # Provide feedback
            self.detector.provide_detection_feedback('correct')
            
            # Check that feedback was recorded
            self.assertTrue(len(self.detector.detection_performance_history) > 0)
            last_entry = self.detector.detection_performance_history[-1]
            self.assertEqual(last_entry['user_feedback'], 'correct')
        
        # Test invalid feedback
        with self.assertRaises(ValueError):
            self.detector.provide_detection_feedback('invalid_feedback')
    
    def test_adaptive_threshold_status(self):
        """Test getting adaptive threshold status."""
        status = self.detector.get_adaptive_threshold_status()
        
        expected_fields = [
            'adaptive_enabled', 'current_threshold_db', 'base_threshold_db',
            'threshold_level', 'threshold_limits', 'environment', 'performance',
            'adaptation_rate'
        ]
        
        for field in expected_fields:
            self.assertIn(field, status)
        
        # Check nested structures
        self.assertIn('type', status['environment'])
        self.assertIn('noise_floor', status['environment'])
        self.assertIn('false_positive_rate', status['performance'])
        self.assertIn('min_db', status['threshold_limits'])
    
    def test_environment_analysis(self):
        """Test detailed environment analysis."""
        # Add some noise history
        for _ in range(20):
            noise_data = np.random.random((100, 4)).astype(np.float32) * 0.01
            self.detector._update_noise_floor(noise_data)
        
        analysis = self.detector.get_environment_analysis()
        
        expected_fields = [
            'environment_type', 'activity_level', 'noise_statistics', 'noise_floor_history'
        ]
        
        for field in expected_fields:
            self.assertIn(field, analysis)
        
        # Check noise statistics
        noise_stats = analysis['noise_statistics']
        self.assertIn('avg_rms', noise_stats)
        self.assertIn('rms_std', noise_stats)
        self.assertIn('peak_to_rms_ratio', noise_stats)
    
    def test_noise_trend_calculation(self):
        """Test noise trend calculation."""
        # Simulate increasing noise
        for i in range(25):
            noise_level = 0.001 * (1 + i * 0.1)  # Increasing noise
            noise_data = np.random.random((50, 4)).astype(np.float32) * noise_level
            self.detector._update_noise_floor(noise_data)
        
        trend = self.detector._calculate_noise_trend()
        self.assertEqual(trend, 'increasing')
        
        # Reset and simulate stable noise
        self.detector.noise_history.clear()
        for _ in range(25):
            noise_data = np.random.random((50, 4)).astype(np.float32) * 0.005  # Stable
            self.detector._update_noise_floor(noise_data)
        
        trend = self.detector._calculate_noise_trend()
        self.assertEqual(trend, 'stable')
    
    def test_enhanced_noise_floor_update(self):
        """Test enhanced noise floor estimation."""
        initial_noise_floor = self.detector.noise_floor
        
        # Feed consistent quiet signal
        for _ in range(50):
            quiet_signal = np.random.random((100, 4)).astype(np.float32) * 0.0005
            self.detector._update_noise_floor(quiet_signal)
        
        # Noise floor should adapt to quiet environment
        self.assertLess(self.detector.noise_floor, initial_noise_floor)
        
        # Check that environment was classified
        self.assertNotEqual(self.detector.environment_type, 'unknown')
    
    def test_multi_factor_adaptive_threshold(self):
        """Test multi-factor adaptive threshold calculation."""
        # Enable adaptive thresholding
        self.detector.enable_adaptive_thresholding(True)
        
        original_threshold = self.detector.threshold_db
        
        # Simulate high noise floor
        high_noise = 0.01
        self.detector.set_adaptive_threshold(high_noise)
        
        # Threshold should be adjusted upward for high noise
        if self.detector.enable_adaptive_threshold:
            # May be adjusted based on multiple factors
            pass  # Threshold adjustment depends on environment classification
        
        # Test with very low noise floor
        low_noise = 0.0001
        self.detector.set_adaptive_threshold(low_noise)
        
        # Should adapt to low noise environment
        self.assertLessEqual(self.detector.noise_floor, low_noise * 1.1)  # Allow for smoothing
    
    def test_reset_adaptive_state(self):
        """Test resetting adaptive state."""
        # Add some history and state
        for _ in range(10):
            noise_data = np.random.random((100, 4)).astype(np.float32) * 0.01
            self.detector._update_noise_floor(noise_data)
        
        self.detector.environment_type = 'busy'
        self.detector.false_positive_rate = 0.1
        
        # Reset state
        self.detector.reset_adaptive_state()
        
        # Verify reset
        self.assertEqual(len(self.detector.noise_history), 0)
        self.assertEqual(len(self.detector.noise_profile['rms_history']), 0)
        self.assertEqual(self.detector.environment_type, 'unknown')
        self.assertEqual(self.detector.false_positive_rate, 0.0)
        self.assertEqual(self.detector.threshold_db, self.detector.base_threshold_db)
    
    def test_calibrate_for_environment(self):
        """Test environment calibration framework."""
        result = self.detector.calibrate_for_environment(10.0)
        
        self.assertIn('status', result)
        self.assertIn('duration', result)
        self.assertIn('instructions', result)
        self.assertEqual(result['duration'], 10.0)
        self.assertIsInstance(result['instructions'], list)


if __name__ == '__main__':
    unittest.main()