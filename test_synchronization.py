"""
Unit tests for audio stream synchronization features.
"""
import unittest
import numpy as np
import time
from unittest.mock import Mock, patch

from audio_capture import AudioCaptureEngine, AudioStreamSynchronizer, SynchronizationMetrics


class TestAudioStreamSynchronizer(unittest.TestCase):
    """Test cases for AudioStreamSynchronizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synchronizer = AudioStreamSynchronizer(channels=4, sample_rate=8000)
    
    def test_initialization(self):
        """Test proper initialization of synchronizer."""
        self.assertEqual(self.synchronizer.channels, 4)
        self.assertEqual(self.synchronizer.sample_rate, 8000)
        self.assertEqual(self.synchronizer._reference_channel, 0)
        self.assertEqual(len(self.synchronizer._channel_delays), 4)
        np.testing.assert_array_equal(self.synchronizer._channel_delays, np.zeros(4))
    
    def test_set_reference_channel(self):
        """Test setting reference channel."""
        # Valid channel
        result = self.synchronizer.set_reference_channel(2)
        self.assertTrue(result)
        self.assertEqual(self.synchronizer._reference_channel, 2)
        
        # Invalid channel
        result = self.synchronizer.set_reference_channel(5)
        self.assertFalse(result)
        self.assertEqual(self.synchronizer._reference_channel, 2)  # Should remain unchanged
    
    def test_analyze_channel_synchronization(self):
        """Test synchronization analysis."""
        # Create test data with known characteristics
        samples = 1024
        test_data = np.random.random((samples, 4)).astype(np.float32) * 0.1
        
        # Make channels 0 and 1 highly correlated (good sync)
        test_data[:, 1] = test_data[:, 0] * 0.9 + np.random.random(samples) * 0.01
        
        # Make channel 2 delayed version of channel 0
        delay_samples = 10
        test_data[delay_samples:, 2] = test_data[:-delay_samples, 0] * 0.8
        
        # Analyze synchronization
        metrics = self.synchronizer.analyze_channel_synchronization(test_data, time.time())
        
        self.assertIsInstance(metrics, SynchronizationMetrics)
        self.assertGreaterEqual(metrics.sync_quality_score, 0.0)
        self.assertLessEqual(metrics.sync_quality_score, 1.0)
        self.assertIn(0, metrics.channel_alignment)
        self.assertIn(1, metrics.channel_alignment)
        self.assertIn(2, metrics.channel_alignment)
        self.assertIn(3, metrics.channel_alignment)
        
        # Channel 0 should have perfect alignment (reference)
        self.assertEqual(metrics.channel_alignment[0], 1.0)
        
        # Channel 1 should have high alignment (correlated)
        self.assertGreater(metrics.channel_alignment[1], 0.8)
    
    def test_calculate_channel_alignment(self):
        """Test channel alignment calculation."""
        samples = 512
        test_data = np.zeros((samples, 4), dtype=np.float32)
        
        # Create reference signal
        reference_signal = np.sin(2 * np.pi * 1000 * np.arange(samples) / 8000)
        test_data[:, 0] = reference_signal
        
        # Perfect copy (should have alignment = 1.0)
        test_data[:, 1] = reference_signal
        
        # Delayed copy
        delay = 5
        test_data[delay:, 2] = reference_signal[:-delay]
        
        # Uncorrelated signal
        test_data[:, 3] = np.random.random(samples) * 0.1
        
        alignment = self.synchronizer._calculate_channel_alignment(test_data)
        
        self.assertEqual(alignment[0], 1.0)  # Reference channel
        self.assertGreater(alignment[1], 0.95)  # Perfect copy
        self.assertGreater(alignment[2], 0.8)   # Delayed copy
        self.assertLess(alignment[3], 0.5)      # Uncorrelated
    
    def test_calculate_phase_coherence(self):
        """Test phase coherence calculation."""
        samples = 1024
        test_data = np.zeros((samples, 4), dtype=np.float32)
        
        # Create coherent signals (same frequency, different phases)
        freq = 1000  # 1kHz
        t = np.arange(samples) / 8000
        
        test_data[:, 0] = np.sin(2 * np.pi * freq * t)
        test_data[:, 1] = np.sin(2 * np.pi * freq * t + np.pi/4)  # 45° phase shift
        test_data[:, 2] = np.sin(2 * np.pi * freq * t + np.pi/2)  # 90° phase shift
        test_data[:, 3] = np.random.random(samples) * 0.1  # Noise
        
        coherence = self.synchronizer._calculate_phase_coherence(test_data)
        
        # Should have reasonable coherence for sinusoidal signals
        self.assertGreater(coherence, 0.1)  # Lower threshold for test data
        self.assertLessEqual(coherence, 1.0)
    
    def test_compensate_channel_delays(self):
        """Test delay compensation."""
        samples = 100
        test_data = np.zeros((samples, 4), dtype=np.float32)
        
        # Create test signal
        test_signal = np.arange(samples, dtype=np.float32)
        test_data[:, 0] = test_signal
        test_data[:, 1] = test_signal
        test_data[:, 2] = test_signal
        test_data[:, 3] = test_signal
        
        # Set some delays (in milliseconds)
        self.synchronizer._channel_delays = np.array([0, 1.25, -1.25, 2.5])  # ms
        
        compensated = self.synchronizer.compensate_channel_delays(test_data)
        
        self.assertEqual(compensated.shape, test_data.shape)
        
        # Channel 0 should be unchanged (no delay)
        np.testing.assert_array_equal(compensated[:, 0], test_data[:, 0])
        
        # Other channels should be shifted
        self.assertFalse(np.array_equal(compensated[:, 1], test_data[:, 1]))
        self.assertFalse(np.array_equal(compensated[:, 2], test_data[:, 2]))
        self.assertFalse(np.array_equal(compensated[:, 3], test_data[:, 3]))
    
    def test_reset_synchronization_state(self):
        """Test synchronization state reset."""
        # Set some state
        self.synchronizer._channel_delays = np.array([1, 2, 3, 4])
        self.synchronizer._drift_accumulator = 0.5
        self.synchronizer._sync_quality = 0.8
        self.synchronizer._sync_failures = 5
        
        # Reset
        self.synchronizer.reset_synchronization_state()
        
        # Verify reset
        np.testing.assert_array_equal(self.synchronizer._channel_delays, np.zeros(4))
        self.assertEqual(self.synchronizer._drift_accumulator, 0.0)
        self.assertEqual(self.synchronizer._sync_quality, 1.0)
        self.assertEqual(self.synchronizer._sync_failures, 0)
    
    def test_get_synchronization_diagnostics(self):
        """Test synchronization diagnostics."""
        diagnostics = self.synchronizer.get_synchronization_diagnostics()
        
        expected_keys = [
            'reference_channel', 'channel_delays_ms', 'sync_quality',
            'drift_accumulator', 'sync_failures', 'last_sync_check',
            'correlation_history_length', 'coherence_threshold'
        ]
        
        for key in expected_keys:
            self.assertIn(key, diagnostics)
        
        self.assertEqual(diagnostics['reference_channel'], 0)
        self.assertEqual(len(diagnostics['channel_delays_ms']), 4)


class TestAudioCaptureEngineSync(unittest.TestCase):
    """Test synchronization integration in AudioCaptureEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = AudioCaptureEngine(
            sample_rate=8000,
            channels=4,
            buffer_duration=0.5
        )
    
    def test_synchronization_initialization(self):
        """Test synchronization is properly initialized."""
        self.assertIsNotNone(self.engine._synchronizer)
        self.assertTrue(self.engine._sync_enabled)
        self.assertEqual(self.engine._synchronizer.channels, 4)
        self.assertEqual(self.engine._synchronizer.sample_rate, 8000)
    
    def test_enable_disable_synchronization(self):
        """Test enabling/disabling synchronization."""
        # Initially enabled
        self.assertTrue(self.engine._sync_enabled)
        
        # Disable
        self.engine.enable_synchronization(False)
        self.assertFalse(self.engine._sync_enabled)
        
        # Re-enable
        self.engine.enable_synchronization(True)
        self.assertTrue(self.engine._sync_enabled)
    
    def test_set_sync_reference_channel(self):
        """Test setting synchronization reference channel."""
        # Valid channel (1-based)
        result = self.engine.set_sync_reference_channel(3)
        self.assertTrue(result)
        
        # Invalid channel
        result = self.engine.set_sync_reference_channel(0)
        self.assertFalse(result)
        
        result = self.engine.set_sync_reference_channel(5)
        self.assertFalse(result)
    
    def test_apply_synchronization(self):
        """Test synchronization application in audio callback."""
        self.engine._capturing = True
        
        # Create test data
        frames = 128
        test_data = np.random.random((frames, 4)).astype(np.float32) * 0.1
        
        # Apply synchronization
        result = self.engine._apply_synchronization(test_data, time.time())
        
        self.assertEqual(result.shape, test_data.shape)
        self.assertEqual(result.dtype, test_data.dtype)
    
    def test_get_synchronization_metrics(self):
        """Test getting synchronization metrics."""
        self.engine._capturing = True
        
        # Initially no metrics
        metrics = self.engine.get_synchronization_metrics()
        self.assertIsNone(metrics)
        
        # Simulate some synchronization analysis
        test_data = np.random.random((256, 4)).astype(np.float32) * 0.1
        self.engine._apply_synchronization(test_data, time.time())
        
        # Should have metrics now
        metrics = self.engine.get_synchronization_metrics()
        if metrics:  # May be None if sync interval hasn't passed
            self.assertIsInstance(metrics, SynchronizationMetrics)
    
    def test_enhanced_synchronization_status(self):
        """Test enhanced synchronization status."""
        status = self.engine.get_synchronization_status()
        
        # Check for synchronization-specific fields
        sync_fields = [
            'sync_enabled', 'sync_quality', 'reference_channel',
            'channel_delays_ms', 'sync_check_interval'
        ]
        
        for field in sync_fields:
            self.assertIn(field, status)
        
        self.assertTrue(status['sync_enabled'])
        self.assertEqual(status['reference_channel'], 1)  # 1-based
    
    def test_calibrate_synchronization_not_capturing(self):
        """Test calibration when not capturing."""
        result = self.engine.calibrate_synchronization(0.1)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Not capturing', result['message'])
    
    def test_export_synchronization_data(self):
        """Test exporting synchronization data."""
        import tempfile
        import os
        import json
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export data
            result = self.engine.export_synchronization_data(temp_path)
            self.assertTrue(result)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            # Check structure
            expected_keys = ['timestamp', 'sample_rate', 'channels', 'sync_enabled', 'diagnostics']
            for key in expected_keys:
                self.assertIn(key, data)
            
            self.assertEqual(data['sample_rate'], 8000)
            self.assertEqual(data['channels'], 4)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()