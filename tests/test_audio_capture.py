"""
Unit tests for AudioCaptureEngine.
"""
import unittest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock

from audio_capture import AudioCaptureEngine, AudioBuffer


class TestAudioCaptureEngine(unittest.TestCase):
    """Test cases for AudioCaptureEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create engine with small buffer for testing
        self.engine = AudioCaptureEngine(
            sample_rate=8000,  # Lower sample rate for faster tests
            channels=4,        # Fewer channels for testing
            buffer_duration=0.5  # Small buffer
        )
    
    def test_initialization(self):
        """Test proper initialization of AudioCaptureEngine."""
        self.assertEqual(self.engine.sample_rate, 8000)
        self.assertEqual(self.engine.channels, 4)
        self.assertEqual(self.engine.buffer_duration, 0.5)
        self.assertEqual(self.engine.buffer_size, 4000)  # 8000 * 0.5
        self.assertFalse(self.engine.is_capturing())
        
        # Check initial channel status
        status = self.engine.get_channel_status()
        self.assertEqual(len(status), 4)
        for i in range(1, 5):
            self.assertIn(i, status)
    
    def test_buffer_initialization(self):
        """Test audio buffer is properly initialized."""
        self.assertEqual(self.engine._audio_buffer.shape, (4000, 4))
        self.assertEqual(self.engine._buffer_index, 0)
        self.assertEqual(self.engine._samples_captured, 0)
        self.assertEqual(self.engine._buffer_overruns, 0)
    
    @patch('sounddevice.InputStream')
    @patch('sounddevice.query_devices')
    def test_start_capture_success(self, mock_query_devices, mock_input_stream):
        """Test successful audio capture start."""
        # Mock audio devices
        mock_query_devices.return_value = [
            {'name': 'Test Device', 'max_input_channels': 8, 'default_samplerate': 48000}
        ]
        
        # Mock audio stream
        mock_stream = Mock()
        mock_input_stream.return_value = mock_stream
        
        # Start capture
        self.engine.start_capture()
        
        # Verify stream was created and started
        mock_input_stream.assert_called_once()
        mock_stream.start.assert_called_once()
        self.assertTrue(self.engine.is_capturing())
    
    @patch('sounddevice.InputStream')
    def test_start_capture_failure(self, mock_input_stream):
        """Test audio capture start failure."""
        # Mock stream creation failure
        mock_input_stream.side_effect = Exception("Device not available")
        
        # Start capture should raise exception
        with self.assertRaises(Exception):
            self.engine.start_capture()
        
        self.assertFalse(self.engine.is_capturing())
    
    def test_start_capture_already_running(self):
        """Test starting capture when already running."""
        self.engine._capturing = True
        
        with patch('sounddevice.InputStream'):
            self.engine.start_capture()
        
        # Should not create new stream
        self.assertTrue(self.engine.is_capturing())
    
    def test_stop_capture(self):
        """Test stopping audio capture."""
        # Mock running stream
        mock_stream = Mock()
        self.engine._stream = mock_stream
        self.engine._capturing = True
        
        self.engine.stop_capture()
        
        # Verify stream was stopped and closed
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        self.assertFalse(self.engine.is_capturing())
        self.assertIsNone(self.engine._stream)
    
    def test_stop_capture_not_running(self):
        """Test stopping capture when not running."""
        self.assertFalse(self.engine.is_capturing())
        
        # Should not raise exception
        self.engine.stop_capture()
        self.assertFalse(self.engine.is_capturing())
    
    def test_audio_callback_normal(self):
        """Test normal audio callback processing."""
        self.engine._capturing = True
        
        # Create test audio data
        frames = 100
        test_data = np.random.random((frames, 4)).astype(np.float32) * 0.1
        
        # Call audio callback
        self.engine._audio_callback(test_data, frames, None, None)
        
        # Verify data was written to buffer
        self.assertEqual(self.engine._buffer_index, frames)
        self.assertEqual(self.engine._samples_captured, frames)
        np.testing.assert_array_equal(
            self.engine._audio_buffer[:frames], test_data
        )
    
    def test_audio_callback_buffer_wrap(self):
        """Test audio callback with buffer wraparound."""
        self.engine._capturing = True
        
        # Fill buffer almost to the end
        self.engine._buffer_index = self.engine.buffer_size - 50
        
        # Create test data that will cause wraparound
        frames = 100
        test_data = np.random.random((frames, 4)).astype(np.float32) * 0.1
        
        # Call audio callback
        self.engine._audio_callback(test_data, frames, None, None)
        
        # Verify wraparound occurred
        self.assertEqual(self.engine._buffer_index, 50)  # Wrapped around
        self.assertEqual(self.engine._buffer_overruns, 1)
        self.assertEqual(self.engine._samples_captured, frames)
    
    def test_audio_callback_not_capturing(self):
        """Test audio callback when not capturing."""
        self.engine._capturing = False
        
        frames = 100
        test_data = np.random.random((frames, 4)).astype(np.float32)
        
        # Call audio callback
        self.engine._audio_callback(test_data, frames, None, None)
        
        # Verify no data was processed
        self.assertEqual(self.engine._buffer_index, 0)
        self.assertEqual(self.engine._samples_captured, 0)
    
    def test_update_channel_status(self):
        """Test channel status update based on signal."""
        # Test with active signal on some channels
        test_data = np.zeros((100, 4), dtype=np.float32)
        test_data[:, 0] = 0.1  # Channel 1 active
        test_data[:, 2] = 0.05  # Channel 3 active
        # Channels 2 and 4 remain silent
        
        self.engine._update_channel_status_with_history(test_data, time.time())
        
        status = self.engine.get_channel_status()
        self.assertTrue(status[1])   # Channel 1 active
        self.assertFalse(status[2])  # Channel 2 inactive
        self.assertTrue(status[3])   # Channel 3 active
        self.assertFalse(status[4])  # Channel 4 inactive
    
    def test_get_audio_buffer_capturing(self):
        """Test getting audio buffer while capturing."""
        self.engine._capturing = True
        
        # Add some test data to buffer
        test_data = np.random.random((100, 4)).astype(np.float32)
        self.engine._audio_buffer[:100] = test_data
        self.engine._last_audio_time = time.time()
        
        buffer = self.engine.get_audio_buffer()
        
        self.assertIsInstance(buffer, AudioBuffer)
        self.assertEqual(buffer.sample_rate, 8000)
        self.assertEqual(buffer.duration, 0.5)
        self.assertEqual(buffer.data.shape, (4000, 4))
        np.testing.assert_array_equal(buffer.data[:100], test_data)
    
    def test_get_audio_buffer_not_capturing(self):
        """Test getting audio buffer when not capturing."""
        self.engine._capturing = False
        
        buffer = self.engine.get_audio_buffer()
        self.assertIsNone(buffer)
    
    def test_get_capture_statistics(self):
        """Test getting capture statistics."""
        self.engine._samples_captured = 1000
        self.engine._buffer_overruns = 2
        
        stats = self.engine.get_capture_statistics()
        
        # Check that all expected fields are present
        expected_fields = [
            'samples_captured', 'buffer_overruns', 'sync_errors', 'buffer_size',
            'channels', 'sample_rate', 'buffer_utilization', 'drift_compensation',
            'avg_callback_latency_ms', 'max_callback_latency_ms', 'current_segment',
            'active_channels'
        ]
        
        for field in expected_fields:
            self.assertIn(field, stats)
        
        # Check specific values
        self.assertEqual(stats['samples_captured'], 1000)
        self.assertEqual(stats['buffer_overruns'], 2)
        self.assertEqual(stats['buffer_size'], 4000)
        self.assertEqual(stats['channels'], 4)
        self.assertEqual(stats['sample_rate'], 8000)
    
    @patch('sounddevice.query_devices')
    def test_list_audio_devices(self, mock_query_devices):
        """Test listing compatible audio devices."""
        # Mock device list
        mock_devices = [
            {'name': 'Device 1', 'max_input_channels': 8, 'default_samplerate': 48000},
            {'name': 'Device 2', 'max_input_channels': 2, 'default_samplerate': 44100},  # Not enough channels
            {'name': 'Device 3', 'max_input_channels': 4, 'default_samplerate': 48000},
        ]
        mock_query_devices.return_value = mock_devices
        
        devices = self.engine.list_audio_devices()
        
        # Should return only devices with enough channels
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0]['name'], 'Device 1')
        self.assertEqual(devices[1]['name'], 'Device 3')
    
    @patch('sounddevice.InputStream')
    def test_device_compatibility_success(self, mock_input_stream):
        """Test successful device compatibility check."""
        mock_stream = Mock()
        mock_input_stream.return_value = mock_stream
        
        result = self.engine.test_device_compatibility(0)
        
        self.assertTrue(result)
        mock_input_stream.assert_called_once()
        mock_stream.close.assert_called_once()
    
    @patch('sounddevice.InputStream')
    def test_device_compatibility_failure(self, mock_input_stream):
        """Test failed device compatibility check."""
        mock_input_stream.side_effect = Exception("Device error")
        
        result = self.engine.test_device_compatibility(0)
        
        self.assertFalse(result)
    
    def test_thread_safety(self):
        """Test thread safety of buffer operations."""
        import threading
        
        self.engine._capturing = True
        results = []
        
        def write_data():
            """Write data to buffer."""
            for i in range(10):
                test_data = np.ones((10, 4), dtype=np.float32) * i
                self.engine._audio_callback(test_data, 10, None, None)
        
        def read_data():
            """Read data from buffer."""
            for i in range(10):
                buffer = self.engine.get_audio_buffer()
                if buffer:
                    results.append(buffer.data.shape)
        
        # Run concurrent operations
        write_thread = threading.Thread(target=write_data)
        read_thread = threading.Thread(target=read_data)
        
        write_thread.start()
        read_thread.start()
        
        write_thread.join()
        read_thread.join()
        
        # Verify no crashes occurred and some data was read
        self.assertGreater(len(results), 0)
        for shape in results:
            self.assertEqual(shape, (4000, 4))
    
    def test_synchronized_buffer_segment(self):
        """Test synchronized buffer segment extraction."""
        self.engine._capturing = True
        
        # Add test data to buffer
        test_data = np.random.random((1000, 4)).astype(np.float32)
        self.engine._audio_callback(test_data, 1000, None, None)
        
        # Get synchronized segment
        segment = self.engine.get_synchronized_buffer_segment(0.1)  # 100ms segment
        
        self.assertIsInstance(segment, AudioBuffer)
        expected_samples = int(8000 * 0.1)  # 800 samples for 100ms at 8kHz
        self.assertEqual(segment.data.shape[0], expected_samples)
        self.assertEqual(segment.duration, 0.1)
    
    def test_freeze_buffer_on_trigger(self):
        """Test buffer freezing functionality."""
        self.engine._capturing = True
        
        # Add test data
        test_data = np.random.random((500, 4)).astype(np.float32)
        self.engine._audio_callback(test_data, 500, None, None)
        
        # Freeze buffer
        frozen_buffer = self.engine.freeze_buffer_on_trigger()
        
        self.assertIsInstance(frozen_buffer, AudioBuffer)
        self.assertEqual(frozen_buffer.data.shape, (4000, 4))
        
        # Verify it's a snapshot (adding more data shouldn't affect frozen buffer)
        original_data = frozen_buffer.data.copy()
        more_data = np.ones((100, 4), dtype=np.float32)
        self.engine._audio_callback(more_data, 100, None, None)
        
        np.testing.assert_array_equal(frozen_buffer.data, original_data)
    
    def test_enhanced_statistics(self):
        """Test enhanced capture statistics."""
        self.engine._capturing = True
        self.engine._sync_errors = 5
        self.engine._drift_compensation = 0.002  # 2ms drift
        
        stats = self.engine.get_capture_statistics()
        
        # Check new statistics fields
        self.assertIn('sync_errors', stats)
        self.assertIn('drift_compensation', stats)
        self.assertIn('buffer_utilization', stats)
        self.assertIn('avg_callback_latency_ms', stats)
        self.assertIn('active_channels', stats)
        
        self.assertEqual(stats['sync_errors'], 5)
        self.assertEqual(stats['drift_compensation'], 2.0)  # Should be in milliseconds
    
    def test_synchronization_status(self):
        """Test synchronization status reporting."""
        self.engine._capturing = True
        
        # Add some frame data
        for i in range(5):
            test_data = np.random.random((100, 4)).astype(np.float32)
            self.engine._audio_callback(test_data, 100, None, None)
            time.sleep(0.01)  # Small delay between frames
        
        sync_status = self.engine.get_synchronization_status()
        
        self.assertIn('status', sync_status)
        self.assertIn('drift_compensation_ms', sync_status)
        self.assertIn('recent_frame_count', sync_status)
        self.assertIn('timing_jitter', sync_status)
        
        self.assertGreater(sync_status['recent_frame_count'], 0)
    
    def test_channel_health(self):
        """Test channel health monitoring."""
        self.engine._capturing = True
        
        # Create test data with different signal levels per channel
        test_data = np.zeros((100, 4), dtype=np.float32)
        test_data[:, 0] = 0.1  # Strong signal
        test_data[:, 1] = 0.01  # Weak signal
        test_data[:, 2] = 0.0   # No signal
        test_data[:, 3] = np.random.random(100) * 0.05  # Noisy signal
        
        # Process multiple frames to build history
        for _ in range(10):
            self.engine._audio_callback(test_data, 100, None, None)
        
        health = self.engine.get_channel_health()
        
        self.assertEqual(len(health), 4)
        for channel_id in range(1, 5):
            self.assertIn(channel_id, health)
            self.assertIn('status', health[channel_id])
            self.assertIn('active', health[channel_id])
            self.assertIn('avg_rms', health[channel_id])
            self.assertIn('activity_ratio', health[channel_id])
    
    def test_buffer_size_adjustment(self):
        """Test dynamic buffer size adjustment."""
        original_size = self.engine.buffer_size
        
        # Adjust to larger buffer
        result = self.engine.adjust_buffer_size(1.0)  # 1 second
        self.assertTrue(result)
        self.assertEqual(self.engine.buffer_size, 8000)  # 8000 samples at 8kHz
        self.assertEqual(self.engine.buffer_duration, 1.0)
        
        # Test invalid adjustment
        result = self.engine.adjust_buffer_size(-1.0)
        self.assertFalse(result)
        
        # Test adjustment with data preservation
        self.engine._capturing = True
        test_data = np.ones((100, 4), dtype=np.float32)
        self.engine._audio_callback(test_data, 100, None, None)
        
        result = self.engine.adjust_buffer_size(0.25)  # Smaller buffer
        self.assertTrue(result)
        self.assertEqual(self.engine.buffer_size, 2000)  # 2000 samples at 8kHz
    
    def test_health_check(self):
        """Test comprehensive health check."""
        self.engine._capturing = True
        
        # Simulate some issues
        self.engine._buffer_overruns = 2
        self.engine._sync_errors = 50
        self.engine._samples_captured = 8000  # 1 second of data to calculate overrun rate
        
        health = self.engine.perform_health_check()
        
        self.assertIn('overall_status', health)
        self.assertIn('issues', health)
        self.assertIn('warnings', health)
        self.assertIn('recommendations', health)
        
        # Should detect buffer overruns (with sufficient sample data)
        self.assertGreaterEqual(len(health['warnings']) + len(health['issues']), 0)
    
    def test_reset_synchronization(self):
        """Test synchronization reset."""
        # Set some drift compensation
        self.engine._drift_compensation = 0.005
        self.engine._sync_errors = 10
        
        self.engine.reset_synchronization()
        
        self.assertEqual(self.engine._drift_compensation, 0.0)
        self.assertEqual(self.engine._sync_errors, 0)
        self.assertIsNone(self.engine._sync_reference_time)


if __name__ == '__main__':
    unittest.main()