"""
Integration tests for the main gunshot localization pipeline.
"""
import unittest
import numpy as np
import tempfile
import json
import os
import time
from unittest.mock import Mock, patch
from main_pipeline import (
    GunshotLocalizationPipeline, 
    DefaultEventHandler,
    PipelineState,
    DetectionEvent,
    LocalizationResult
)


class TestMainPipeline(unittest.TestCase):
    """Test cases for the main processing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.config_data = {
            "microphone_positions": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [1.5, 0.5, 0.0],
                [0.5, 1.5, 0.0],
                [1.5, 1.5, 0.0]
            ],
            "sample_rate": 48000,
            "sound_speed": 343.0,
            "detector_type": "amplitude",
            "amplitude_threshold": 0.1,
            "pipeline": {
                "buffer_duration": 0.1,
                "detection_threshold": 0.5,
                "max_processing_time": 0.05
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump(self.config_data, self.temp_config)
        self.temp_config.close()
        
        # Create event handler mock
        self.event_handler = Mock(spec=DefaultEventHandler)
        
        # Create pipeline
        self.pipeline = GunshotLocalizationPipeline(
            config_path=self.temp_config.name,
            event_handler=self.event_handler
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary config file
        if os.path.exists(self.temp_config.name):
            os.unlink(self.temp_config.name)
        
        # Shutdown pipeline
        if hasattr(self.pipeline, 'shutdown'):
            self.pipeline.shutdown()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.state, PipelineState.STOPPED)
        self.assertFalse(self.pipeline.is_running)
        self.assertIsNotNone(self.pipeline.config_path)
        self.assertIsNotNone(self.pipeline.event_handler)
    
    @patch('main_pipeline.AudioCaptureEngine')
    def test_pipeline_setup(self, mock_audio_capture):
        """Test pipeline setup process."""
        # Mock audio capture
        mock_audio_instance = Mock()
        mock_audio_capture.return_value = mock_audio_instance
        
        # Setup pipeline
        result = self.pipeline.setup()
        
        # Verify setup success
        self.assertTrue(result)
        self.assertIsNotNone(self.pipeline.config_manager)
        self.assertIsNotNone(self.pipeline.audio_capture)
        self.assertIsNotNone(self.pipeline.gunshot_detector)
        self.assertIsNotNone(self.pipeline.tdoa_localizer)
        self.assertIsNotNone(self.pipeline.intensity_filter)
        self.assertIsNotNone(self.pipeline.channel_selector)
    
    @patch('main_pipeline.AudioCaptureEngine')
    def test_pipeline_start_stop(self, mock_audio_capture):
        """Test pipeline start and stop operations."""
        # Mock audio capture
        mock_audio_instance = Mock()
        mock_audio_instance.start.return_value = True
        mock_audio_instance.stop.return_value = True
        mock_audio_capture.return_value = mock_audio_instance
        
        # Setup pipeline
        self.assertTrue(self.pipeline.setup())
        
        # Test start
        result = self.pipeline.start()
        self.assertTrue(result)
        self.assertEqual(self.pipeline.state, PipelineState.RUNNING)
        self.assertTrue(self.pipeline.is_running)
        
        # Brief pause to let processing thread start
        time.sleep(0.1)
        
        # Test stop
        result = self.pipeline.stop()
        self.assertTrue(result)
        self.assertEqual(self.pipeline.state, PipelineState.STOPPED)
        self.assertFalse(self.pipeline.is_running)
    
    @patch('main_pipeline.AudioCaptureEngine')
    def test_detection_event_creation(self, mock_audio_capture):
        """Test detection event creation."""
        # Mock audio capture
        mock_audio_instance = Mock()
        mock_audio_capture.return_value = mock_audio_instance
        
        # Setup pipeline
        self.assertTrue(self.pipeline.setup())
        
        # Create test audio data
        audio_data = np.random.randn(4800, 8)  # 0.1 seconds at 48kHz, 8 channels
        
        # Mock detector to return high confidence detection
        mock_detection = Mock()
        mock_detection.confidence = 0.8
        self.pipeline.gunshot_detector.detect_gunshot = Mock(return_value=mock_detection)
        
        # Test detection
        detection_event = self.pipeline._detect_gunshot(audio_data)
        
        # Verify detection event
        self.assertIsNotNone(detection_event)
        self.assertIsInstance(detection_event, DetectionEvent)
        self.assertEqual(detection_event.confidence, 0.8)
        self.assertIsNotNone(detection_event.timestamp)
        self.assertTrue(np.array_equal(detection_event.audio_data, audio_data))
    
    @patch('main_pipeline.AudioCaptureEngine')
    def test_channel_selection(self, mock_audio_capture):
        """Test channel selection process."""
        # Mock audio capture
        mock_audio_instance = Mock()
        mock_audio_capture.return_value = mock_audio_instance
        
        # Setup pipeline
        self.assertTrue(self.pipeline.setup())
        
        # Create test audio data
        audio_data = np.random.randn(4800, 8)
        
        # Test channel selection
        channel_selection = self.pipeline._select_channels(audio_data, 0.8)
        
        # Verify channel selection result
        self.assertIsNotNone(channel_selection)
        self.assertIsInstance(channel_selection.selected_channels, list)
        self.assertGreater(len(channel_selection.selected_channels), 0)
        self.assertIsInstance(channel_selection.channel_weights, np.ndarray)
    
    @patch('main_pipeline.AudioCaptureEngine')
    def test_gunshot_localization(self, mock_audio_capture):
        """Test gunshot localization process."""
        # Mock audio capture
        mock_audio_instance = Mock()
        mock_audio_capture.return_value = mock_audio_instance
        
        # Setup pipeline
        self.assertTrue(self.pipeline.setup())
        
        # Create test audio data
        audio_data = np.random.randn(4800, 8)
        
        # Create mock channel selection
        from adaptive_channel_selector import ChannelSelectionResult
        channel_selection = ChannelSelectionResult(
            selected_channels=[0, 1, 2, 3],
            excluded_channels=[4, 5, 6, 7],
            channel_weights=np.ones(8),
            selection_confidence=0.8,
            strategy_used="test",
            fallback_applied=False,
            quality_metrics={},
            timestamp=time.time()
        )
        
        # Test localization
        location_result = self.pipeline._localize_gunshot(audio_data, channel_selection)
        
        # Verify location result
        self.assertIsNotNone(location_result)
        self.assertIsInstance(location_result.x, float)
        self.assertIsInstance(location_result.y, float)
        self.assertIsInstance(location_result.confidence, float)
    
    @patch('main_pipeline.AudioCaptureEngine')
    def test_metrics_tracking(self, mock_audio_capture):
        """Test pipeline metrics tracking."""
        # Mock audio capture
        mock_audio_instance = Mock()
        mock_audio_capture.return_value = mock_audio_instance
        
        # Setup pipeline
        self.assertTrue(self.pipeline.setup())
        
        # Get initial metrics
        initial_metrics = self.pipeline.get_metrics()
        self.assertEqual(initial_metrics.total_detections, 0)
        self.assertEqual(initial_metrics.successful_localizations, 0)
        
        # Create mock result for metrics update
        from tdoa_localizer import LocationResult
        mock_detection = DetectionEvent(
            timestamp=time.time(),
            confidence=0.8,
            audio_data=np.random.randn(4800, 8),
            processing_time=0.01
        )
        
        mock_location = LocationResult(
            x=1.0, y=1.0, z=0.0,
            confidence=0.7,
            residual_error=0.1,
            timestamp=time.time(),
            microphones_used=[0, 1, 2, 3],
            tdoa_matrix=None,
            correlation_peaks=None
        )
        
        from adaptive_channel_selector import ChannelSelectionResult
        mock_channel_selection = ChannelSelectionResult(
            selected_channels=[0, 1, 2, 3],
            excluded_channels=[],
            channel_weights=np.ones(8),
            selection_confidence=0.8,
            strategy_used="test",
            fallback_applied=False,
            quality_metrics={},
            timestamp=time.time()
        )
        
        mock_result = LocalizationResult(
            detection_event=mock_detection,
            location_result=mock_location,
            channel_selection=mock_channel_selection,
            total_processing_time=0.02,
            pipeline_latency=0.02,
            quality_metrics={}
        )
        
        # Update metrics
        self.pipeline._update_metrics(mock_result)
        
        # Verify metrics update
        updated_metrics = self.pipeline.get_metrics()
        self.assertEqual(updated_metrics.total_detections, 1)
        self.assertEqual(updated_metrics.successful_localizations, 1)
        self.assertGreater(updated_metrics.average_processing_time, 0)
    
    @patch('main_pipeline.AudioCaptureEngine')
    def test_system_status(self, mock_audio_capture):
        """Test system status reporting."""
        # Mock audio capture
        mock_audio_instance = Mock()
        mock_audio_capture.return_value = mock_audio_instance
        
        # Setup pipeline
        self.assertTrue(self.pipeline.setup())
        
        # Get system status
        status = self.pipeline.get_system_status()
        
        # Verify status structure
        self.assertIn('pipeline_state', status)
        self.assertIn('processing_mode', status)
        self.assertIn('is_running', status)
        self.assertIn('metrics', status)
        self.assertIn('component_status', status)
        
        # Verify component status
        expected_components = [
            'audio_capture', 'gunshot_detector', 'tdoa_localizer',
            'intensity_filter', 'channel_selector'
        ]
        for component in expected_components:
            self.assertIn(component, status['component_status'])
    
    def test_event_handler_integration(self):
        """Test event handler integration."""
        # Create pipeline with mock event handler
        pipeline = GunshotLocalizationPipeline(
            config_path=self.temp_config.name,
            event_handler=self.event_handler
        )
        
        # Test state change notification
        pipeline._set_state(PipelineState.RUNNING)
        
        # Verify event handler was called
        self.event_handler.on_state_change.assert_called_once()
        args = self.event_handler.on_state_change.call_args[0]
        self.assertEqual(args[0], PipelineState.STOPPED)  # old state
        self.assertEqual(args[1], PipelineState.RUNNING)  # new state


class TestDefaultEventHandler(unittest.TestCase):
    """Test cases for the default event handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = DefaultEventHandler()
    
    def test_event_handler_methods(self):
        """Test event handler methods don't raise exceptions."""
        # Create mock objects
        mock_detection = Mock(spec=DetectionEvent)
        mock_detection.confidence = 0.8
        
        mock_result = Mock(spec=LocalizationResult)
        mock_result.location_result.x = 1.0
        mock_result.location_result.y = 1.0
        mock_result.location_result.confidence = 0.7
        
        # Test methods (should not raise exceptions)
        try:
            self.handler.on_detection(mock_detection)
            self.handler.on_localization(mock_result)
            self.handler.on_error(Exception("test error"), "test_context")
            self.handler.on_state_change(PipelineState.STOPPED, PipelineState.RUNNING)
        except Exception as e:
            self.fail(f"Event handler methods should not raise exceptions: {e}")


def create_test_suite():
    """Create test suite for main pipeline."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestMainPipeline))
    suite.addTest(unittest.makeSuite(TestDefaultEventHandler))
    
    return suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")