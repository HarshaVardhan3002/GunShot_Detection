"""
Demo script for the main gunshot localization pipeline.
"""
import numpy as np
import json
import tempfile
import os
import time
import logging
from main_pipeline import GunshotLocalizationPipeline, DefaultEventHandler, PipelineState


def create_test_config():
    """Create a test configuration file."""
    config_data = {
        "microphones": [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0},
            {"id": 3, "x": 0.0, "y": 1.0, "z": 0.0},
            {"id": 4, "x": 1.0, "y": 1.0, "z": 0.0},
            {"id": 5, "x": 0.5, "y": 0.5, "z": 1.0},
            {"id": 6, "x": 1.5, "y": 0.5, "z": 0.0},
            {"id": 7, "x": 0.5, "y": 1.5, "z": 0.0},
            {"id": 8, "x": 1.5, "y": 1.5, "z": 0.0}
        ],
        "system": {
            "sample_rate": 48000,
            "sound_speed": 343.0,
            "detection_threshold_db": -20.0,
            "buffer_duration": 0.5,
            "min_confidence": 0.5
        }
    }
    
    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    )
    json.dump(config_data, temp_config, indent=2)
    temp_config.close()
    
    return temp_config.name


class TestEventHandler(DefaultEventHandler):
    """Test event handler that tracks events."""
    
    def __init__(self):
        super().__init__()
        self.detections = []
        self.localizations = []
        self.errors = []
        self.state_changes = []
    
    def on_detection(self, event):
        """Handle gunshot detection event."""
        super().on_detection(event)
        self.detections.append(event)
        print(f"Detection: confidence={event.confidence:.3f}, method={event.detection_method}")
    
    def on_localization(self, result):
        """Handle successful localization result."""
        super().on_localization(result)
        self.localizations.append(result)
        loc = result.location_result
        print(f"Localization: ({loc.x:.2f}, {loc.y:.2f}) confidence={loc.confidence:.3f}")
    
    def on_error(self, error, context):
        """Handle pipeline errors."""
        super().on_error(error, context)
        self.errors.append((error, context))
        print(f"Error in {context}: {error}")
    
    def on_state_change(self, old_state, new_state):
        """Handle pipeline state changes."""
        super().on_state_change(old_state, new_state)
        self.state_changes.append((old_state, new_state))
        print(f"State change: {old_state.value} -> {new_state.value}")


def test_pipeline_setup():
    """Test pipeline setup and configuration."""
    print("=== Testing Pipeline Setup ===")
    
    # Create test configuration
    config_path = create_test_config()
    
    try:
        # Create event handler
        event_handler = TestEventHandler()
        
        # Create pipeline
        pipeline = GunshotLocalizationPipeline(
            config_path=config_path,
            event_handler=event_handler
        )
        
        print(f"Initial state: {pipeline.state.value}")
        
        # Setup pipeline
        print("Setting up pipeline...")
        success = pipeline.setup()
        print(f"Setup result: {success}")
        
        if success:
            # Check component initialization
            components = [
                ('config_manager', pipeline.config_manager),
                ('audio_capture', pipeline.audio_capture),
                ('gunshot_detector', pipeline.gunshot_detector),
                ('tdoa_localizer', pipeline.tdoa_localizer),
                ('intensity_filter', pipeline.intensity_filter),
                ('channel_selector', pipeline.channel_selector)
            ]
            
            print("\nComponent status:")
            for name, component in components:
                status = "✓ Initialized" if component is not None else "✗ Not initialized"
                print(f"  {name}: {status}")
            
            # Get system status
            print("\nSystem status:")
            status = pipeline.get_system_status()
            print(f"  Pipeline state: {status['pipeline_state']}")
            print(f"  Processing mode: {status['processing_mode']}")
            print(f"  Running: {status['is_running']}")
            
            # Get metrics
            print("\nMetrics:")
            metrics = pipeline.get_metrics()
            print(f"  Total detections: {metrics.total_detections}")
            print(f"  Successful localizations: {metrics.successful_localizations}")
            print(f"  Average processing time: {metrics.average_processing_time:.4f}s")
            
            return pipeline, event_handler
        else:
            print("Pipeline setup failed")
            return None, None
            
    finally:
        # Clean up config file
        if os.path.exists(config_path):
            os.unlink(config_path)


def test_component_integration(pipeline, event_handler):
    """Test individual component integration."""
    print("\n=== Testing Component Integration ===")
    
    if pipeline is None:
        print("Pipeline not available for testing")
        return
    
    # Create test audio data
    sample_rate = 48000
    duration = 0.1  # 100ms
    num_samples = int(sample_rate * duration)
    num_channels = 8
    
    # Generate synthetic gunshot-like signal
    t = np.linspace(0, duration, num_samples)
    
    # Create a sharp impulse followed by decay (gunshot-like)
    impulse = np.exp(-t * 50) * np.sin(2 * np.pi * 1000 * t)  # 1kHz with decay
    
    # Add some noise
    noise = np.random.randn(num_samples) * 0.1
    signal = impulse + noise
    
    # Create multi-channel data with time delays (simulating source location)
    audio_data = np.zeros((num_samples, num_channels))
    source_pos = np.array([0.5, 0.5, 0.0])  # Source at center
    
    # Get microphone positions
    mic_positions = pipeline.config_manager.get_microphone_positions()
    
    for i, mic in enumerate(mic_positions):
        mic_pos = np.array([mic.x, mic.y, mic.z])
        distance = np.linalg.norm(source_pos - mic_pos)
        delay_samples = int(distance / 343.0 * sample_rate)  # Sound speed delay
        
        if delay_samples < num_samples:
            # Apply delay and some attenuation
            delayed_signal = np.zeros(num_samples)
            delayed_signal[delay_samples:] = signal[:num_samples-delay_samples]
            attenuation = 1.0 / (1.0 + distance)  # Simple distance attenuation
            audio_data[:, i] = delayed_signal * attenuation
        else:
            audio_data[:, i] = noise  # Just noise if delay too large
    
    print(f"Created test audio: {audio_data.shape} samples")
    
    # Test gunshot detection
    print("\nTesting gunshot detection...")
    detection_event = pipeline._detect_gunshot(audio_data)
    if detection_event:
        print(f"  Detection successful: confidence={detection_event.confidence:.3f}")
    else:
        print("  No detection (confidence too low)")
    
    # Test channel selection
    print("\nTesting channel selection...")
    channel_selection = pipeline._select_channels(audio_data, 0.8)
    print(f"  Selected channels: {channel_selection.selected_channels}")
    print(f"  Selection confidence: {channel_selection.selection_confidence:.3f}")
    
    # Test localization
    print("\nTesting localization...")
    location_result = pipeline._localize_gunshot(audio_data, channel_selection)
    print(f"  Estimated location: ({location_result.x:.2f}, {location_result.y:.2f})")
    print(f"  Localization confidence: {location_result.confidence:.3f}")
    print(f"  Residual error: {location_result.residual_error:.4f}")
    
    # Compare with actual source position
    estimated_pos = np.array([location_result.x, location_result.y])
    actual_pos = source_pos[:2]
    error_distance = np.linalg.norm(estimated_pos - actual_pos)
    print(f"  Localization error: {error_distance:.3f}m")


def test_pipeline_metrics(pipeline, event_handler):
    """Test pipeline metrics and monitoring."""
    print("\n=== Testing Pipeline Metrics ===")
    
    if pipeline is None:
        print("Pipeline not available for testing")
        return
    
    # Reset metrics
    pipeline.reset_metrics()
    initial_metrics = pipeline.get_metrics()
    print(f"Initial metrics reset: {initial_metrics.total_detections} detections")
    
    # Simulate some processing results
    from tdoa_localizer import LocationResult
    from adaptive_channel_selector import ChannelSelectionResult
    from main_pipeline import DetectionEvent, LocalizationResult
    
    # Create mock results
    for i in range(5):
        # Mock detection event
        detection_event = DetectionEvent(
            timestamp=time.time(),
            confidence=0.7 + i * 0.05,
            audio_data=np.random.randn(4800, 8),
            processing_time=0.01 + i * 0.002
        )
        
        # Mock location result
        location_result = LocationResult(
            x=i * 0.2, y=i * 0.2, z=0.0,
            confidence=0.6 + i * 0.05,
            residual_error=0.1 - i * 0.01,
            timestamp=time.time(),
            microphones_used=[0, 1, 2, 3],
            tdoa_matrix=None,
            correlation_peaks=None
        )
        
        # Mock channel selection
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
        
        # Create complete result
        complete_result = LocalizationResult(
            detection_event=detection_event,
            location_result=location_result,
            channel_selection=channel_selection,
            total_processing_time=0.02 + i * 0.003,
            pipeline_latency=0.02 + i * 0.003,
            quality_metrics={}
        )
        
        # Update metrics
        pipeline._update_metrics(complete_result)
    
    # Check updated metrics
    final_metrics = pipeline.get_metrics()
    print(f"Final metrics:")
    print(f"  Total detections: {final_metrics.total_detections}")
    print(f"  Successful localizations: {final_metrics.successful_localizations}")
    print(f"  Failed localizations: {final_metrics.failed_localizations}")
    print(f"  Average processing time: {final_metrics.average_processing_time:.4f}s")
    print(f"  Average latency: {final_metrics.average_latency:.4f}s")
    print(f"  Localization accuracy: {final_metrics.localization_accuracy:.3f}")
    
    # Check recent results
    recent_results = pipeline.get_recent_results(3)
    print(f"  Recent results count: {len(recent_results)}")


def main():
    """Main demo function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Gunshot Localization Pipeline Demo")
    print("=" * 40)
    
    try:
        # Test pipeline setup
        pipeline, event_handler = test_pipeline_setup()
        
        if pipeline is not None:
            # Test component integration
            test_component_integration(pipeline, event_handler)
            
            # Test metrics
            test_pipeline_metrics(pipeline, event_handler)
            
            # Show event handler results
            print(f"\n=== Event Handler Summary ===")
            print(f"Detections: {len(event_handler.detections)}")
            print(f"Localizations: {len(event_handler.localizations)}")
            print(f"Errors: {len(event_handler.errors)}")
            print(f"State changes: {len(event_handler.state_changes)}")
            
            # Cleanup
            pipeline.shutdown()
            print("\nPipeline shutdown complete")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()