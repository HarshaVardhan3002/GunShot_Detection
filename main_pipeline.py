"""
Main processing pipeline for gunshot localization system.
"""
import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
from abc import ABC, abstractmethod

# Import all the modules we've built
from config_manager import ConfigurationManager
from audio_capture import AudioCaptureEngine
from gunshot_detector import AmplitudeBasedDetector
from tdoa_localizer import CrossCorrelationTDoALocalizer, LocationResult
from intensity_filter import RMSIntensityFilter
from adaptive_channel_selector import AdaptiveChannelSelector, ChannelSelectionResult
from output_formatter import RealTimeOutputManager, OutputFormat
from structured_logger import StructuredLogger, PerformanceTracker
from error_handler import ErrorHandlingSystem, ErrorSeverity, ErrorCategory, handle_component_error
from diagnostics import DiagnosticsManager


class PipelineState(Enum):
    """Pipeline execution states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ProcessingMode(Enum):
    """Processing modes."""
    CONTINUOUS = "continuous"
    EVENT_DRIVEN = "event_driven"
    BATCH = "batch"


@dataclass
class DetectionEvent:
    """Container for gunshot detection events."""
    timestamp: float
    confidence: float
    audio_data: np.ndarray
    channel_weights: Optional[np.ndarray] = None
    selected_channels: Optional[List[int]] = None
    detection_method: str = "unknown"
    processing_time: float = 0.0


@dataclass
class LocalizationResult:
    """Container for complete localization results."""
    detection_event: DetectionEvent
    location_result: LocationResult
    channel_selection: ChannelSelectionResult
    total_processing_time: float
    pipeline_latency: float
    quality_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
    """Performance metrics for the pipeline."""
    total_detections: int = 0
    successful_localizations: int = 0
    failed_localizations: int = 0
    average_processing_time: float = 0.0
    average_latency: float = 0.0
    detection_rate: float = 0.0  # Detections per second
    localization_accuracy: float = 0.0
    uptime: float = 0.0
    last_reset_time: float = field(default_factory=time.time)


class PipelineEventHandler(ABC):
    """Abstract base class for pipeline event handlers."""
    
    @abstractmethod
    def on_detection(self, event: DetectionEvent) -> None:
        """Handle gunshot detection event."""
        pass
    
    @abstractmethod
    def on_localization(self, result: LocalizationResult) -> None:
        """Handle successful localization result."""
        pass
    
    @abstractmethod
    def on_error(self, error: Exception, context: str) -> None:
        """Handle pipeline errors."""
        pass
    
    @abstractmethod
    def on_state_change(self, old_state: PipelineState, new_state: PipelineState) -> None:
        """Handle pipeline state changes."""
        pass


class GunshotLocalizationPipeline:
    """Main processing pipeline for gunshot localization."""
    
    def __init__(self, config_path: str, event_handler: Optional[PipelineEventHandler] = None,
                 output_format: OutputFormat = OutputFormat.CONSOLE,
                 output_file: Optional[str] = None,
                 log_dir: str = "logs"):
        """
        Initialize the gunshot localization pipeline.
        
        Args:
            config_path: Path to configuration file
            event_handler: Optional event handler for pipeline events
            output_format: Real-time output format
            output_file: Optional output file path
            log_dir: Directory for log files
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.event_handler = event_handler
        
        # Pipeline state
        self.state = PipelineState.STOPPED
        self.processing_mode = ProcessingMode.EVENT_DRIVEN
        self.is_running = False
        self.processing_thread = None
        
        # Components (will be initialized in setup)
        self.config_manager = None
        self.audio_capture = None
        self.gunshot_detector = None
        self.tdoa_localizer = None
        self.intensity_filter = None
        self.channel_selector = None
        
        # Processing parameters
        self.buffer_duration = 0.5  # 500ms buffer for processing
        self.detection_threshold = 0.5  # Minimum confidence for processing
        self.max_processing_time = 0.1  # 100ms max processing time
        
        # Performance monitoring
        self.metrics = PipelineMetrics()
        self.recent_results = deque(maxlen=100)
        self.performance_history = deque(maxlen=1000)
        
        # Threading and synchronization
        self.processing_lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Initialize output and logging systems
        self.output_manager = RealTimeOutputManager(
            output_format=output_format,
            enable_file_output=output_file is not None,
            output_file=output_file
        )
        
        self.structured_logger = StructuredLogger(
            name="gunshot_pipeline",
            log_dir=log_dir,
            enable_console=False,  # Use output_manager for console
            enable_file=True,
            enable_performance_log=True
        )
        
        self.performance_tracker = PerformanceTracker(self.structured_logger)
        
        # Initialize error handling system
        self.error_handler = ErrorHandlingSystem()
        
        # Initialize diagnostics manager
        self.diagnostics_manager = DiagnosticsManager(pipeline=self)
        
        # Register components for health monitoring
        self._register_components_for_monitoring()
        
        self.logger.info(f"Pipeline initialized with config: {config_path}")
    
    def setup(self) -> bool:
        """
        Set up all pipeline components.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self._set_state(PipelineState.STARTING)
            
            # Load configuration with error handling
            self.logger.info("Loading configuration...")
            try:
                self.config_manager = ConfigurationManager()
                success = self.config_manager.load_config(self.config_path)
                if not success:
                    self.logger.warning("Failed to load configuration, using defaults")
                
                # Get configuration data
                system_config = self.config_manager.get_system_config()
                mic_positions = self.config_manager.get_microphone_positions()
                
            except Exception as e:
                if not self._handle_component_error('config_manager', e, {'config_path': self.config_path}):
                    raise Exception(f"Critical configuration error: {e}")
                # Use minimal default config if recovery failed
                system_config = None
                mic_positions = []
            
            # Initialize audio capture with error handling
            self.logger.info("Initializing audio capture...")
            try:
                if system_config:
                    self.audio_capture = AudioCaptureEngine(
                        sample_rate=system_config.sample_rate,
                        channels=len(mic_positions),
                        buffer_duration=self.buffer_duration
                    )
                else:
                    # Fallback configuration
                    self.audio_capture = AudioCaptureEngine(
                        sample_rate=48000,
                        channels=8,
                        buffer_duration=self.buffer_duration
                    )
            except Exception as e:
                if not self._handle_component_error('audio_capture', e, {'channels': len(mic_positions) if mic_positions else 8}):
                    raise Exception(f"Critical audio capture error: {e}")
            
            # Initialize gunshot detection with error handling
            self.logger.info("Initializing gunshot detection...")
            try:
                self.gunshot_detector = self._create_gunshot_detector(system_config)
            except Exception as e:
                if not self._handle_component_error('gunshot_detector', e, {'system_config': system_config}):
                    raise Exception(f"Critical gunshot detector error: {e}")
            
            # Initialize TDoA localization
            self.logger.info("Initializing TDoA localization...")
            # Convert MicrophonePosition objects to coordinate tuples
            mic_coords = [(mic.x, mic.y, mic.z) for mic in mic_positions]
            self.tdoa_localizer = CrossCorrelationTDoALocalizer(
                microphone_positions=mic_coords,
                sample_rate=system_config.sample_rate,
                sound_speed=system_config.sound_speed
            )
            
            # Initialize intensity filtering
            self.logger.info("Initializing intensity filter...")
            self.intensity_filter = RMSIntensityFilter(
                sample_rate=system_config.sample_rate
            )
            
            # Initialize adaptive channel selection
            self.logger.info("Initializing adaptive channel selector...")
            self.channel_selector = AdaptiveChannelSelector(
                num_channels=len(mic_positions),
                intensity_filter=self.intensity_filter
            )
            
            # Configure components based on config
            self._configure_components(system_config, mic_positions)
            
            self.logger.info("Pipeline setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline setup failed: {e}")
            self._set_state(PipelineState.ERROR)
            if self.event_handler:
                self.event_handler.on_error(e, "setup")
            return False
    
    def _create_gunshot_detector(self, system_config) -> Any:
        """Create gunshot detector based on configuration."""
        # For now, only amplitude-based detector is available
        return AmplitudeBasedDetector(
            sample_rate=system_config.sample_rate,
            channels=8,  # Fixed for 8-microphone array
            threshold_db=system_config.detection_threshold_db
        )
    
    def _configure_components(self, system_config, mic_positions) -> None:
        """Configure components based on configuration."""
        # Configure pipeline parameters from system config
        self.buffer_duration = system_config.buffer_duration
        self.detection_threshold = system_config.min_confidence
        
        # Configure components if they have configuration methods
        if hasattr(self.gunshot_detector, 'configure_detector_parameters'):
            self.gunshot_detector.configure_detector_parameters(
                threshold_db=system_config.detection_threshold_db
            )
        
        if hasattr(self.tdoa_localizer, 'configure_correlation_parameters'):
            self.tdoa_localizer.configure_correlation_parameters(
                sound_speed=system_config.sound_speed
            )
        
        if hasattr(self.intensity_filter, 'configure_filter_parameters'):
            self.intensity_filter.configure_filter_parameters(
                sample_rate=system_config.sample_rate
            )
        
        if hasattr(self.channel_selector, 'configure_selection_parameters'):
            self.channel_selector.configure_selection_parameters(
                min_confidence=system_config.min_confidence
            )
    
    def start(self) -> bool:
        """
        Start the processing pipeline.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.state == PipelineState.RUNNING:
            self.logger.warning("Pipeline is already running")
            return True
        
        if not self._verify_components():
            self.logger.error("Component verification failed")
            return False
        
        try:
            # Start audio capture with error handling
            try:
                self.audio_capture.start_capture()
            except Exception as e:
                context = {'buffer_duration': self.buffer_duration}
                if not self._handle_component_error('audio_capture', e, context):
                    self.logger.error(f"Failed to start audio capture: {e}")
                    return False
            
            # Start processing thread
            self.is_running = True
            self.stop_event.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="GunshotPipelineProcessor"
            )
            self.processing_thread.start()
            
            # Reset metrics
            self.metrics = PipelineMetrics()
            self.metrics.last_reset_time = time.time()
            
            # Start error monitoring
            self.error_handler.start_health_monitoring()
            
            # Start diagnostics monitoring
            self.diagnostics_manager.start_monitoring()
            
            self._set_state(PipelineState.RUNNING)
            self.logger.info("Pipeline started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            self._set_state(PipelineState.ERROR)
            if self.event_handler:
                self.event_handler.on_error(e, "start")
            return False
    
    def stop(self) -> bool:
        """
        Stop the processing pipeline.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self.state == PipelineState.STOPPED:
            self.logger.warning("Pipeline is already stopped")
            return True
        
        try:
            self._set_state(PipelineState.STOPPING)
            
            # Signal stop to processing thread
            self.is_running = False
            self.stop_event.set()
            
            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
                if self.processing_thread.is_alive():
                    self.logger.warning("Processing thread did not stop gracefully")
            
            # Stop audio capture
            if self.audio_capture:
                try:
                    self.audio_capture.stop_capture()
                except Exception as e:
                    self._handle_component_error('audio_capture', e, {'operation': 'stop'})
            
            # Stop error monitoring
            if hasattr(self, 'error_handler'):
                self.error_handler.stop_health_monitoring()
            
            # Stop diagnostics monitoring
            if hasattr(self, 'diagnostics_manager'):
                self.diagnostics_manager.stop_monitoring()
            
            self._set_state(PipelineState.STOPPED)
            self.logger.info("Pipeline stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop pipeline: {e}")
            self._set_state(PipelineState.ERROR)
            if self.event_handler:
                self.event_handler.on_error(e, "stop")
            return False
    
    def _verify_components(self) -> bool:
        """Verify all components are properly initialized."""
        components = [
            ('config_manager', self.config_manager),
            ('audio_capture', self.audio_capture),
            ('gunshot_detector', self.gunshot_detector),
            ('tdoa_localizer', self.tdoa_localizer),
            ('intensity_filter', self.intensity_filter),
            ('channel_selector', self.channel_selector)
        ]
        
        for name, component in components:
            if component is None:
                self.logger.error(f"Component {name} is not initialized")
                return False
        
        return True
    
    def _set_state(self, new_state: PipelineState) -> None:
        """Set pipeline state and notify event handler."""
        old_state = self.state
        self.state = new_state
        
        if old_state != new_state:
            self.logger.info(f"Pipeline state changed: {old_state.value} -> {new_state.value}")
            if self.event_handler:
                self.event_handler.on_state_change(old_state, new_state)
    
    def _processing_loop(self) -> None:
        """Main processing loop."""
        self.logger.info("Processing loop started")
        
        try:
            while self.is_running and not self.stop_event.is_set():
                try:
                    # Get audio data from capture engine
                    audio_data = self._get_audio_data()
                    if audio_data is None:
                        time.sleep(0.001)  # 1ms sleep to prevent busy waiting
                        continue
                    
                    # Process audio data
                    self._process_audio_data(audio_data)
                    
                    # Update uptime
                    self.metrics.uptime = time.time() - self.metrics.last_reset_time
                    
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}")
                    if self.event_handler:
                        self.event_handler.on_error(e, "processing_loop")
                    
                    # Brief pause before continuing
                    time.sleep(0.01)
        
        except Exception as e:
            self.logger.error(f"Fatal error in processing loop: {e}")
            self._set_state(PipelineState.ERROR)
            if self.event_handler:
                self.event_handler.on_error(e, "processing_loop_fatal")
        
        self.logger.info("Processing loop ended")
    
    def _get_audio_data(self) -> Optional[np.ndarray]:
        """Get audio data from capture engine."""
        try:
            # Get latest audio buffer
            if hasattr(self.audio_capture, 'get_latest_buffer'):
                return self.audio_capture.get_latest_buffer()
            elif hasattr(self.audio_capture, 'read_buffer'):
                return self.audio_capture.read_buffer()
            else:
                # Fallback: assume audio_capture has a method to get data
                return None
        except Exception as e:
            self.logger.debug(f"No audio data available: {e}")
            return None
    
    def _process_audio_data(self, audio_data: np.ndarray) -> None:
        """Process audio data through the complete pipeline."""
        processing_start_time = time.time()
        
        try:
            with self.processing_lock:
                # Step 1: Gunshot Detection
                detection_result = self._detect_gunshot(audio_data)
                if detection_result is None:
                    return  # No gunshot detected
                
                # Step 2: Channel Selection
                channel_selection = self._select_channels(audio_data, detection_result.confidence)
                
                # Step 3: TDoA Calculation and Triangulation
                location_result = self._localize_gunshot(audio_data, channel_selection)
                
                # Step 4: Create complete result
                processing_time = time.time() - processing_start_time
                pipeline_latency = processing_time  # Simplified latency calculation
                
                complete_result = LocalizationResult(
                    detection_event=detection_result,
                    location_result=location_result,
                    channel_selection=channel_selection,
                    total_processing_time=processing_time,
                    pipeline_latency=pipeline_latency,
                    quality_metrics=self._calculate_quality_metrics(
                        detection_result, channel_selection, location_result
                    )
                )
                
                # Step 5: Update metrics and provide feedback
                self._update_metrics(complete_result)
                self._provide_feedback(complete_result)
                
                # Step 6: Output results and logging
                self._output_results(complete_result)
                self._log_results(complete_result)
                
                # Step 7: Store result and notify handler
                self.recent_results.append(complete_result)
                if self.event_handler:
                    self.event_handler.on_localization(complete_result)
                
                self.logger.debug(f"Gunshot localized at ({location_result.x:.2f}, {location_result.y:.2f}) "
                                f"with confidence {location_result.confidence:.3f} in {processing_time*1000:.1f}ms")
        
        except Exception as e:
            self.logger.error(f"Error processing audio data: {e}")
            if self.event_handler:
                self.event_handler.on_error(e, "process_audio_data")
    
    def _detect_gunshot(self, audio_data: np.ndarray) -> Optional[DetectionEvent]:
        """Detect gunshot in audio data with error handling."""
        try:
            detection_start_time = time.time()
            
            # Run gunshot detection
            detection_result = self.gunshot_detector.detect_gunshot(audio_data)
            
            # Extract detection information (returns tuple: is_detected, confidence, metadata)
            is_detected, confidence, metadata = detection_result
            
            # Check if detection meets threshold
            if is_detected and confidence >= self.detection_threshold:
                
                processing_time = time.time() - detection_start_time
                
                event = DetectionEvent(
                    timestamp=time.time(),
                    confidence=confidence,
                    audio_data=audio_data,
                    detection_method=type(self.gunshot_detector).__name__,
                    processing_time=processing_time
                )
                
                # Notify event handler
                if self.event_handler:
                    self.event_handler.on_detection(event)
                
                # Report successful operation
                self.error_handler.report_error(
                    component='gunshot_detector',
                    error=Exception("Success"),  # Dummy success event
                    severity=ErrorSeverity.LOW,
                    category=ErrorCategory.SOFTWARE
                )
                
                return event
            
            return None
            
        except Exception as e:
            # Handle detection error with recovery
            context = {
                'audio_shape': audio_data.shape if audio_data is not None else None,
                'detection_threshold': self.detection_threshold
            }
            
            recovery_successful = self._handle_component_error('gunshot_detector', e, context)
            
            if not recovery_successful:
                self.logger.error(f"Gunshot detection failed: {e}")
            
            return None
    
    def _select_channels(self, audio_data: np.ndarray, detection_confidence: float) -> ChannelSelectionResult:
        """Select optimal channels for localization with error handling."""
        try:
            result = self.channel_selector.select_channels(
                audio_data,
                detection_confidence=detection_confidence
            )
            
            # Report successful operation
            self.error_handler.report_error(
                component='channel_selector',
                error=Exception("Success"),  # Dummy success event
                severity=ErrorSeverity.LOW,
                category=ErrorCategory.SOFTWARE
            )
            
            return result
            
        except Exception as e:
            # Handle channel selection error with recovery
            context = {
                'audio_shape': audio_data.shape if audio_data is not None else None,
                'detection_confidence': detection_confidence,
                'total_channels': audio_data.shape[1] if audio_data is not None else 0
            }
            
            recovery_successful = self._handle_component_error('channel_selector', e, context)
            
            # Always return fallback selection to keep system running
            fallback_channels = min(4, audio_data.shape[1]) if audio_data is not None else 4
            
            self.logger.warning(f"Channel selection failed, using fallback: {list(range(fallback_channels))}")
            
            return ChannelSelectionResult(
                selected_channels=list(range(fallback_channels)),
                excluded_channels=[],
                channel_weights=np.ones(audio_data.shape[1]) if audio_data is not None else np.ones(8),
                selection_confidence=0.3,  # Low confidence for fallback
                strategy_used="fallback",
                fallback_applied=True,
                quality_metrics={'error_recovery': True},
                timestamp=time.time()
            )
    
    def _localize_gunshot(self, audio_data: np.ndarray, channel_selection: ChannelSelectionResult) -> LocationResult:
        """Localize gunshot using TDoA analysis with error handling."""
        try:
            # Use all channels for TDoA calculation (the localizer handles channel selection internally)
            # Calculate TDoA matrix
            tdoa_matrix = self.tdoa_localizer.calculate_tdoa(audio_data)
            
            # Triangulate source location
            location_result = self.tdoa_localizer.triangulate_source(tdoa_matrix)
            
            # Report successful operation if confidence is reasonable
            if hasattr(location_result, 'confidence') and location_result.confidence > 0.1:
                self.error_handler.report_error(
                    component='tdoa_localizer',
                    error=Exception("Success"),  # Dummy success event
                    severity=ErrorSeverity.LOW,
                    category=ErrorCategory.SOFTWARE
                )
            
            return location_result
            
        except Exception as e:
            # Handle localization error with recovery
            context = {
                'audio_shape': audio_data.shape if audio_data is not None else None,
                'selected_channels': channel_selection.selected_channels,
                'selection_confidence': channel_selection.selection_confidence,
                'microphone_count': len(self.config_manager.get_microphone_positions()) if self.config_manager else 8
            }
            
            recovery_successful = self._handle_component_error('tdoa_localizer', e, context)
            
            # Return fallback result to keep system running
            self.logger.warning("Localization failed, returning fallback result")
            
            return LocationResult(
                x=0.0,
                y=0.0,
                z=0.0,
                confidence=0.0,
                residual_error=float('inf'),
                timestamp=time.time(),
                microphones_used=channel_selection.selected_channels,
                tdoa_matrix=None,
                correlation_peaks=None
            )
    
    def _calculate_quality_metrics(self, detection_event: DetectionEvent,
                                 channel_selection: ChannelSelectionResult,
                                 location_result: LocationResult) -> Dict[str, Any]:
        """Calculate quality metrics for the complete result."""
        return {
            'detection_confidence': detection_event.confidence,
            'selection_confidence': channel_selection.selection_confidence,
            'localization_confidence': location_result.confidence,
            'channels_used': len(channel_selection.selected_channels),
            'fallback_applied': channel_selection.fallback_applied,
            'residual_error': location_result.residual_error,
            'processing_time': detection_event.processing_time
        }
    
    def _update_metrics(self, result: LocalizationResult) -> None:
        """Update pipeline performance metrics."""
        self.metrics.total_detections += 1
        
        if result.location_result.confidence > 0.1:  # Successful localization
            self.metrics.successful_localizations += 1
        else:
            self.metrics.failed_localizations += 1
        
        # Update average processing time
        n = self.metrics.total_detections
        current_avg = self.metrics.average_processing_time
        new_avg = (current_avg * (n - 1) + result.total_processing_time) / n
        self.metrics.average_processing_time = new_avg
        
        # Update average latency
        current_latency_avg = self.metrics.average_latency
        new_latency_avg = (current_latency_avg * (n - 1) + result.pipeline_latency) / n
        self.metrics.average_latency = new_latency_avg
        
        # Calculate detection rate (detections per second)
        uptime = time.time() - self.metrics.last_reset_time
        if uptime > 0:
            self.metrics.detection_rate = self.metrics.total_detections / uptime
        
        # Calculate localization accuracy (success rate)
        if self.metrics.total_detections > 0:
            self.metrics.localization_accuracy = (
                self.metrics.successful_localizations / self.metrics.total_detections
            )
        
        # Store in performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'processing_time': result.total_processing_time,
            'latency': result.pipeline_latency,
            'confidence': result.location_result.confidence,
            'success': result.location_result.confidence > 0.1
        })
    
    def _provide_feedback(self, result: LocalizationResult) -> None:
        """Provide feedback to adaptive components."""
        try:
            # Provide feedback to channel selector
            self.channel_selector.update_performance_feedback(
                selected_channels=result.channel_selection.selected_channels,
                triangulation_confidence=result.location_result.confidence,
                triangulation_error=result.location_result.residual_error
            )
        except Exception as e:
            self.logger.debug(f"Failed to provide feedback: {e}")
    
    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        # Update uptime
        self.metrics.uptime = time.time() - self.metrics.last_reset_time
        return self.metrics
    
    def get_recent_results(self, count: int = 10) -> List[LocalizationResult]:
        """Get recent localization results."""
        return list(self.recent_results)[-count:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'pipeline_state': self.state.value,
            'processing_mode': self.processing_mode.value,
            'is_running': self.is_running,
            'metrics': self.get_metrics().__dict__,
            'component_status': {}
        }
        
        # Check component status
        components = [
            ('audio_capture', self.audio_capture),
            ('gunshot_detector', self.gunshot_detector),
            ('tdoa_localizer', self.tdoa_localizer),
            ('intensity_filter', self.intensity_filter),
            ('channel_selector', self.channel_selector)
        ]
        
        for name, component in components:
            if component is not None:
                status['component_status'][name] = 'initialized'
                # Add component-specific status if available
                if hasattr(component, 'get_status'):
                    status['component_status'][name] = component.get_status()
            else:
                status['component_status'][name] = 'not_initialized'
        
        return status
    
    def reset_metrics(self) -> None:
        """Reset pipeline metrics."""
        self.metrics = PipelineMetrics()
        self.metrics.last_reset_time = time.time()
        self.performance_history.clear()
        self.logger.info("Pipeline metrics reset")
    
    def process_audio_file(self, audio_file_path: str) -> List[LocalizationResult]:
        """
        Process an audio file for testing/analysis purposes.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            List of localization results
        """
        # This would be implemented for offline processing
        # For now, return empty list
        self.logger.warning("Audio file processing not yet implemented")
        return []
    
    def shutdown(self) -> None:
        """Shutdown the pipeline and cleanup resources."""
        self.logger.info("Shutting down pipeline...")
        
        # Stop the pipeline
        self.stop()
        
        # Cleanup components
        if self.audio_capture:
            if hasattr(self.audio_capture, 'cleanup'):
                self.audio_capture.cleanup()
        
        # Clear data structures
        self.recent_results.clear()
        self.performance_history.clear()
        
        # Close output and logging systems
        if hasattr(self, 'output_manager'):
            self.output_manager.close()
        
        if hasattr(self, 'structured_logger'):
            self.structured_logger.close()
        
        # Shutdown error handler
        if hasattr(self, 'error_handler'):
            self.error_handler.shutdown()
        
        # Shutdown diagnostics manager
        if hasattr(self, 'diagnostics_manager'):
            self.diagnostics_manager.shutdown()
        
        self.logger.info("Pipeline shutdown complete")
    
    def _output_results(self, result: LocalizationResult) -> None:
        """Output results using the real-time output manager."""
        try:
            # Output detection event
            detection = result.detection_event
            self.output_manager.output_detection(
                confidence=detection.confidence,
                detection_method=detection.detection_method,
                processing_time=detection.processing_time
            )
            
            # Output localization result
            location = result.location_result
            self.output_manager.output_localization(
                x=location.x,
                y=location.y,
                z=location.z,
                confidence=location.confidence,
                error=location.residual_error,
                processing_time=result.total_processing_time,
                channels_used=len(result.channel_selection.selected_channels)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to output results: {e}")
    
    def _log_results(self, result: LocalizationResult) -> None:
        """Log results using the structured logger."""
        try:
            # Log detection event
            detection = result.detection_event
            self.structured_logger.log_detection(
                confidence=detection.confidence,
                detection_method=detection.detection_method,
                processing_time=detection.processing_time,
                channels=list(range(detection.audio_data.shape[1])),
                metadata=result.quality_metrics
            )
            
            # Log localization result
            location = result.location_result
            self.structured_logger.log_localization(
                x=location.x,
                y=location.y,
                z=location.z,
                confidence=location.confidence,
                error=location.residual_error,
                processing_time=result.total_processing_time,
                channels_used=result.channel_selection.selected_channels,
                method="cross_correlation_tdoa",
                metadata=result.quality_metrics
            )
            
            # Log performance metrics periodically
            if self.metrics.total_detections % 10 == 0:  # Every 10 detections
                self._log_performance_metrics()
                
        except Exception as e:
            self.logger.error(f"Failed to log results: {e}")
    
    def _log_performance_metrics(self) -> None:
        """Log performance metrics."""
        try:
            metrics = self.get_metrics()
            
            # Calculate resource usage
            import psutil
            process = psutil.Process()
            resource_usage = {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'threads': process.num_threads()
            }
            
            # Component processing times (estimated)
            component_times = {
                'detection': self.metrics.average_processing_time * 0.3,
                'channel_selection': self.metrics.average_processing_time * 0.2,
                'localization': self.metrics.average_processing_time * 0.5
            }
            
            self.structured_logger.log_performance(
                latency_ms=metrics.average_latency * 1000,
                accuracy=metrics.localization_accuracy,
                throughput=metrics.detection_rate,
                resource_usage=resource_usage,
                component_times={k: v*1000 for k, v in component_times.items()}
            )
            
            # Also output performance to console
            self.output_manager.output_performance(metrics.__dict__)
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metrics: {e}")
    
    def output_system_status(self) -> None:
        """Output current system status."""
        try:
            status = self.get_system_status()
            self.output_manager.output_system_status(status)
            
            # Log component status
            for component, status_info in status['component_status'].items():
                self.structured_logger.log_component_status(
                    component=component,
                    status=status_info,
                    metrics={}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to output system status: {e}")
    
    def set_output_format(self, format_type: OutputFormat, output_file: Optional[str] = None) -> None:
        """Change output format dynamically."""
        try:
            # Close current output manager
            self.output_manager.close()
            
            # Create new output manager
            self.output_manager = RealTimeOutputManager(
                output_format=format_type,
                enable_file_output=output_file is not None,
                output_file=output_file
            )
            
            self.logger.info(f"Output format changed to {format_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to change output format: {e}")
    
    def get_output_statistics(self) -> Dict[str, Any]:
        """Get output and logging statistics."""
        try:
            output_stats = self.output_manager.get_output_stats()
            log_stats = self.structured_logger.get_log_statistics()
            
            return {
                'output_statistics': output_stats,
                'logging_statistics': log_stats,
                'pipeline_metrics': self.get_metrics().__dict__
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get output statistics: {e}")
            return {}
    
    def _register_components_for_monitoring(self) -> None:
        """Register all components for error monitoring."""
        components = [
            'config_manager',
            'audio_capture', 
            'gunshot_detector',
            'tdoa_localizer',
            'intensity_filter',
            'channel_selector',
            'output_manager',
            'structured_logger'
        ]
        
        for component in components:
            self.error_handler.register_component(component)
    
    def _handle_component_error(self, component: str, error: Exception, 
                              context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle component error with recovery."""
        try:
            # Report error to error handling system
            recovery_successful = handle_component_error(
                self.error_handler, component, error, context
            )
            
            # Log error to structured logger
            self.structured_logger.log_error(
                error=str(error),
                context=component,
                component=component,
                exception=error,
                additional_data=context
            )
            
            # Output error if severe
            if not recovery_successful:
                self.output_manager.output_error(str(error), component)
            
            return recovery_successful
            
        except Exception as e:
            self.logger.error(f"Error in error handling: {e}")
            return False
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        try:
            # Get error handler health
            error_health = self.error_handler.get_system_health()
            error_stats = self.error_handler.get_error_statistics()
            
            # Get pipeline metrics
            pipeline_metrics = self.get_metrics()
            
            # Get component status
            component_status = self.get_system_status()
            
            # Calculate overall system health
            overall_health = {
                'pipeline_health': {
                    'state': self.state.value,
                    'is_running': self.is_running,
                    'uptime': pipeline_metrics.uptime,
                    'success_rate': pipeline_metrics.localization_accuracy,
                    'average_latency': pipeline_metrics.average_latency
                },
                'error_health': error_health,
                'error_statistics': error_stats,
                'component_status': component_status['component_status'],
                'recommendations': self._generate_health_recommendations(error_health, error_stats)
            }
            
            return overall_health
            
        except Exception as e:
            self.logger.error(f"Failed to get system health status: {e}")
            return {'error': str(e)}
    
    def _generate_health_recommendations(self, error_health: Dict[str, Any], 
                                       error_stats: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on system status."""
        recommendations = []
        
        # Check overall health score
        if error_health['overall_health_score'] < 0.7:
            recommendations.append("System health is degraded - consider restarting problematic components")
        
        # Check error rates
        if error_stats['recent_errors_1h'] > 20:
            recommendations.append("High error rate detected - investigate root causes")
        
        # Check for critical errors
        if error_health['critical_errors'] > 0:
            recommendations.append("Critical errors detected - system restart may be required")
        
        # Check component health
        for component, health in error_health['component_health'].items():
            if not health['is_healthy']:
                recommendations.append(f"Component '{component}' is unhealthy - check logs and restart if needed")
            elif health['health_score'] < 0.5:
                recommendations.append(f"Component '{component}' has low health score - monitor closely")
        
        # Check for degraded mode
        if error_health['system_degraded']:
            recommendations.append("System is in degraded mode - some functionality may be limited")
        
        if not recommendations:
            recommendations.append("System health is good - no immediate action required")
        
        return recommendations
    
    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics."""
        try:
            if hasattr(self, 'diagnostics_manager'):
                diagnostics = self.diagnostics_manager.get_system_diagnostics()
                return {
                    'diagnostics': diagnostics,
                    'microphone_report': self.diagnostics_manager.get_microphone_connectivity_report(),
                    'performance_summary': self.diagnostics_manager.get_performance_summary(hours=1.0)
                }
            else:
                return {'error': 'Diagnostics manager not available'}
        except Exception as e:
            return {'error': f'Failed to get diagnostics: {e}'}
    
    def export_diagnostics_report(self, format_type: str = 'json') -> str:
        """Export diagnostics report in specified format."""
        try:
            if hasattr(self, 'diagnostics_manager'):
                return self.diagnostics_manager.export_diagnostics_report(format_type)
            else:
                return json.dumps({'error': 'Diagnostics manager not available'})
        except Exception as e:
            return json.dumps({'error': f'Failed to export diagnostics: {e}'})


class DefaultEventHandler(PipelineEventHandler):
    """Default event handler that logs events."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DefaultEventHandler")
    
    def on_detection(self, event: DetectionEvent) -> None:
        """Handle gunshot detection event."""
        self.logger.info(f"Gunshot detected with confidence {event.confidence:.3f}")
    
    def on_localization(self, result: LocalizationResult) -> None:
        """Handle successful localization result."""
        loc = result.location_result
        self.logger.info(f"Gunshot localized at ({loc.x:.2f}, {loc.y:.2f}) "
                        f"with confidence {loc.confidence:.3f}")
    
    def on_error(self, error: Exception, context: str) -> None:
        """Handle pipeline errors."""
        self.logger.error(f"Pipeline error in {context}: {error}")
    
    def on_state_change(self, old_state: PipelineState, new_state: PipelineState) -> None:
        """Handle pipeline state changes."""
        self.logger.info(f"Pipeline state: {old_state.value} -> {new_state.value}")


# Convenience function for creating and running the pipeline
def create_pipeline(config_path: str, event_handler: Optional[PipelineEventHandler] = None) -> GunshotLocalizationPipeline:
    """
    Create a gunshot localization pipeline.
    
    Args:
        config_path: Path to configuration file
        event_handler: Optional event handler
        
    Returns:
        Configured pipeline instance
    """
    if event_handler is None:
        event_handler = DefaultEventHandler()
    
    pipeline = GunshotLocalizationPipeline(config_path, event_handler)
    
    return pipeline