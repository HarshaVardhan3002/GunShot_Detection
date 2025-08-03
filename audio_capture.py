"""
Audio capture module for multi-channel synchronized recording.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import threading
import time
import logging
try:
    import sounddevice as sd
except ImportError:
    sd = None
from collections import deque


@dataclass
class AudioBuffer:
    """Container for audio data with metadata."""
    data: np.ndarray  # Shape: (samples, channels)
    timestamp: float
    sample_rate: int
    duration: float


@dataclass
class SynchronizationMetrics:
    """Container for synchronization quality metrics."""
    clock_drift_ms: float
    phase_coherence: float
    channel_alignment: Dict[int, float]
    sync_quality_score: float
    reference_channel: int
    last_sync_time: float


class AudioStreamSynchronizer:
    """Advanced audio stream synchronization manager."""
    
    def __init__(self, channels: int, sample_rate: int):
        """
        Initialize synchronization manager.
        
        Args:
            channels: Number of audio channels
            sample_rate: Audio sampling rate
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
        
        # Synchronization state
        self._reference_channel = 0  # Channel used as timing reference
        self._channel_delays = np.zeros(channels)  # Per-channel delay compensation
        self._sync_lock = threading.Lock()
        
        # Cross-correlation analysis for sync detection
        self._correlation_window = 1024  # Samples for correlation analysis
        self._correlation_history = deque(maxlen=100)
        
        # Phase alignment tracking
        self._phase_history = {i: deque(maxlen=50) for i in range(channels)}
        self._coherence_threshold = 0.8  # Minimum coherence for good sync
        
        # Timing analysis
        self._sample_timestamps = deque(maxlen=1000)
        self._clock_reference = None
        self._drift_accumulator = 0.0
        
        # Synchronization quality metrics
        self._sync_quality = 1.0
        self._last_sync_check = time.time()
        self._sync_failures = 0
        
        self.logger.info(f"AudioStreamSynchronizer initialized for {channels} channels at {sample_rate}Hz")
    
    def analyze_channel_synchronization(self, audio_data: np.ndarray, timestamp: float) -> SynchronizationMetrics:
        """
        Analyze synchronization quality across channels.
        
        Args:
            audio_data: Multi-channel audio data (samples, channels)
            timestamp: Current timestamp
            
        Returns:
            SynchronizationMetrics with analysis results
        """
        if audio_data.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {audio_data.shape[1]}")
        
        with self._sync_lock:
            # Update timing reference
            self._update_timing_reference(timestamp)
            
            # Calculate cross-channel correlations
            channel_alignment = self._calculate_channel_alignment(audio_data)
            
            # Analyze phase coherence
            phase_coherence = self._calculate_phase_coherence(audio_data)
            
            # Calculate clock drift
            clock_drift = self._calculate_clock_drift(timestamp)
            
            # Compute overall sync quality score
            sync_quality = self._compute_sync_quality_score(
                channel_alignment, phase_coherence, abs(clock_drift)
            )
            
            # Update internal state
            self._sync_quality = sync_quality
            self._last_sync_check = timestamp
            
            return SynchronizationMetrics(
                clock_drift_ms=clock_drift * 1000,
                phase_coherence=phase_coherence,
                channel_alignment=channel_alignment,
                sync_quality_score=sync_quality,
                reference_channel=self._reference_channel,
                last_sync_time=timestamp
            )
    
    def _update_timing_reference(self, timestamp: float) -> None:
        """Update timing reference for drift calculation."""
        if self._clock_reference is None:
            self._clock_reference = timestamp
        
        self._sample_timestamps.append(timestamp)
    
    def _calculate_channel_alignment(self, audio_data: np.ndarray) -> Dict[int, float]:
        """
        Calculate alignment between channels using cross-correlation.
        
        Args:
            audio_data: Multi-channel audio data
            
        Returns:
            Dictionary of channel alignment scores
        """
        alignment = {}
        reference_channel = audio_data[:, self._reference_channel]
        
        for ch in range(self.channels):
            if ch == self._reference_channel:
                alignment[ch] = 1.0
                continue
            
            channel_data = audio_data[:, ch]
            
            # Calculate cross-correlation
            correlation = np.correlate(reference_channel, channel_data, mode='full')
            max_corr_idx = np.argmax(np.abs(correlation))
            max_correlation = correlation[max_corr_idx]
            
            # Normalize correlation
            ref_energy = np.sum(reference_channel ** 2)
            ch_energy = np.sum(channel_data ** 2)
            
            if ref_energy > 0 and ch_energy > 0:
                normalized_corr = abs(max_correlation) / np.sqrt(ref_energy * ch_energy)
                alignment[ch] = min(normalized_corr, 1.0)
            else:
                alignment[ch] = 0.0
            
            # Calculate delay offset
            delay_samples = max_corr_idx - len(channel_data) + 1
            delay_ms = (delay_samples / self.sample_rate) * 1000
            
            # Store delay for compensation
            self._channel_delays[ch] = delay_ms
        
        return alignment
    
    def _calculate_phase_coherence(self, audio_data: np.ndarray) -> float:
        """
        Calculate phase coherence across channels using FFT analysis.
        
        Args:
            audio_data: Multi-channel audio data
            
        Returns:
            Phase coherence score (0-1)
        """
        if audio_data.shape[0] < 64:  # Need minimum samples for FFT
            return self._sync_quality  # Return previous quality
        
        # Apply window to reduce spectral leakage
        window = np.hanning(audio_data.shape[0])
        windowed_data = audio_data * window[:, np.newaxis]
        
        # Calculate FFT for each channel
        ffts = np.fft.fft(windowed_data, axis=0)
        
        # Focus on frequency range relevant for gunshots (100Hz - 8kHz)
        freq_bins = np.fft.fftfreq(audio_data.shape[0], 1/self.sample_rate)
        freq_mask = (np.abs(freq_bins) >= 100) & (np.abs(freq_bins) <= 8000)
        
        if not np.any(freq_mask):
            return self._sync_quality
        
        # Calculate phase differences relative to reference channel
        reference_phase = np.angle(ffts[freq_mask, self._reference_channel])
        coherence_scores = []
        
        for ch in range(self.channels):
            if ch == self._reference_channel:
                continue
            
            channel_phase = np.angle(ffts[freq_mask, ch])
            phase_diff = np.abs(reference_phase - channel_phase)
            
            # Wrap phase differences to [-π, π]
            phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
            
            # Calculate coherence (lower phase difference = higher coherence)
            coherence = np.mean(np.cos(phase_diff))
            coherence_scores.append(max(coherence, 0))  # Ensure non-negative
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _calculate_clock_drift(self, current_timestamp: float) -> float:
        """
        Calculate clock drift based on timestamp analysis.
        
        Args:
            current_timestamp: Current timestamp
            
        Returns:
            Clock drift in seconds
        """
        if len(self._sample_timestamps) < 10:
            return 0.0
        
        # Calculate expected vs actual timing
        timestamps = list(self._sample_timestamps)
        time_diffs = np.diff(timestamps)
        
        # Expected time difference based on sample rate and block size
        expected_diff = self._correlation_window / self.sample_rate
        
        # Calculate drift as deviation from expected timing
        actual_avg_diff = np.mean(time_diffs)
        drift = actual_avg_diff - expected_diff
        
        # Apply exponential smoothing
        self._drift_accumulator = 0.9 * self._drift_accumulator + 0.1 * drift
        
        return self._drift_accumulator
    
    def _compute_sync_quality_score(self, alignment: Dict[int, float], 
                                   coherence: float, drift: float) -> float:
        """
        Compute overall synchronization quality score.
        
        Args:
            alignment: Channel alignment scores
            coherence: Phase coherence score
            drift: Clock drift magnitude
            
        Returns:
            Overall sync quality score (0-1)
        """
        # Weight factors for different aspects
        alignment_weight = 0.4
        coherence_weight = 0.4
        timing_weight = 0.2
        
        # Calculate alignment score
        alignment_scores = [score for ch, score in alignment.items() if ch != self._reference_channel]
        avg_alignment = np.mean(alignment_scores) if alignment_scores else 1.0
        
        # Calculate timing score (penalize high drift)
        timing_score = max(0, 1.0 - abs(drift) * 1000)  # Convert to ms for scoring
        
        # Combine scores
        quality_score = (
            alignment_weight * avg_alignment +
            coherence_weight * coherence +
            timing_weight * timing_score
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def compensate_channel_delays(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply delay compensation to align channels.
        
        Args:
            audio_data: Multi-channel audio data
            
        Returns:
            Delay-compensated audio data
        """
        if np.all(self._channel_delays == 0):
            return audio_data  # No compensation needed
        
        compensated_data = audio_data.copy()
        
        for ch in range(self.channels):
            delay_samples = int(self._channel_delays[ch] * self.sample_rate / 1000)
            
            if delay_samples > 0:
                # Positive delay - shift channel forward (add zeros at beginning)
                compensated_data[delay_samples:, ch] = audio_data[:-delay_samples, ch]
                compensated_data[:delay_samples, ch] = 0
            elif delay_samples < 0:
                # Negative delay - shift channel backward (add zeros at end)
                delay_samples = abs(delay_samples)
                compensated_data[:-delay_samples, ch] = audio_data[delay_samples:, ch]
                compensated_data[-delay_samples:, ch] = 0
        
        return compensated_data
    
    def set_reference_channel(self, channel: int) -> bool:
        """
        Set the reference channel for synchronization.
        
        Args:
            channel: Channel index to use as reference (0-based)
            
        Returns:
            True if successful
        """
        if 0 <= channel < self.channels:
            with self._sync_lock:
                self._reference_channel = channel
                # Reset delay compensation when reference changes
                self._channel_delays = np.zeros(self.channels)
                self.logger.info(f"Reference channel set to {channel}")
            return True
        else:
            self.logger.error(f"Invalid reference channel: {channel}")
            return False
    
    def get_synchronization_diagnostics(self) -> Dict[str, any]:
        """
        Get detailed synchronization diagnostics.
        
        Returns:
            Dictionary with diagnostic information
        """
        with self._sync_lock:
            return {
                'reference_channel': self._reference_channel,
                'channel_delays_ms': self._channel_delays.tolist(),
                'sync_quality': self._sync_quality,
                'drift_accumulator': self._drift_accumulator,
                'sync_failures': self._sync_failures,
                'last_sync_check': self._last_sync_check,
                'correlation_history_length': len(self._correlation_history),
                'coherence_threshold': self._coherence_threshold
            }
    
    def reset_synchronization_state(self) -> None:
        """Reset all synchronization state."""
        with self._sync_lock:
            self._channel_delays = np.zeros(self.channels)
            self._correlation_history.clear()
            for ch_history in self._phase_history.values():
                ch_history.clear()
            self._sample_timestamps.clear()
            self._clock_reference = None
            self._drift_accumulator = 0.0
            self._sync_quality = 1.0
            self._sync_failures = 0
            
        self.logger.info("Synchronization state reset")


class AudioCaptureInterface(ABC):
    """Abstract interface for audio capture engines."""
    
    @abstractmethod
    def start_capture(self) -> None:
        """Start audio capture."""
        pass
    
    @abstractmethod
    def stop_capture(self) -> None:
        """Stop audio capture."""
        pass
    
    @abstractmethod
    def get_audio_buffer(self) -> Optional[AudioBuffer]:
        """Get current audio buffer."""
        pass
    
    @abstractmethod
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        pass
    
    @abstractmethod
    def get_channel_status(self) -> Dict[int, bool]:
        """Get status of each channel."""
        pass


class AudioCaptureEngine(AudioCaptureInterface):
    """Concrete implementation of multi-channel audio capture."""
    
    def __init__(self, sample_rate: int = 48000, channels: int = 8, 
                 buffer_duration: float = 2.0, device_id: Optional[int] = None):
        """
        Initialize audio capture engine.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            channels: Number of audio channels to capture
            buffer_duration: Duration of circular buffer in seconds
            device_id: Optional specific audio device ID
        """
        if sd is None:
            raise ImportError("sounddevice library is required for audio capture. Install with: pip install sounddevice")
        
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_duration = buffer_duration
        self.device_id = device_id
        
        # Calculate buffer size
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # Circular buffer for audio data
        self._audio_buffer = np.zeros((self.buffer_size, channels), dtype=np.float32)
        self._buffer_index = 0
        self._buffer_lock = threading.RLock()  # Reentrant lock for nested operations
        
        # Multiple buffer system for better synchronization
        self._buffer_segments = 4  # Number of buffer segments
        self._segment_size = self.buffer_size // self._buffer_segments
        self._current_segment = 0
        self._segment_timestamps = [0.0] * self._buffer_segments
        
        # Capture state
        self._capturing = False
        self._capture_thread = None
        self._stream = None
        self._stop_event = threading.Event()
        
        # Channel status tracking with history
        self._channel_status = {i: True for i in range(1, channels + 1)}
        self._channel_history = {i: deque(maxlen=100) for i in range(1, channels + 1)}
        self._last_audio_time = time.time()
        
        # Advanced timing and synchronization
        self._timing_lock = threading.Lock()
        self._frame_timestamps = deque(maxlen=1000)  # Store recent frame timestamps
        self._drift_compensation = 0.0
        self._sync_reference_time = None
        
        # Audio stream synchronization
        self._synchronizer = AudioStreamSynchronizer(channels, sample_rate)
        self._sync_enabled = True
        self._sync_check_interval = 0.1  # Check sync every 100ms
        self._last_sync_analysis = 0.0
        self._sync_metrics_history = deque(maxlen=100)
        
        # Statistics and monitoring
        self._samples_captured = 0
        self._buffer_overruns = 0
        self._sync_errors = 0
        self._callback_latency = deque(maxlen=100)
        
        # Buffer overflow protection
        self._max_buffer_size = int(sample_rate * buffer_duration * 2)  # 2x safety margin
        self._buffer_resize_threshold = 0.9  # Resize when 90% full
        
        self.logger.info(f"AudioCaptureEngine initialized: {sample_rate}Hz, {channels}ch, {buffer_duration}s buffer")
    
    def start_capture(self) -> None:
        """Start audio capture."""
        if self._capturing:
            self.logger.warning("Audio capture already running")
            return
        
        try:
            # List available devices for debugging
            devices = sd.query_devices()
            self.logger.debug(f"Available audio devices: {len(devices)}")
            
            # Configure audio stream
            stream_params = {
                'samplerate': self.sample_rate,
                'channels': self.channels,
                'dtype': np.float32,
                'callback': self._audio_callback,
                'blocksize': 1024,  # Small block size for low latency
                'latency': 'low'
            }
            
            if self.device_id is not None:
                stream_params['device'] = self.device_id
            
            # Start audio stream
            self._stream = sd.InputStream(**stream_params)
            self._stream.start()
            
            self._capturing = True
            self._last_audio_time = time.time()
            
            self.logger.info("Audio capture started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            self._capturing = False
            raise
    
    def stop_capture(self) -> None:
        """Stop audio capture."""
        if not self._capturing:
            self.logger.warning("Audio capture not running")
            return
        
        try:
            self._capturing = False
            
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            
            self.logger.info("Audio capture stopped")
            self.logger.info(f"Capture statistics: {self._samples_captured} samples, {self._buffer_overruns} overruns")
            
        except Exception as e:
            self.logger.error(f"Error stopping audio capture: {e}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """
        Enhanced audio stream callback with synchronization and drift compensation.
        
        Args:
            indata: Input audio data (frames, channels)
            frames: Number of frames
            time_info: Timing information
            status: Stream status
        """
        callback_start_time = time.time()
        
        if status:
            self.logger.warning(f"Audio stream status: {status}")
            if status.input_overflow:
                self._buffer_overruns += 1
        
        if not self._capturing:
            return
        
        try:
            # Process timing information for synchronization
            current_time = callback_start_time
            if time_info and hasattr(time_info, 'inputBufferAdcTime'):
                # Use hardware timestamp if available
                hardware_time = time_info.inputBufferAdcTime
                if self._sync_reference_time is None:
                    self._sync_reference_time = current_time - hardware_time
                
                # Calculate drift compensation
                expected_time = self._sync_reference_time + hardware_time
                drift = current_time - expected_time
                self._drift_compensation = 0.95 * self._drift_compensation + 0.05 * drift
            
            with self._buffer_lock:
                # Enhanced buffer management with overflow protection
                if self._buffer_index + frames > self.buffer_size:
                    # Check if we need to resize buffer
                    if self._samples_captured > self._max_buffer_size * self._buffer_resize_threshold:
                        self._handle_buffer_overflow()
                    
                    # Wrap around circular buffer
                    first_part = self.buffer_size - self._buffer_index
                    second_part = frames - first_part
                    
                    self._audio_buffer[self._buffer_index:] = indata[:first_part]
                    if second_part > 0:
                        self._audio_buffer[:second_part] = indata[first_part:]
                        self._buffer_index = second_part
                        self._buffer_overruns += 1
                    else:
                        self._buffer_index = 0
                else:
                    # Normal buffer write
                    self._audio_buffer[self._buffer_index:self._buffer_index + frames] = indata
                    self._buffer_index += frames
                
                # Update segment tracking
                current_segment = self._buffer_index // self._segment_size
                if current_segment != self._current_segment:
                    self._segment_timestamps[current_segment] = current_time
                    self._current_segment = current_segment
                
                self._samples_captured += frames
                self._last_audio_time = current_time
                
                # Store frame timestamp for synchronization analysis
                with self._timing_lock:
                    self._frame_timestamps.append({
                        'timestamp': current_time,
                        'frames': frames,
                        'buffer_index': self._buffer_index,
                        'drift': self._drift_compensation
                    })
                
                # Apply synchronization compensation if enabled
                processed_data = indata
                if self._sync_enabled and frames >= 64:  # Need minimum samples for sync analysis
                    processed_data = self._apply_synchronization(indata, current_time)
                
                # Update channel status with history
                self._update_channel_status_with_history(processed_data, current_time)
                
                # Monitor callback latency
                callback_latency = time.time() - callback_start_time
                self._callback_latency.append(callback_latency)
                
                # Detect synchronization issues
                if callback_latency > 0.01:  # 10ms threshold
                    self._sync_errors += 1
                    if self._sync_errors % 100 == 0:
                        self.logger.warning(f"High callback latency detected: {callback_latency*1000:.1f}ms")
                
        except Exception as e:
            self.logger.error(f"Error in audio callback: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _update_channel_status_with_history(self, audio_data: np.ndarray, timestamp: float) -> None:
        """
        Update channel status with historical tracking.
        
        Args:
            audio_data: Recent audio data (frames, channels)
            timestamp: Current timestamp
        """
        # Calculate multiple signal metrics for each channel
        rms_values = np.sqrt(np.mean(audio_data**2, axis=0))
        peak_values = np.max(np.abs(audio_data), axis=0)
        
        # Adaptive threshold based on recent history
        base_threshold = 1e-6
        
        for i in range(self.channels):
            channel_id = i + 1
            
            # Calculate current signal metrics
            rms = rms_values[i]
            peak = peak_values[i]
            
            # Store in history
            self._channel_history[channel_id].append({
                'timestamp': timestamp,
                'rms': rms,
                'peak': peak,
                'active': rms > base_threshold
            })
            
            # Determine channel status with hysteresis
            if len(self._channel_history[channel_id]) > 10:
                recent_activity = [h['active'] for h in list(self._channel_history[channel_id])[-10:]]
                activity_ratio = sum(recent_activity) / len(recent_activity)
                
                # Use hysteresis to prevent flickering
                if self._channel_status[channel_id]:
                    # Currently active - require low activity to deactivate
                    self._channel_status[channel_id] = activity_ratio > 0.2
                else:
                    # Currently inactive - require high activity to activate
                    self._channel_status[channel_id] = activity_ratio > 0.8
            else:
                # Not enough history - use simple threshold
                self._channel_status[channel_id] = rms > base_threshold
    
    def _handle_buffer_overflow(self) -> None:
        """Handle buffer overflow by adjusting buffer management."""
        self.logger.warning("Buffer overflow detected - implementing overflow protection")
        
        # Reset buffer index to prevent memory issues
        self._buffer_index = 0
        self._buffer_overruns += 1
        
        # Clear old data from the beginning of buffer
        self._audio_buffer[:self._segment_size] = 0
        
        # Reset segment tracking
        self._current_segment = 0
        self._segment_timestamps = [time.time()] * self._buffer_segments
    
    def _apply_synchronization(self, audio_data: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Apply synchronization analysis and compensation.
        
        Args:
            audio_data: Input audio data
            timestamp: Current timestamp
            
        Returns:
            Synchronized audio data
        """
        try:
            # Perform synchronization analysis periodically
            if timestamp - self._last_sync_analysis >= self._sync_check_interval:
                sync_metrics = self._synchronizer.analyze_channel_synchronization(
                    audio_data, timestamp
                )
                
                # Store metrics for history
                self._sync_metrics_history.append(sync_metrics)
                self._last_sync_analysis = timestamp
                
                # Log sync issues if quality is poor
                if sync_metrics.sync_quality_score < 0.7:
                    self.logger.warning(
                        f"Poor synchronization quality: {sync_metrics.sync_quality_score:.2f}, "
                        f"drift: {sync_metrics.clock_drift_ms:.2f}ms"
                    )
            
            # Apply delay compensation
            compensated_data = self._synchronizer.compensate_channel_delays(audio_data)
            
            return compensated_data
            
        except Exception as e:
            self.logger.error(f"Error in synchronization processing: {e}")
            return audio_data  # Return original data on error
    
    def get_audio_buffer(self) -> Optional[AudioBuffer]:
        """
        Get current audio buffer with enhanced synchronization.
        
        Returns:
            AudioBuffer with current data or None if not capturing
        """
        if not self._capturing:
            return None
        
        try:
            with self._buffer_lock:
                # Create a copy of the current buffer
                buffer_copy = self._audio_buffer.copy()
                timestamp = self._last_audio_time - self._drift_compensation
                buffer_index = self._buffer_index
            
            return AudioBuffer(
                data=buffer_copy,
                timestamp=timestamp,
                sample_rate=self.sample_rate,
                duration=self.buffer_duration
            )
            
        except Exception as e:
            self.logger.error(f"Error getting audio buffer: {e}")
            return None
    
    def get_synchronized_buffer_segment(self, segment_duration: float = 0.5) -> Optional[AudioBuffer]:
        """
        Get a synchronized buffer segment for analysis.
        
        Args:
            segment_duration: Duration of segment to extract in seconds
            
        Returns:
            AudioBuffer with synchronized segment data
        """
        if not self._capturing:
            return None
        
        try:
            segment_samples = int(self.sample_rate * segment_duration)
            if segment_samples > self.buffer_size:
                segment_samples = self.buffer_size
            
            with self._buffer_lock:
                # Calculate start position for most recent complete segment
                start_index = (self._buffer_index - segment_samples) % self.buffer_size
                
                # Extract segment data
                if start_index + segment_samples <= self.buffer_size:
                    # Segment doesn't wrap around
                    segment_data = self._audio_buffer[start_index:start_index + segment_samples].copy()
                else:
                    # Segment wraps around buffer
                    first_part = self.buffer_size - start_index
                    second_part = segment_samples - first_part
                    segment_data = np.vstack([
                        self._audio_buffer[start_index:],
                        self._audio_buffer[:second_part]
                    ])
                
                # Calculate precise timestamp for this segment
                segment_timestamp = self._last_audio_time - (segment_samples / self.sample_rate)
                segment_timestamp -= self._drift_compensation
            
            return AudioBuffer(
                data=segment_data,
                timestamp=segment_timestamp,
                sample_rate=self.sample_rate,
                duration=segment_duration
            )
            
        except Exception as e:
            self.logger.error(f"Error getting synchronized buffer segment: {e}")
            return None
    
    def freeze_buffer_on_trigger(self) -> Optional[AudioBuffer]:
        """
        Freeze current buffer state for event analysis.
        
        Returns:
            Frozen AudioBuffer for analysis
        """
        if not self._capturing:
            return None
        
        try:
            with self._buffer_lock:
                # Create immutable snapshot of current state
                frozen_buffer = self._audio_buffer.copy()
                frozen_timestamp = self._last_audio_time - self._drift_compensation
                frozen_index = self._buffer_index
                
                # Mark buffer as frozen for analysis
                self.logger.debug(f"Buffer frozen at index {frozen_index}, timestamp {frozen_timestamp}")
            
            return AudioBuffer(
                data=frozen_buffer,
                timestamp=frozen_timestamp,
                sample_rate=self.sample_rate,
                duration=self.buffer_duration
            )
            
        except Exception as e:
            self.logger.error(f"Error freezing buffer: {e}")
            return None
    
    def is_capturing(self) -> bool:
        """Check if currently capturing."""
        return self._capturing
    
    def get_channel_status(self) -> Dict[int, bool]:
        """Get status of each channel."""
        return self._channel_status.copy()
    
    def get_capture_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive capture statistics.
        
        Returns:
            Dictionary with detailed capture statistics
        """
        with self._timing_lock:
            avg_latency = np.mean(self._callback_latency) if self._callback_latency else 0
            max_latency = np.max(self._callback_latency) if self._callback_latency else 0
            
        return {
            'samples_captured': self._samples_captured,
            'buffer_overruns': self._buffer_overruns,
            'sync_errors': self._sync_errors,
            'buffer_size': self.buffer_size,
            'channels': self.channels,
            'sample_rate': self.sample_rate,
            'buffer_utilization': (self._buffer_index / self.buffer_size) * 100,
            'drift_compensation': self._drift_compensation * 1000,  # in milliseconds
            'avg_callback_latency_ms': avg_latency * 1000,
            'max_callback_latency_ms': max_latency * 1000,
            'current_segment': self._current_segment,
            'active_channels': sum(1 for status in self._channel_status.values() if status)
        }
    
    def get_synchronization_status(self) -> Dict[str, any]:
        """
        Get detailed synchronization status.
        
        Returns:
            Dictionary with synchronization information
        """
        with self._timing_lock:
            recent_frames = list(self._frame_timestamps)[-10:] if self._frame_timestamps else []
            
        if not recent_frames:
            return {'status': 'no_data'}
        
        # Calculate timing statistics
        frame_intervals = []
        for i in range(1, len(recent_frames)):
            interval = recent_frames[i]['timestamp'] - recent_frames[i-1]['timestamp']
            frame_intervals.append(interval)
        
        expected_interval = 1024 / self.sample_rate  # Based on blocksize
        
        return {
            'status': 'synchronized' if abs(self._drift_compensation) < 0.001 else 'drift_detected',
            'drift_compensation_ms': self._drift_compensation * 1000,
            'sync_reference_set': self._sync_reference_time is not None,
            'recent_frame_count': len(recent_frames),
            'avg_frame_interval': np.mean(frame_intervals) if frame_intervals else 0,
            'expected_frame_interval': expected_interval,
            'timing_jitter': np.std(frame_intervals) if frame_intervals else 0,
            'sync_errors': self._sync_errors
        }
    
    def get_channel_health(self) -> Dict[int, Dict[str, any]]:
        """
        Get detailed health information for each channel.
        
        Returns:
            Dictionary with per-channel health metrics
        """
        channel_health = {}
        
        for channel_id in range(1, self.channels + 1):
            history = list(self._channel_history[channel_id])
            
            if not history:
                channel_health[channel_id] = {
                    'status': 'no_data',
                    'active': self._channel_status[channel_id]
                }
                continue
            
            # Calculate health metrics from history
            recent_rms = [h['rms'] for h in history[-50:]]  # Last 50 samples
            recent_peaks = [h['peak'] for h in history[-50:]]
            activity_history = [h['active'] for h in history[-100:]]  # Last 100 samples
            
            channel_health[channel_id] = {
                'status': 'healthy' if self._channel_status[channel_id] else 'inactive',
                'active': self._channel_status[channel_id],
                'avg_rms': np.mean(recent_rms) if recent_rms else 0,
                'max_peak': np.max(recent_peaks) if recent_peaks else 0,
                'activity_ratio': np.mean(activity_history) if activity_history else 0,
                'signal_stability': 1.0 - np.std(recent_rms) if len(recent_rms) > 1 else 1.0,
                'samples_in_history': len(history)
            }
        
        return channel_health
    
    def list_audio_devices(self) -> List[Dict]:
        """
        List available audio devices.
        
        Returns:
            List of audio device information
        """
        try:
            devices = sd.query_devices()
            device_list = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] >= self.channels:
                    device_list.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            return device_list
            
        except Exception as e:
            self.logger.error(f"Error listing audio devices: {e}")
            return []
    
    def test_device_compatibility(self, device_id: int) -> bool:
        """
        Test if a device is compatible with our requirements.
        
        Args:
            device_id: Device ID to test
            
        Returns:
            True if device is compatible
        """
        try:
            # Try to create a test stream
            test_stream = sd.InputStream(
                device=device_id,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=1024
            )
            
            test_stream.close()
            return True
            
        except Exception as e:
            self.logger.debug(f"Device {device_id} not compatible: {e}")
            return False
    
    def adjust_buffer_size(self, new_duration: float) -> bool:
        """
        Dynamically adjust buffer size during operation.
        
        Args:
            new_duration: New buffer duration in seconds
            
        Returns:
            True if adjustment successful
        """
        if new_duration <= 0 or new_duration > 10:  # Reasonable limits
            self.logger.error(f"Invalid buffer duration: {new_duration}")
            return False
        
        try:
            new_buffer_size = int(self.sample_rate * new_duration)
            
            with self._buffer_lock:
                # Save current data if capturing
                old_data = None
                if self._capturing and self._samples_captured > 0:
                    old_data = self._audio_buffer[:self._buffer_index].copy()
                
                # Create new buffer
                self._audio_buffer = np.zeros((new_buffer_size, self.channels), dtype=np.float32)
                self.buffer_size = new_buffer_size
                self.buffer_duration = new_duration
                
                # Restore data if possible
                if old_data is not None:
                    copy_size = min(old_data.shape[0], new_buffer_size)
                    self._audio_buffer[:copy_size] = old_data[:copy_size]
                    self._buffer_index = copy_size
                else:
                    self._buffer_index = 0
                
                # Update segment tracking
                self._segment_size = new_buffer_size // self._buffer_segments
                self._current_segment = 0
                self._segment_timestamps = [time.time()] * self._buffer_segments
            
            self.logger.info(f"Buffer size adjusted to {new_duration}s ({new_buffer_size} samples)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adjusting buffer size: {e}")
            return False
    
    def reset_synchronization(self) -> None:
        """Reset synchronization state and drift compensation."""
        with self._timing_lock:
            self._drift_compensation = 0.0
            self._sync_reference_time = None
            self._frame_timestamps.clear()
            self._sync_errors = 0
            
        self.logger.info("Synchronization state reset")
    
    def perform_health_check(self) -> Dict[str, any]:
        """
        Perform comprehensive health check of the audio capture system.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            'overall_status': 'healthy',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check capture status
        if not self._capturing:
            health_status['issues'].append('Audio capture not running')
            health_status['overall_status'] = 'error'
        
        # Check buffer overruns
        if self._buffer_overruns > 0:
            overrun_rate = self._buffer_overruns / max(1, self._samples_captured / self.sample_rate)
            if overrun_rate > 0.1:  # More than 0.1 overruns per second
                health_status['issues'].append(f'High buffer overrun rate: {overrun_rate:.2f}/sec')
                health_status['overall_status'] = 'warning'
            else:
                health_status['warnings'].append(f'Buffer overruns detected: {self._buffer_overruns}')
        
        # Check synchronization
        if abs(self._drift_compensation) > 0.005:  # 5ms drift
            health_status['warnings'].append(f'Clock drift detected: {self._drift_compensation*1000:.1f}ms')
            if health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'warning'
        
        # Check callback latency
        if self._callback_latency:
            avg_latency = np.mean(self._callback_latency)
            if avg_latency > 0.01:  # 10ms
                health_status['issues'].append(f'High callback latency: {avg_latency*1000:.1f}ms')
                health_status['overall_status'] = 'warning'
        
        # Check channel health
        inactive_channels = [ch for ch, status in self._channel_status.items() if not status]
        if len(inactive_channels) > self.channels // 2:
            health_status['issues'].append(f'Many inactive channels: {inactive_channels}')
            health_status['overall_status'] = 'warning'
        elif inactive_channels:
            health_status['warnings'].append(f'Some inactive channels: {inactive_channels}')
        
        # Generate recommendations
        if self._buffer_overruns > 0:
            health_status['recommendations'].append('Consider increasing buffer size or reducing system load')
        
        if self._sync_errors > 100:
            health_status['recommendations'].append('Consider resetting synchronization or checking audio hardware')
        
        if len(inactive_channels) > 0:
            health_status['recommendations'].append('Check microphone connections and hardware')
        
        return health_status
    
    def enable_synchronization(self, enabled: bool = True) -> None:
        """
        Enable or disable audio stream synchronization.
        
        Args:
            enabled: True to enable synchronization
        """
        self._sync_enabled = enabled
        if enabled:
            self.logger.info("Audio stream synchronization enabled")
        else:
            self.logger.info("Audio stream synchronization disabled")
    
    def set_sync_reference_channel(self, channel: int) -> bool:
        """
        Set the reference channel for synchronization.
        
        Args:
            channel: Channel number (1-based) to use as reference
            
        Returns:
            True if successful
        """
        if 1 <= channel <= self.channels:
            success = self._synchronizer.set_reference_channel(channel - 1)  # Convert to 0-based
            if success:
                self.logger.info(f"Synchronization reference set to channel {channel}")
            return success
        else:
            self.logger.error(f"Invalid reference channel: {channel} (must be 1-{self.channels})")
            return False
    
    def get_synchronization_metrics(self) -> Optional[SynchronizationMetrics]:
        """
        Get the latest synchronization metrics.
        
        Returns:
            Latest SynchronizationMetrics or None if not available
        """
        if self._sync_metrics_history:
            return self._sync_metrics_history[-1]
        return None
    
    def get_synchronization_status(self) -> Dict[str, any]:
        """
        Get detailed synchronization status (enhanced version).
        
        Returns:
            Dictionary with synchronization information
        """
        with self._timing_lock:
            recent_frames = list(self._frame_timestamps)[-10:] if self._frame_timestamps else []
        
        base_status = super().get_synchronization_status() if hasattr(super(), 'get_synchronization_status') else {}
        
        # Add synchronization-specific status
        sync_diagnostics = self._synchronizer.get_synchronization_diagnostics()
        latest_metrics = self.get_synchronization_metrics()
        
        enhanced_status = {
            **base_status,
            'sync_enabled': self._sync_enabled,
            'sync_quality': sync_diagnostics['sync_quality'],
            'reference_channel': sync_diagnostics['reference_channel'] + 1,  # Convert to 1-based
            'channel_delays_ms': sync_diagnostics['channel_delays_ms'],
            'sync_check_interval': self._sync_check_interval,
            'metrics_history_length': len(self._sync_metrics_history)
        }
        
        if latest_metrics:
            enhanced_status.update({
                'latest_phase_coherence': latest_metrics.phase_coherence,
                'latest_sync_score': latest_metrics.sync_quality_score,
                'latest_clock_drift_ms': latest_metrics.clock_drift_ms
            })
        
        return enhanced_status
    
    def calibrate_synchronization(self, calibration_duration: float = 2.0) -> Dict[str, any]:
        """
        Perform synchronization calibration using ambient noise.
        
        Args:
            calibration_duration: Duration of calibration in seconds
            
        Returns:
            Calibration results
        """
        if not self._capturing:
            return {'status': 'error', 'message': 'Not capturing audio'}
        
        self.logger.info(f"Starting synchronization calibration for {calibration_duration}s")
        
        # Reset synchronization state
        self._synchronizer.reset_synchronization_state()
        
        # Collect calibration data
        start_time = time.time()
        calibration_metrics = []
        
        while time.time() - start_time < calibration_duration:
            time.sleep(0.1)  # Wait for data
            
            # Get recent buffer segment for analysis
            segment = self.get_synchronized_buffer_segment(0.1)
            if segment and segment.data.shape[0] >= 64:
                try:
                    metrics = self._synchronizer.analyze_channel_synchronization(
                        segment.data, segment.timestamp
                    )
                    calibration_metrics.append(metrics)
                except Exception as e:
                    self.logger.warning(f"Calibration analysis error: {e}")
        
        if not calibration_metrics:
            return {'status': 'error', 'message': 'No calibration data collected'}
        
        # Analyze calibration results
        sync_scores = [m.sync_quality_score for m in calibration_metrics]
        coherence_scores = [m.phase_coherence for m in calibration_metrics]
        drift_values = [m.clock_drift_ms for m in calibration_metrics]
        
        # Find best reference channel based on alignment scores
        best_ref_channel = 0
        best_avg_alignment = 0
        
        for ch in range(self.channels):
            self._synchronizer.set_reference_channel(ch)
            
            # Re-analyze with this reference
            test_metrics = []
            for i, segment_data in enumerate([m for m in calibration_metrics[-5:]]):  # Use last 5 samples
                if hasattr(segment_data, 'channel_alignment'):
                    alignment_scores = list(segment_data.channel_alignment.values())
                    avg_alignment = np.mean([s for i, s in enumerate(alignment_scores) if i != ch])
                    test_metrics.append(avg_alignment)
            
            if test_metrics:
                avg_alignment = np.mean(test_metrics)
                if avg_alignment > best_avg_alignment:
                    best_avg_alignment = avg_alignment
                    best_ref_channel = ch
        
        # Set optimal reference channel
        self._synchronizer.set_reference_channel(best_ref_channel)
        
        calibration_results = {
            'status': 'success',
            'duration': calibration_duration,
            'samples_analyzed': len(calibration_metrics),
            'optimal_reference_channel': best_ref_channel + 1,  # Convert to 1-based
            'avg_sync_quality': np.mean(sync_scores),
            'avg_phase_coherence': np.mean(coherence_scores),
            'avg_clock_drift_ms': np.mean(drift_values),
            'sync_stability': 1.0 - np.std(sync_scores),
            'recommendations': []
        }
        
        # Generate recommendations
        if calibration_results['avg_sync_quality'] < 0.8:
            calibration_results['recommendations'].append(
                "Consider checking microphone connections or reducing electromagnetic interference"
            )
        
        if abs(calibration_results['avg_clock_drift_ms']) > 1.0:
            calibration_results['recommendations'].append(
                "Significant clock drift detected - consider using external clock synchronization"
            )
        
        if calibration_results['avg_phase_coherence'] < 0.7:
            calibration_results['recommendations'].append(
                "Poor phase coherence - check for acoustic coupling between microphones"
            )
        
        self.logger.info(f"Synchronization calibration completed: quality={calibration_results['avg_sync_quality']:.2f}")
        
        return calibration_results
    
    def export_synchronization_data(self, filepath: str) -> bool:
        """
        Export synchronization metrics to file for analysis.
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if successful
        """
        try:
            import json
            
            export_data = {
                'timestamp': time.time(),
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'sync_enabled': self._sync_enabled,
                'diagnostics': self._synchronizer.get_synchronization_diagnostics(),
                'metrics_history': []
            }
            
            # Export recent metrics
            for metrics in list(self._sync_metrics_history)[-50:]:  # Last 50 entries
                export_data['metrics_history'].append({
                    'timestamp': float(metrics.last_sync_time),
                    'clock_drift_ms': float(metrics.clock_drift_ms),
                    'phase_coherence': float(metrics.phase_coherence),
                    'sync_quality_score': float(metrics.sync_quality_score),
                    'reference_channel': int(metrics.reference_channel),
                    'channel_alignment': {str(k): float(v) for k, v in metrics.channel_alignment.items()}
                })
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Synchronization data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting synchronization data: {e}")
            return False