"""
Gunshot detection module using signal analysis.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import time
import logging
from collections import deque


@dataclass
class DetectionEvent:
    """Container for gunshot detection results."""
    timestamp: float
    confidence: float
    peak_amplitude: float
    frequency_profile: Dict[str, float]
    triggered_channels: List[int]
    duration_ms: float
    rise_time_ms: float
    signal_to_noise_ratio: float


class GunshotDetectorInterface(ABC):
    """Abstract interface for gunshot detection."""
    
    @abstractmethod
    def detect_gunshot(self, audio_data: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect gunshot in audio data.
        
        Args:
            audio_data: Multi-channel audio data (samples, channels)
            
        Returns:
            Tuple of (detected, confidence, metadata)
        """
        pass
    
    @abstractmethod
    def set_adaptive_threshold(self, noise_floor: float) -> None:
        """Set adaptive detection threshold."""
        pass
    
    @abstractmethod
    def get_detection_confidence(self) -> float:
        """Get current detection confidence."""
        pass


class AmplitudeBasedDetector(GunshotDetectorInterface):
    """Amplitude-based gunshot detection using signal characteristics."""
    
    def __init__(self, sample_rate: int = 48000, channels: int = 8, 
                 threshold_db: float = -20.0):
        """
        Initialize amplitude-based gunshot detector.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            channels: Number of audio channels
            threshold_db: Detection threshold in dB
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.threshold_db = threshold_db
        
        # Convert dB threshold to linear amplitude
        self.threshold_linear = 10 ** (threshold_db / 20.0)
        
        # Detection parameters
        self.min_duration_ms = 10    # Minimum gunshot duration
        self.max_duration_ms = 500   # Maximum gunshot duration
        self.min_rise_time_ms = 1    # Minimum rise time
        self.max_rise_time_ms = 50   # Maximum rise time
        
        # Enhanced adaptive threshold parameters
        self.noise_floor = 0.001     # Current noise floor estimate
        self.noise_history = deque(maxlen=1000)  # Recent noise samples
        self.adaptation_rate = 0.01  # Rate of threshold adaptation
        
        # Advanced adaptive thresholding
        self.enable_adaptive_threshold = True
        self.base_threshold_db = threshold_db  # Original threshold for reference
        self.min_threshold_db = threshold_db - 10  # Minimum adaptive threshold
        self.max_threshold_db = threshold_db + 10  # Maximum adaptive threshold
        
        # Environmental adaptation
        self.environment_type = 'unknown'  # Current environment classification
        self.noise_profile = {
            'rms_history': deque(maxlen=500),
            'peak_history': deque(maxlen=500),
            'spectral_history': deque(maxlen=100)
        }
        
        # Time-based adaptation
        self.time_of_day_factor = 1.0  # Adjustment based on time patterns
        self.activity_level = 'normal'  # Current activity level (quiet, normal, busy)
        
        # Statistical adaptation
        self.false_positive_rate = 0.0
        self.missed_detection_rate = 0.0
        self.detection_performance_history = deque(maxlen=100)
        
        # Multi-level thresholding
        self.threshold_levels = {
            'conservative': threshold_db + 5,   # High confidence threshold
            'normal': threshold_db,             # Standard threshold
            'sensitive': threshold_db - 5       # Low threshold for quiet environments
        }
        self.current_threshold_level = 'normal'
        
        # Detection state
        self.last_detection_time = 0.0
        self.detection_cooldown = 0.5  # Minimum time between detections (seconds)
        self.current_confidence = 0.0
        
        # Signal analysis buffers
        self.analysis_window_ms = 100  # Analysis window duration
        self.analysis_window_samples = int(sample_rate * self.analysis_window_ms / 1000)
        
        # Frequency domain analysis parameters
        self.enable_frequency_analysis = True
        self.fft_window_size = 1024  # FFT window size
        self.frequency_bands = {
            'low': (50, 500),      # Low frequency band (50-500 Hz)
            'mid_low': (500, 1500), # Mid-low frequency band (500-1500 Hz)
            'mid': (1500, 4000),   # Mid frequency band (1500-4000 Hz) - gunshot dominant
            'high': (4000, 8000),  # High frequency band (4000-8000 Hz)
            'very_high': (8000, sample_rate//2)  # Very high frequency band
        }
        
        # Gunshot frequency signature (typical energy distribution)
        self.gunshot_signature = {
            'low': 0.15,      # 15% energy in low frequencies
            'mid_low': 0.25,  # 25% energy in mid-low frequencies  
            'mid': 0.45,      # 45% energy in mid frequencies (dominant)
            'high': 0.12,     # 12% energy in high frequencies
            'very_high': 0.03 # 3% energy in very high frequencies
        }
        
        # Frequency analysis state
        self.frequency_history = deque(maxlen=50)  # Recent frequency profiles
        self.spectral_templates = {}  # Learned spectral templates
        
        # Statistics
        self.total_detections = 0
        self.false_positive_count = 0
        self.detection_history = deque(maxlen=100)
        
        self.logger.info(f"AmplitudeBasedDetector initialized: {sample_rate}Hz, {channels}ch, {threshold_db}dB")
    
    def detect_gunshot(self, audio_data: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect gunshot using amplitude-based analysis.
        
        Args:
            audio_data: Multi-channel audio data (samples, channels)
            
        Returns:
            Tuple of (detected, confidence, metadata)
        """
        if audio_data.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {audio_data.shape[1]}")
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_detection_time < self.detection_cooldown:
            return False, 0.0, {'reason': 'cooldown_period'}
        
        try:
            # Update noise floor estimate
            self._update_noise_floor(audio_data)
            
            # Analyze each channel for potential gunshot
            channel_results = []
            for ch in range(self.channels):
                channel_data = audio_data[:, ch]
                amplitude_result = self._analyze_channel_amplitude(channel_data, ch)
                
                # Add frequency domain analysis if enabled
                if self.enable_frequency_analysis and len(channel_data) >= self.fft_window_size:
                    frequency_result = self._analyze_channel_frequency(channel_data, ch)
                    # Combine amplitude and frequency results
                    combined_result = self._combine_amplitude_frequency_analysis(
                        amplitude_result, frequency_result
                    )
                    channel_results.append(combined_result)
                else:
                    channel_results.append(amplitude_result)
            
            # Combine channel results
            detection_result = self._combine_channel_results(channel_results, current_time)
            
            # Update detection history
            if detection_result[0]:  # If detection occurred
                self.last_detection_time = current_time
                self.total_detections += 1
                
                detection_event = DetectionEvent(
                    timestamp=current_time,
                    confidence=detection_result[1],
                    peak_amplitude=detection_result[2].get('peak_amplitude', 0.0),
                    frequency_profile=detection_result[2].get('frequency_profile', {}),
                    triggered_channels=detection_result[2].get('triggered_channels', []),
                    duration_ms=detection_result[2].get('duration_ms', 0.0),
                    rise_time_ms=detection_result[2].get('rise_time_ms', 0.0),
                    signal_to_noise_ratio=detection_result[2].get('snr', 0.0)
                )
                
                self.detection_history.append(detection_event)
                self.logger.info(f"Gunshot detected: confidence={detection_result[1]:.2f}, channels={detection_result[2].get('triggered_channels', [])}")
                
                # Update performance tracking
                self._update_detection_performance(True, detection_result[1])
            else:
                # Track non-detections for performance analysis
                self._update_detection_performance(False, detection_result[1])
            
            self.current_confidence = detection_result[1]
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Error in gunshot detection: {e}")
            return False, 0.0, {'error': str(e)}
    
    def _analyze_channel_amplitude(self, channel_data: np.ndarray, channel_id: int) -> Dict:
        """
        Analyze amplitude characteristics of a single channel.
        
        Args:
            channel_data: Single channel audio data
            channel_id: Channel identifier
            
        Returns:
            Dictionary with analysis results
        """
        # Calculate basic amplitude metrics
        rms_amplitude = np.sqrt(np.mean(channel_data ** 2))
        peak_amplitude = np.max(np.abs(channel_data))
        
        # Calculate signal-to-noise ratio
        snr = peak_amplitude / max(self.noise_floor, 1e-10)
        snr_db = 20 * np.log10(snr) if snr > 0 else -100
        
        # Check if amplitude exceeds threshold
        amplitude_trigger = peak_amplitude > self.threshold_linear
        
        # Debug logging for threshold comparison
        if peak_amplitude > 0.01:  # Only log significant signals
            self.logger.debug(f"Channel {channel_id}: peak={peak_amplitude:.4f}, threshold={self.threshold_linear:.4f}, trigger={amplitude_trigger}")
        
        # Analyze temporal characteristics
        temporal_analysis = self._analyze_temporal_characteristics(channel_data)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_amplitude_confidence(
            peak_amplitude, rms_amplitude, snr_db, temporal_analysis
        )
        
        return {
            'channel_id': channel_id,
            'peak_amplitude': peak_amplitude,
            'rms_amplitude': rms_amplitude,
            'snr_db': snr_db,
            'amplitude_trigger': amplitude_trigger,
            'confidence': confidence,
            'temporal_analysis': temporal_analysis
        }
    
    def _analyze_temporal_characteristics(self, signal: np.ndarray) -> Dict:
        """
        Analyze temporal characteristics of the signal.
        
        Args:
            signal: Audio signal data
            
        Returns:
            Dictionary with temporal analysis results
        """
        # Find signal envelope using Hilbert transform approximation
        envelope = self._calculate_envelope(signal)
        
        # Find peak location
        peak_idx = np.argmax(envelope)
        peak_value = envelope[peak_idx]
        
        # Calculate duration above threshold
        threshold_value = peak_value * 0.1  # 10% of peak
        above_threshold = envelope > threshold_value
        
        if np.any(above_threshold):
            # Find start and end of signal
            signal_start = np.where(above_threshold)[0][0]
            signal_end = np.where(above_threshold)[0][-1]
            
            duration_samples = signal_end - signal_start + 1
            duration_ms = (duration_samples / self.sample_rate) * 1000
            
            # Calculate rise time (10% to 90% of peak)
            rise_threshold_low = peak_value * 0.1
            rise_threshold_high = peak_value * 0.9
            
            rise_start_idx = signal_start
            rise_end_idx = peak_idx
            
            # Find more precise rise time boundaries
            for i in range(signal_start, peak_idx):
                if envelope[i] >= rise_threshold_low:
                    rise_start_idx = i
                    break
            
            for i in range(rise_start_idx, peak_idx):
                if envelope[i] >= rise_threshold_high:
                    rise_end_idx = i
                    break
            
            rise_time_samples = rise_end_idx - rise_start_idx
            rise_time_ms = (rise_time_samples / self.sample_rate) * 1000
            
        else:
            duration_ms = 0.0
            rise_time_ms = 0.0
        
        return {
            'duration_ms': duration_ms,
            'rise_time_ms': rise_time_ms,
            'peak_position': peak_idx / len(signal),
            'envelope_shape_factor': self._calculate_shape_factor(envelope)
        }
    
    def _calculate_envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        Calculate signal envelope using moving average of absolute values.
        
        Args:
            signal: Input signal
            
        Returns:
            Signal envelope
        """
        # Use absolute value
        abs_signal = np.abs(signal)
        
        # Apply smoothing with moving average
        window_size = max(1, int(self.sample_rate * 0.001))  # 1ms window
        
        if len(abs_signal) < window_size:
            return abs_signal
        
        # Simple moving average
        envelope = np.convolve(abs_signal, np.ones(window_size) / window_size, mode='same')
        
        return envelope
    
    def _calculate_shape_factor(self, envelope: np.ndarray) -> float:
        """
        Calculate shape factor of the envelope (measure of impulsiveness).
        
        Args:
            envelope: Signal envelope
            
        Returns:
            Shape factor (higher = more impulsive)
        """
        if len(envelope) == 0:
            return 0.0
        
        # Calculate RMS and peak
        rms = np.sqrt(np.mean(envelope ** 2))
        peak = np.max(envelope)
        
        # Shape factor = peak / RMS (higher for impulsive signals)
        if rms > 0:
            return peak / rms
        else:
            return 0.0
    
    def _calculate_amplitude_confidence(self, peak_amp: float, rms_amp: float, 
                                     snr_db: float, temporal_analysis: Dict) -> float:
        """
        Calculate confidence score based on amplitude characteristics.
        
        Args:
            peak_amp: Peak amplitude
            rms_amp: RMS amplitude
            snr_db: Signal-to-noise ratio in dB
            temporal_analysis: Temporal analysis results
            
        Returns:
            Confidence score (0-1)
        """
        confidence_factors = []
        
        # Factor 1: SNR-based confidence
        if snr_db > 20:  # Very strong signal
            snr_confidence = 1.0
        elif snr_db > 10:  # Strong signal
            snr_confidence = 0.8
        elif snr_db > 0:   # Above noise floor
            snr_confidence = 0.6
        else:              # Below noise floor
            snr_confidence = 0.0
        
        confidence_factors.append(snr_confidence)
        
        # Factor 2: Duration-based confidence
        duration_ms = temporal_analysis.get('duration_ms', 0)
        if self.min_duration_ms <= duration_ms <= self.max_duration_ms:
            duration_confidence = 1.0
        elif duration_ms < self.min_duration_ms:
            duration_confidence = max(0, duration_ms / self.min_duration_ms)
        else:  # Too long
            duration_confidence = max(0, 1.0 - (duration_ms - self.max_duration_ms) / 1000)
        
        confidence_factors.append(duration_confidence)
        
        # Factor 3: Rise time-based confidence
        rise_time_ms = temporal_analysis.get('rise_time_ms', 0)
        if self.min_rise_time_ms <= rise_time_ms <= self.max_rise_time_ms:
            rise_confidence = 1.0
        elif rise_time_ms < self.min_rise_time_ms:
            rise_confidence = 0.5  # Very fast rise (could be gunshot)
        else:  # Too slow
            rise_confidence = max(0, 1.0 - (rise_time_ms - self.max_rise_time_ms) / 100)
        
        confidence_factors.append(rise_confidence)
        
        # Factor 4: Shape factor (impulsiveness)
        shape_factor = temporal_analysis.get('envelope_shape_factor', 1.0)
        if shape_factor > 3.0:  # Very impulsive
            shape_confidence = 1.0
        elif shape_factor > 2.0:  # Moderately impulsive
            shape_confidence = 0.8
        elif shape_factor > 1.5:  # Somewhat impulsive
            shape_confidence = 0.6
        else:  # Not impulsive
            shape_confidence = 0.3
        
        confidence_factors.append(shape_confidence)
        
        # Combine factors with weights
        weights = [0.3, 0.25, 0.25, 0.2]  # SNR, duration, rise time, shape
        weighted_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return max(0.0, min(1.0, weighted_confidence))
    
    def _analyze_channel_frequency(self, channel_data: np.ndarray, channel_id: int) -> Dict:
        """
        Analyze frequency domain characteristics of a single channel.
        
        Args:
            channel_data: Single channel audio data
            channel_id: Channel identifier
            
        Returns:
            Dictionary with frequency analysis results
        """
        # Ensure we have enough data for FFT
        if len(channel_data) < self.fft_window_size:
            return {
                'frequency_profile': {},
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'spectral_flatness': 0.0,
                'gunshot_similarity': 0.0,
                'frequency_confidence': 0.0
            }
        
        # Find the most energetic segment for analysis
        segment_start = self._find_peak_segment(channel_data)
        segment_end = min(segment_start + self.fft_window_size, len(channel_data))
        analysis_segment = channel_data[segment_start:segment_end]
        
        # Pad if necessary
        if len(analysis_segment) < self.fft_window_size:
            analysis_segment = np.pad(analysis_segment, 
                                    (0, self.fft_window_size - len(analysis_segment)), 
                                    'constant')
        
        # Apply window to reduce spectral leakage
        windowed_segment = analysis_segment * np.hanning(len(analysis_segment))
        
        # Compute FFT
        fft_result = np.fft.fft(windowed_segment)
        magnitude_spectrum = np.abs(fft_result[:len(fft_result)//2])
        
        # Calculate frequency bins
        freqs = np.fft.fftfreq(len(windowed_segment), 1/self.sample_rate)[:len(magnitude_spectrum)]
        
        # Calculate frequency band energies
        frequency_profile = self._calculate_frequency_bands(magnitude_spectrum, freqs)
        
        # Calculate spectral features
        spectral_features = self._calculate_spectral_features(magnitude_spectrum, freqs)
        
        # Compare with gunshot signature
        gunshot_similarity = self._calculate_gunshot_similarity(frequency_profile)
        
        # Calculate frequency-based confidence
        frequency_confidence = self._calculate_frequency_confidence(
            frequency_profile, spectral_features, gunshot_similarity
        )
        
        return {
            'frequency_profile': frequency_profile,
            'spectral_centroid': spectral_features['centroid'],
            'spectral_rolloff': spectral_features['rolloff'],
            'spectral_flatness': spectral_features['flatness'],
            'gunshot_similarity': gunshot_similarity,
            'frequency_confidence': frequency_confidence,
            'dominant_frequency': spectral_features['dominant_freq']
        }
    
    def _find_peak_segment(self, signal: np.ndarray) -> int:
        """
        Find the segment with highest energy for frequency analysis.
        
        Args:
            signal: Input signal
            
        Returns:
            Start index of peak segment
        """
        if len(signal) <= self.fft_window_size:
            return 0
        
        # Calculate energy in overlapping windows
        hop_size = self.fft_window_size // 4
        max_energy = 0
        best_start = 0
        
        for start in range(0, len(signal) - self.fft_window_size, hop_size):
            segment = signal[start:start + self.fft_window_size]
            energy = np.sum(segment ** 2)
            
            if energy > max_energy:
                max_energy = energy
                best_start = start
        
        return best_start
    
    def _calculate_frequency_bands(self, magnitude_spectrum: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
        """
        Calculate energy in different frequency bands.
        
        Args:
            magnitude_spectrum: FFT magnitude spectrum
            freqs: Frequency bins
            
        Returns:
            Dictionary with energy in each frequency band
        """
        total_energy = np.sum(magnitude_spectrum ** 2)
        if total_energy == 0:
            return {band: 0.0 for band in self.frequency_bands.keys()}
        
        band_energies = {}
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Find frequency bin indices
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_energy = np.sum(magnitude_spectrum[band_mask] ** 2)
            
            # Normalize by total energy
            band_energies[band_name] = band_energy / total_energy
        
        return band_energies
    
    def _calculate_spectral_features(self, magnitude_spectrum: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
        """
        Calculate spectral features for signal characterization.
        
        Args:
            magnitude_spectrum: FFT magnitude spectrum
            freqs: Frequency bins
            
        Returns:
            Dictionary with spectral features
        """
        # Spectral centroid (center of mass of spectrum)
        if np.sum(magnitude_spectrum) > 0:
            spectral_centroid = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
        else:
            spectral_centroid = 0.0
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(magnitude_spectrum ** 2)
        total_energy = cumulative_energy[-1]
        
        if total_energy > 0:
            rolloff_threshold = 0.85 * total_energy
            rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        else:
            spectral_rolloff = 0.0
        
        # Spectral flatness (measure of how noise-like vs tonal the spectrum is)
        # Geometric mean / Arithmetic mean
        magnitude_spectrum_positive = magnitude_spectrum[magnitude_spectrum > 0]
        if len(magnitude_spectrum_positive) > 0:
            geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum_positive)))
            arithmetic_mean = np.mean(magnitude_spectrum_positive)
            spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
        else:
            spectral_flatness = 0.0
        
        # Dominant frequency
        dominant_idx = np.argmax(magnitude_spectrum)
        dominant_freq = freqs[dominant_idx]
        
        return {
            'centroid': spectral_centroid,
            'rolloff': spectral_rolloff,
            'flatness': spectral_flatness,
            'dominant_freq': dominant_freq
        }
    
    def _calculate_gunshot_similarity(self, frequency_profile: Dict[str, float]) -> float:
        """
        Calculate similarity to typical gunshot frequency signature.
        
        Args:
            frequency_profile: Energy distribution across frequency bands
            
        Returns:
            Similarity score (0-1)
        """
        if not frequency_profile:
            return 0.0
        
        # Calculate normalized cross-correlation with gunshot signature
        profile_values = np.array([frequency_profile.get(band, 0.0) for band in self.gunshot_signature.keys()])
        signature_values = np.array(list(self.gunshot_signature.values()))
        
        # Normalize vectors
        profile_norm = np.linalg.norm(profile_values)
        signature_norm = np.linalg.norm(signature_values)
        
        if profile_norm == 0 or signature_norm == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(profile_values, signature_values) / (profile_norm * signature_norm)
        
        # Convert to 0-1 range (cosine similarity is -1 to 1)
        return max(0.0, similarity)
    
    def _calculate_frequency_confidence(self, frequency_profile: Dict[str, float], 
                                      spectral_features: Dict[str, float], 
                                      gunshot_similarity: float) -> float:
        """
        Calculate confidence based on frequency domain characteristics.
        
        Args:
            frequency_profile: Energy distribution across bands
            spectral_features: Calculated spectral features
            gunshot_similarity: Similarity to gunshot signature
            
        Returns:
            Frequency-based confidence score (0-1)
        """
        confidence_factors = []
        
        # Factor 1: Gunshot signature similarity
        similarity_confidence = gunshot_similarity
        confidence_factors.append(similarity_confidence)
        
        # Factor 2: Dominant frequency in gunshot range (500-4000 Hz)
        dominant_freq = spectral_features['dominant_freq']
        if 500 <= dominant_freq <= 4000:
            freq_confidence = 1.0
        elif 200 <= dominant_freq <= 6000:  # Extended range
            freq_confidence = 0.7
        else:
            freq_confidence = 0.3
        
        confidence_factors.append(freq_confidence)
        
        # Factor 3: Mid-frequency dominance (gunshots have strong mid-frequency content)
        mid_energy = frequency_profile.get('mid', 0.0)
        if mid_energy > 0.3:  # Strong mid-frequency content
            mid_confidence = 1.0
        elif mid_energy > 0.2:
            mid_confidence = 0.8
        elif mid_energy > 0.1:
            mid_confidence = 0.6
        else:
            mid_confidence = 0.2
        
        confidence_factors.append(mid_confidence)
        
        # Factor 4: Spectral rolloff (gunshots typically have rolloff around 2-5 kHz)
        rolloff = spectral_features['rolloff']
        if 2000 <= rolloff <= 5000:
            rolloff_confidence = 1.0
        elif 1000 <= rolloff <= 7000:
            rolloff_confidence = 0.7
        else:
            rolloff_confidence = 0.4
        
        confidence_factors.append(rolloff_confidence)
        
        # Factor 5: Spectral flatness (gunshots are more tonal than noise)
        flatness = spectral_features['flatness']
        if flatness < 0.3:  # More tonal
            flatness_confidence = 1.0
        elif flatness < 0.5:
            flatness_confidence = 0.8
        elif flatness < 0.7:
            flatness_confidence = 0.6
        else:  # Too flat (noise-like)
            flatness_confidence = 0.3
        
        confidence_factors.append(flatness_confidence)
        
        # Combine factors with weights
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Signature, dominant freq, mid energy, rolloff, flatness
        weighted_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return max(0.0, min(1.0, weighted_confidence))
    
    def _combine_amplitude_frequency_analysis(self, amplitude_result: Dict, frequency_result: Dict) -> Dict:
        """
        Combine amplitude and frequency analysis results.
        
        Args:
            amplitude_result: Results from amplitude analysis
            frequency_result: Results from frequency analysis
            
        Returns:
            Combined analysis results
        """
        # Combine confidences with weighting
        amplitude_confidence = amplitude_result.get('confidence', 0.0)
        frequency_confidence = frequency_result.get('frequency_confidence', 0.0)
        
        # Weight amplitude analysis more heavily (it's more reliable for detection)
        combined_confidence = 0.7 * amplitude_confidence + 0.3 * frequency_confidence
        
        # Update the amplitude result with frequency information
        combined_result = amplitude_result.copy()
        combined_result.update({
            'frequency_analysis': frequency_result,
            'combined_confidence': combined_confidence,
            'confidence': combined_confidence  # Override original confidence
        })
        
        return combined_result
    
    def _combine_channel_results(self, channel_results: List[Dict], timestamp: float) -> Tuple[bool, float, Dict]:
        """
        Combine results from multiple channels to make final detection decision.
        
        Args:
            channel_results: List of per-channel analysis results
            timestamp: Current timestamp
            
        Returns:
            Tuple of (detected, confidence, metadata)
        """
        # Find channels that triggered
        triggered_channels = []
        channel_confidences = []
        
        for result in channel_results:
            if result['amplitude_trigger'] and result['confidence'] > 0.3:
                triggered_channels.append(result['channel_id'])
                channel_confidences.append(result['confidence'])
        
        # Require at least one channel to trigger
        if not triggered_channels:
            return False, 0.0, {'reason': 'no_channels_triggered'}
        
        # Calculate overall confidence
        if len(triggered_channels) >= 3:
            # Multiple channels - high confidence
            overall_confidence = min(1.0, np.mean(channel_confidences) * 1.2)
        elif len(triggered_channels) == 2:
            # Two channels - moderate confidence
            overall_confidence = np.mean(channel_confidences)
        else:
            # Single channel - lower confidence
            overall_confidence = channel_confidences[0] * 0.8
        
        # Find best channel result for metadata
        best_channel_idx = np.argmax([r['confidence'] for r in channel_results])
        best_result = channel_results[best_channel_idx]
        
        # Compile metadata
        metadata = {
            'triggered_channels': [ch + 1 for ch in triggered_channels],  # Convert to 1-based
            'peak_amplitude': best_result['peak_amplitude'],
            'snr': best_result['snr_db'],
            'duration_ms': best_result['temporal_analysis']['duration_ms'],
            'rise_time_ms': best_result['temporal_analysis']['rise_time_ms'],
            'shape_factor': best_result['temporal_analysis']['envelope_shape_factor'],
            'noise_floor': self.noise_floor,
            'detection_method': 'amplitude_frequency_combined' if self.enable_frequency_analysis else 'amplitude_based'
        }
        
        # Add frequency analysis results if available
        if 'frequency_analysis' in best_result:
            freq_analysis = best_result['frequency_analysis']
            metadata.update({
                'frequency_profile': freq_analysis['frequency_profile'],
                'spectral_centroid': freq_analysis['spectral_centroid'],
                'spectral_rolloff': freq_analysis['spectral_rolloff'],
                'gunshot_similarity': freq_analysis['gunshot_similarity'],
                'dominant_frequency': freq_analysis['dominant_frequency']
            })
        
        # Make final detection decision
        detection_threshold = 0.5  # Minimum confidence for detection
        detected = overall_confidence >= detection_threshold
        
        return detected, overall_confidence, metadata
    
    def _update_noise_floor(self, audio_data: np.ndarray) -> None:
        """
        Enhanced noise floor estimation with environmental analysis.
        
        Args:
            audio_data: Recent audio data
        """
        # Calculate comprehensive noise statistics
        channel_rms = np.sqrt(np.mean(audio_data ** 2, axis=0))
        channel_peaks = np.max(np.abs(audio_data), axis=0)
        
        # Store in noise profile
        current_rms = np.mean(channel_rms)  # Average across channels
        current_peak = np.max(channel_peaks)
        
        self.noise_profile['rms_history'].append(current_rms)
        self.noise_profile['peak_history'].append(current_peak)
        
        # Update basic noise floor
        current_noise = np.min(channel_rms)  # Quietest channel
        self.noise_history.append(current_noise)
        
        if len(self.noise_history) > 10:
            # Enhanced noise floor calculation
            recent_noise_samples = list(self.noise_history)[-100:]
            
            # Use multiple percentiles for robust estimation
            p10 = np.percentile(recent_noise_samples, 10)  # Very quiet periods
            p25 = np.percentile(recent_noise_samples, 25)  # Quiet periods
            p50 = np.percentile(recent_noise_samples, 50)  # Median
            
            # Weighted combination favoring quieter periods
            robust_noise_floor = 0.5 * p10 + 0.3 * p25 + 0.2 * p50
            
            # Apply exponential smoothing
            self.noise_floor = (1 - self.adaptation_rate) * self.noise_floor + self.adaptation_rate * robust_noise_floor
            
            # Classify environment and adapt threshold
            if self.enable_adaptive_threshold:
                self._classify_environment()
                self._adapt_threshold_to_environment()
    
    def _classify_environment(self) -> None:
        """
        Classify the current acoustic environment based on noise characteristics.
        """
        if len(self.noise_profile['rms_history']) < 50:
            return  # Need sufficient history
        
        recent_rms = list(self.noise_profile['rms_history'])[-50:]
        recent_peaks = list(self.noise_profile['peak_history'])[-50:]
        
        # Calculate environment metrics
        avg_rms = np.mean(recent_rms)
        rms_std = np.std(recent_rms)
        avg_peak = np.mean(recent_peaks)
        peak_to_rms_ratio = avg_peak / max(avg_rms, 1e-10)
        
        # Classify environment
        if avg_rms < 0.001 and rms_std < 0.0005:
            new_environment = 'very_quiet'  # Library, empty room
            new_activity = 'quiet'
        elif avg_rms < 0.005 and rms_std < 0.002:
            new_environment = 'quiet'       # Residential area at night
            new_activity = 'quiet'
        elif avg_rms < 0.02 and rms_std < 0.01:
            new_environment = 'moderate'    # Office, residential day
            new_activity = 'normal'
        elif avg_rms < 0.05 and rms_std < 0.02:
            new_environment = 'busy'        # Street, commercial area
            new_activity = 'busy'
        else:
            new_environment = 'very_busy'   # Construction, traffic
            new_activity = 'busy'
        
        # Check for impulsive environment (many transients)
        if peak_to_rms_ratio > 5.0 and rms_std > avg_rms * 0.5:
            new_environment += '_impulsive'
        
        # Update environment if changed
        if new_environment != self.environment_type:
            self.environment_type = new_environment
            self.activity_level = new_activity
            self.logger.info(f"Environment classified as: {new_environment} (activity: {new_activity})")
    
    def _adapt_threshold_to_environment(self) -> None:
        """
        Adapt detection threshold based on current environment.
        """
        # Base adaptation on environment type
        if 'very_quiet' in self.environment_type:
            # Lower threshold in very quiet environments
            adaptive_adjustment = -8  # More sensitive
            self.current_threshold_level = 'sensitive'
        elif 'quiet' in self.environment_type:
            # Slightly lower threshold in quiet environments
            adaptive_adjustment = -3
            self.current_threshold_level = 'sensitive'
        elif 'moderate' in self.environment_type:
            # Standard threshold
            adaptive_adjustment = 0
            self.current_threshold_level = 'normal'
        elif 'busy' in self.environment_type:
            # Higher threshold in busy environments
            adaptive_adjustment = +3
            self.current_threshold_level = 'normal'
        elif 'very_busy' in self.environment_type:
            # Much higher threshold in very busy environments
            adaptive_adjustment = +8
            self.current_threshold_level = 'conservative'
        else:
            adaptive_adjustment = 0
        
        # Additional adjustment for impulsive environments
        if 'impulsive' in self.environment_type:
            adaptive_adjustment += 2  # Slightly higher threshold
        
        # Apply noise floor based adjustment
        noise_floor_db = 20 * np.log10(max(self.noise_floor, 1e-10))
        if noise_floor_db > -40:  # High noise floor
            adaptive_adjustment += min(5, (noise_floor_db + 40) / 4)
        
        # Calculate new threshold
        new_threshold_db = self.base_threshold_db + adaptive_adjustment
        
        # Apply limits
        new_threshold_db = max(self.min_threshold_db, 
                              min(self.max_threshold_db, new_threshold_db))
        
        # Update threshold if significantly different
        if abs(new_threshold_db - self.threshold_db) > 0.5:
            old_threshold = self.threshold_db
            self.threshold_db = new_threshold_db
            self.threshold_linear = 10 ** (self.threshold_db / 20.0)
            
            self.logger.info(f"Adaptive threshold: {old_threshold:.1f}dB → {new_threshold_db:.1f}dB "
                           f"(environment: {self.environment_type})")
    
    def _calculate_time_based_adjustment(self) -> float:
        """
        Calculate threshold adjustment based on time patterns.
        
        Returns:
            Adjustment factor for threshold
        """
        current_hour = time.localtime().tm_hour
        
        # Typical noise patterns throughout the day
        if 22 <= current_hour or current_hour <= 6:  # Night time (10 PM - 6 AM)
            return -2  # More sensitive at night
        elif 6 <= current_hour <= 8 or 17 <= current_hour <= 19:  # Rush hours
            return +2  # Less sensitive during busy periods
        elif 9 <= current_hour <= 17:  # Business hours
            return +1  # Slightly less sensitive during day
        else:
            return 0   # Standard sensitivity
    
    def _update_detection_performance(self, detected: bool, confidence: float, 
                                    user_feedback: Optional[str] = None) -> None:
        """
        Update detection performance statistics for adaptive learning.
        
        Args:
            detected: Whether detection occurred
            confidence: Detection confidence
            user_feedback: Optional user feedback ('correct', 'false_positive', 'missed')
        """
        performance_entry = {
            'timestamp': time.time(),
            'detected': detected,
            'confidence': confidence,
            'threshold_db': self.threshold_db,
            'environment': self.environment_type,
            'user_feedback': user_feedback
        }
        
        self.detection_performance_history.append(performance_entry)
        
        # Update performance metrics if feedback provided
        if user_feedback:
            recent_entries = list(self.detection_performance_history)[-50:]  # Last 50 detections
            
            false_positives = sum(1 for e in recent_entries 
                                if e.get('user_feedback') == 'false_positive')
            missed_detections = sum(1 for e in recent_entries 
                                  if e.get('user_feedback') == 'missed')
            total_feedback = sum(1 for e in recent_entries 
                               if e.get('user_feedback') is not None)
            
            if total_feedback > 0:
                self.false_positive_rate = false_positives / total_feedback
                self.missed_detection_rate = missed_detections / total_feedback
                
                # Adapt threshold based on performance
                if self.false_positive_rate > 0.2:  # Too many false positives
                    self._adjust_threshold_for_performance(+2)  # Increase threshold
                elif self.missed_detection_rate > 0.2:  # Too many missed detections
                    self._adjust_threshold_for_performance(-2)  # Decrease threshold
    
    def _adjust_threshold_for_performance(self, adjustment_db: float) -> None:
        """
        Adjust threshold based on performance feedback.
        
        Args:
            adjustment_db: Threshold adjustment in dB
        """
        new_base_threshold = self.base_threshold_db + adjustment_db
        new_base_threshold = max(self.min_threshold_db, 
                               min(self.max_threshold_db, new_base_threshold))
        
        if abs(new_base_threshold - self.base_threshold_db) > 0.1:
            self.base_threshold_db = new_base_threshold
            self.logger.info(f"Base threshold adjusted to {new_base_threshold:.1f}dB based on performance feedback")
    
    def set_adaptive_threshold(self, noise_floor: float) -> None:
        """
        Enhanced adaptive threshold setting with multi-factor analysis.
        
        Args:
            noise_floor: Current noise floor estimate
        """
        if noise_floor > 0:
            self.noise_floor = noise_floor
            
            if self.enable_adaptive_threshold:
                # Multi-factor adaptive threshold calculation
                
                # Factor 1: Noise floor based adjustment
                noise_floor_db = 20 * np.log10(max(noise_floor, 1e-10))
                noise_adjustment = max(0, (noise_floor_db + 60) / 10)  # Scale from -60dB
                
                # Factor 2: Time-based adjustment
                time_adjustment = self._calculate_time_based_adjustment()
                
                # Factor 3: Environment-based adjustment (already applied in _adapt_threshold_to_environment)
                
                # Factor 4: Performance-based adjustment
                performance_adjustment = 0
                if self.false_positive_rate > 0.15:
                    performance_adjustment += 1
                if self.missed_detection_rate > 0.15:
                    performance_adjustment -= 1
                
                # Combine adjustments
                total_adjustment = noise_adjustment + time_adjustment + performance_adjustment
                
                # Apply to base threshold
                new_threshold_db = self.base_threshold_db + total_adjustment
                
                # Apply limits
                new_threshold_db = max(self.min_threshold_db, 
                                     min(self.max_threshold_db, new_threshold_db))
                
                # Update if significantly different
                if abs(new_threshold_db - self.threshold_db) > 0.2:
                    old_threshold = self.threshold_db
                    self.threshold_db = new_threshold_db
                    self.threshold_linear = 10 ** (self.threshold_db / 20.0)
                    
                    self.logger.debug(f"Multi-factor adaptive threshold: {old_threshold:.1f}dB → {new_threshold_db:.1f}dB "
                                    f"(noise: {noise_adjustment:.1f}, time: {time_adjustment:.1f}, "
                                    f"perf: {performance_adjustment:.1f})")
            else:
                # Simple threshold adjustment (original behavior)
                self.threshold_linear = max(self.threshold_linear, noise_floor * 3.0)
                self.threshold_db = 20 * np.log10(self.threshold_linear)
                
                self.logger.debug(f"Simple adaptive threshold: {self.threshold_db:.1f}dB (noise floor: {noise_floor:.6f})")
    
    def get_detection_confidence(self) -> float:
        """Get current detection confidence."""
        return self.current_confidence
    
    def get_detection_statistics(self) -> Dict:
        """
        Get detection statistics and performance metrics.
        
        Returns:
            Dictionary with detection statistics
        """
        recent_detections = list(self.detection_history)[-10:]  # Last 10 detections
        
        if recent_detections:
            avg_confidence = np.mean([d.confidence for d in recent_detections])
            avg_snr = np.mean([d.signal_to_noise_ratio for d in recent_detections])
            avg_duration = np.mean([d.duration_ms for d in recent_detections])
        else:
            avg_confidence = 0.0
            avg_snr = 0.0
            avg_duration = 0.0
        
        return {
            'total_detections': self.total_detections,
            'false_positives': self.false_positive_count,
            'detection_rate': len(recent_detections) / max(1, len(self.detection_history)),
            'avg_confidence': avg_confidence,
            'avg_snr_db': avg_snr,
            'avg_duration_ms': avg_duration,
            'current_threshold_db': self.threshold_db,
            'current_noise_floor': self.noise_floor,
            'cooldown_period': self.detection_cooldown
        }
    
    def reset_detection_state(self) -> None:
        """Reset detection state and statistics."""
        self.last_detection_time = 0.0
        self.current_confidence = 0.0
        self.total_detections = 0
        self.false_positive_count = 0
        self.detection_history.clear()
        self.noise_history.clear()
        self.noise_floor = 0.001
        
        self.logger.info("Detection state reset")
    
    def configure_detection_parameters(self, **kwargs) -> None:
        """
        Configure detection parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'threshold_db' in kwargs:
            self.threshold_db = kwargs['threshold_db']
            self.threshold_linear = 10 ** (self.threshold_db / 20.0)
        
        if 'min_duration_ms' in kwargs:
            self.min_duration_ms = kwargs['min_duration_ms']
        
        if 'max_duration_ms' in kwargs:
            self.max_duration_ms = kwargs['max_duration_ms']
        
        if 'detection_cooldown' in kwargs:
            self.detection_cooldown = kwargs['detection_cooldown']
        
        if 'adaptation_rate' in kwargs:
            self.adaptation_rate = kwargs['adaptation_rate']
        
        self.logger.info(f"Detection parameters updated: {kwargs}")
    
    def enable_adaptive_thresholding(self, enabled: bool = True) -> None:
        """
        Enable or disable adaptive thresholding.
        
        Args:
            enabled: True to enable adaptive thresholding
        """
        self.enable_adaptive_threshold = enabled
        if not enabled:
            # Reset to base threshold
            self.threshold_db = self.base_threshold_db
            self.threshold_linear = 10 ** (self.threshold_db / 20.0)
        
        self.logger.info(f"Adaptive thresholding {'enabled' if enabled else 'disabled'}")
    
    def configure_adaptive_limits(self, min_threshold_db: float, max_threshold_db: float) -> None:
        """
        Configure adaptive threshold limits.
        
        Args:
            min_threshold_db: Minimum allowed threshold in dB
            max_threshold_db: Maximum allowed threshold in dB
        """
        if min_threshold_db >= max_threshold_db:
            raise ValueError("Minimum threshold must be less than maximum threshold")
        
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        
        # Ensure current threshold is within limits
        self.threshold_db = max(min_threshold_db, min(max_threshold_db, self.threshold_db))
        self.threshold_linear = 10 ** (self.threshold_db / 20.0)
        
        self.logger.info(f"Adaptive threshold limits: {min_threshold_db:.1f}dB to {max_threshold_db:.1f}dB")
    
    def set_threshold_level(self, level: str) -> None:
        """
        Set detection threshold level.
        
        Args:
            level: Threshold level ('conservative', 'normal', 'sensitive')
        """
        if level not in self.threshold_levels:
            raise ValueError(f"Invalid threshold level: {level}. Must be one of {list(self.threshold_levels.keys())}")
        
        self.current_threshold_level = level
        new_threshold_db = self.threshold_levels[level]
        
        # Apply limits
        new_threshold_db = max(self.min_threshold_db, min(self.max_threshold_db, new_threshold_db))
        
        self.threshold_db = new_threshold_db
        self.threshold_linear = 10 ** (self.threshold_db / 20.0)
        
        self.logger.info(f"Threshold level set to '{level}': {new_threshold_db:.1f}dB")
    
    def provide_detection_feedback(self, feedback: str) -> None:
        """
        Provide feedback on the last detection for adaptive learning.
        
        Args:
            feedback: Feedback type ('correct', 'false_positive', 'missed')
        """
        valid_feedback = ['correct', 'false_positive', 'missed']
        if feedback not in valid_feedback:
            raise ValueError(f"Invalid feedback: {feedback}. Must be one of {valid_feedback}")
        
        # Update the last detection entry with feedback
        if self.detection_performance_history:
            self.detection_performance_history[-1]['user_feedback'] = feedback
            self._update_detection_performance(
                self.detection_performance_history[-1]['detected'],
                self.detection_performance_history[-1]['confidence'],
                feedback
            )
            
            self.logger.info(f"Detection feedback received: {feedback}")
        else:
            self.logger.warning("No recent detection to provide feedback for")
    
    def get_adaptive_threshold_status(self) -> Dict:
        """
        Get comprehensive adaptive threshold status.
        
        Returns:
            Dictionary with adaptive threshold information
        """
        return {
            'adaptive_enabled': self.enable_adaptive_threshold,
            'current_threshold_db': self.threshold_db,
            'base_threshold_db': self.base_threshold_db,
            'threshold_level': self.current_threshold_level,
            'threshold_limits': {
                'min_db': self.min_threshold_db,
                'max_db': self.max_threshold_db
            },
            'environment': {
                'type': self.environment_type,
                'activity_level': self.activity_level,
                'noise_floor': self.noise_floor,
                'noise_floor_db': 20 * np.log10(max(self.noise_floor, 1e-10))
            },
            'performance': {
                'false_positive_rate': self.false_positive_rate,
                'missed_detection_rate': self.missed_detection_rate,
                'total_detections': len(self.detection_performance_history)
            },
            'adaptation_rate': self.adaptation_rate
        }
    
    def get_environment_analysis(self) -> Dict:
        """
        Get detailed environment analysis.
        
        Returns:
            Dictionary with environment analysis results
        """
        if len(self.noise_profile['rms_history']) < 10:
            return {'status': 'insufficient_data'}
        
        recent_rms = list(self.noise_profile['rms_history'])[-100:]
        recent_peaks = list(self.noise_profile['peak_history'])[-100:]
        
        return {
            'environment_type': self.environment_type,
            'activity_level': self.activity_level,
            'noise_statistics': {
                'avg_rms': np.mean(recent_rms),
                'rms_std': np.std(recent_rms),
                'avg_peak': np.mean(recent_peaks),
                'peak_to_rms_ratio': np.mean(recent_peaks) / max(np.mean(recent_rms), 1e-10),
                'samples_analyzed': len(recent_rms)
            },
            'noise_floor_history': {
                'current': self.noise_floor,
                'min': np.min(list(self.noise_history)[-100:]) if self.noise_history else 0,
                'max': np.max(list(self.noise_history)[-100:]) if self.noise_history else 0,
                'trend': self._calculate_noise_trend()
            }
        }
    
    def _calculate_noise_trend(self) -> str:
        """
        Calculate noise floor trend over recent history.
        
        Returns:
            Trend description ('increasing', 'decreasing', 'stable')
        """
        if len(self.noise_history) < 20:
            return 'insufficient_data'
        
        recent_noise = list(self.noise_history)[-20:]
        first_half = np.mean(recent_noise[:10])
        second_half = np.mean(recent_noise[10:])
        
        change_ratio = (second_half - first_half) / max(first_half, 1e-10)
        
        if change_ratio > 0.2:
            return 'increasing'
        elif change_ratio < -0.2:
            return 'decreasing'
        else:
            return 'stable'
    
    def reset_adaptive_state(self) -> None:
        """Reset adaptive threshold state and learning history."""
        self.noise_floor = 0.001
        self.noise_history.clear()
        self.noise_profile['rms_history'].clear()
        self.noise_profile['peak_history'].clear()
        self.noise_profile['spectral_history'].clear()
        
        self.environment_type = 'unknown'
        self.activity_level = 'normal'
        self.false_positive_rate = 0.0
        self.missed_detection_rate = 0.0
        self.detection_performance_history.clear()
        
        # Reset to base threshold
        self.threshold_db = self.base_threshold_db
        self.threshold_linear = 10 ** (self.threshold_db / 20.0)
        self.current_threshold_level = 'normal'
        
        self.logger.info("Adaptive threshold state reset")
    
    def calibrate_for_environment(self, calibration_duration: float = 30.0) -> Dict:
        """
        Perform environment calibration for optimal threshold setting.
        
        Args:
            calibration_duration: Duration of calibration in seconds
            
        Returns:
            Calibration results
        """
        self.logger.info(f"Starting environment calibration for {calibration_duration}s")
        
        # Reset state for clean calibration
        self.reset_adaptive_state()
        
        calibration_start = time.time()
        noise_samples = []
        
        # Note: This method would typically be called with real audio data
        # For now, we'll return the calibration framework
        
        return {
            'status': 'calibration_framework_ready',
            'duration': calibration_duration,
            'instructions': [
                'Feed audio data to detector during calibration period',
                'Ensure representative background noise is present',
                'Avoid gunshots or loud transients during calibration',
                'Call get_adaptive_threshold_status() after calibration'
            ],
            'recommended_threshold_adjustment': 0.0
        }
    
    def set_frequency_analysis_enabled(self, enabled: bool = True) -> None:
        """
        Enable or disable frequency domain analysis.
        
        Args:
            enabled: True to enable frequency analysis
        """
        self.enable_frequency_analysis = enabled
        self.logger.info(f"Frequency analysis {'enabled' if enabled else 'disabled'}")
    
    def configure_frequency_bands(self, frequency_bands: Dict[str, Tuple[float, float]]) -> None:
        """
        Configure frequency bands for analysis.
        
        Args:
            frequency_bands: Dictionary mapping band names to (low_freq, high_freq) tuples
        """
        self.frequency_bands = frequency_bands.copy()
        self.logger.info(f"Frequency bands updated: {frequency_bands}")
    
    def set_gunshot_signature(self, signature: Dict[str, float]) -> None:
        """
        Set the expected gunshot frequency signature.
        
        Args:
            signature: Dictionary mapping band names to expected energy ratios
        """
        # Normalize signature to sum to 1.0
        total_energy = sum(signature.values())
        if total_energy > 0:
            self.gunshot_signature = {band: energy/total_energy for band, energy in signature.items()}
        else:
            self.gunshot_signature = signature.copy()
        
        self.logger.info(f"Gunshot signature updated: {self.gunshot_signature}")
    
    def get_frequency_statistics(self) -> Dict:
        """
        Get frequency analysis statistics.
        
        Returns:
            Dictionary with frequency analysis statistics
        """
        if not self.frequency_history:
            return {
                'frequency_analysis_enabled': self.enable_frequency_analysis,
                'samples_analyzed': 0,
                'avg_gunshot_similarity': 0.0,
                'avg_spectral_centroid': 0.0,
                'frequency_bands': self.frequency_bands
            }
        
        recent_profiles = list(self.frequency_history)
        
        # Calculate average similarity scores
        similarities = []
        centroids = []
        
        for profile in recent_profiles:
            if 'gunshot_similarity' in profile:
                similarities.append(profile['gunshot_similarity'])
            if 'spectral_centroid' in profile:
                centroids.append(profile['spectral_centroid'])
        
        return {
            'frequency_analysis_enabled': self.enable_frequency_analysis,
            'samples_analyzed': len(recent_profiles),
            'avg_gunshot_similarity': np.mean(similarities) if similarities else 0.0,
            'avg_spectral_centroid': np.mean(centroids) if centroids else 0.0,
            'frequency_bands': self.frequency_bands
        }
    
    def analyze_frequency_profile(self, audio_data: np.ndarray, channel: int = 0) -> Dict:
        """
        Analyze frequency profile of audio data for diagnostic purposes.
        
        Args:
            audio_data: Multi-channel audio data
            channel: Channel to analyze (0-based)
            
        Returns:
            Detailed frequency analysis results
        """
        if channel >= audio_data.shape[1]:
            raise ValueError(f"Channel {channel} not available (only {audio_data.shape[1]} channels)")
        
        channel_data = audio_data[:, channel]
        
        if len(channel_data) < self.fft_window_size:
            return {'error': 'Insufficient data for frequency analysis'}
        
        # Perform frequency analysis
        frequency_result = self._analyze_channel_frequency(channel_data, channel)
        
        # Store in history for statistics
        self.frequency_history.append(frequency_result)
        
        return frequency_result