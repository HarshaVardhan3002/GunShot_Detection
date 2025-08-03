"""
Signal intensity filtering and channel weighting module.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
from scipy import signal


@dataclass
class ChannelQualityMetrics:
    """Container for channel quality metrics."""
    rms_amplitude: float
    snr_db: float
    signal_power: float
    noise_power: float
    peak_amplitude: float
    dynamic_range: float
    spectral_centroid: float
    zero_crossing_rate: float
    weight: float
    is_valid: bool


class IntensityFilterInterface(ABC):
    """Abstract interface for intensity-based filtering."""
    
    @abstractmethod
    def calculate_weights(self, audio_channels: np.ndarray) -> np.ndarray:
        """
        Calculate channel weights based on signal intensity.
        
        Args:
            audio_channels: Multi-channel audio data (samples, channels)
            
        Returns:
            Weight array for each channel
        """
        pass
    
    @abstractmethod
    def filter_low_snr_channels(self, weights: np.ndarray, threshold: float = 0.3) -> List[int]:
        """
        Filter out channels with low signal-to-noise ratio.
        
        Args:
            weights: Channel weight array
            threshold: Minimum weight threshold
            
        Returns:
            List of valid channel indices
        """
        pass
    
    @abstractmethod
    def estimate_noise_floor(self, audio_data: np.ndarray) -> float:
        """Estimate background noise floor."""
        pass


class RMSIntensityFilter(IntensityFilterInterface):
    """Intensity filter based on RMS amplitude and signal quality metrics."""
    
    def __init__(self, sample_rate: int = 48000, noise_estimation_method: str = 'percentile'):
        """
        Initialize RMS intensity filter.
        
        Args:
            sample_rate: Audio sampling rate in Hz
            noise_estimation_method: Method for noise floor estimation ('percentile', 'minimum', 'adaptive')
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.noise_estimation_method = noise_estimation_method
        
        # Filter parameters
        self.min_snr_db = 6.0  # Minimum SNR for valid channel (6dB)
        self.min_rms_threshold = 1e-6  # Minimum RMS amplitude
        self.noise_percentile = 10  # Percentile for noise floor estimation
        self.spectral_analysis_enabled = True
        
        # Weighting parameters
        self.rms_weight = 0.4      # Weight for RMS amplitude
        self.snr_weight = 0.3      # Weight for SNR
        self.spectral_weight = 0.2  # Weight for spectral features
        self.dynamic_weight = 0.1   # Weight for dynamic range
        
        # History tracking
        self.quality_history = deque(maxlen=100)
        self.noise_floor_history = deque(maxlen=50)
        
        # Adaptive parameters
        self.enable_adaptive_thresholds = True
        self.adaptation_rate = 0.1
        
        self.logger.info(f"RMS Intensity Filter initialized: {sample_rate}Hz, method={noise_estimation_method}")
    
    def calculate_weights(self, audio_channels: np.ndarray) -> np.ndarray:
        """
        Calculate channel weights based on comprehensive signal quality metrics.
        
        Args:
            audio_channels: Multi-channel audio data (samples, channels)
            
        Returns:
            Weight array for each channel (0-1, higher is better)
        """
        if len(audio_channels.shape) != 2:
            raise ValueError("Audio channels must be 2D array (samples, channels)")
        
        num_samples, num_channels = audio_channels.shape
        
        if num_samples < 10:
            self.logger.warning(f"Very short audio segment: {num_samples} samples")
            return np.ones(num_channels) * 0.5  # Default moderate weights
        
        # Calculate quality metrics for each channel
        channel_metrics = []
        for ch in range(num_channels):
            metrics = self._calculate_channel_quality_metrics(audio_channels[:, ch])
            channel_metrics.append(metrics)
        
        # Calculate weights based on metrics
        weights = self._calculate_weights_from_metrics(channel_metrics)
        
        # Apply adaptive adjustments if enabled
        if self.enable_adaptive_thresholds:
            weights = self._apply_adaptive_adjustments(weights, channel_metrics)
        
        # Store metrics for analysis
        self.quality_history.append(channel_metrics)
        
        # Log quality summary
        avg_snr = np.mean([m.snr_db for m in channel_metrics])
        valid_channels = sum(1 for m in channel_metrics if m.is_valid)
        self.logger.debug(f"Channel quality: avg_snr={avg_snr:.1f}dB, valid={valid_channels}/{num_channels}")
        
        return weights
    
    def _calculate_channel_quality_metrics(self, channel_data: np.ndarray) -> ChannelQualityMetrics:
        """
        Calculate comprehensive quality metrics for a single channel.
        
        Args:
            channel_data: Single channel audio data
            
        Returns:
            ChannelQualityMetrics object with all calculated metrics
        """
        # Basic amplitude metrics
        rms_amplitude = np.sqrt(np.mean(channel_data ** 2))
        peak_amplitude = np.max(np.abs(channel_data))
        
        # Signal and noise power estimation
        signal_power, noise_power = self._estimate_signal_noise_power(channel_data)
        
        # SNR calculation
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(max(snr_linear, 1e-10))
        else:
            snr_db = 60.0  # Very high SNR if no noise detected
        
        # Dynamic range
        if peak_amplitude > 0:
            dynamic_range = 20 * np.log10(peak_amplitude / max(rms_amplitude, 1e-10))
        else:
            dynamic_range = 0.0
        
        # Spectral features (if enabled)
        if self.spectral_analysis_enabled and len(channel_data) > 64:
            spectral_centroid = self._calculate_spectral_centroid(channel_data)
            zero_crossing_rate = self._calculate_zero_crossing_rate(channel_data)
        else:
            spectral_centroid = 1000.0  # Default 1kHz
            zero_crossing_rate = 0.1    # Default moderate rate
        
        # Validity check
        is_valid = (rms_amplitude >= self.min_rms_threshold and 
                   snr_db >= self.min_snr_db and
                   not np.isnan(rms_amplitude) and
                   not np.isinf(rms_amplitude))
        
        return ChannelQualityMetrics(
            rms_amplitude=rms_amplitude,
            snr_db=snr_db,
            signal_power=signal_power,
            noise_power=noise_power,
            peak_amplitude=peak_amplitude,
            dynamic_range=dynamic_range,
            spectral_centroid=spectral_centroid,
            zero_crossing_rate=zero_crossing_rate,
            weight=0.0,  # Will be calculated later
            is_valid=is_valid
        )
    
    def _estimate_signal_noise_power(self, channel_data: np.ndarray) -> Tuple[float, float]:
        """
        Estimate signal and noise power for a channel.
        
        Args:
            channel_data: Single channel audio data
            
        Returns:
            Tuple of (signal_power, noise_power)
        """
        # Calculate overall power
        total_power = np.mean(channel_data ** 2)
        
        if self.noise_estimation_method == 'percentile':
            # Use percentile-based noise estimation
            noise_floor = self.estimate_noise_floor(channel_data)
            noise_power = noise_floor ** 2
            signal_power = max(0, total_power - noise_power)
            
        elif self.noise_estimation_method == 'minimum':
            # Use minimum segments for noise estimation
            segment_size = min(len(channel_data) // 10, 480)  # 10ms segments at 48kHz
            if segment_size > 10:
                segments = []
                for i in range(0, len(channel_data) - segment_size, segment_size):
                    segment_power = np.mean(channel_data[i:i+segment_size] ** 2)
                    segments.append(segment_power)
                
                noise_power = np.min(segments) if segments else total_power * 0.1
                signal_power = max(0, total_power - noise_power)
            else:
                noise_power = total_power * 0.1
                signal_power = total_power * 0.9
                
        elif self.noise_estimation_method == 'adaptive':
            # Use adaptive noise estimation based on signal statistics
            # Assume first and last 10% are more likely to contain noise
            noise_samples = int(len(channel_data) * 0.1)
            if noise_samples > 10:
                noise_start = channel_data[:noise_samples]
                noise_end = channel_data[-noise_samples:]
                noise_data = np.concatenate([noise_start, noise_end])
                noise_power = np.mean(noise_data ** 2)
                signal_power = max(0, total_power - noise_power)
            else:
                noise_power = total_power * 0.1
                signal_power = total_power * 0.9
        else:
            # Default: assume 10% noise, 90% signal
            noise_power = total_power * 0.1
            signal_power = total_power * 0.9
        
        return signal_power, noise_power
    
    def _calculate_spectral_centroid(self, channel_data: np.ndarray) -> float:
        """
        Calculate spectral centroid (brightness measure).
        
        Args:
            channel_data: Single channel audio data
            
        Returns:
            Spectral centroid in Hz
        """
        # Apply window to reduce spectral leakage
        windowed_data = channel_data * signal.windows.hann(len(channel_data))
        
        # Calculate FFT
        fft = np.fft.fft(windowed_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(channel_data), 1/self.sample_rate)[:len(magnitude)]
        
        # Calculate spectral centroid
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 1000.0  # Default 1kHz
        
        return max(0, spectral_centroid)
    
    def _calculate_zero_crossing_rate(self, channel_data: np.ndarray) -> float:
        """
        Calculate zero crossing rate (measure of signal noisiness).
        
        Args:
            channel_data: Single channel audio data
            
        Returns:
            Zero crossing rate (0-1)
        """
        if len(channel_data) < 2:
            return 0.0
        
        # Count sign changes
        sign_changes = np.sum(np.diff(np.sign(channel_data)) != 0)
        zcr = sign_changes / (len(channel_data) - 1)
        
        return min(1.0, zcr)
    
    def _calculate_weights_from_metrics(self, channel_metrics: List[ChannelQualityMetrics]) -> np.ndarray:
        """
        Calculate channel weights from quality metrics.
        
        Args:
            channel_metrics: List of quality metrics for each channel
            
        Returns:
            Normalized weight array
        """
        num_channels = len(channel_metrics)
        weights = np.zeros(num_channels)
        
        # Extract metrics for normalization
        rms_values = np.array([m.rms_amplitude for m in channel_metrics])
        snr_values = np.array([m.snr_db for m in channel_metrics])
        spectral_values = np.array([m.spectral_centroid for m in channel_metrics])
        dynamic_values = np.array([m.dynamic_range for m in channel_metrics])
        
        # Normalize metrics to 0-1 range
        rms_norm = self._normalize_metric(rms_values, 'rms')
        snr_norm = self._normalize_metric(snr_values, 'snr')
        spectral_norm = self._normalize_metric(spectral_values, 'spectral')
        dynamic_norm = self._normalize_metric(dynamic_values, 'dynamic')
        
        # Calculate weighted combination
        for i, metrics in enumerate(channel_metrics):
            if metrics.is_valid:
                weight = (self.rms_weight * rms_norm[i] +
                         self.snr_weight * snr_norm[i] +
                         self.spectral_weight * spectral_norm[i] +
                         self.dynamic_weight * dynamic_norm[i])
                weights[i] = max(0.0, min(1.0, weight))
            else:
                weights[i] = 0.0
            
            # Update metrics with calculated weight
            channel_metrics[i].weight = weights[i]
        
        # Ensure at least one channel has non-zero weight
        if np.sum(weights) == 0:
            self.logger.warning("All channels have zero weight, using uniform weights")
            weights = np.ones(num_channels) / num_channels
        else:
            # Normalize weights to sum to number of channels (for averaging)
            weights = weights * num_channels / np.sum(weights)
        
        return weights
    
    def _normalize_metric(self, values: np.ndarray, metric_type: str) -> np.ndarray:
        """
        Normalize metric values to 0-1 range.
        
        Args:
            values: Array of metric values
            metric_type: Type of metric for appropriate normalization
            
        Returns:
            Normalized values (0-1)
        """
        if len(values) == 0:
            return np.array([])
        
        if metric_type == 'rms':
            # RMS: use log scale for better dynamic range
            log_values = np.log10(np.maximum(values, 1e-10))
            min_log, max_log = np.min(log_values), np.max(log_values)
            if max_log > min_log:
                normalized = (log_values - min_log) / (max_log - min_log)
            else:
                normalized = np.ones_like(values) * 0.5
                
        elif metric_type == 'snr':
            # SNR: sigmoid-like normalization
            # Good SNR is > 10dB, excellent is > 20dB
            normalized = 1.0 / (1.0 + np.exp(-(values - 10) / 5))
            
        elif metric_type == 'spectral':
            # Spectral centroid: prefer moderate values (500-2000 Hz for gunshots)
            target_freq = 1000.0
            deviation = np.abs(values - target_freq) / target_freq
            normalized = np.exp(-deviation)
            
        elif metric_type == 'dynamic':
            # Dynamic range: higher is generally better
            normalized = 1.0 / (1.0 + np.exp(-(values - 10) / 5))
            
        else:
            # Default: min-max normalization
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
            else:
                normalized = np.ones_like(values) * 0.5
        
        return np.clip(normalized, 0.0, 1.0)
    
    def _apply_adaptive_adjustments(self, weights: np.ndarray, 
                                  channel_metrics: List[ChannelQualityMetrics]) -> np.ndarray:
        """
        Apply adaptive adjustments to weights based on historical performance.
        
        Args:
            weights: Current channel weights
            channel_metrics: Current channel quality metrics
            
        Returns:
            Adjusted weights
        """
        if len(self.quality_history) < 5:
            return weights  # Need more history for adaptation
        
        adjusted_weights = weights.copy()
        
        # Calculate historical performance for each channel
        for ch in range(len(weights)):
            # Get historical SNR values for this channel
            historical_snr = []
            for history_entry in list(self.quality_history)[-10:]:  # Last 10 entries
                if ch < len(history_entry):
                    historical_snr.append(history_entry[ch].snr_db)
            
            if len(historical_snr) >= 3:
                # Calculate trend
                recent_snr = np.mean(historical_snr[-3:])
                older_snr = np.mean(historical_snr[:-3])
                
                # Adjust weight based on trend
                if recent_snr > older_snr + 2:  # Improving
                    adjustment = 1.0 + self.adaptation_rate
                elif recent_snr < older_snr - 2:  # Degrading
                    adjustment = 1.0 - self.adaptation_rate
                else:
                    adjustment = 1.0
                
                adjusted_weights[ch] *= adjustment
        
        # Re-normalize
        if np.sum(adjusted_weights) > 0:
            adjusted_weights = adjusted_weights * len(weights) / np.sum(adjusted_weights)
        
        return np.clip(adjusted_weights, 0.0, 2.0)  # Limit maximum weight
    
    def filter_low_snr_channels(self, weights: np.ndarray, threshold: float = 0.3) -> List[int]:
        """
        Filter out channels with low signal-to-noise ratio.
        
        Args:
            weights: Channel weight array
            threshold: Minimum weight threshold
            
        Returns:
            List of valid channel indices
        """
        valid_channels = []
        
        for ch, weight in enumerate(weights):
            if weight >= threshold:
                valid_channels.append(ch)
        
        # Ensure we have at least one channel
        if len(valid_channels) == 0 and len(weights) > 0:
            # Use the best channel even if below threshold
            best_channel = np.argmax(weights)
            valid_channels = [best_channel]
            self.logger.warning(f"All channels below threshold, using best channel {best_channel}")
        
        self.logger.debug(f"Filtered channels: {len(valid_channels)}/{len(weights)} valid")
        return valid_channels
    
    def estimate_noise_floor(self, audio_data: np.ndarray) -> float:
        """
        Estimate background noise floor.
        
        Args:
            audio_data: Audio data for noise estimation
            
        Returns:
            Estimated noise floor (RMS amplitude)
        """
        if len(audio_data) == 0:
            return 1e-6
        
        if self.noise_estimation_method == 'percentile':
            # Use percentile of absolute values
            abs_values = np.abs(audio_data)
            noise_floor = np.percentile(abs_values, self.noise_percentile)
            
        elif self.noise_estimation_method == 'minimum':
            # Use minimum RMS of segments
            segment_size = min(len(audio_data) // 20, 240)  # 5ms segments
            if segment_size > 5:
                segment_rms = []
                for i in range(0, len(audio_data) - segment_size, segment_size):
                    segment = audio_data[i:i+segment_size]
                    rms = np.sqrt(np.mean(segment ** 2))
                    segment_rms.append(rms)
                
                noise_floor = np.min(segment_rms) if segment_rms else np.sqrt(np.mean(audio_data ** 2)) * 0.1
            else:
                noise_floor = np.sqrt(np.mean(audio_data ** 2)) * 0.1
                
        elif self.noise_estimation_method == 'adaptive':
            # Adaptive estimation based on signal distribution
            sorted_abs = np.sort(np.abs(audio_data))
            # Use lower quartile as noise estimate
            noise_floor = sorted_abs[len(sorted_abs) // 4]
            
        else:
            # Default: use 10th percentile
            noise_floor = np.percentile(np.abs(audio_data), 10)
        
        # Store in history for trend analysis
        self.noise_floor_history.append(noise_floor)
        
        return max(noise_floor, 1e-10)  # Avoid zero noise floor
    
    def get_channel_quality_report(self) -> Dict:
        """
        Get comprehensive channel quality report.
        
        Returns:
            Dictionary with quality statistics and recommendations
        """
        if not self.quality_history:
            return {'status': 'no_data', 'message': 'No quality data available'}
        
        latest_metrics = self.quality_history[-1]
        num_channels = len(latest_metrics)
        
        # Calculate statistics
        valid_channels = sum(1 for m in latest_metrics if m.is_valid)
        avg_snr = np.mean([m.snr_db for m in latest_metrics])
        avg_rms = np.mean([m.rms_amplitude for m in latest_metrics])
        avg_weight = np.mean([m.weight for m in latest_metrics])
        
        # Channel-specific details
        channel_details = []
        for i, metrics in enumerate(latest_metrics):
            channel_details.append({
                'channel': i,
                'rms_amplitude': metrics.rms_amplitude,
                'snr_db': metrics.snr_db,
                'weight': metrics.weight,
                'is_valid': metrics.is_valid,
                'spectral_centroid': metrics.spectral_centroid,
                'dynamic_range': metrics.dynamic_range
            })
        
        # Overall assessment
        if valid_channels == 0:
            quality_status = 'poor'
            recommendation = 'Check microphone connections and signal levels'
        elif valid_channels < num_channels * 0.5:
            quality_status = 'marginal'
            recommendation = 'Some microphones may have issues'
        elif avg_snr < 10:
            quality_status = 'fair'
            recommendation = 'Consider noise reduction or better microphone placement'
        elif avg_snr > 20:
            quality_status = 'excellent'
            recommendation = 'Signal quality is optimal'
        else:
            quality_status = 'good'
            recommendation = 'Signal quality is acceptable'
        
        return {
            'status': quality_status,
            'recommendation': recommendation,
            'summary': {
                'total_channels': num_channels,
                'valid_channels': valid_channels,
                'avg_snr_db': avg_snr,
                'avg_rms_amplitude': avg_rms,
                'avg_weight': avg_weight
            },
            'channels': channel_details,
            'noise_floor_trend': list(self.noise_floor_history)[-10:] if self.noise_floor_history else []
        }
    
    def configure_filter_parameters(self, **kwargs) -> None:
        """
        Configure filter parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'min_snr_db' in kwargs:
            self.min_snr_db = float(kwargs['min_snr_db'])
        
        if 'min_rms_threshold' in kwargs:
            self.min_rms_threshold = float(kwargs['min_rms_threshold'])
        
        if 'noise_estimation_method' in kwargs:
            if kwargs['noise_estimation_method'] in ['percentile', 'minimum', 'adaptive']:
                self.noise_estimation_method = kwargs['noise_estimation_method']
            else:
                raise ValueError("Invalid noise estimation method")
        
        if 'spectral_analysis_enabled' in kwargs:
            self.spectral_analysis_enabled = bool(kwargs['spectral_analysis_enabled'])
        
        if 'enable_adaptive_thresholds' in kwargs:
            self.enable_adaptive_thresholds = bool(kwargs['enable_adaptive_thresholds'])
        
        # Weight parameters
        weight_params = ['rms_weight', 'snr_weight', 'spectral_weight', 'dynamic_weight']
        for param in weight_params:
            if param in kwargs:
                setattr(self, param, float(kwargs[param]))
        
        # Normalize weights to sum to 1.0
        total_weight = self.rms_weight + self.snr_weight + self.spectral_weight + self.dynamic_weight
        if total_weight > 0:
            self.rms_weight /= total_weight
            self.snr_weight /= total_weight
            self.spectral_weight /= total_weight
            self.dynamic_weight /= total_weight
        
        self.logger.info(f"Filter parameters updated: {kwargs}")
    
    def reset_history(self) -> None:
        """Reset quality and noise floor history."""
        self.quality_history.clear()
        self.noise_floor_history.clear()
        self.logger.info("Filter history reset")