"""
Time Difference of Arrival (TDoA) localization module.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import time
import logging
from collections import deque
from scipy import signal
from scipy.optimize import least_squares


@dataclass
class LocationResult:
    """Container for triangulation results."""
    x: float
    y: float
    confidence: float
    residual_error: float
    timestamp: float
    microphones_used: List[int]
    z: float = 0.0
    tdoa_matrix: Optional[np.ndarray] = None
    correlation_peaks: Optional[Dict[str, float]] = None


@dataclass
class MicrophonePosition:
    """Container for microphone position data."""
    id: int
    x: float
    y: float
    z: float = 0.0


class TDoALocalizerInterface(ABC):
    """Abstract interface for TDoA localization."""
    
    @abstractmethod
    def calculate_tdoa(self, audio_channels: np.ndarray) -> np.ndarray:
        """
        Calculate Time Difference of Arrival between microphone pairs.
        
        Args:
            audio_channels: Multi-channel audio data (samples, channels)
            
        Returns:
            TDoA matrix for all microphone pairs
        """
        pass
    
    @abstractmethod
    def triangulate_source(self, tdoa_matrix: np.ndarray) -> LocationResult:
        """
        Triangulate source location from TDoA data.
        
        Args:
            tdoa_matrix: Time differences between microphone pairs
            
        Returns:
            LocationResult with estimated coordinates and confidence
        """
        pass
    
    @abstractmethod
    def estimate_confidence(self, residuals: np.ndarray) -> float:
        """Estimate confidence in triangulation result."""
        pass


class CrossCorrelationTDoALocalizer(TDoALocalizerInterface):
    """TDoA localization using cross-correlation analysis."""
    
    def __init__(self, microphone_positions: List[MicrophonePosition], 
                 sample_rate: int = 48000, sound_speed: float = 343.0):
        """
        Initialize cross-correlation TDoA localizer.
        
        Args:
            microphone_positions: List of microphone positions
            sample_rate: Audio sampling rate in Hz
            sound_speed: Speed of sound in m/s
        """
        self.logger = logging.getLogger(__name__)
        self.microphone_positions = microphone_positions
        self.sample_rate = sample_rate
        self.sound_speed = sound_speed
        self.num_mics = len(microphone_positions)
        
        if self.num_mics < 3:
            raise ValueError("At least 3 microphones required for localization")
        
        # Cross-correlation parameters
        self.correlation_method = 'fft'  # 'fft' or 'direct'
        self.max_delay_samples = int(0.1 * sample_rate)  # Maximum 100ms delay
        self.interpolation_factor = 4  # Sub-sample interpolation factor
        self.window_function = 'hann'  # Window function for correlation
        
        # Signal preprocessing
        self.enable_preprocessing = True
        self.bandpass_filter = None  # Will be initialized on first use
        self.filter_low_freq = 100   # Hz
        self.filter_high_freq = 8000 # Hz
        
        # Quality control
        self.min_correlation_threshold = 0.3  # Minimum correlation for valid TDoA
        self.max_tdoa_seconds = 0.05  # Maximum reasonable TDoA (50ms)
        
        # Performance tracking
        self.correlation_history = deque(maxlen=100)
        self.tdoa_history = deque(maxlen=100)
        
        # Create microphone pair combinations
        self.mic_pairs = []
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                self.mic_pairs.append((i, j))
        
        self.logger.info(f"TDoA localizer initialized: {self.num_mics} mics, {len(self.mic_pairs)} pairs")
    
    def calculate_tdoa(self, audio_channels: np.ndarray) -> np.ndarray:
        """
        Calculate Time Difference of Arrival using cross-correlation.
        
        Args:
            audio_channels: Multi-channel audio data (samples, channels)
            
        Returns:
            TDoA matrix where tdoa_matrix[i,j] = time delay from mic i to mic j
        """
        if audio_channels.shape[1] != self.num_mics:
            raise ValueError(f"Expected {self.num_mics} channels, got {audio_channels.shape[1]}")
        
        # Initialize TDoA matrix
        tdoa_matrix = np.zeros((self.num_mics, self.num_mics))
        correlation_peaks = {}
        
        # Preprocess signals if enabled
        if self.enable_preprocessing:
            processed_channels = self._preprocess_signals(audio_channels)
        else:
            processed_channels = audio_channels
        
        # Calculate TDoA for each microphone pair
        for i, j in self.mic_pairs:
            signal_i = processed_channels[:, i]
            signal_j = processed_channels[:, j]
            
            # Calculate cross-correlation
            tdoa_seconds, correlation_peak = self._cross_correlate_signals(signal_i, signal_j)
            
            # Store results
            tdoa_matrix[i, j] = tdoa_seconds
            tdoa_matrix[j, i] = -tdoa_seconds  # Symmetric matrix
            correlation_peaks[f"{i}-{j}"] = correlation_peak
            
            # Quality check
            if abs(tdoa_seconds) > self.max_tdoa_seconds:
                self.logger.warning(f"Large TDoA detected for mics {i}-{j}: {tdoa_seconds*1000:.1f}ms")
            
            if correlation_peak < self.min_correlation_threshold:
                self.logger.warning(f"Low correlation for mics {i}-{j}: {correlation_peak:.3f}")
        
        # Store for analysis
        self.correlation_history.append(correlation_peaks)
        self.tdoa_history.append(tdoa_matrix.copy())
        
        return tdoa_matrix
    
    def _preprocess_signals(self, audio_channels: np.ndarray) -> np.ndarray:
        """
        Preprocess audio signals for better correlation.
        
        Args:
            audio_channels: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Initialize bandpass filter if needed
        if self.bandpass_filter is None:
            nyquist = self.sample_rate / 2
            low_norm = max(0.01, self.filter_low_freq / nyquist)  # Ensure > 0
            high_norm = min(0.99, self.filter_high_freq / nyquist)  # Ensure < 1
            
            # Only create filter if we have valid frequency range
            if low_norm < high_norm:
                # Design Butterworth bandpass filter
                self.bandpass_filter = signal.butter(4, [low_norm, high_norm], btype='band')
            else:
                # Skip filtering if invalid range
                self.bandpass_filter = None
        
        # Apply bandpass filter to each channel if filter is valid
        if self.bandpass_filter is not None:
            filtered_channels = np.zeros_like(audio_channels)
            for ch in range(audio_channels.shape[1]):
                filtered_channels[:, ch] = signal.filtfilt(
                    self.bandpass_filter[0], self.bandpass_filter[1], audio_channels[:, ch]
                )
        else:
            # No filtering
            filtered_channels = audio_channels.copy()
        
        # Apply window function to reduce edge effects
        if self.window_function and len(filtered_channels) > 100:
            window = signal.get_window(self.window_function, len(filtered_channels))
            filtered_channels = filtered_channels * window[:, np.newaxis]
        
        return filtered_channels
    
    def _cross_correlate_signals(self, signal1: np.ndarray, signal2: np.ndarray) -> Tuple[float, float]:
        """
        Cross-correlate two signals to find time delay.
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            Tuple of (time_delay_seconds, correlation_peak_value)
        """
        # Ensure signals have the same length
        min_length = min(len(signal1), len(signal2))
        signal1 = signal1[:min_length]
        signal2 = signal2[:min_length]
        
        if self.correlation_method == 'fft':
            # FFT-based cross-correlation (faster for long signals)
            correlation = signal.correlate(signal1, signal2, mode='full')
        else:
            # Direct cross-correlation
            correlation = np.correlate(signal1, signal2, mode='full')
        
        # Find peak correlation
        peak_index = np.argmax(np.abs(correlation))
        peak_value = correlation[peak_index]
        
        # Convert to normalized correlation
        signal1_energy = np.sum(signal1 ** 2)
        signal2_energy = np.sum(signal2 ** 2)
        
        if signal1_energy > 0 and signal2_energy > 0:
            normalized_peak = abs(peak_value) / np.sqrt(signal1_energy * signal2_energy)
        else:
            normalized_peak = 0.0
        
        # Calculate time delay
        # For full correlation, the zero-lag is at index len(signal2)-1
        zero_lag_index = len(signal2) - 1
        delay_samples = peak_index - zero_lag_index
        
        # Apply sub-sample interpolation for higher precision
        if self.interpolation_factor > 1 and len(correlation) > 2:
            interpolated_delay = self._interpolate_peak(correlation, peak_index)
            # Only use interpolation if it's reasonable
            if abs(interpolated_delay - delay_samples) < 2:
                delay_samples = interpolated_delay
        
        # Limit delay to reasonable range
        max_delay_samples = min(self.max_delay_samples, len(signal1) // 2)
        delay_samples = np.clip(delay_samples, -max_delay_samples, max_delay_samples)
        
        # Convert to time
        time_delay = delay_samples / self.sample_rate
        
        return time_delay, normalized_peak
    
    def _interpolate_peak(self, correlation: np.ndarray, peak_index: int) -> float:
        """
        Interpolate correlation peak for sub-sample precision.
        
        Args:
            correlation: Cross-correlation result
            peak_index: Index of correlation peak
            
        Returns:
            Interpolated peak position
        """
        if peak_index <= 0 or peak_index >= len(correlation) - 1:
            return float(peak_index)
        
        # Use parabolic interpolation
        y1 = abs(correlation[peak_index - 1])
        y2 = abs(correlation[peak_index])
        y3 = abs(correlation[peak_index + 1])
        
        # Parabolic interpolation formula
        if y1 + y3 - 2 * y2 != 0:
            offset = 0.5 * (y1 - y3) / (y1 + y3 - 2 * y2)
            interpolated_peak = peak_index + offset
        else:
            interpolated_peak = float(peak_index)
        
        return interpolated_peak
    
    def triangulate_source(self, tdoa_matrix: np.ndarray) -> LocationResult:
        """
        Triangulate source location from TDoA matrix using multilateration.
        
        Args:
            tdoa_matrix: Time differences between microphone pairs
            
        Returns:
            LocationResult with estimated coordinates and confidence
        """
        try:
            # Validate input
            if tdoa_matrix.shape != (self.num_mics, self.num_mics):
                raise ValueError(f"TDoA matrix shape {tdoa_matrix.shape} doesn't match {self.num_mics} microphones")
            
            # Select best microphones for triangulation
            selected_mics, quality_scores = self._select_microphones_for_triangulation(tdoa_matrix)
            
            # Enhanced validation and error handling
            validation_result = self._validate_tdoa_matrix(tdoa_matrix)
            if not validation_result['valid']:
                self.logger.warning(f"TDoA matrix validation failed: {validation_result['reason']}")
                return self._create_failed_result(tdoa_matrix, f"tdoa_validation: {validation_result['reason']}")
            
            # Check minimum microphone requirements
            min_mics_result = self._check_minimum_microphone_requirements(selected_mics, quality_scores)
            if not min_mics_result['sufficient']:
                self.logger.warning(f"Insufficient microphones: {min_mics_result['reason']}")
                return self._create_failed_result(tdoa_matrix, f"insufficient_mics: {min_mics_result['reason']}")
            
            # Perform multilateration with fallback methods
            result = self._solve_multilateration_with_fallbacks(tdoa_matrix, selected_mics)
            
            # Enhanced solution validation
            solution_validation = self._validate_solution_quality(result, selected_mics, quality_scores)
            if not solution_validation['valid']:
                self.logger.warning(f"Solution validation failed: {solution_validation['reason']}")
                return self._create_failed_result(tdoa_matrix, f"solution_validation: {solution_validation['reason']}")
            
            # Calculate enhanced confidence with quality metrics
            confidence = self._calculate_enhanced_confidence(result, quality_scores, solution_validation)
            
            # Create final result
            return LocationResult(
                x=result['position'][0],
                y=result['position'][1],
                z=result['position'][2] if len(result['position']) > 2 else 0.0,
                confidence=confidence,
                residual_error=result['residual_error'],
                timestamp=time.time(),
                microphones_used=selected_mics,
                tdoa_matrix=tdoa_matrix,
                correlation_peaks=self.correlation_history[-1] if self.correlation_history else None
            )
            
        except Exception as e:
            self.logger.error(f"Triangulation failed: {e}")
            return self._create_failed_result(tdoa_matrix, f"error: {str(e)}")
    
    def _select_microphones_for_triangulation(self, tdoa_matrix: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        Select the best microphones for triangulation based on TDoA quality.
        
        Args:
            tdoa_matrix: Time differences between microphone pairs
            
        Returns:
            Tuple of (selected_microphone_indices, quality_scores)
        """
        # Calculate quality score for each microphone based on:
        # 1. Number of valid TDoA measurements
        # 2. Consistency of TDoA values
        # 3. Correlation quality from history
        
        quality_scores = []
        correlation_peaks = self.correlation_history[-1] if self.correlation_history else {}
        
        for mic_id in range(self.num_mics):
            # Count valid TDoA measurements for this microphone
            valid_tdoas = 0
            tdoa_consistency = 0.0
            correlation_quality = 0.0
            
            for other_mic in range(self.num_mics):
                if mic_id != other_mic:
                    tdoa = tdoa_matrix[mic_id, other_mic]
                    
                    # Check if TDoA is reasonable
                    if abs(tdoa) <= self.max_tdoa_seconds:
                        valid_tdoas += 1
                        
                        # Get correlation quality if available
                        pair_key = f"{min(mic_id, other_mic)}-{max(mic_id, other_mic)}"
                        if pair_key in correlation_peaks:
                            correlation_quality += correlation_peaks[pair_key]
            
            # Calculate consistency (lower variance in TDoA values is better)
            mic_tdoas = []
            for other_mic in range(self.num_mics):
                if mic_id != other_mic:
                    tdoa = tdoa_matrix[mic_id, other_mic]
                    if abs(tdoa) <= self.max_tdoa_seconds:
                        mic_tdoas.append(abs(tdoa))
            
            if len(mic_tdoas) > 1:
                tdoa_consistency = 1.0 / (1.0 + np.std(mic_tdoas))
            else:
                tdoa_consistency = 0.5
            
            # Normalize correlation quality
            if valid_tdoas > 0:
                correlation_quality /= valid_tdoas
            
            # Combined quality score
            quality = 0.4 * (valid_tdoas / (self.num_mics - 1)) + \
                     0.3 * tdoa_consistency + \
                     0.3 * correlation_quality
            
            quality_scores.append(quality)
        
        # Select microphones with highest quality scores
        # Need at least 3 for 2D, 4 for 3D triangulation
        min_mics = 4 if self.num_mics >= 4 else 3
        max_mics = min(8, self.num_mics)  # Don't use too many to avoid over-constraint
        
        # Sort by quality and select best ones
        mic_quality_pairs = list(zip(range(self.num_mics), quality_scores))
        mic_quality_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select microphones above quality threshold
        quality_threshold = 0.3
        selected_mics = []
        selected_qualities = []
        
        for mic_id, quality in mic_quality_pairs:
            if quality >= quality_threshold and len(selected_mics) < max_mics:
                selected_mics.append(mic_id)
                selected_qualities.append(quality)
        
        # Ensure minimum number of microphones
        if len(selected_mics) < min_mics:
            # Add more microphones even if below threshold
            for mic_id, quality in mic_quality_pairs:
                if mic_id not in selected_mics and len(selected_mics) < min_mics:
                    selected_mics.append(mic_id)
                    selected_qualities.append(quality)
        
        self.logger.debug(f"Selected {len(selected_mics)} microphones for triangulation: {selected_mics}")
        return selected_mics, selected_qualities
    
    def _solve_multilateration(self, tdoa_matrix: np.ndarray, selected_mics: List[int]) -> Dict:
        """
        Solve multilateration using nonlinear least squares optimization.
        
        Args:
            tdoa_matrix: Time differences between microphone pairs
            selected_mics: List of microphone indices to use
            
        Returns:
            Dictionary with solution details
        """
        # Use first microphone as reference
        ref_mic = selected_mics[0]
        other_mics = selected_mics[1:]
        
        # Get microphone positions
        ref_pos = np.array([self.microphone_positions[ref_mic].x, 
                           self.microphone_positions[ref_mic].y,
                           self.microphone_positions[ref_mic].z])
        
        other_positions = []
        tdoa_measurements = []
        
        for mic_id in other_mics:
            pos = np.array([self.microphone_positions[mic_id].x,
                           self.microphone_positions[mic_id].y, 
                           self.microphone_positions[mic_id].z])
            other_positions.append(pos)
            
            # TDoA from reference microphone to this microphone
            tdoa = tdoa_matrix[ref_mic, mic_id]
            tdoa_measurements.append(tdoa)
        
        other_positions = np.array(other_positions)
        tdoa_measurements = np.array(tdoa_measurements)
        
        # Define objective function for least squares
        def objective_function(source_pos):
            """Calculate residuals for least squares optimization."""
            residuals = []
            
            # Distance from source to reference microphone
            ref_distance = np.linalg.norm(source_pos - ref_pos)
            
            for i, other_pos in enumerate(other_positions):
                # Distance from source to other microphone
                other_distance = np.linalg.norm(source_pos - other_pos)
                
                # Expected TDoA based on current source position
                expected_tdoa = (other_distance - ref_distance) / self.sound_speed
                
                # Residual between expected and measured TDoA
                residual = expected_tdoa - tdoa_measurements[i]
                residuals.append(residual)
            
            return np.array(residuals)
        
        # Initial guess - centroid of microphone array
        all_positions = np.array([[mic.x, mic.y, mic.z] for mic in self.microphone_positions])
        initial_guess = np.mean(all_positions, axis=0)
        
        # Try multiple initial guesses to avoid local minima
        best_result = None
        best_cost = float('inf')
        
        initial_guesses = [
            initial_guess,  # Centroid
            ref_pos + np.array([1.0, 0.0, 0.0]),  # 1m east of reference
            ref_pos + np.array([0.0, 1.0, 0.0]),  # 1m north of reference
            ref_pos + np.array([-1.0, 0.0, 0.0]), # 1m west of reference
            ref_pos + np.array([0.0, -1.0, 0.0])  # 1m south of reference
        ]
        
        for guess in initial_guesses:
            try:
                # Solve using least squares
                result = least_squares(
                    objective_function,
                    guess,
                    method='lm',  # Levenberg-Marquardt algorithm
                    max_nfev=1000,  # Maximum function evaluations
                    ftol=1e-8,      # Function tolerance
                    xtol=1e-8       # Parameter tolerance
                )
                
                if result.success and result.cost < best_cost:
                    best_result = result
                    best_cost = result.cost
                    
            except Exception as e:
                self.logger.debug(f"Optimization failed for guess {guess}: {e}")
                continue
        
        if best_result is None or not best_result.success:
            raise RuntimeError("Multilateration optimization failed to converge")
        
        # Calculate final residual error
        final_residuals = objective_function(best_result.x)
        residual_error = np.sqrt(np.mean(final_residuals ** 2))  # RMS error
        
        return {
            'position': best_result.x,
            'residuals': final_residuals,
            'residual_error': residual_error,
            'optimization_result': best_result,
            'cost': best_result.cost,
            'iterations': best_result.nfev
        }
    
    def _validate_geometric_constraints(self, solution: Dict, selected_mics: List[int]) -> bool:
        """
        Validate that the solution satisfies geometric constraints.
        
        Args:
            solution: Solution dictionary from multilateration
            selected_mics: Microphones used in solution
            
        Returns:
            True if solution is geometrically valid
        """
        source_pos = solution['position']
        
        # Check if solution is within reasonable bounds
        # Calculate bounding box of microphone array
        mic_positions = np.array([[self.microphone_positions[i].x, 
                                  self.microphone_positions[i].y,
                                  self.microphone_positions[i].z] for i in selected_mics])
        
        min_bounds = np.min(mic_positions, axis=0)
        max_bounds = np.max(mic_positions, axis=0)
        
        # Allow solution to be outside array but within reasonable distance
        array_size = np.max(max_bounds - min_bounds)
        max_distance = max(10.0, 5 * array_size)  # 5x array size or 10m, whichever is larger
        
        extended_min = min_bounds - max_distance
        extended_max = max_bounds + max_distance
        
        # Check bounds
        if not np.all(source_pos >= extended_min) or not np.all(source_pos <= extended_max):
            self.logger.debug(f"Solution {source_pos} outside reasonable bounds {extended_min} to {extended_max}")
            return False
        
        # Check that residual error is reasonable
        max_acceptable_error = 0.05  # 50ms equivalent distance error (more lenient)
        if solution['residual_error'] > max_acceptable_error:
            self.logger.debug(f"Residual error {solution['residual_error']} exceeds threshold {max_acceptable_error}")
            return False
        
        # Check that distances to microphones are physically reasonable
        for mic_id in selected_mics:
            mic_pos = np.array([self.microphone_positions[mic_id].x,
                               self.microphone_positions[mic_id].y,
                               self.microphone_positions[mic_id].z])
            distance = np.linalg.norm(source_pos - mic_pos)
            
            # Maximum reasonable distance (sound travels this in max_tdoa_seconds)
            max_reasonable_distance = self.sound_speed * self.max_tdoa_seconds * 5  # More lenient
            
            if distance > max_reasonable_distance:
                self.logger.debug(f"Distance to mic {mic_id}: {distance}m exceeds reasonable limit {max_reasonable_distance}m")
                return False
        
        return True
    
    def _calculate_solution_confidence(self, solution: Dict, quality_scores: List[float]) -> float:
        """
        Calculate confidence in the triangulation solution.
        
        Args:
            solution: Solution dictionary from multilateration
            quality_scores: Quality scores of microphones used
            
        Returns:
            Confidence score (0-1)
        """
        # Factors affecting confidence:
        # 1. Residual error (lower is better)
        # 2. Microphone quality scores
        # 3. Optimization convergence quality
        # 4. Geometric dilution of precision (GDOP)
        
        # Residual error component (exponential decay)
        error_confidence = np.exp(-solution['residual_error'] * 100)
        
        # Microphone quality component
        quality_confidence = np.mean(quality_scores) if quality_scores else 0.5
        
        # Optimization convergence component
        convergence_confidence = 1.0 if solution['cost'] < 1e-6 else np.exp(-solution['cost'] * 10)
        
        # Geometric dilution component (simplified)
        # Better geometry gives higher confidence
        num_mics = len(quality_scores)
        geometry_confidence = min(1.0, num_mics / 4.0)  # More mics generally better
        
        # Combined confidence (weighted average)
        confidence = (0.4 * error_confidence + 
                     0.3 * quality_confidence + 
                     0.2 * convergence_confidence + 
                     0.1 * geometry_confidence)
        
        return max(0.0, min(1.0, confidence))
    
    def _create_failed_result(self, tdoa_matrix: np.ndarray, reason: str) -> LocationResult:
        """
        Create a LocationResult for failed triangulation.
        
        Args:
            tdoa_matrix: Original TDoA matrix
            reason: Reason for failure
            
        Returns:
            LocationResult indicating failure
        """
        return LocationResult(
            x=0.0,
            y=0.0,
            z=0.0,
            confidence=0.0,
            residual_error=float('inf'),
            timestamp=time.time(),
            microphones_used=[],
            tdoa_matrix=tdoa_matrix,
            correlation_peaks=self.correlation_history[-1] if self.correlation_history else None
        )
    
    def _validate_tdoa_matrix(self, tdoa_matrix: np.ndarray) -> Dict:
        """
        Validate TDoA matrix quality and consistency.
        
        Args:
            tdoa_matrix: Time differences between microphone pairs
            
        Returns:
            Dictionary with validation result and details
        """
        validation_result = {
            'valid': True,
            'reason': '',
            'quality_score': 1.0,
            'issues': []
        }
        
        # Check for NaN or infinite values
        if np.any(np.isnan(tdoa_matrix)) or np.any(np.isinf(tdoa_matrix)):
            validation_result['valid'] = False
            validation_result['reason'] = 'Matrix contains NaN or infinite values'
            validation_result['issues'].append('invalid_values')
            return validation_result
        
        # Check diagonal is zero (or very close)
        diagonal_error = np.max(np.abs(np.diag(tdoa_matrix)))
        if diagonal_error > 1e-6:
            validation_result['issues'].append('non_zero_diagonal')
            validation_result['quality_score'] *= 0.9
        
        # Check antisymmetry: tdoa[i,j] = -tdoa[j,i]
        antisymmetry_error = np.max(np.abs(tdoa_matrix + tdoa_matrix.T))
        if antisymmetry_error > 1e-6:
            validation_result['issues'].append('not_antisymmetric')
            validation_result['quality_score'] *= 0.8
        
        # Check for unreasonably large TDoAs
        max_reasonable_tdoa = self.max_tdoa_seconds
        large_tdoas = np.abs(tdoa_matrix) > max_reasonable_tdoa
        np.fill_diagonal(large_tdoas, False)  # Ignore diagonal
        
        if np.any(large_tdoas):
            large_count = np.sum(large_tdoas)
            total_pairs = self.num_mics * (self.num_mics - 1)
            large_fraction = large_count / total_pairs
            
            if large_fraction > 0.3:  # More than 30% are unreasonable
                validation_result['valid'] = False
                validation_result['reason'] = f'Too many unreasonable TDoAs: {large_count}/{total_pairs}'
                validation_result['issues'].append('excessive_large_tdoas')
                return validation_result  # Return early to prioritize this over consistency
            else:
                validation_result['issues'].append('some_large_tdoas')
                validation_result['quality_score'] *= (1.0 - large_fraction)
        
        # Check for consistency using triangle inequality
        consistency_violations = 0
        total_triangles = 0
        
        for i in range(self.num_mics):
            for j in range(i + 1, self.num_mics):
                for k in range(j + 1, self.num_mics):
                    # Triangle inequality: |tdoa_ij + tdoa_jk - tdoa_ik| should be small
                    expected_tdoa_ik = tdoa_matrix[i, j] + tdoa_matrix[j, k]
                    actual_tdoa_ik = tdoa_matrix[i, k]
                    triangle_error = abs(expected_tdoa_ik - actual_tdoa_ik)
                    
                    if triangle_error > 0.001:  # 1ms tolerance
                        consistency_violations += 1
                    
                    total_triangles += 1
        
        if total_triangles > 0:
            consistency_ratio = consistency_violations / total_triangles
            if consistency_ratio > 0.3:  # More than 30% violations
                validation_result['valid'] = False
                validation_result['reason'] = f'Poor TDoA consistency: {consistency_violations}/{total_triangles} violations'
                validation_result['issues'].append('poor_consistency')
            elif consistency_ratio > 0.1:
                validation_result['issues'].append('moderate_inconsistency')
                validation_result['quality_score'] *= (1.0 - consistency_ratio)
        
        return validation_result
    
    def _check_minimum_microphone_requirements(self, selected_mics: List[int], quality_scores: List[float]) -> Dict:
        """
        Check if we have sufficient microphones for reliable triangulation.
        
        Args:
            selected_mics: List of selected microphone indices
            quality_scores: Quality scores for selected microphones
            
        Returns:
            Dictionary with sufficiency result and details
        """
        result = {
            'sufficient': True,
            'reason': '',
            'recommended_count': 4,
            'quality_assessment': 'good'
        }
        
        num_selected = len(selected_mics)
        
        # Absolute minimum: 3 microphones for 2D triangulation
        if num_selected < 3:
            result['sufficient'] = False
            result['reason'] = f'Only {num_selected} microphones available, minimum 3 required'
            return result
        
        # Recommended minimum: 4 microphones for robust 2D triangulation
        if num_selected < 4:
            # Check if the 3 microphones have very high quality
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            if avg_quality < 0.7:
                result['sufficient'] = False
                result['reason'] = f'Only {num_selected} microphones with average quality {avg_quality:.2f}, need 4+ or higher quality'
                return result
            else:
                result['quality_assessment'] = 'marginal'
                result['reason'] = f'Using minimum {num_selected} microphones with high quality'
                return result  # Return early to ensure marginal assessment
        
        # Check geometric diversity (microphones shouldn't be too clustered)
        if num_selected >= 3:
            mic_positions = np.array([[self.microphone_positions[i].x, 
                                     self.microphone_positions[i].y] for i in selected_mics])
            
            # Calculate area of convex hull (simplified for small sets)
            if num_selected == 3:
                # Triangle area
                p1, p2, p3 = mic_positions
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
            else:
                # Approximate area using bounding box
                min_coords = np.min(mic_positions, axis=0)
                max_coords = np.max(mic_positions, axis=0)
                area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
            
            # Check if microphones are too clustered
            min_area = 0.01  # Minimum 0.01 m² area (very small for tight clustering)
            if area < min_area:
                result['sufficient'] = False
                result['reason'] = f'Microphones too clustered (area: {area:.3f}m²), need more geometric diversity'
                return result
        
        # Assess overall quality
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            min_quality = np.min(quality_scores)
            
            if avg_quality >= 0.8 and min_quality >= 0.6:
                result['quality_assessment'] = 'excellent'
            elif avg_quality >= 0.6 and min_quality >= 0.4:
                result['quality_assessment'] = 'good'
            elif avg_quality >= 0.4 and min_quality >= 0.2:
                result['quality_assessment'] = 'fair'
            else:
                result['quality_assessment'] = 'poor'
                if num_selected < 6:  # Need more microphones if quality is poor
                    result['sufficient'] = False
                    result['reason'] = f'Poor microphone quality (avg: {avg_quality:.2f}), need more microphones'
        
        return result
    
    def _solve_multilateration_with_fallbacks(self, tdoa_matrix: np.ndarray, selected_mics: List[int]) -> Dict:
        """
        Solve multilateration with multiple fallback methods for robustness.
        
        Args:
            tdoa_matrix: Time differences between microphone pairs
            selected_mics: List of microphone indices to use
            
        Returns:
            Dictionary with solution details
        """
        # Try primary method first
        try:
            result = self._solve_multilateration(tdoa_matrix, selected_mics)
            
            # Check if primary solution is good enough
            if result['residual_error'] < 0.01 and result['optimization_result'].success:
                result['method_used'] = 'primary_least_squares'
                return result
            
        except Exception as e:
            self.logger.debug(f"Primary multilateration failed: {e}")
        
        # Fallback 1: Try with different microphone subset
        if len(selected_mics) > 4:
            try:
                # Use best 4 microphones
                best_4_mics = selected_mics[:4]
                result = self._solve_multilateration(tdoa_matrix, best_4_mics)
                
                if result['residual_error'] < 0.02:
                    result['method_used'] = 'fallback_best_4_mics'
                    return result
                    
            except Exception as e:
                self.logger.debug(f"Fallback method 1 failed: {e}")
        
        # Fallback 2: Relaxed optimization parameters
        try:
            result = self._solve_multilateration_relaxed(tdoa_matrix, selected_mics)
            
            if result['residual_error'] < 0.05:
                result['method_used'] = 'fallback_relaxed_optimization'
                return result
                
        except Exception as e:
            self.logger.debug(f"Fallback method 2 failed: {e}")
        
        # Fallback 3: Geometric method (if available)
        try:
            result = self._solve_geometric_method(tdoa_matrix, selected_mics)
            result['method_used'] = 'fallback_geometric'
            return result
            
        except Exception as e:
            self.logger.debug(f"Fallback method 3 failed: {e}")
        
        # If all methods fail, raise exception
        raise RuntimeError("All multilateration methods failed to converge")
    
    def _solve_multilateration_relaxed(self, tdoa_matrix: np.ndarray, selected_mics: List[int]) -> Dict:
        """
        Solve multilateration with relaxed optimization parameters.
        
        Args:
            tdoa_matrix: Time differences between microphone pairs
            selected_mics: List of microphone indices to use
            
        Returns:
            Dictionary with solution details
        """
        # Use the same approach as _solve_multilateration but with relaxed parameters
        ref_mic = selected_mics[0]
        other_mics = selected_mics[1:]
        
        ref_pos = np.array([self.microphone_positions[ref_mic].x, 
                           self.microphone_positions[ref_mic].y,
                           self.microphone_positions[ref_mic].z])
        
        other_positions = []
        tdoa_measurements = []
        
        for mic_id in other_mics:
            pos = np.array([self.microphone_positions[mic_id].x,
                           self.microphone_positions[mic_id].y, 
                           self.microphone_positions[mic_id].z])
            other_positions.append(pos)
            tdoa_measurements.append(tdoa_matrix[ref_mic, mic_id])
        
        other_positions = np.array(other_positions)
        tdoa_measurements = np.array(tdoa_measurements)
        
        def objective_function(source_pos):
            residuals = []
            ref_distance = np.linalg.norm(source_pos - ref_pos)
            
            for i, other_pos in enumerate(other_positions):
                other_distance = np.linalg.norm(source_pos - other_pos)
                expected_tdoa = (other_distance - ref_distance) / self.sound_speed
                residual = expected_tdoa - tdoa_measurements[i]
                residuals.append(residual)
            
            return np.array(residuals)
        
        # Initial guess
        all_positions = np.array([[mic.x, mic.y, mic.z] for mic in self.microphone_positions])
        initial_guess = np.mean(all_positions, axis=0)
        
        # Relaxed optimization with more iterations and tolerance
        result = least_squares(
            objective_function,
            initial_guess,
            method='lm',
            max_nfev=2000,  # More iterations
            ftol=1e-6,      # Relaxed function tolerance
            xtol=1e-6       # Relaxed parameter tolerance
        )
        
        if not result.success:
            raise RuntimeError("Relaxed multilateration optimization failed to converge")
        
        final_residuals = objective_function(result.x)
        residual_error = np.sqrt(np.mean(final_residuals ** 2))
        
        return {
            'position': result.x,
            'residuals': final_residuals,
            'residual_error': residual_error,
            'optimization_result': result,
            'cost': result.cost,
            'iterations': result.nfev
        }
    
    def _solve_geometric_method(self, tdoa_matrix: np.ndarray, selected_mics: List[int]) -> Dict:
        """
        Solve using geometric method as final fallback.
        
        Args:
            tdoa_matrix: Time differences between microphone pairs
            selected_mics: List of microphone indices to use
            
        Returns:
            Dictionary with solution details
        """
        # Simple geometric approach using first 3 microphones
        if len(selected_mics) < 3:
            raise RuntimeError("Need at least 3 microphones for geometric method")
        
        # Use first 3 microphones
        mic_ids = selected_mics[:3]
        
        # Get positions and TDoAs
        positions = []
        for mic_id in mic_ids:
            pos = [self.microphone_positions[mic_id].x, 
                   self.microphone_positions[mic_id].y]
            positions.append(pos)
        
        # Simple 2D geometric solution (approximate)
        # This is a simplified implementation - in practice, you'd use more sophisticated geometry
        x_est = np.mean([pos[0] for pos in positions])
        y_est = np.mean([pos[1] for pos in positions])
        
        # Calculate residual error
        residuals = []
        for i in range(len(mic_ids)):
            for j in range(i + 1, len(mic_ids)):
                pos_i = positions[i]
                pos_j = positions[j]
                
                dist_i = np.sqrt((x_est - pos_i[0])**2 + (y_est - pos_i[1])**2)
                dist_j = np.sqrt((x_est - pos_j[0])**2 + (y_est - pos_j[1])**2)
                
                expected_tdoa = (dist_j - dist_i) / self.sound_speed
                actual_tdoa = tdoa_matrix[mic_ids[i], mic_ids[j]]
                
                residuals.append(expected_tdoa - actual_tdoa)
        
        residual_error = np.sqrt(np.mean(np.array(residuals) ** 2))
        
        return {
            'position': np.array([x_est, y_est, 0.0]),
            'residuals': np.array(residuals),
            'residual_error': residual_error,
            'optimization_result': None,
            'cost': residual_error,
            'iterations': 1
        }
    
    def _validate_solution_quality(self, solution: Dict, selected_mics: List[int], quality_scores: List[float]) -> Dict:
        """
        Comprehensive solution quality validation.
        
        Args:
            solution: Solution dictionary from multilateration
            selected_mics: Microphones used in solution
            quality_scores: Quality scores of microphones used
            
        Returns:
            Dictionary with validation result and quality metrics
        """
        validation_result = {
            'valid': True,
            'reason': '',
            'quality_metrics': {},
            'warnings': []
        }
        
        # Check geometric constraints (existing method)
        if not self._validate_geometric_constraints(solution, selected_mics):
            validation_result['valid'] = False
            validation_result['reason'] = 'Failed geometric constraints'
            return validation_result
        
        # Check convergence quality
        if solution['optimization_result'] is not None:
            opt_result = solution['optimization_result']
            
            if not opt_result.success:
                validation_result['warnings'].append('optimization_did_not_converge')
            
            if opt_result.nfev > 500:  # Too many iterations
                validation_result['warnings'].append('slow_convergence')
            
            validation_result['quality_metrics']['convergence_iterations'] = opt_result.nfev
            validation_result['quality_metrics']['optimization_success'] = opt_result.success
        
        # Check residual error quality
        residual_error = solution['residual_error']
        validation_result['quality_metrics']['residual_error'] = residual_error
        
        if residual_error > 0.02:  # 20ms equivalent
            validation_result['warnings'].append('high_residual_error')
        
        if residual_error > 0.05:  # 50ms equivalent - too high
            validation_result['valid'] = False
            validation_result['reason'] = f'Residual error too high: {residual_error:.4f}'
            return validation_result
        
        # Check solution stability (if we have multiple attempts)
        validation_result['quality_metrics']['solution_stability'] = self._assess_solution_stability(solution)
        
        # Check microphone quality contribution
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            min_quality = np.min(quality_scores)
            
            validation_result['quality_metrics']['avg_microphone_quality'] = avg_quality
            validation_result['quality_metrics']['min_microphone_quality'] = min_quality
            
            if avg_quality < 0.3:
                validation_result['warnings'].append('low_average_microphone_quality')
            
            if min_quality < 0.1:
                validation_result['warnings'].append('very_poor_microphone_present')
        
        # Check geometric dilution of precision (GDOP)
        gdop = self._calculate_gdop(selected_mics, solution['position'])
        validation_result['quality_metrics']['gdop'] = gdop
        
        if gdop > 10:
            validation_result['warnings'].append('poor_geometric_dilution')
        
        if gdop > 50:
            validation_result['valid'] = False
            validation_result['reason'] = f'Geometric dilution too poor: {gdop:.1f}'
            return validation_result
        
        return validation_result
    
    def _assess_solution_stability(self, solution: Dict) -> float:
        """
        Assess stability of the solution.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Stability score (0-1, higher is better)
        """
        # For now, base stability on residual consistency
        residuals = solution['residuals']
        
        if len(residuals) < 2:
            return 1.0
        
        # Lower variance in residuals indicates more stable solution
        residual_variance = np.var(residuals)
        stability = np.exp(-residual_variance * 1000)  # More sensitive to variance
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_gdop(self, selected_mics: List[int], source_position: np.ndarray) -> float:
        """
        Calculate Geometric Dilution of Precision (GDOP).
        
        Args:
            selected_mics: List of microphone indices used
            source_position: Estimated source position
            
        Returns:
            GDOP value (lower is better)
        """
        if len(selected_mics) < 4:
            return 1.0  # Can't calculate proper GDOP with < 4 mics
        
        # Build geometry matrix
        H = []
        
        for mic_id in selected_mics[1:]:  # Skip reference microphone
            mic_pos = np.array([self.microphone_positions[mic_id].x,
                               self.microphone_positions[mic_id].y,
                               self.microphone_positions[mic_id].z])
            
            ref_pos = np.array([self.microphone_positions[selected_mics[0]].x,
                               self.microphone_positions[selected_mics[0]].y,
                               self.microphone_positions[selected_mics[0]].z])
            
            # Unit vector from source to microphone
            r_mic = np.linalg.norm(source_position - mic_pos)
            r_ref = np.linalg.norm(source_position - ref_pos)
            
            if r_mic > 0 and r_ref > 0:
                unit_mic = (source_position - mic_pos) / r_mic
                unit_ref = (source_position - ref_pos) / r_ref
                
                # Difference of unit vectors
                h_row = unit_mic - unit_ref
                H.append(h_row[:2])  # Use only x, y for 2D
        
        if len(H) < 3:
            return 100.0  # Poor geometry
        
        H = np.array(H)
        
        try:
            # GDOP = sqrt(trace((H^T * H)^-1))
            HTH = np.dot(H.T, H)
            HTH_inv = np.linalg.inv(HTH)
            gdop = np.sqrt(np.trace(HTH_inv))
            
            return gdop
            
        except np.linalg.LinAlgError:
            return 100.0  # Singular matrix - poor geometry
    
    def _calculate_enhanced_confidence(self, solution: Dict, quality_scores: List[float], 
                                     solution_validation: Dict) -> float:
        """
        Calculate enhanced confidence score incorporating all quality metrics.
        
        Args:
            solution: Solution dictionary from multilateration
            quality_scores: Quality scores of microphones used
            solution_validation: Validation result dictionary
            
        Returns:
            Enhanced confidence score (0-1)
        """
        # Base confidence from original method
        base_confidence = self._calculate_solution_confidence(solution, quality_scores)
        
        # Quality metrics from validation
        quality_metrics = solution_validation.get('quality_metrics', {})
        warnings = solution_validation.get('warnings', [])
        
        # Confidence adjustments based on quality metrics
        confidence_multiplier = 1.0
        
        # Residual error adjustment
        residual_error = quality_metrics.get('residual_error', 0.0)
        if residual_error < 0.005:  # Very low error
            confidence_multiplier *= 1.1
        elif residual_error > 0.02:  # High error
            confidence_multiplier *= 0.8
        
        # Convergence quality adjustment
        if quality_metrics.get('optimization_success', True):
            confidence_multiplier *= 1.05
        else:
            confidence_multiplier *= 0.7
        
        # GDOP adjustment
        gdop = quality_metrics.get('gdop', 1.0)
        if gdop < 2.0:  # Excellent geometry
            confidence_multiplier *= 1.1
        elif gdop > 10.0:  # Poor geometry
            confidence_multiplier *= 0.6
        
        # Solution stability adjustment
        stability = quality_metrics.get('solution_stability', 1.0)
        confidence_multiplier *= (0.8 + 0.2 * stability)
        
        # Warning penalties
        warning_penalty = 1.0
        for warning in warnings:
            if warning in ['high_residual_error', 'poor_geometric_dilution']:
                warning_penalty *= 0.8
            elif warning in ['low_average_microphone_quality', 'slow_convergence']:
                warning_penalty *= 0.9
            elif warning in ['very_poor_microphone_present']:
                warning_penalty *= 0.7
        
        # Calculate final confidence
        enhanced_confidence = base_confidence * confidence_multiplier * warning_penalty
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, enhanced_confidence))
    
    def estimate_confidence(self, residuals: np.ndarray) -> float:
        """
        Estimate confidence in TDoA calculation.
        
        Args:
            residuals: Residual errors from correlation
            
        Returns:
            Confidence score (0-1)
        """
        if len(residuals) == 0:
            return 0.0
        
        # Base confidence on residual magnitude
        mean_residual = np.mean(np.abs(residuals))
        confidence = np.exp(-mean_residual * 10)  # Exponential decay
        
        return max(0.0, min(1.0, confidence))
    
    def get_correlation_statistics(self) -> Dict:
        """
        Get cross-correlation performance statistics.
        
        Returns:
            Dictionary with correlation statistics
        """
        if not self.correlation_history:
            return {
                'samples_processed': 0,
                'avg_correlation': 0.0,
                'min_correlation': 0.0,
                'max_correlation': 0.0,
                'correlation_pairs': 0
            }
        
        # Collect all correlation values
        all_correlations = []
        for correlation_dict in self.correlation_history:
            all_correlations.extend(correlation_dict.values())
        
        return {
            'samples_processed': len(self.correlation_history),
            'avg_correlation': np.mean(all_correlations) if all_correlations else 0.0,
            'min_correlation': np.min(all_correlations) if all_correlations else 0.0,
            'max_correlation': np.max(all_correlations) if all_correlations else 0.0,
            'correlation_pairs': len(self.mic_pairs),
            'total_correlations': len(all_correlations)
        }
    
    def get_tdoa_statistics(self) -> Dict:
        """
        Get TDoA calculation statistics.
        
        Returns:
            Dictionary with TDoA statistics
        """
        if not self.tdoa_history:
            return {
                'samples_processed': 0,
                'avg_tdoa_magnitude': 0.0,
                'max_tdoa_magnitude': 0.0,
                'tdoa_consistency': 0.0
            }
        
        # Analyze TDoA consistency
        recent_tdoas = list(self.tdoa_history)[-10:]  # Last 10 samples
        
        if len(recent_tdoas) > 1:
            # Calculate standard deviation across recent samples
            tdoa_stack = np.stack(recent_tdoas)
            tdoa_std = np.std(tdoa_stack, axis=0)
            consistency = 1.0 - np.mean(tdoa_std) / self.max_tdoa_seconds
            consistency = max(0.0, min(1.0, consistency))
        else:
            consistency = 1.0
        
        # Get magnitude statistics
        latest_tdoa = self.tdoa_history[-1]
        non_zero_tdoas = latest_tdoa[latest_tdoa != 0]
        
        return {
            'samples_processed': len(self.tdoa_history),
            'avg_tdoa_magnitude': np.mean(np.abs(non_zero_tdoas)) if len(non_zero_tdoas) > 0 else 0.0,
            'max_tdoa_magnitude': np.max(np.abs(non_zero_tdoas)) if len(non_zero_tdoas) > 0 else 0.0,
            'tdoa_consistency': consistency,
            'microphone_pairs': len(self.mic_pairs)
        }
    
    def configure_correlation_parameters(self, **kwargs) -> None:
        """
        Configure cross-correlation parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'correlation_method' in kwargs:
            if kwargs['correlation_method'] in ['fft', 'direct']:
                self.correlation_method = kwargs['correlation_method']
            else:
                raise ValueError("correlation_method must be 'fft' or 'direct'")
        
        if 'max_delay_samples' in kwargs:
            self.max_delay_samples = int(kwargs['max_delay_samples'])
        
        if 'interpolation_factor' in kwargs:
            self.interpolation_factor = int(kwargs['interpolation_factor'])
        
        if 'min_correlation_threshold' in kwargs:
            self.min_correlation_threshold = float(kwargs['min_correlation_threshold'])
        
        if 'max_tdoa_seconds' in kwargs:
            self.max_tdoa_seconds = float(kwargs['max_tdoa_seconds'])
        
        if 'enable_preprocessing' in kwargs:
            self.enable_preprocessing = bool(kwargs['enable_preprocessing'])
        
        if 'filter_low_freq' in kwargs:
            self.filter_low_freq = float(kwargs['filter_low_freq'])
            self.bandpass_filter = None  # Reset filter
        
        if 'filter_high_freq' in kwargs:
            self.filter_high_freq = float(kwargs['filter_high_freq'])
            self.bandpass_filter = None  # Reset filter
        
        self.logger.info(f"Correlation parameters updated: {kwargs}")
    
    def analyze_signal_quality(self, audio_channels: np.ndarray) -> Dict:
        """
        Analyze signal quality for TDoA calculation.
        
        Args:
            audio_channels: Multi-channel audio data
            
        Returns:
            Dictionary with signal quality metrics
        """
        quality_metrics = {
            'channel_snr': [],
            'channel_energy': [],
            'cross_channel_correlation': [],
            'signal_bandwidth': [],
            'overall_quality': 0.0
        }
        
        # Analyze each channel
        for ch in range(audio_channels.shape[1]):
            channel_data = audio_channels[:, ch]
            
            # Signal-to-noise ratio estimation
            signal_power = np.mean(channel_data ** 2)
            noise_power = np.mean(channel_data[:100] ** 2)  # Assume first 100 samples are noise
            snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            quality_metrics['channel_snr'].append(snr)
            
            # Signal energy
            energy = np.sum(channel_data ** 2)
            quality_metrics['channel_energy'].append(energy)
            
            # Signal bandwidth (using FFT)
            fft = np.fft.fft(channel_data)
            freqs = np.fft.fftfreq(len(channel_data), 1/self.sample_rate)
            magnitude = np.abs(fft[:len(fft)//2])
            
            # Calculate bandwidth (frequency range containing 90% of energy)
            cumulative_energy = np.cumsum(magnitude ** 2)
            total_energy = cumulative_energy[-1]
            
            if total_energy > 0:
                low_idx = np.where(cumulative_energy >= 0.05 * total_energy)[0]
                high_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
                
                if len(low_idx) > 0 and len(high_idx) > 0:
                    bandwidth = freqs[high_idx[0]] - freqs[low_idx[0]]
                else:
                    bandwidth = 0
            else:
                bandwidth = 0
            
            quality_metrics['signal_bandwidth'].append(bandwidth)
        
        # Cross-channel correlation analysis
        for i in range(audio_channels.shape[1]):
            for j in range(i + 1, audio_channels.shape[1]):
                correlation = np.corrcoef(audio_channels[:, i], audio_channels[:, j])[0, 1]
                quality_metrics['cross_channel_correlation'].append(abs(correlation))
        
        # Overall quality score
        avg_snr = np.mean(quality_metrics['channel_snr'])
        avg_correlation = np.mean(quality_metrics['cross_channel_correlation'])
        avg_bandwidth = np.mean(quality_metrics['signal_bandwidth'])
        
        # Normalize and combine metrics
        snr_score = min(1.0, max(0.0, (avg_snr + 10) / 30))  # -10 to 20 dB range
        correlation_score = avg_correlation
        bandwidth_score = min(1.0, avg_bandwidth / 4000)  # Up to 4kHz
        
        overall_quality = 0.5 * snr_score + 0.3 * correlation_score + 0.2 * bandwidth_score
        quality_metrics['overall_quality'] = overall_quality
        
        return quality_metrics
    
    def reset_history(self) -> None:
        """Reset correlation and TDoA history."""
        self.correlation_history.clear()
        self.tdoa_history.clear()
        self.logger.info("TDoA history reset")