"""
Adaptive channel selection module for dynamic microphone management.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import logging
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time

from intensity_filter import RMSIntensityFilter, ChannelQualityMetrics


class SelectionStrategy(Enum):
    """Channel selection strategies."""
    QUALITY_BASED = "quality_based"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    GEOMETRIC_OPTIMAL = "geometric_optimal"


@dataclass
class ChannelSelectionResult:
    """Result of channel selection process."""
    selected_channels: List[int]
    excluded_channels: List[int]
    channel_weights: np.ndarray
    selection_confidence: float
    strategy_used: SelectionStrategy
    fallback_applied: bool
    quality_metrics: Dict[int, ChannelQualityMetrics]
    timestamp: float


@dataclass
class AdaptationState:
    """State information for adaptive behavior."""
    recent_detections: List[float]  # Recent detection confidences
    channel_performance_history: Dict[int, List[float]]  # Channel performance over time
    environmental_noise_level: float
    adaptation_rate: float
    last_update_time: float


class AdaptiveChannelSelectorInterface(ABC):
    """Abstract interface for adaptive channel selection."""
    
    @abstractmethod
    def select_channels(self, audio_channels: np.ndarray, 
                       detection_confidence: float = 1.0,
                       required_channels: Optional[int] = None) -> ChannelSelectionResult:
        """
        Select optimal channels for processing.
        
        Args:
            audio_channels: Multi-channel audio data
            detection_confidence: Confidence from gunshot detection
            required_channels: Minimum number of channels required
            
        Returns:
            ChannelSelectionResult with selected channels and metadata
        """
        pass
    
    @abstractmethod
    def update_performance_feedback(self, selected_channels: List[int], 
                                  triangulation_confidence: float,
                                  triangulation_error: float) -> None:
        """Update performance feedback for adaptive learning."""
        pass
    
    @abstractmethod
    def get_selection_statistics(self) -> Dict:
        """Get statistics about channel selection performance."""
        pass


class AdaptiveChannelSelector(AdaptiveChannelSelectorInterface):
    """Advanced adaptive channel selector with multiple strategies and learning."""
    
    def __init__(self, num_channels: int, intensity_filter: Optional[RMSIntensityFilter] = None):
        """
        Initialize adaptive channel selector.
        
        Args:
            num_channels: Total number of available channels
            intensity_filter: Optional intensity filter instance
        """
        self.logger = logging.getLogger(__name__)
        self.num_channels = num_channels
        
        # Initialize intensity filter if not provided
        self.intensity_filter = intensity_filter or RMSIntensityFilter()
        
        # Selection parameters
        self.min_channels = max(3, min(4, num_channels))  # Minimum channels for triangulation
        self.preferred_channels = min(6, num_channels)    # Preferred number of channels
        self.max_channels = num_channels                  # Maximum channels to use
        
        # Quality thresholds
        self.base_quality_threshold = 0.3    # Base quality threshold
        self.confidence_threshold = 0.5      # Detection confidence threshold
        self.snr_threshold = 6.0             # Minimum SNR in dB
        
        # Adaptive parameters
        self.adaptation_enabled = True
        self.adaptation_rate = 0.1
        self.learning_window = 50  # Number of samples for learning
        
        # Selection strategies
        self.primary_strategy = SelectionStrategy.CONFIDENCE_WEIGHTED
        self.fallback_strategies = [
            SelectionStrategy.QUALITY_BASED,
            SelectionStrategy.ADAPTIVE_THRESHOLD,
            SelectionStrategy.GEOMETRIC_OPTIMAL
        ]
        
        # State tracking
        self.adaptation_state = AdaptationState(
            recent_detections=[],
            channel_performance_history={i: [] for i in range(num_channels)},
            environmental_noise_level=0.1,
            adaptation_rate=self.adaptation_rate,
            last_update_time=time.time()
        )
        
        # History tracking
        self.selection_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
        
        # Statistics
        self.stats = {
            'total_selections': 0,
            'fallback_count': 0,
            'strategy_usage': {strategy: 0 for strategy in SelectionStrategy},
            'average_channels_selected': 0.0,
            'adaptation_events': 0
        }
        
        self.logger.info(f"Adaptive channel selector initialized: {num_channels} channels")
    
    def select_channels(self, audio_channels: np.ndarray, 
                       detection_confidence: float = 1.0,
                       required_channels: Optional[int] = None) -> ChannelSelectionResult:
        """
        Select optimal channels using adaptive strategies.
        
        Args:
            audio_channels: Multi-channel audio data (samples, channels)
            detection_confidence: Confidence from gunshot detection (0-1)
            required_channels: Minimum number of channels required
            
        Returns:
            ChannelSelectionResult with selected channels and metadata
        """
        if audio_channels.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {audio_channels.shape[1]}")
        
        # Update adaptation state
        self._update_adaptation_state(detection_confidence)
        
        # Calculate channel weights using intensity filter
        channel_weights = self.intensity_filter.calculate_weights(audio_channels)
        
        # Get quality metrics
        quality_metrics = self._get_channel_quality_metrics()
        
        # Determine required channels
        if required_channels is None:
            required_channels = self._determine_required_channels(detection_confidence)
        
        # Try primary strategy first
        result = self._try_selection_strategy(
            self.primary_strategy, 
            channel_weights, 
            quality_metrics,
            detection_confidence,
            required_channels
        )
        
        # Try fallback strategies if primary fails
        if result is None or len(result.selected_channels) < required_channels:
            for strategy in self.fallback_strategies:
                fallback_result = self._try_selection_strategy(
                    strategy,
                    channel_weights,
                    quality_metrics,
                    detection_confidence,
                    required_channels
                )
                
                if (fallback_result is not None and 
                    len(fallback_result.selected_channels) >= required_channels):
                    result = fallback_result
                    result.fallback_applied = True
                    self.stats['fallback_count'] += 1
                    break
        
        # Final fallback: use best available channels
        if result is None or len(result.selected_channels) < self.min_channels:
            result = self._emergency_fallback_selection(
                channel_weights, 
                quality_metrics,
                detection_confidence
            )
        
        # Update statistics
        self._update_statistics(result)
        
        # Store in history
        self.selection_history.append(result)
        
        self.logger.debug(f"Selected {len(result.selected_channels)} channels using {result.strategy_used.value}")
        
        return result
    
    def _update_adaptation_state(self, detection_confidence: float) -> None:
        """Update the adaptation state with new information."""
        current_time = time.time()
        
        # Update recent detections
        self.adaptation_state.recent_detections.append(detection_confidence)
        if len(self.adaptation_state.recent_detections) > self.learning_window:
            self.adaptation_state.recent_detections.pop(0)
        
        # Update environmental noise level (simple moving average)
        if hasattr(self.intensity_filter, 'noise_floor_history') and self.intensity_filter.noise_floor_history:
            recent_noise = list(self.intensity_filter.noise_floor_history)[-10:]
            self.adaptation_state.environmental_noise_level = np.mean(recent_noise)
        
        # Adapt thresholds based on recent performance
        if self.adaptation_enabled and len(self.adaptation_state.recent_detections) > 10:
            self._adapt_thresholds()
        
        self.adaptation_state.last_update_time = current_time
    
    def _adapt_thresholds(self) -> None:
        """Adapt selection thresholds based on recent performance."""
        recent_confidence = np.mean(self.adaptation_state.recent_detections[-10:])
        
        # Adapt quality threshold based on detection confidence
        if recent_confidence > 0.8:
            # High confidence detections - can be more selective
            target_threshold = self.base_quality_threshold * 1.2
        elif recent_confidence < 0.4:
            # Low confidence detections - be less selective
            target_threshold = self.base_quality_threshold * 0.8
        else:
            # Moderate confidence - use base threshold
            target_threshold = self.base_quality_threshold
        
        # Smooth adaptation
        current_threshold = getattr(self, 'current_quality_threshold', self.base_quality_threshold)
        self.current_quality_threshold = (
            current_threshold * (1 - self.adaptation_rate) + 
            target_threshold * self.adaptation_rate
        )
        
        # Adapt based on environmental noise
        noise_factor = min(2.0, self.adaptation_state.environmental_noise_level / 0.1)
        self.current_quality_threshold *= noise_factor
        
        # Keep within reasonable bounds
        self.current_quality_threshold = np.clip(self.current_quality_threshold, 0.1, 0.8)
        
        self.stats['adaptation_events'] += 1
        self.logger.debug(f"Adapted quality threshold to {self.current_quality_threshold:.3f}")
    
    def _get_channel_quality_metrics(self) -> Dict[int, ChannelQualityMetrics]:
        """Get quality metrics for all channels."""
        quality_metrics = {}
        
        if (hasattr(self.intensity_filter, 'quality_history') and 
            self.intensity_filter.quality_history):
            latest_metrics = self.intensity_filter.quality_history[-1]
            
            for i, metrics in enumerate(latest_metrics):
                if i < self.num_channels:
                    quality_metrics[i] = metrics
        
        return quality_metrics
    
    def _determine_required_channels(self, detection_confidence: float) -> int:
        """Determine the number of channels required based on detection confidence."""
        if detection_confidence > 0.8:
            # High confidence - can use fewer channels
            return self.min_channels
        elif detection_confidence > 0.5:
            # Moderate confidence - use preferred number
            return min(self.preferred_channels, self.num_channels)
        else:
            # Low confidence - use more channels for robustness
            return min(self.max_channels, self.num_channels)
    
    def _try_selection_strategy(self, strategy: SelectionStrategy,
                               channel_weights: np.ndarray,
                               quality_metrics: Dict[int, ChannelQualityMetrics],
                               detection_confidence: float,
                               required_channels: int) -> Optional[ChannelSelectionResult]:
        """Try a specific selection strategy."""
        try:
            if strategy == SelectionStrategy.QUALITY_BASED:
                return self._quality_based_selection(
                    channel_weights, quality_metrics, required_channels
                )
            elif strategy == SelectionStrategy.CONFIDENCE_WEIGHTED:
                return self._confidence_weighted_selection(
                    channel_weights, quality_metrics, detection_confidence, required_channels
                )
            elif strategy == SelectionStrategy.ADAPTIVE_THRESHOLD:
                return self._adaptive_threshold_selection(
                    channel_weights, quality_metrics, required_channels
                )
            elif strategy == SelectionStrategy.GEOMETRIC_OPTIMAL:
                return self._geometric_optimal_selection(
                    channel_weights, quality_metrics, required_channels
                )
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Strategy {strategy.value} failed: {e}")
            return None
    
    def _quality_based_selection(self, channel_weights: np.ndarray,
                                quality_metrics: Dict[int, ChannelQualityMetrics],
                                required_channels: int) -> ChannelSelectionResult:
        """Select channels based purely on quality metrics."""
        # Sort channels by weight (quality)
        channel_indices = np.argsort(channel_weights)[::-1]  # Highest first
        
        selected_channels = []
        excluded_channels = []
        
        threshold = getattr(self, 'current_quality_threshold', self.base_quality_threshold)
        
        for ch in channel_indices:
            if len(selected_channels) < required_channels:
                if channel_weights[ch] >= threshold:
                    selected_channels.append(ch)
                else:
                    excluded_channels.append(ch)
            else:
                excluded_channels.append(ch)
        
        # If we don't have enough channels, lower the threshold
        if len(selected_channels) < self.min_channels:
            selected_channels = channel_indices[:self.min_channels].tolist()
            excluded_channels = channel_indices[self.min_channels:].tolist()
        
        return ChannelSelectionResult(
            selected_channels=selected_channels,
            excluded_channels=excluded_channels,
            channel_weights=channel_weights,
            selection_confidence=np.mean(channel_weights[selected_channels]) if selected_channels else 0.0,
            strategy_used=SelectionStrategy.QUALITY_BASED,
            fallback_applied=False,
            quality_metrics=quality_metrics,
            timestamp=time.time()
        )
    
    def _confidence_weighted_selection(self, channel_weights: np.ndarray,
                                     quality_metrics: Dict[int, ChannelQualityMetrics],
                                     detection_confidence: float,
                                     required_channels: int) -> ChannelSelectionResult:
        """Select channels with detection confidence weighting."""
        # Adjust channel weights based on detection confidence
        confidence_factor = 0.5 + 0.5 * detection_confidence  # 0.5 to 1.0
        adjusted_weights = channel_weights * confidence_factor
        
        # Add bonus for channels with good historical performance
        for ch in range(self.num_channels):
            if ch in self.adaptation_state.channel_performance_history:
                history = self.adaptation_state.channel_performance_history[ch]
                if len(history) > 5:
                    avg_performance = np.mean(history[-10:])  # Last 10 samples
                    performance_bonus = (avg_performance - 0.5) * 0.2  # -0.1 to +0.1
                    adjusted_weights[ch] += performance_bonus
        
        # Select top channels
        channel_indices = np.argsort(adjusted_weights)[::-1]
        
        # Dynamic threshold based on detection confidence
        base_threshold = getattr(self, 'current_quality_threshold', self.base_quality_threshold)
        dynamic_threshold = base_threshold * (0.7 + 0.3 * detection_confidence)
        
        selected_channels = []
        excluded_channels = []
        
        for ch in channel_indices:
            if len(selected_channels) < required_channels:
                if adjusted_weights[ch] >= dynamic_threshold:
                    selected_channels.append(ch)
                else:
                    excluded_channels.append(ch)
            else:
                excluded_channels.append(ch)
        
        # Ensure minimum channels
        if len(selected_channels) < self.min_channels:
            needed = self.min_channels - len(selected_channels)
            additional = [ch for ch in channel_indices if ch not in selected_channels][:needed]
            selected_channels.extend(additional)
            excluded_channels = [ch for ch in excluded_channels if ch not in additional]
        
        return ChannelSelectionResult(
            selected_channels=selected_channels,
            excluded_channels=excluded_channels,
            channel_weights=adjusted_weights,
            selection_confidence=np.mean(adjusted_weights[selected_channels]) if selected_channels else 0.0,
            strategy_used=SelectionStrategy.CONFIDENCE_WEIGHTED,
            fallback_applied=False,
            quality_metrics=quality_metrics,
            timestamp=time.time()
        )
    
    def _adaptive_threshold_selection(self, channel_weights: np.ndarray,
                                    quality_metrics: Dict[int, ChannelQualityMetrics],
                                    required_channels: int) -> ChannelSelectionResult:
        """Select channels using adaptive thresholding."""
        # Calculate adaptive threshold based on weight distribution
        if np.max(channel_weights) > 0:
            # Use percentile-based threshold
            threshold_percentile = max(20, 100 - (required_channels / self.num_channels) * 100)
            adaptive_threshold = np.percentile(channel_weights, threshold_percentile)
            
            # Adjust based on environmental conditions
            noise_factor = min(1.5, self.adaptation_state.environmental_noise_level / 0.1)
            adaptive_threshold /= noise_factor
            
            # Ensure reasonable bounds
            min_threshold = 0.1
            max_threshold = np.max(channel_weights) * 0.8
            adaptive_threshold = np.clip(adaptive_threshold, min_threshold, max_threshold)
        else:
            adaptive_threshold = 0.1
        
        # Select channels above adaptive threshold
        selected_channels = []
        excluded_channels = []
        
        for ch in range(self.num_channels):
            if channel_weights[ch] >= adaptive_threshold:
                selected_channels.append(ch)
            else:
                excluded_channels.append(ch)
        
        # Sort selected channels by weight
        selected_channels.sort(key=lambda ch: channel_weights[ch], reverse=True)
        
        # Limit to required number
        if len(selected_channels) > required_channels:
            excess = selected_channels[required_channels:]
            selected_channels = selected_channels[:required_channels]
            excluded_channels.extend(excess)
        
        # Ensure minimum channels
        if len(selected_channels) < self.min_channels:
            all_channels = list(range(self.num_channels))
            all_channels.sort(key=lambda ch: channel_weights[ch], reverse=True)
            selected_channels = all_channels[:self.min_channels]
            excluded_channels = all_channels[self.min_channels:]
        
        return ChannelSelectionResult(
            selected_channels=selected_channels,
            excluded_channels=excluded_channels,
            channel_weights=channel_weights,
            selection_confidence=adaptive_threshold,
            strategy_used=SelectionStrategy.ADAPTIVE_THRESHOLD,
            fallback_applied=False,
            quality_metrics=quality_metrics,
            timestamp=time.time()
        )
    
    def _geometric_optimal_selection(self, channel_weights: np.ndarray,
                                   quality_metrics: Dict[int, ChannelQualityMetrics],
                                   required_channels: int) -> ChannelSelectionResult:
        """Select channels for optimal geometric configuration."""
        # This is a simplified geometric selection
        # In a real implementation, you'd consider microphone positions
        
        # For now, select channels that are spread across the array
        # Assume channels are arranged in order around a circle
        
        if required_channels >= self.num_channels:
            # Use all channels
            selected_channels = list(range(self.num_channels))
            excluded_channels = []
        else:
            # Select evenly spaced channels with quality weighting
            step = self.num_channels / required_channels
            candidate_channels = []
            
            for i in range(required_channels):
                base_idx = int(i * step)
                # Consider nearby channels and pick the best quality one
                candidates = []
                for offset in [-1, 0, 1]:
                    idx = (base_idx + offset) % self.num_channels
                    candidates.append((idx, channel_weights[idx]))
                
                # Pick the best candidate
                best_idx = max(candidates, key=lambda x: x[1])[0]
                candidate_channels.append(best_idx)
            
            # Remove duplicates and sort
            selected_channels = sorted(list(set(candidate_channels)))
            
            # If we have fewer than required due to duplicates, add more
            if len(selected_channels) < required_channels:
                remaining = [ch for ch in range(self.num_channels) if ch not in selected_channels]
                remaining.sort(key=lambda ch: channel_weights[ch], reverse=True)
                needed = required_channels - len(selected_channels)
                selected_channels.extend(remaining[:needed])
            
            excluded_channels = [ch for ch in range(self.num_channels) if ch not in selected_channels]
        
        return ChannelSelectionResult(
            selected_channels=selected_channels,
            excluded_channels=excluded_channels,
            channel_weights=channel_weights,
            selection_confidence=np.mean(channel_weights[selected_channels]) if selected_channels else 0.0,
            strategy_used=SelectionStrategy.GEOMETRIC_OPTIMAL,
            fallback_applied=False,
            quality_metrics=quality_metrics,
            timestamp=time.time()
        )
    
    def _emergency_fallback_selection(self, channel_weights: np.ndarray,
                                    quality_metrics: Dict[int, ChannelQualityMetrics],
                                    detection_confidence: float) -> ChannelSelectionResult:
        """Emergency fallback when all strategies fail."""
        # Simply select the best available channels regardless of thresholds
        channel_indices = np.argsort(channel_weights)[::-1]
        selected_channels = channel_indices[:self.min_channels].tolist()
        excluded_channels = channel_indices[self.min_channels:].tolist()
        
        self.logger.warning(f"Emergency fallback selection used: {selected_channels}")
        
        return ChannelSelectionResult(
            selected_channels=selected_channels,
            excluded_channels=excluded_channels,
            channel_weights=channel_weights,
            selection_confidence=0.1,  # Low confidence for emergency fallback
            strategy_used=SelectionStrategy.QUALITY_BASED,  # Default strategy
            fallback_applied=True,
            quality_metrics=quality_metrics,
            timestamp=time.time()
        )
    
    def update_performance_feedback(self, selected_channels: List[int], 
                                  triangulation_confidence: float,
                                  triangulation_error: float) -> None:
        """
        Update performance feedback for adaptive learning.
        
        Args:
            selected_channels: Channels that were used for triangulation
            triangulation_confidence: Confidence of triangulation result (0-1)
            triangulation_error: Error in triangulation (lower is better)
        """
        # Convert error to performance score (0-1, higher is better)
        # Assume error is in meters, with 0.1m being excellent, 1.0m being poor
        max_acceptable_error = 1.0
        performance_score = max(0.0, 1.0 - (triangulation_error / max_acceptable_error))
        
        # Combine with triangulation confidence
        combined_performance = 0.7 * triangulation_confidence + 0.3 * performance_score
        
        # Update performance history for selected channels
        for ch in selected_channels:
            if ch in self.adaptation_state.channel_performance_history:
                history = self.adaptation_state.channel_performance_history[ch]
                history.append(combined_performance)
                
                # Keep only recent history
                if len(history) > self.learning_window:
                    history.pop(0)
        
        # Store overall performance
        self.performance_history.append({
            'channels': selected_channels,
            'confidence': triangulation_confidence,
            'error': triangulation_error,
            'performance': combined_performance,
            'timestamp': time.time()
        })
        
        self.logger.debug(f"Updated performance feedback: confidence={triangulation_confidence:.3f}, "
                         f"error={triangulation_error:.3f}m, performance={combined_performance:.3f}")
    
    def _update_statistics(self, result: ChannelSelectionResult) -> None:
        """Update selection statistics."""
        self.stats['total_selections'] += 1
        self.stats['strategy_usage'][result.strategy_used] += 1
        
        # Update average channels selected
        current_avg = self.stats['average_channels_selected']
        n = self.stats['total_selections']
        new_avg = (current_avg * (n - 1) + len(result.selected_channels)) / n
        self.stats['average_channels_selected'] = new_avg
    
    def get_selection_statistics(self) -> Dict:
        """Get comprehensive statistics about channel selection performance."""
        stats = self.stats.copy()
        
        # Add recent performance metrics
        if self.performance_history:
            recent_performance = list(self.performance_history)[-20:]  # Last 20 selections
            stats['recent_performance'] = {
                'avg_confidence': np.mean([p['confidence'] for p in recent_performance]),
                'avg_error': np.mean([p['error'] for p in recent_performance]),
                'avg_performance': np.mean([p['performance'] for p in recent_performance])
            }
        
        # Add channel usage statistics
        channel_usage = {ch: 0 for ch in range(self.num_channels)}
        for selection in list(self.selection_history)[-50:]:  # Last 50 selections
            for ch in selection.selected_channels:
                channel_usage[ch] += 1
        
        stats['channel_usage'] = channel_usage
        
        # Add adaptation state
        stats['adaptation_state'] = {
            'current_threshold': getattr(self, 'current_quality_threshold', self.base_quality_threshold),
            'environmental_noise': self.adaptation_state.environmental_noise_level,
            'recent_detections_avg': np.mean(self.adaptation_state.recent_detections) if self.adaptation_state.recent_detections else 0.0
        }
        
        return stats
    
    def configure_selection_parameters(self, **kwargs) -> None:
        """
        Configure selection parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'min_channels' in kwargs:
            self.min_channels = max(3, int(kwargs['min_channels']))
        
        if 'preferred_channels' in kwargs:
            self.preferred_channels = int(kwargs['preferred_channels'])
        
        if 'base_quality_threshold' in kwargs:
            self.base_quality_threshold = float(kwargs['base_quality_threshold'])
        
        if 'confidence_threshold' in kwargs:
            self.confidence_threshold = float(kwargs['confidence_threshold'])
        
        if 'adaptation_enabled' in kwargs:
            self.adaptation_enabled = bool(kwargs['adaptation_enabled'])
        
        if 'adaptation_rate' in kwargs:
            self.adaptation_rate = float(kwargs['adaptation_rate'])
            self.adaptation_state.adaptation_rate = self.adaptation_rate
        
        if 'primary_strategy' in kwargs:
            strategy_name = kwargs['primary_strategy']
            if hasattr(SelectionStrategy, strategy_name.upper()):
                self.primary_strategy = SelectionStrategy(strategy_name.lower())
        
        self.logger.info(f"Selection parameters updated: {kwargs}")
    
    def reset_adaptation_state(self) -> None:
        """Reset the adaptation state."""
        self.adaptation_state = AdaptationState(
            recent_detections=[],
            channel_performance_history={i: [] for i in range(self.num_channels)},
            environmental_noise_level=0.1,
            adaptation_rate=self.adaptation_rate,
            last_update_time=time.time()
        )
        
        # Reset current threshold to base
        if hasattr(self, 'current_quality_threshold'):
            delattr(self, 'current_quality_threshold')
        
        self.logger.info("Adaptation state reset")
    
    def get_channel_recommendations(self) -> Dict:
        """Get recommendations for channel configuration and maintenance."""
        recommendations = {
            'status': 'good',
            'issues': [],
            'suggestions': []
        }
        
        # Analyze channel usage patterns
        if self.selection_history:
            recent_selections = list(self.selection_history)[-20:]
            
            # Check for consistently excluded channels
            all_channels = set(range(self.num_channels))
            frequently_excluded = set()
            
            for ch in all_channels:
                exclusion_count = sum(1 for sel in recent_selections if ch in sel.excluded_channels)
                if exclusion_count > len(recent_selections) * 0.8:  # Excluded >80% of time
                    frequently_excluded.add(ch)
            
            if frequently_excluded:
                recommendations['issues'].append(f"Channels {list(frequently_excluded)} frequently excluded")
                recommendations['suggestions'].append("Check microphone connections and positioning")
                recommendations['status'] = 'warning'
            
            # Check for low selection confidence
            avg_confidence = np.mean([sel.selection_confidence for sel in recent_selections])
            if avg_confidence < 0.4:
                recommendations['issues'].append(f"Low average selection confidence: {avg_confidence:.2f}")
                recommendations['suggestions'].append("Consider improving microphone quality or reducing noise")
                recommendations['status'] = 'warning'
            
            # Check fallback usage
            fallback_rate = sum(1 for sel in recent_selections if sel.fallback_applied) / len(recent_selections)
            if fallback_rate > 0.3:
                recommendations['issues'].append(f"High fallback usage: {fallback_rate:.1%}")
                recommendations['suggestions'].append("Primary selection strategy may need adjustment")
                recommendations['status'] = 'warning'
        
        # Check adaptation effectiveness
        if self.adaptation_enabled and len(self.performance_history) > 10:
            recent_performance = [p['performance'] for p in list(self.performance_history)[-10:]]
            if np.mean(recent_performance) < 0.5:
                recommendations['issues'].append("Poor recent triangulation performance")
                recommendations['suggestions'].append("Consider recalibrating microphone array")
                recommendations['status'] = 'error'
        
        return recommendations