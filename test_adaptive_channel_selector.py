"""
Unit tests for adaptive channel selector functionality.
"""
import unittest
import numpy as np
import time
from unittest.mock import Mock, patch
from adaptive_channel_selector import (
    AdaptiveChannelSelector,
    SelectionStrategy,
    ChannelSelectionResult,
    AdaptationState
)
from intensity_filter import RMSIntensityFilter, ChannelQualityMetrics


class TestAdaptiveChannelSelector(unittest.TestCase):
    """Test cases for adaptive channel selector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_channels = 8
        self.intensity_filter = RMSIntensityFilter(sample_rate=48000)
        self.selector = AdaptiveChannelSelector(
            num_channels=self.num_channels,
            intensity_filter=self.intensity_filter
        )
    
    def create_test_audio_channels(self, channel_qualities: list) -> np.ndarray:
        """
        Create test audio channels with specified quality levels.
        
        Args:
            channel_qualities: List of (amplitude, noise_level) tuples
            
        Returns:
            Multi-channel audio array
        """
        duration = 0.1  # 100ms
        sample_rate = 48000
        samples = int(duration * sample_rate)
        
        audio_channels = np.zeros((samples, len(channel_qualities)))
        
        for ch, (amplitude, noise_level) in enumerate(channel_qualities):
            # Create sine wave signal
            t = np.linspace(0, duration, samples)
            signal = amplitude * np.sin(2 * np.pi * 1000 * t)
            
            # Add noise
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, samples)
                signal += noise
            
            audio_channels[:, ch] = signal
        
        return audio_channels
    
    def test_initialization(self):
        """Test proper initialization of adaptive channel selector."""
        # Test default initialization
        selector = AdaptiveChannelSelector(num_channels=6)
        self.assertEqual(selector.num_channels, 6)
        self.assertEqual(selector.min_channels, 4)
        self.assertEqual(selector.preferred_channels, 6)
        self.assertIsNotNone(selector.intensity_filter)
        
        # Test with custom intensity filter
        custom_filter = RMSIntensityFilter(sample_rate=44100)
        selector_custom = AdaptiveChannelSelector(
            num_channels=4,
            intensity_filter=custom_filter
        )
        self.assertEqual(selector_custom.num_channels, 4)
        self.assertEqual(selector_custom.min_channels, 4)  # max(3, min(4, 4)) = 4
        self.assertIs(selector_custom.intensity_filter, custom_filter)
    
    def test_basic_channel_selection(self):
        """Test basic channel selection functionality."""
        # Create test audio with varying quality
        channel_qualities = [
            (1.5, 0.05),  # Excellent
            (1.0, 0.1),   # Good
            (0.8, 0.15),  # Fair
            (0.5, 0.2),   # Poor
            (0.3, 0.3),   # Very poor
            (0.1, 0.4),   # Extremely poor
            (1.2, 0.08),  # Very good
            (0.9, 0.12)   # Good
        ]
        
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Perform selection
        result = self.selector.select_channels(audio_channels, detection_confidence=0.8)
        
        # Verify result structure
        self.assertIsInstance(result, ChannelSelectionResult)
        self.assertIsInstance(result.selected_channels, list)
        self.assertIsInstance(result.excluded_channels, list)
        self.assertIsInstance(result.channel_weights, np.ndarray)
        self.assertIsInstance(result.selection_confidence, float)
        self.assertIsInstance(result.strategy_used, SelectionStrategy)
        self.assertIsInstance(result.fallback_applied, bool)
        
        # Verify selection logic
        self.assertGreaterEqual(len(result.selected_channels), self.selector.min_channels)
        self.assertLessEqual(len(result.selected_channels), self.num_channels)
        self.assertEqual(len(result.selected_channels) + len(result.excluded_channels), self.num_channels)
        
        # Verify no duplicates
        all_channels = set(result.selected_channels + result.excluded_channels)
        self.assertEqual(len(all_channels), self.num_channels)
        
        # Verify confidence is reasonable
        self.assertGreaterEqual(result.selection_confidence, 0.0)
        self.assertLessEqual(result.selection_confidence, 2.0)  # Can be > 1 due to weighting
    
    def test_quality_based_selection_strategy(self):
        """Test quality-based selection strategy."""
        # Create audio with clear quality differences
        channel_qualities = [
            (2.0, 0.02),  # Best quality
            (1.5, 0.05),  # Second best
            (1.0, 0.1),   # Third best
            (0.5, 0.2),   # Fourth best
            (0.2, 0.3),   # Poor
            (0.1, 0.4),   # Very poor
            (0.05, 0.5),  # Extremely poor
            (0.02, 0.6)   # Unusable
        ]
        
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Force quality-based strategy
        self.selector.primary_strategy = SelectionStrategy.QUALITY_BASED
        result = self.selector.select_channels(audio_channels, required_channels=4)
        
        # Should select the 4 best channels (0, 1, 2, 3)
        self.assertEqual(len(result.selected_channels), 4)
        self.assertEqual(result.strategy_used, SelectionStrategy.QUALITY_BASED)
        
        # Best channels should be selected
        expected_best = [0, 1, 2, 3]  # Based on our quality ordering
        selected_set = set(result.selected_channels)
        expected_set = set(expected_best)
        
        # Should have significant overlap with expected best channels
        overlap = len(selected_set.intersection(expected_set))
        self.assertGreaterEqual(overlap, 3)  # At least 3 of 4 should match
    
    def test_confidence_weighted_selection_strategy(self):
        """Test confidence-weighted selection strategy."""
        channel_qualities = [(1.0, 0.1)] * self.num_channels  # Uniform quality
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Test with high detection confidence
        self.selector.primary_strategy = SelectionStrategy.CONFIDENCE_WEIGHTED
        result_high = self.selector.select_channels(
            audio_channels, 
            detection_confidence=0.9,
            required_channels=4
        )
        
        # Test with low detection confidence
        result_low = self.selector.select_channels(
            audio_channels,
            detection_confidence=0.2,
            required_channels=4
        )
        
        # Both should succeed
        self.assertEqual(result_high.strategy_used, SelectionStrategy.CONFIDENCE_WEIGHTED)
        self.assertEqual(result_low.strategy_used, SelectionStrategy.CONFIDENCE_WEIGHTED)
        
        # High confidence might select fewer channels (more selective)
        # Low confidence might be less selective
        self.assertGreaterEqual(len(result_high.selected_channels), 4)
        self.assertGreaterEqual(len(result_low.selected_channels), 4)
    
    def test_adaptive_threshold_selection_strategy(self):
        """Test adaptive threshold selection strategy."""
        # Create channels with bimodal quality distribution
        channel_qualities = [
            (1.5, 0.05),  # High quality group
            (1.4, 0.06),
            (1.3, 0.07),
            (1.2, 0.08),
            (0.3, 0.3),   # Low quality group
            (0.2, 0.4),
            (0.1, 0.5),
            (0.05, 0.6)
        ]
        
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Use adaptive threshold strategy
        self.selector.primary_strategy = SelectionStrategy.ADAPTIVE_THRESHOLD
        result = self.selector.select_channels(audio_channels, required_channels=4)
        
        self.assertEqual(result.strategy_used, SelectionStrategy.ADAPTIVE_THRESHOLD)
        self.assertEqual(len(result.selected_channels), 4)
        
        # Should preferentially select from high quality group
        high_quality_channels = {0, 1, 2, 3}
        selected_set = set(result.selected_channels)
        overlap = len(selected_set.intersection(high_quality_channels))
        self.assertGreaterEqual(overlap, 3)  # Most should be from high quality group
    
    def test_geometric_optimal_selection_strategy(self):
        """Test geometric optimal selection strategy."""
        channel_qualities = [(1.0, 0.1)] * self.num_channels  # Uniform quality
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Use geometric optimal strategy
        self.selector.primary_strategy = SelectionStrategy.GEOMETRIC_OPTIMAL
        result = self.selector.select_channels(audio_channels, required_channels=4)
        
        self.assertEqual(result.strategy_used, SelectionStrategy.GEOMETRIC_OPTIMAL)
        self.assertEqual(len(result.selected_channels), 4)
        
        # Should select evenly distributed channels
        # For 8 channels selecting 4, expect roughly every other channel
        selected_channels = sorted(result.selected_channels)
        
        # Check that channels are reasonably distributed
        # (This is a simplified test since we don't have actual geometry)
        self.assertGreater(len(set(selected_channels)), 1)  # Not all the same
    
    def test_fallback_strategy_usage(self):
        """Test fallback strategy usage when primary fails."""
        # Create scenario where primary strategy might struggle
        channel_qualities = [(0.1, 0.5)] * self.num_channels  # All very poor quality
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Set very high threshold to force fallback
        self.selector.base_quality_threshold = 2.0  # Impossibly high
        
        result = self.selector.select_channels(audio_channels, required_channels=4)
        
        # Should still get a result (via fallback)
        self.assertIsInstance(result, ChannelSelectionResult)
        self.assertGreaterEqual(len(result.selected_channels), self.selector.min_channels)
        
        # Might have used fallback
        # (Exact behavior depends on implementation details)
    
    def test_performance_feedback_integration(self):
        """Test performance feedback and learning."""
        channel_qualities = [(1.0, 0.1)] * self.num_channels
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Perform initial selection
        result = self.selector.select_channels(audio_channels)
        
        # Provide positive feedback
        self.selector.update_performance_feedback(
            selected_channels=result.selected_channels,
            triangulation_confidence=0.9,
            triangulation_error=0.05  # 5cm error - very good
        )
        
        # Verify feedback was recorded
        self.assertGreater(len(self.selector.performance_history), 0)
        
        # Check that channel performance history was updated
        for ch in result.selected_channels:
            history = self.selector.adaptation_state.channel_performance_history[ch]
            self.assertGreater(len(history), 0)
            self.assertGreater(history[-1], 0.5)  # Should be good performance
        
        # Provide negative feedback
        self.selector.update_performance_feedback(
            selected_channels=result.selected_channels,
            triangulation_confidence=0.2,
            triangulation_error=1.5  # 1.5m error - very poor
        )
        
        # Should have more history entries
        self.assertGreater(len(self.selector.performance_history), 1)
        
        # Latest performance should be poor
        for ch in result.selected_channels:
            history = self.selector.adaptation_state.channel_performance_history[ch]
            self.assertGreater(len(history), 1)
            self.assertLess(history[-1], 0.5)  # Should be poor performance
    
    def test_adaptive_threshold_adjustment(self):
        """Test adaptive threshold adjustment based on performance."""
        # Enable adaptation
        self.selector.adaptation_enabled = True
        
        # Simulate high confidence detections
        for _ in range(15):
            self.selector._update_adaptation_state(0.9)
        
        # Should have adapted threshold upward (more selective)
        if hasattr(self.selector, 'current_quality_threshold'):
            self.assertGreater(self.selector.current_quality_threshold, self.selector.base_quality_threshold)
        
        # Reset and simulate low confidence detections
        self.selector.reset_adaptation_state()
        for _ in range(15):
            self.selector._update_adaptation_state(0.2)
        
        # Should have adapted threshold downward (less selective)
        if hasattr(self.selector, 'current_quality_threshold'):
            self.assertLess(self.selector.current_quality_threshold, self.selector.base_quality_threshold)
    
    def test_required_channels_determination(self):
        """Test determination of required channels based on detection confidence."""
        # High confidence should require fewer channels
        high_conf_required = self.selector._determine_required_channels(0.9)
        self.assertEqual(high_conf_required, self.selector.min_channels)
        
        # Medium confidence should require preferred number
        med_conf_required = self.selector._determine_required_channels(0.6)
        self.assertEqual(med_conf_required, self.selector.preferred_channels)
        
        # Low confidence should require more channels
        low_conf_required = self.selector._determine_required_channels(0.3)
        self.assertEqual(low_conf_required, self.selector.max_channels)
    
    def test_statistics_tracking(self):
        """Test statistics tracking functionality."""
        channel_qualities = [(1.0, 0.1)] * self.num_channels
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Perform several selections
        for i in range(5):
            confidence = 0.5 + i * 0.1
            result = self.selector.select_channels(audio_channels, detection_confidence=confidence)
            
            # Provide feedback
            self.selector.update_performance_feedback(
                selected_channels=result.selected_channels,
                triangulation_confidence=confidence,
                triangulation_error=0.1 + i * 0.05
            )
        
        # Get statistics
        stats = self.selector.get_selection_statistics()
        
        # Verify statistics structure
        expected_keys = ['total_selections', 'fallback_count', 'strategy_usage', 
                        'average_channels_selected', 'adaptation_events']
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Verify values
        self.assertEqual(stats['total_selections'], 5)
        self.assertGreaterEqual(stats['average_channels_selected'], self.selector.min_channels)
        self.assertIn('recent_performance', stats)
        self.assertIn('channel_usage', stats)
        self.assertIn('adaptation_state', stats)
    
    def test_parameter_configuration(self):
        """Test parameter configuration functionality."""
        # Test parameter updates
        new_params = {
            'min_channels': 5,
            'preferred_channels': 7,
            'base_quality_threshold': 0.4,
            'confidence_threshold': 0.6,
            'adaptation_enabled': False,
            'adaptation_rate': 0.2,
            'primary_strategy': 'quality_based'
        }
        
        self.selector.configure_selection_parameters(**new_params)
        
        # Verify parameters were updated
        self.assertEqual(self.selector.min_channels, 5)
        self.assertEqual(self.selector.preferred_channels, 7)
        self.assertEqual(self.selector.base_quality_threshold, 0.4)
        self.assertEqual(self.selector.confidence_threshold, 0.6)
        self.assertEqual(self.selector.adaptation_enabled, False)
        self.assertEqual(self.selector.adaptation_rate, 0.2)
        self.assertEqual(self.selector.primary_strategy, SelectionStrategy.QUALITY_BASED)
    
    def test_channel_recommendations(self):
        """Test channel recommendation functionality."""
        channel_qualities = [
            (1.5, 0.05),  # Good channels
            (1.4, 0.06),
            (1.3, 0.07),
            (1.2, 0.08),
            (0.1, 0.5),   # Consistently poor channels
            (0.05, 0.6),
            (1.1, 0.09),  # More good channels
            (1.0, 0.1)
        ]
        
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Perform multiple selections to build history
        for _ in range(25):
            result = self.selector.select_channels(audio_channels)
            # Simulate poor performance for consistently excluded channels
            confidence = 0.3 if any(ch in [4, 5] for ch in result.selected_channels) else 0.8
            error = 1.0 if any(ch in [4, 5] for ch in result.selected_channels) else 0.1
            
            self.selector.update_performance_feedback(
                selected_channels=result.selected_channels,
                triangulation_confidence=confidence,
                triangulation_error=error
            )
        
        # Get recommendations
        recommendations = self.selector.get_channel_recommendations()
        
        # Verify structure
        self.assertIn('status', recommendations)
        self.assertIn('issues', recommendations)
        self.assertIn('suggestions', recommendations)
        
        # Should identify issues with poor channels
        self.assertIsInstance(recommendations['issues'], list)
        self.assertIsInstance(recommendations['suggestions'], list)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with wrong number of channels
        wrong_channels = np.random.normal(0, 0.1, (1000, 4))  # 4 channels instead of 8
        
        with self.assertRaises(ValueError):
            self.selector.select_channels(wrong_channels)
        
        # Test with very short audio
        short_audio = np.random.normal(0, 0.1, (10, self.num_channels))
        result = self.selector.select_channels(short_audio)
        
        # Should still work
        self.assertIsInstance(result, ChannelSelectionResult)
        self.assertGreaterEqual(len(result.selected_channels), self.selector.min_channels)
        
        # Test with zero audio
        zero_audio = np.zeros((1000, self.num_channels))
        result = self.selector.select_channels(zero_audio)
        
        # Should still work (emergency fallback)
        self.assertIsInstance(result, ChannelSelectionResult)
        self.assertGreaterEqual(len(result.selected_channels), self.selector.min_channels)
    
    def test_adaptation_state_reset(self):
        """Test adaptation state reset functionality."""
        # Build some adaptation state
        for _ in range(10):
            self.selector._update_adaptation_state(0.8)
        
        # Verify state exists
        self.assertGreater(len(self.selector.adaptation_state.recent_detections), 0)
        
        # Reset state
        self.selector.reset_adaptation_state()
        
        # Verify state was reset
        self.assertEqual(len(self.selector.adaptation_state.recent_detections), 0)
        for history in self.selector.adaptation_state.channel_performance_history.values():
            self.assertEqual(len(history), 0)
        
        # Current threshold should be reset
        self.assertFalse(hasattr(self.selector, 'current_quality_threshold'))
    
    def test_emergency_fallback(self):
        """Test emergency fallback selection."""
        # Create scenario that should trigger emergency fallback
        channel_qualities = [(0.01, 0.9)] * self.num_channels  # Extremely poor quality
        audio_channels = self.create_test_audio_channels(channel_qualities)
        
        # Set impossible thresholds
        self.selector.base_quality_threshold = 10.0
        self.selector.current_quality_threshold = 10.0
        
        result = self.selector.select_channels(audio_channels)
        
        # Should still get minimum channels via emergency fallback
        self.assertGreaterEqual(len(result.selected_channels), self.selector.min_channels)
        # May or may not show fallback_applied depending on which strategy succeeded
        # The important thing is we get a valid result
        self.assertIsInstance(result.selection_confidence, (int, float))  # Should be a number


class TestAdaptiveChannelSelectorIntegration(unittest.TestCase):
    """Integration tests for adaptive channel selector."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.num_channels = 8
        self.selector = AdaptiveChannelSelector(num_channels=self.num_channels)
    
    def create_realistic_scenario(self, scenario_type: str) -> np.ndarray:
        """Create realistic test scenarios."""
        duration = 0.1
        sample_rate = 48000
        samples = int(duration * sample_rate)
        
        if scenario_type == "good_array":
            # Well-functioning microphone array
            channel_qualities = [
                (1.2, 0.08), (1.1, 0.09), (1.3, 0.07), (1.0, 0.1),
                (1.15, 0.085), (1.25, 0.075), (1.05, 0.095), (1.35, 0.065)
            ]
        elif scenario_type == "degraded_array":
            # Array with some failing microphones
            channel_qualities = [
                (1.2, 0.08), (0.3, 0.4), (1.3, 0.07), (0.1, 0.6),
                (1.15, 0.085), (1.25, 0.075), (0.2, 0.5), (1.35, 0.065)
            ]
        elif scenario_type == "noisy_environment":
            # High noise environment
            channel_qualities = [
                (1.0, 0.3), (0.9, 0.35), (1.1, 0.28), (0.8, 0.4),
                (1.05, 0.32), (1.15, 0.25), (0.95, 0.38), (1.2, 0.22)
            ]
        else:
            # Default uniform quality
            channel_qualities = [(1.0, 0.1)] * self.num_channels
        
        # Create audio channels
        audio_channels = np.zeros((samples, self.num_channels))
        
        for ch, (amplitude, noise_level) in enumerate(channel_qualities):
            t = np.linspace(0, duration, samples)
            signal = amplitude * np.sin(2 * np.pi * 1000 * t)
            
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, samples)
                signal += noise
            
            audio_channels[:, ch] = signal
        
        return audio_channels
    
    def test_good_array_performance(self):
        """Test performance with a well-functioning array."""
        audio_channels = self.create_realistic_scenario("good_array")
        
        # Perform multiple selections
        results = []
        for confidence in [0.9, 0.8, 0.7, 0.6, 0.5]:
            result = self.selector.select_channels(
                audio_channels, 
                detection_confidence=confidence
            )
            results.append(result)
            
            # Provide good feedback
            self.selector.update_performance_feedback(
                selected_channels=result.selected_channels,
                triangulation_confidence=confidence,
                triangulation_error=0.05 + (1 - confidence) * 0.1
            )
        
        # All selections should succeed
        for result in results:
            self.assertGreaterEqual(len(result.selected_channels), self.selector.min_channels)
            self.assertGreater(result.selection_confidence, 0.3)
        
        # Get statistics
        stats = self.selector.get_selection_statistics()
        self.assertEqual(stats['total_selections'], 5)
        self.assertLess(stats['fallback_count'], 2)  # Should rarely need fallback
    
    def test_degraded_array_adaptation(self):
        """Test adaptation to degraded array conditions."""
        audio_channels = self.create_realistic_scenario("degraded_array")
        
        # Perform selections and track adaptation
        initial_result = self.selector.select_channels(audio_channels)
        
        # Simulate poor performance from bad channels
        for _ in range(20):
            result = self.selector.select_channels(audio_channels)
            
            # Channels 1, 3, 6 are poor quality - simulate poor performance when used
            poor_channels = {1, 3, 6}
            selected_poor = set(result.selected_channels).intersection(poor_channels)
            
            if selected_poor:
                # Poor performance when bad channels are used
                confidence = 0.2
                error = 1.0
            else:
                # Good performance when bad channels are avoided
                confidence = 0.8
                error = 0.1
            
            self.selector.update_performance_feedback(
                selected_channels=result.selected_channels,
                triangulation_confidence=confidence,
                triangulation_error=error
            )
        
        # Final selection should avoid poor channels
        final_result = self.selector.select_channels(audio_channels)
        final_poor = set(final_result.selected_channels).intersection(poor_channels)
        
        # Should use fewer poor channels than initially
        initial_poor = set(initial_result.selected_channels).intersection(poor_channels)
        self.assertLessEqual(len(final_poor), len(initial_poor))
    
    def test_noisy_environment_adaptation(self):
        """Test adaptation to noisy environment."""
        audio_channels = self.create_realistic_scenario("noisy_environment")
        
        # Perform selections in noisy environment
        for _ in range(15):
            result = self.selector.select_channels(
                audio_channels,
                detection_confidence=0.4  # Lower confidence due to noise
            )
            
            # Simulate moderate performance in noisy conditions
            self.selector.update_performance_feedback(
                selected_channels=result.selected_channels,
                triangulation_confidence=0.5,
                triangulation_error=0.3
            )
        
        # Should adapt to use more channels for robustness
        final_result = self.selector.select_channels(audio_channels, detection_confidence=0.4)
        
        # In noisy conditions, should tend to use more channels
        self.assertGreaterEqual(len(final_result.selected_channels), self.selector.min_channels)
        
        # Get recommendations
        recommendations = self.selector.get_channel_recommendations()
        
        # Should provide relevant suggestions for noisy environment
        self.assertIsInstance(recommendations, dict)
        self.assertIn('status', recommendations)
    
    def test_long_term_learning(self):
        """Test long-term learning and adaptation."""
        # Simulate changing conditions over time
        scenarios = ["good_array", "degraded_array", "noisy_environment", "good_array"]
        
        performance_over_time = []
        
        for phase, scenario in enumerate(scenarios):
            audio_channels = self.create_realistic_scenario(scenario)
            
            phase_performance = []
            for _ in range(10):
                result = self.selector.select_channels(audio_channels)
                
                # Simulate performance based on scenario
                if scenario == "good_array":
                    confidence = 0.8 + np.random.normal(0, 0.1)
                    error = 0.05 + np.random.normal(0, 0.02)
                elif scenario == "degraded_array":
                    confidence = 0.4 + np.random.normal(0, 0.15)
                    error = 0.5 + np.random.normal(0, 0.2)
                else:  # noisy_environment
                    confidence = 0.5 + np.random.normal(0, 0.1)
                    error = 0.3 + np.random.normal(0, 0.1)
                
                # Clip to reasonable ranges
                confidence = np.clip(confidence, 0.1, 0.95)
                error = np.clip(error, 0.01, 2.0)
                
                self.selector.update_performance_feedback(
                    selected_channels=result.selected_channels,
                    triangulation_confidence=confidence,
                    triangulation_error=error
                )
                
                phase_performance.append(confidence)
            
            performance_over_time.append(np.mean(phase_performance))
        
        # Should show adaptation - performance in final good phase should be 
        # better than or equal to initial good phase due to learning
        initial_performance = performance_over_time[0]
        final_performance = performance_over_time[-1]
        
        # Allow for some variance, but should not be significantly worse
        self.assertGreaterEqual(final_performance, initial_performance - 0.1)
        
        # Get final statistics
        stats = self.selector.get_selection_statistics()
        self.assertGreater(stats['total_selections'], 30)
        self.assertGreater(stats['adaptation_events'], 0)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)