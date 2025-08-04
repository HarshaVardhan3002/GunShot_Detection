"""
Integration and Performance Test Suite for Gunshot Localization System.
"""
import unittest
import numpy as np
import time
from gunshot_detector import AmplitudeBasedDetector
from tdoa_localizer import CrossCorrelationTDoALocalizer


class TestLatencyValidation(unittest.TestCase):
    """Tests to verify the 500ms processing latency requirement."""
    
    def setUp(self):
        """Set up latency test fixtures."""
        self.sample_rate = 48000
        self.channels = 8
        self.target_latency_ms = 500
        
        # Create test audio data
        duration = 0.2
        num_samples = int(self.sample_rate * duration)
        
        # Create realistic gunshot signal
        t = np.linspace(0, duration, num_samples)
        signal = np.exp(-t * 50) * np.sin(2 * np.pi * 1500 * t)
        noise = np.random.randn(num_samples) * 0.03
        base_signal = signal + noise
        
        # Multi-channel with delays
        self.test_audio = np.zeros((num_samples, self.channels))
        for i in range(self.channels):
            delay = int(i * 0.0005 * self.sample_rate)  # 0.5ms delays
            if delay < num_samples:
                delayed_signal = np.zeros(num_samples)
                delayed_signal[delay:] = base_signal[:num_samples-delay]
                self.test_audio[:, i] = delayed_signal * (1.0 / (1.0 + i * 0.05))
    
    def test_detection_latency(self):
        """Test gunshot detection latency."""
        detector = AmplitudeBasedDetector(
            sample_rate=self.sample_rate,
            channels=self.channels,
            threshold_db=-20.0
        )
        
        # Measure detection latency over multiple iterations
        latencies = []
        for _ in range(50):
            start_time = time.perf_counter()
            is_detected, confidence, metadata = detector.detect_gunshot(self.test_audio)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        print(f"Detection latency - Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")
        
        # Verify latency requirements
        self.assertLess(avg_latency, 50)  # Average should be much less than 500ms
        self.assertLess(max_latency, 200)  # Even worst case should be acceptable
    
    def test_localization_latency(self):
        """Test TDoA localization latency."""
        mic_positions = [(i*0.5, i*0.5, 0.0) for i in range(self.channels)]
        localizer = CrossCorrelationTDoALocalizer(
            microphone_positions=mic_positions,
            sample_rate=self.sample_rate,
            sound_speed=343.0
        )
        
        # Measure localization latency
        latencies = []
        for _ in range(20):  # Fewer iterations as this is more computationally intensive
            start_time = time.perf_counter()
            
            tdoa_matrix = localizer.calculate_tdoa(self.test_audio)
            location_result = localizer.triangulate_source(tdoa_matrix)
            
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        print(f"Localization latency - Avg: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")
        
        # Verify latency requirements
        self.assertLess(avg_latency, 300)  # Should be well under 500ms
        self.assertLess(max_latency, self.target_latency_ms)  # Meet hard requirement


if __name__ == '__main__':
    unittest.main(verbosity=2)