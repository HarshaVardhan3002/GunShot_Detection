#!/usr/bin/env python3
"""
Performance and Integration Test Runner for Gunshot Localization System.
This script runs comprehensive tests to validate system performance and accuracy.
"""
import time
import numpy as np
import psutil
from gunshot_detector import AmplitudeBasedDetector
from tdoa_localizer import CrossCorrelationTDoALocalizer
from intensity_filter import RMSIntensityFilter
from adaptive_channel_selector import AdaptiveChannelSelector


def generate_test_audio(sample_rate=48000, channels=8, duration=0.2):
    """Generate test audio data for performance testing."""
    num_samples = int(sample_rate * duration)
    
    # Create realistic gunshot signal
    t = np.linspace(0, duration, num_samples)
    signal = np.exp(-t * 30) * np.sin(2 * np.pi * 1500 * t) * 2.0  # Stronger signal
    noise = np.random.randn(num_samples) * 0.01  # Less noise
    base_signal = signal + noise
    
    # Multi-channel with delays
    audio_data = np.zeros((num_samples, channels))
    for i in range(channels):
        delay = int(i * 0.0005 * sample_rate)  # 0.5ms delays
        if delay < num_samples:
            delayed_signal = np.zeros(num_samples)
            delayed_signal[delay:] = base_signal[:num_samples-delay]
            audio_data[:, i] = delayed_signal * (1.0 / (1.0 + i * 0.05))
    
    return audio_data


def test_detection_latency():
    """Test gunshot detection latency."""
    print("Testing Detection Latency...")
    print("-" * 40)
    
    sample_rate = 48000
    channels = 8
    test_audio = generate_test_audio(sample_rate, channels)
    
    detector = AmplitudeBasedDetector(
        sample_rate=sample_rate,
        channels=channels,
        threshold_db=-20.0
    )
    
    # Measure detection latency over multiple iterations
    latencies = []
    detections = 0
    
    for i in range(100):
        start_time = time.perf_counter()
        is_detected, confidence, metadata = detector.detect_gunshot(test_audio)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        if is_detected:
            detections += 1
    
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    p95_latency = np.percentile(latencies, 95)
    detection_rate = detections / len(latencies)
    
    print(f"Detection Results:")
    print(f"  Iterations: {len(latencies)}")
    print(f"  Detection rate: {detection_rate:.1%}")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Min latency: {min_latency:.2f}ms")
    print(f"  Max latency: {max_latency:.2f}ms")
    print(f"  95th percentile: {p95_latency:.2f}ms")
    
    # Check requirements
    latency_ok = avg_latency < 50 and max_latency < 200
    detection_ok = detection_rate > 0.8
    
    print(f"  Latency requirement (< 50ms avg): {'PASS' if latency_ok else 'FAIL'}")
    print(f"  Detection requirement (> 80%): {'PASS' if detection_ok else 'FAIL'}")
    
    return latency_ok and detection_ok


def test_localization_latency():
    """Test TDoA localization latency."""
    print("\nTesting Localization Latency...")
    print("-" * 40)
    
    sample_rate = 48000
    channels = 8
    test_audio = generate_test_audio(sample_rate, channels)
    
    mic_positions = [(i*0.5, i*0.5, 0.0) for i in range(channels)]
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=sample_rate,
        sound_speed=343.0
    )
    
    # Measure localization latency
    latencies = []
    successful_localizations = 0
    
    for i in range(50):  # Fewer iterations as this is more computationally intensive
        start_time = time.perf_counter()
        
        tdoa_matrix = localizer.calculate_tdoa(test_audio)
        location_result = localizer.triangulate_source(tdoa_matrix)
        
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        if location_result and location_result.confidence > 0:
            successful_localizations += 1
    
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    p95_latency = np.percentile(latencies, 95)
    success_rate = successful_localizations / len(latencies)
    
    print(f"Localization Results:")
    print(f"  Iterations: {len(latencies)}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Min latency: {min_latency:.2f}ms")
    print(f"  Max latency: {max_latency:.2f}ms")
    print(f"  95th percentile: {p95_latency:.2f}ms")
    
    # Check requirements (500ms target)
    latency_ok = avg_latency < 300 and max_latency < 500
    success_ok = success_rate > 0.6
    
    print(f"  Latency requirement (< 300ms avg, < 500ms max): {'PASS' if latency_ok else 'FAIL'}")
    print(f"  Success requirement (> 60%): {'PASS' if success_ok else 'FAIL'}")
    
    return latency_ok and success_ok


def test_end_to_end_latency():
    """Test complete end-to-end processing latency."""
    print("\nTesting End-to-End Latency...")
    print("-" * 40)
    
    sample_rate = 48000
    channels = 8
    test_audio = generate_test_audio(sample_rate, channels)
    
    # Create all components
    detector = AmplitudeBasedDetector(
        sample_rate=sample_rate,
        channels=channels,
        threshold_db=-20.0
    )
    
    mic_positions = [(i*0.5, i*0.5, 0.0) for i in range(channels)]
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=sample_rate,
        sound_speed=343.0
    )
    
    intensity_filter = RMSIntensityFilter(sample_rate=sample_rate)
    channel_selector = AdaptiveChannelSelector(
        num_channels=channels,
        intensity_filter=intensity_filter
    )
    
    # Measure end-to-end latency
    latencies = []
    successful_localizations = 0
    
    for i in range(30):
        start_time = time.perf_counter()
        
        # Complete processing pipeline
        is_detected, confidence, metadata = detector.detect_gunshot(test_audio)
        
        if is_detected:
            channel_result = channel_selector.select_channels(
                test_audio, detection_confidence=confidence
            )
            
            if channel_result and len(channel_result.selected_channels) >= 4:
                tdoa_matrix = localizer.calculate_tdoa(test_audio)
                location_result = localizer.triangulate_source(tdoa_matrix)
                
                if location_result and location_result.confidence > 0:
                    successful_localizations += 1
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    p95_latency = np.percentile(latencies, 95)
    success_rate = successful_localizations / len(latencies)
    
    print(f"End-to-End Results:")
    print(f"  Iterations: {len(latencies)}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Average latency: {avg_latency:.2f}ms")
    print(f"  Min latency: {min_latency:.2f}ms")
    print(f"  Max latency: {max_latency:.2f}ms")
    print(f"  95th percentile: {p95_latency:.2f}ms")
    
    # Check requirements (500ms target)
    latency_ok = avg_latency < 500 and max_latency < 600
    success_ok = success_rate > 0.7
    
    print(f"  Latency requirement (< 500ms avg): {'PASS' if latency_ok else 'FAIL'}")
    print(f"  Success requirement (> 70%): {'PASS' if success_ok else 'FAIL'}")
    
    return latency_ok and success_ok


def test_memory_usage():
    """Test memory usage during processing."""
    print("\nTesting Memory Usage...")
    print("-" * 40)
    
    sample_rate = 48000
    channels = 8
    test_audio = generate_test_audio(sample_rate, channels)
    
    detector = AmplitudeBasedDetector(
        sample_rate=sample_rate,
        channels=channels,
        threshold_db=-20.0
    )
    
    mic_positions = [(i*0.5, i*0.5, 0.0) for i in range(channels)]
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=sample_rate,
        sound_speed=343.0
    )
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run many iterations to check memory usage
    iterations = 200
    for i in range(iterations):
        is_detected, confidence, metadata = detector.detect_gunshot(test_audio)
        if is_detected:
            tdoa_matrix = localizer.calculate_tdoa(test_audio)
            location_result = localizer.triangulate_source(tdoa_matrix)
        
        # Check memory every 50 iterations
        if i % 50 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"  Iteration {i}: Memory usage: {current_memory:.1f}MB "
                  f"(+{current_memory - initial_memory:.1f}MB)")
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"Memory Usage Results:")
    print(f"  Initial memory: {initial_memory:.1f}MB")
    print(f"  Final memory: {final_memory:.1f}MB")
    print(f"  Memory increase: {memory_increase:.1f}MB")
    print(f"  Iterations: {iterations}")
    
    # Check memory usage is reasonable
    memory_ok = memory_increase < 50  # Less than 50MB increase
    
    print(f"  Memory requirement (< 50MB increase): {'PASS' if memory_ok else 'FAIL'}")
    
    return memory_ok


def test_stress_operation():
    """Test system under stress conditions."""
    print("\nTesting Stress Operation...")
    print("-" * 40)
    
    sample_rate = 48000
    channels = 8
    test_audio = generate_test_audio(sample_rate, channels, duration=0.1)  # Shorter for stress test
    
    detector = AmplitudeBasedDetector(
        sample_rate=sample_rate,
        channels=channels,
        threshold_db=-20.0
    )
    
    mic_positions = [(i*0.6, i*0.6, 0.0) for i in range(channels)]
    localizer = CrossCorrelationTDoALocalizer(
        microphone_positions=mic_positions,
        sample_rate=sample_rate,
        sound_speed=343.0
    )
    
    # Stress test parameters
    duration_seconds = 10  # 10 seconds of stress testing
    events_per_second = 10  # High event rate
    total_events = duration_seconds * events_per_second
    
    print(f"Running stress test: {total_events} events over {duration_seconds} seconds")
    
    start_time = time.time()
    successful_detections = 0
    successful_localizations = 0
    processing_times = []
    
    for i in range(total_events):
        event_start = time.time()
        
        # Detection and localization
        is_detected, confidence, metadata = detector.detect_gunshot(test_audio)
        if is_detected:
            successful_detections += 1
            
            tdoa_matrix = localizer.calculate_tdoa(test_audio)
            location_result = localizer.triangulate_source(tdoa_matrix)
            
            if location_result and location_result.confidence > 0:
                successful_localizations += 1
        
        event_end = time.time()
        processing_time = (event_end - event_start) * 1000
        processing_times.append(processing_time)
        
        # Control event rate
        target_interval = 1.0 / events_per_second
        elapsed = event_end - event_start
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        
        # Progress reporting
        if i % 20 == 0 and i > 0:
            elapsed_time = time.time() - start_time
            print(f"  Progress: {i}/{total_events} events, {elapsed_time:.1f}s elapsed")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Analyze results
    detection_rate = successful_detections / total_events
    localization_rate = successful_localizations / total_events
    avg_processing_time = np.mean(processing_times)
    max_processing_time = np.max(processing_times)
    
    print(f"Stress Test Results:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Events processed: {total_events}")
    print(f"  Detection rate: {detection_rate:.1%}")
    print(f"  Localization rate: {localization_rate:.1%}")
    print(f"  Avg processing time: {avg_processing_time:.1f}ms")
    print(f"  Max processing time: {max_processing_time:.1f}ms")
    
    # Check stress test requirements
    detection_ok = detection_rate > 0.7
    localization_ok = localization_rate > 0.5
    performance_ok = avg_processing_time < 100 and max_processing_time < 500
    
    print(f"  Detection requirement (> 70%): {'PASS' if detection_ok else 'FAIL'}")
    print(f"  Localization requirement (> 50%): {'PASS' if localization_ok else 'FAIL'}")
    print(f"  Performance requirement: {'PASS' if performance_ok else 'FAIL'}")
    
    return detection_ok and localization_ok and performance_ok


def main():
    """Run all performance and integration tests."""
    print("Gunshot Localization System - Performance & Integration Tests")
    print("=" * 70)
    
    # Run all tests
    tests = [
        ("Detection Latency", test_detection_latency),
        ("Localization Latency", test_localization_latency),
        ("End-to-End Latency", test_end_to_end_latency),
        ("Memory Usage", test_memory_usage),
        ("Stress Operation", test_stress_operation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    overall_success = passed == total
    print(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
    
    return overall_success


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)