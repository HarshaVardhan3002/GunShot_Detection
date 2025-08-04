#!/usr/bin/env python3
"""
Demo script for enhanced AudioCaptureEngine threading and synchronization features.
"""
import time
import numpy as np
import threading
from audio_capture import AudioCaptureEngine


def simulate_audio_processing(engine, duration=5):
    """Simulate continuous audio processing."""
    print(f"Simulating audio processing for {duration} seconds...")
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration:
        # Simulate audio callback with synthetic data
        frames = 1024
        synthetic_audio = np.random.random((frames, 8)).astype(np.float32) * 0.1
        
        # Add some "events" to specific channels
        if frame_count % 100 == 0:  # Every 100th frame
            synthetic_audio[:, 0] *= 5  # Spike on channel 1
            synthetic_audio[:, 3] *= 3  # Spike on channel 4
        
        engine._audio_callback(synthetic_audio, frames, None, None)
        frame_count += 1
        
        # Small delay to simulate real-time processing
        time.sleep(0.02)  # ~50 FPS
    
    print(f"Processed {frame_count} frames")


def monitor_system_health(engine, duration=5):
    """Monitor system health during operation."""
    print("Starting health monitoring...")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Get comprehensive statistics
        stats = engine.get_capture_statistics()
        sync_status = engine.get_synchronization_status()
        health = engine.perform_health_check()
        
        print(f"\n--- Health Report (t={time.time()-start_time:.1f}s) ---")
        print(f"Samples captured: {stats['samples_captured']}")
        print(f"Buffer utilization: {stats['buffer_utilization']:.1f}%")
        print(f"Active channels: {stats['active_channels']}/{stats['channels']}")
        print(f"Sync status: {sync_status.get('status', 'unknown')}")
        print(f"Overall health: {health['overall_status']}")
        
        if health['warnings']:
            print(f"Warnings: {', '.join(health['warnings'])}")
        
        if health['issues']:
            print(f"Issues: {', '.join(health['issues'])}")
        
        time.sleep(1)  # Update every second


def test_buffer_operations(engine):
    """Test various buffer operations."""
    print("\n=== Testing Buffer Operations ===")
    
    # Test synchronized buffer segment
    segment = engine.get_synchronized_buffer_segment(0.5)
    if segment:
        print(f"✓ Synchronized segment: {segment.data.shape} samples, {segment.duration}s")
    else:
        print("✗ Failed to get synchronized segment")
    
    # Test buffer freezing
    frozen = engine.freeze_buffer_on_trigger()
    if frozen:
        print(f"✓ Frozen buffer: {frozen.data.shape} samples")
        print(f"  Timestamp: {frozen.timestamp}")
    else:
        print("✗ Failed to freeze buffer")
    
    # Test buffer size adjustment
    original_size = engine.buffer_size
    if engine.adjust_buffer_size(1.5):  # 1.5 seconds
        print(f"✓ Buffer size adjusted: {original_size} → {engine.buffer_size} samples")
        
        # Adjust back
        engine.adjust_buffer_size(2.0)
        print(f"✓ Buffer size restored: {engine.buffer_size} samples")
    else:
        print("✗ Failed to adjust buffer size")


def test_channel_monitoring(engine):
    """Test channel health monitoring."""
    print("\n=== Testing Channel Monitoring ===")
    
    # Simulate different channel conditions
    test_scenarios = [
        ("Normal operation", np.random.random((1000, 8)) * 0.1),
        ("Channel 1 failure", np.random.random((1000, 8)) * 0.1),
        ("High noise", np.random.random((1000, 8)) * 0.5),
        ("Mixed conditions", np.random.random((1000, 8)) * 0.1)
    ]
    
    # Modify scenarios
    test_scenarios[1][1][:, 0] = 0  # Channel 1 silent
    test_scenarios[3][1][:, 2] = 0  # Channel 3 silent
    test_scenarios[3][1][:, 4] *= 10  # Channel 5 very loud
    
    for scenario_name, test_data in test_scenarios:
        print(f"\nScenario: {scenario_name}")
        
        # Process several frames to build history
        for i in range(10):
            frame_data = test_data[i*100:(i+1)*100]
            engine._audio_callback(frame_data, 100, None, None)
        
        # Check channel health
        channel_health = engine.get_channel_health()
        
        active_channels = [ch for ch, health in channel_health.items() if health['active']]
        inactive_channels = [ch for ch, health in channel_health.items() if not health['active']]
        
        print(f"  Active channels: {active_channels}")
        if inactive_channels:
            print(f"  Inactive channels: {inactive_channels}")
        
        # Show detailed health for first few channels
        for ch in range(1, min(4, len(channel_health) + 1)):
            health = channel_health[ch]
            print(f"  Ch{ch}: RMS={health['avg_rms']:.4f}, Activity={health['activity_ratio']:.2f}")


def main():
    """Main demo function."""
    print("Enhanced AudioCaptureEngine Threading & Synchronization Demo")
    print("=" * 60)
    
    # Create engine with enhanced features
    engine = AudioCaptureEngine(
        sample_rate=48000,
        channels=8,
        buffer_duration=2.0
    )
    
    print(f"Initialized: {engine.sample_rate}Hz, {engine.channels} channels")
    print(f"Buffer size: {engine.buffer_size} samples ({engine.buffer_duration}s)")
    
    # Enable capturing mode for testing
    engine._capturing = True
    
    try:
        # Test buffer operations
        test_buffer_operations(engine)
        
        # Test channel monitoring
        test_channel_monitoring(engine)
        
        print("\n=== Starting Concurrent Processing Demo ===")
        
        # Start concurrent threads
        audio_thread = threading.Thread(
            target=simulate_audio_processing, 
            args=(engine, 5)
        )
        
        monitor_thread = threading.Thread(
            target=monitor_system_health, 
            args=(engine, 5)
        )
        
        # Start threads
        audio_thread.start()
        monitor_thread.start()
        
        # Wait for completion
        audio_thread.join()
        monitor_thread.join()
        
        print("\n=== Final System Status ===")
        
        # Final statistics
        final_stats = engine.get_capture_statistics()
        final_health = engine.perform_health_check()
        
        print(f"Total samples processed: {final_stats['samples_captured']}")
        print(f"Buffer overruns: {final_stats['buffer_overruns']}")
        print(f"Sync errors: {final_stats['sync_errors']}")
        print(f"Average callback latency: {final_stats['avg_callback_latency_ms']:.2f}ms")
        print(f"Final health status: {final_health['overall_status']}")
        
        if final_health['recommendations']:
            print("Recommendations:")
            for rec in final_health['recommendations']:
                print(f"  • {rec}")
        
        print("\n✓ Demo completed successfully!")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        engine._capturing = False


if __name__ == "__main__":
    main()