"""
Demo script for error handling and recovery system.
"""
import time
import threading
from error_handler import (
    ErrorHandlingSystem, ErrorSeverity, ErrorCategory,
    handle_component_error, MicrophoneFailureHandler,
    ProcessingTimeoutHandler, ResourceExhaustionHandler
)


def demo_basic_error_handling():
    """Demo basic error handling functionality."""
    print("=== Basic Error Handling Demo ===")
    
    # Create error handling system
    error_handler = ErrorHandlingSystem()
    
    # Register components
    components = ["audio_capture", "gunshot_detector", "tdoa_localizer", "channel_selector"]
    for component in components:
        error_handler.register_component(component)
        print(f"Registered component: {component}")
    
    print(f"\nInitial system health: {error_handler.get_system_health()['overall_health_score']:.2f}")
    
    # Simulate various errors
    print("\n1. Simulating software errors...")
    
    # Low severity error
    error_handler.report_error(
        component="gunshot_detector",
        error=ValueError("Invalid threshold parameter"),
        severity=ErrorSeverity.LOW,
        category=ErrorCategory.SOFTWARE,
        context={"threshold": -1.0}
    )
    
    # Medium severity error
    error_handler.report_error(
        component="tdoa_localizer",
        error=RuntimeError("Convergence failed"),
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.SOFTWARE,
        context={"iterations": 100, "residual": 0.5}
    )
    
    print(f"Health after software errors: {error_handler.get_system_health()['overall_health_score']:.2f}")
    
    # Hardware error
    print("\n2. Simulating hardware error...")
    error_handler.report_error(
        component="audio_capture",
        error=IOError("Microphone 3 disconnected"),
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.HARDWARE,
        context={"microphone_id": 3, "total_microphones": 8}
    )
    
    print(f"Health after hardware error: {error_handler.get_system_health()['overall_health_score']:.2f}")
    
    # Critical error
    print("\n3. Simulating critical error...")
    error_handler.report_error(
        component="audio_capture",
        error=MemoryError("Out of memory"),
        severity=ErrorSeverity.CRITICAL,
        category=ErrorCategory.RESOURCE,
        context={"memory_usage": "95%"}
    )
    
    health = error_handler.get_system_health()
    print(f"Health after critical error: {health['overall_health_score']:.2f}")
    print(f"System degraded: {health['system_degraded']}")
    print(f"Critical errors: {health['critical_errors']}")
    
    # Show error statistics
    stats = error_handler.get_error_statistics()
    print(f"\nError Statistics:")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  Errors by category: {stats['errors_by_category']}")
    print(f"  Errors by severity: {stats['errors_by_severity']}")
    print(f"  Most problematic component: {stats['most_problematic_component']}")
    
    error_handler.shutdown()


def demo_microphone_failure_recovery():
    """Demo microphone failure recovery."""
    print("\n=== Microphone Failure Recovery Demo ===")
    
    error_handler = ErrorHandlingSystem()
    error_handler.register_component("audio_capture")
    
    print("Simulating progressive microphone failures...")
    
    # Simulate microphone failures one by one
    for mic_id in range(1, 6):  # Fail 5 microphones
        print(f"\nFailing microphone {mic_id}...")
        
        success = error_handler.report_error(
            component="audio_capture",
            error=IOError(f"Microphone {mic_id} failed"),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.HARDWARE,
            context={
                "microphone_id": mic_id,
                "total_microphones": 8
            }
        )
        
        remaining_mics = 8 - mic_id
        print(f"  Recovery successful: {success}")
        print(f"  Remaining microphones: {remaining_mics}")
        
        if not success:
            print("  ⚠️  System cannot continue with insufficient microphones!")
            break
        elif remaining_mics == 4:
            print("  ⚠️  At minimum microphone threshold!")
    
    # Show final health
    health = error_handler.get_system_health()
    print(f"\nFinal system health: {health['overall_health_score']:.2f}")
    
    error_handler.shutdown()


def demo_timeout_retry_mechanism():
    """Demo timeout retry mechanism."""
    print("\n=== Timeout Retry Mechanism Demo ===")
    
    error_handler = ErrorHandlingSystem()
    error_handler.register_component("processor")
    
    print("Simulating processing timeouts with retry logic...")
    
    # Simulate repeated timeouts for the same operation
    for attempt in range(5):
        print(f"\nAttempt {attempt + 1}: Processing timeout...")
        
        success = error_handler.report_error(
            component="processor",
            error=TimeoutError("Processing timeout exceeded"),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
            context={
                "operation": "triangulation",
                "timeout_ms": 100,
                "attempt": attempt + 1
            }
        )
        
        print(f"  Recovery successful: {success}")
        
        if not success:
            print("  ⚠️  Max retries exceeded - operation failed!")
            break
        
        time.sleep(0.1)  # Brief pause between attempts
    
    error_handler.shutdown()


def demo_resource_exhaustion_handling():
    """Demo resource exhaustion handling."""
    print("\n=== Resource Exhaustion Handling Demo ===")
    
    error_handler = ErrorHandlingSystem()
    error_handler.register_component("memory_manager")
    
    print("Simulating memory exhaustion...")
    
    success = error_handler.report_error(
        component="memory_manager",
        error=MemoryError("Insufficient memory for buffer allocation"),
        severity=ErrorSeverity.CRITICAL,
        category=ErrorCategory.RESOURCE,
        context={
            "requested_memory_mb": 500,
            "available_memory_mb": 100,
            "buffer_size": 1024000
        }
    )
    
    print(f"Recovery successful: {success}")
    
    if success:
        print("✅ Memory management recovered - buffers reduced")
    else:
        print("❌ Memory recovery failed - system restart may be needed")
    
    error_handler.shutdown()


def demo_health_monitoring():
    """Demo continuous health monitoring."""
    print("\n=== Health Monitoring Demo ===")
    
    error_handler = ErrorHandlingSystem()
    
    # Register multiple components
    components = ["audio", "detector", "localizer", "output"]
    for comp in components:
        error_handler.register_component(comp)
    
    # Start health monitoring
    error_handler.start_health_monitoring()
    print("Health monitoring started...")
    
    # Simulate errors over time
    def simulate_errors():
        """Simulate errors in background thread."""
        time.sleep(1)
        
        # Simulate various error patterns
        error_patterns = [
            ("audio", ValueError("Signal processing error")),
            ("detector", TimeoutError("Detection timeout")),
            ("localizer", RuntimeError("Triangulation failed")),
            ("output", IOError("Output device error")),
            ("audio", IOError("Microphone disconnected"))
        ]
        
        for component, error in error_patterns:
            handle_component_error(error_handler, component, error)
            time.sleep(0.5)
    
    # Start error simulation in background
    error_thread = threading.Thread(target=simulate_errors)
    error_thread.start()
    
    # Monitor health for a few seconds
    for i in range(6):
        time.sleep(1)
        health = error_handler.get_system_health()
        stats = error_handler.get_error_statistics()
        
        print(f"\nHealth Check {i+1}:")
        print(f"  Overall Health: {health['overall_health_score']:.2f}")
        print(f"  Total Errors: {stats['total_errors']}")
        print(f"  Recent Errors: {stats['recent_errors_1h']}")
        print(f"  System Degraded: {health['system_degraded']}")
        
        # Show component health
        for comp_name, comp_health in health['component_health'].items():
            status = "✅" if comp_health['is_healthy'] else "❌"
            print(f"    {comp_name}: {status} (score: {comp_health['health_score']:.2f})")
    
    # Wait for error simulation to complete
    error_thread.join()
    
    # Stop monitoring
    error_handler.stop_health_monitoring()
    print("\nHealth monitoring stopped")
    
    # Final statistics
    final_stats = error_handler.get_error_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total Errors: {final_stats['total_errors']}")
    print(f"  Error Rate: {final_stats['error_rate_per_hour']:.1f} errors/hour")
    print(f"  Most Problematic: {final_stats['most_problematic_component']}")
    
    error_handler.shutdown()


def demo_system_degradation():
    """Demo system degradation and recovery."""
    print("\n=== System Degradation Demo ===")
    
    error_handler = ErrorHandlingSystem()
    error_handler.register_component("critical_system")
    
    print("Simulating escalating system problems...")
    
    # Start with minor issues
    print("\n1. Minor issues (should recover):")
    for i in range(3):
        error_handler.report_error(
            component="critical_system",
            error=ValueError(f"Minor error {i+1}"),
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.SOFTWARE
        )
    
    health = error_handler.get_system_health()
    print(f"   Health after minor issues: {health['overall_health_score']:.2f}")
    
    # Escalate to serious problems
    print("\n2. Serious problems (degraded mode):")
    for i in range(2):
        error_handler.report_error(
            component="critical_system",
            error=RuntimeError(f"Serious error {i+1}"),
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SOFTWARE
        )
    
    health = error_handler.get_system_health()
    print(f"   Health after serious problems: {health['overall_health_score']:.2f}")
    print(f"   System degraded: {health['system_degraded']}")
    
    # Critical failures
    print("\n3. Critical failures (restart needed):")
    for i in range(3):
        error_handler.report_error(
            component="critical_system",
            error=SystemError(f"Critical failure {i+1}"),
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.RESOURCE
        )
    
    health = error_handler.get_system_health()
    print(f"   Health after critical failures: {health['overall_health_score']:.2f}")
    print(f"   Critical errors: {health['critical_errors']}")
    
    if health['critical_errors'] >= 3:
        print("   🚨 SYSTEM RESTART RECOMMENDED 🚨")
    
    # Show recovery recommendations
    print("\n4. Recovery recommendations:")
    # This would be implemented in the main pipeline
    recommendations = [
        "Restart critical_system component",
        "Check hardware connections",
        "Review system logs for root cause",
        "Consider system reboot if problems persist"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    error_handler.shutdown()


def demo_error_recovery_strategies():
    """Demo different error recovery strategies."""
    print("\n=== Error Recovery Strategies Demo ===")
    
    error_handler = ErrorHandlingSystem()
    
    # Register components
    components = ["audio_hw", "processing", "network", "storage"]
    for comp in components:
        error_handler.register_component(comp)
    
    print("Testing different recovery strategies...")
    
    # Strategy 1: Ignore and continue (low severity)
    print("\n1. IGNORE strategy (low severity):")
    success = error_handler.report_error(
        component="processing",
        error=UserWarning("Minor processing anomaly"),
        severity=ErrorSeverity.LOW,
        category=ErrorCategory.SOFTWARE
    )
    print(f"   Result: {'✅ Continued' if success else '❌ Failed'}")
    
    # Strategy 2: Retry (timeout errors)
    print("\n2. RETRY strategy (timeout):")
    success = error_handler.report_error(
        component="network",
        error=TimeoutError("Network request timeout"),
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.TIMEOUT
    )
    print(f"   Result: {'✅ Retried' if success else '❌ Failed'}")
    
    # Strategy 3: Graceful degradation (hardware failure)
    print("\n3. GRACEFUL DEGRADATION strategy (hardware):")
    success = error_handler.report_error(
        component="audio_hw",
        error=IOError("Audio device malfunction"),
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.HARDWARE,
        context={"microphone_id": 2, "total_microphones": 8}
    )
    print(f"   Result: {'✅ Degraded gracefully' if success else '❌ Failed'}")
    
    # Strategy 4: System restart (critical failure)
    print("\n4. RESTART strategy (critical):")
    success = error_handler.report_error(
        component="storage",
        error=OSError("Disk full - cannot write logs"),
        severity=ErrorSeverity.CRITICAL,
        category=ErrorCategory.RESOURCE
    )
    print(f"   Result: {'✅ Restart initiated' if success else '❌ System failure'}")
    
    # Show final system state
    health = error_handler.get_system_health()
    print(f"\nFinal system state:")
    print(f"   Overall health: {health['overall_health_score']:.2f}")
    print(f"   System degraded: {health['system_degraded']}")
    print(f"   Components with issues: {sum(1 for h in health['component_health'].values() if not h['is_healthy'])}")
    
    error_handler.shutdown()


def main():
    """Main demo function."""
    print("Gunshot Localization Error Handling & Recovery Demo")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_basic_error_handling()
        demo_microphone_failure_recovery()
        demo_timeout_retry_mechanism()
        demo_resource_exhaustion_handling()
        demo_health_monitoring()
        demo_system_degradation()
        demo_error_recovery_strategies()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Comprehensive error classification and handling")
        print("✅ Component health monitoring and tracking")
        print("✅ Automatic recovery strategies (retry, fallback, degradation)")
        print("✅ Microphone failure graceful degradation")
        print("✅ Timeout handling with retry logic")
        print("✅ Resource exhaustion recovery")
        print("✅ Continuous health monitoring")
        print("✅ System degradation detection")
        print("✅ Error statistics and trend analysis")
        print("✅ Recovery recommendations")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()