"""
Demo script for diagnostics and status reporting system.
"""
import time
import json
import tempfile
from unittest.mock import Mock
from diagnostics import DiagnosticsManager, get_quick_system_status


def demo_quick_system_status():
    """Demo quick system status functionality."""
    print("=== Quick System Status Demo ===")
    
    print("\n1. Basic system status:")
    status = get_quick_system_status()
    
    health_symbol = "✅" if status['system_healthy'] else "❌"
    print(f"   System Health: {health_symbol}")
    print(f"   CPU Usage: {status['cpu_usage_percent']:.1f}%")
    print(f"   Memory Usage: {status['memory_usage_percent']:.1f}%")
    
    if status.get('issues'):
        print("   Issues:")
        for issue in status['issues']:
            print(f"     • {issue}")
    
    print("\n2. With mock pipeline:")
    mock_pipeline = Mock()
    mock_pipeline.get_system_status.return_value = {
        'pipeline_state': 'running',
        'is_running': True
    }
    
    status_with_pipeline = get_quick_system_status(pipeline=mock_pipeline)
    print(f"   Pipeline State: {status_with_pipeline['pipeline_state']}")
    print(f"   Pipeline Running: {status_with_pipeline['pipeline_running']}")


def demo_diagnostics_manager():
    """Demo comprehensive diagnostics manager."""
    print("\n=== Diagnostics Manager Demo ===")
    
    # Create mock pipeline
    mock_pipeline = Mock()
    mock_pipeline.config_manager = Mock()
    
    # Mock microphone positions
    mock_mics = [
        Mock(id=1, x=0.0, y=0.0, z=0.0),
        Mock(id=2, x=1.0, y=0.0, z=0.0),
        Mock(id=3, x=0.0, y=1.0, z=0.0),
        Mock(id=4, x=1.0, y=1.0, z=0.0)
    ]
    mock_pipeline.config_manager.get_microphone_positions.return_value = mock_mics
    
    # Mock pipeline metrics
    mock_metrics = Mock()
    mock_metrics.average_latency = 0.025
    mock_metrics.detection_rate = 5.0
    mock_metrics.localization_accuracy = 0.85
    mock_pipeline.get_metrics.return_value = mock_metrics
    
    # Mock error handler
    mock_error_handler = Mock()
    mock_error_handler.get_system_health.return_value = {
        'overall_health_score': 0.9,
        'system_degraded': False,
        'critical_errors': 0
    }
    mock_error_handler.get_error_statistics.return_value = {
        'total_errors': 5,
        'recent_errors_1h': 2,
        'error_rate_per_hour': 1.5
    }
    mock_pipeline.error_handler = mock_error_handler
    
    # Create diagnostics manager
    diagnostics_manager = DiagnosticsManager(pipeline=mock_pipeline, update_interval=0.1)
    
    try:
        print("\n1. Starting diagnostics monitoring...")
        diagnostics_manager.start_monitoring()
        
        # Let it run for a bit
        time.sleep(0.5)
        
        print("   ✅ Monitoring started successfully")
        
        print("\n2. Collecting system diagnostics...")
        diagnostics = diagnostics_manager.get_system_diagnostics()
        
        print(f"   Overall Status: {diagnostics.overall_status.value}")
        print(f"   Overall Health Score: {diagnostics.overall_health_score:.2f}")
        print(f"   Components Monitored: {len(diagnostics.components)}")
        print(f"   Microphones Tracked: {len(diagnostics.microphones)}")
        
        # Show component details
        print("\n3. Component Status Details:")
        for component in diagnostics.components:
            status_symbol = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '❌',
                'offline': '⭕'
            }.get(component.status.value, '❓')
            
            print(f"   {status_symbol} {component.name}: {component.status.value} "
                  f"(Health: {component.health_score:.2f})")
            
            if component.issues:
                for issue in component.issues:
                    print(f"     - {issue}")
        
        print("\n4. Performance Metrics:")
        perf = diagnostics.performance
        print(f"   CPU Usage: {perf.cpu_usage_percent:.1f}%")
        print(f"   Memory Usage: {perf.memory_usage_percent:.1f}%")
        print(f"   Processing Latency: {perf.processing_latency_ms:.1f}ms")
        print(f"   Detection Rate: {perf.detection_rate_per_second:.1f}/sec")
        print(f"   Localization Accuracy: {perf.localization_accuracy:.1%}")
        
        print("\n5. Microphone Connectivity Report:")
        mic_report = diagnostics_manager.get_microphone_connectivity_report()
        print(f"   Total Microphones: {mic_report['total_microphones']}")
        print(f"   Connected: {mic_report['connected_microphones']}")
        print(f"   Disconnected: {mic_report['disconnected_microphones']}")
        print(f"   Average Signal Quality: {mic_report['average_signal_quality']:.2f}")
        
        # Show individual microphone status
        for mic in mic_report['microphones'][:2]:  # Show first 2
            status_symbol = "✅" if mic['connected'] else "❌"
            print(f"     {status_symbol} Mic {mic['id']}: Quality {mic['signal_quality']:.2f}, "
                  f"Noise {mic['noise_level_db']:.1f}dB")
        
        print("\n6. Performance Summary (last hour):")
        perf_summary = diagnostics_manager.get_performance_summary(hours=1.0)
        
        if 'error' not in perf_summary:
            print(f"   Data Points: {perf_summary['data_points']}")
            print(f"   CPU Usage: {perf_summary['cpu_usage']['average']:.1f}% avg "
                  f"({perf_summary['cpu_usage']['min']:.1f}% - {perf_summary['cpu_usage']['max']:.1f}%)")
            print(f"   Memory Usage: {perf_summary['memory_usage']['average']:.1f}% avg")
            print(f"   Processing Latency: {perf_summary['processing_latency_ms']['average']:.1f}ms avg")
        else:
            print(f"   {perf_summary['error']}")
        
        print("\n7. Exporting diagnostics report...")
        
        # JSON export
        json_report = diagnostics_manager.export_diagnostics_report('json')
        print(f"   JSON Report Length: {len(json_report)} characters")
        
        # Text export
        text_report = diagnostics_manager.export_diagnostics_report('text')
        print(f"   Text Report Length: {len(text_report)} characters")
        print("   Text Report Preview:")
        lines = text_report.split('\n')[:5]
        for line in lines:
            print(f"     {line}")
        print("     ... (truncated)")
        
    finally:
        print("\n8. Stopping diagnostics monitoring...")
        diagnostics_manager.stop_monitoring()
        diagnostics_manager.shutdown()
        print("   ✅ Diagnostics manager shutdown complete")


def demo_performance_monitoring():
    """Demo performance monitoring over time."""
    print("\n=== Performance Monitoring Demo ===")
    
    # Create diagnostics manager
    diagnostics_manager = DiagnosticsManager(update_interval=0.1)
    
    try:
        print("\n1. Starting performance monitoring...")
        diagnostics_manager.start_monitoring()
        
        # Monitor for a few seconds
        print("   Collecting performance data...")
        for i in range(5):
            time.sleep(0.2)
            print(f"   Data point {i+1}/5 collected")
        
        print("\n2. Performance history analysis:")
        if diagnostics_manager.performance_history:
            latest = diagnostics_manager.performance_history[-1]
            print(f"   Latest CPU Usage: {latest.cpu_usage_percent:.1f}%")
            print(f"   Latest Memory Usage: {latest.memory_usage_percent:.1f}%")
            print(f"   Data Points Collected: {len(diagnostics_manager.performance_history)}")
            
            # Show trend
            if len(diagnostics_manager.performance_history) >= 2:
                first = diagnostics_manager.performance_history[0]
                cpu_trend = latest.cpu_usage_percent - first.cpu_usage_percent
                trend_symbol = "📈" if cpu_trend > 0 else "📉" if cpu_trend < 0 else "➡️"
                print(f"   CPU Trend: {trend_symbol} {cpu_trend:+.1f}%")
        
        print("\n3. Alert checking simulation:")
        # Simulate high resource usage
        import psutil
        current_cpu = psutil.cpu_percent()
        if current_cpu > 50:
            print(f"   ⚠️ Simulated alert: High CPU usage detected ({current_cpu:.1f}%)")
        else:
            print(f"   ✅ No alerts: CPU usage normal ({current_cpu:.1f}%)")
        
    finally:
        diagnostics_manager.shutdown()


def demo_error_integration():
    """Demo integration with error handling system."""
    print("\n=== Error Integration Demo ===")
    
    # Create mock pipeline with error handler
    mock_pipeline = Mock()
    
    # Mock error handler with some errors
    mock_error_handler = Mock()
    mock_error_handler.get_system_health.return_value = {
        'overall_health_score': 0.7,  # Degraded
        'system_degraded': True,
        'critical_errors': 1,
        'component_health': {
            'audio_capture': {
                'is_healthy': False,
                'health_score': 0.5,
                'error_count': 3
            }
        }
    }
    mock_error_handler.get_error_statistics.return_value = {
        'total_errors': 15,
        'recent_errors_1h': 8,
        'error_rate_per_hour': 12.0,
        'most_problematic_component': 'audio_capture'
    }
    mock_pipeline.error_handler = mock_error_handler
    
    # Create diagnostics manager
    diagnostics_manager = DiagnosticsManager(pipeline=mock_pipeline)
    
    try:
        print("\n1. Updating diagnostics with error information...")
        diagnostics_manager._update_diagnostics()
        
        print("\n2. System status with errors:")
        diagnostics = diagnostics_manager.get_system_diagnostics()
        
        print(f"   Overall Status: {diagnostics.overall_status.value}")
        print(f"   Overall Health Score: {diagnostics.overall_health_score:.2f}")
        
        if diagnostics.alerts:
            print("   Recent Alerts:")
            for alert in diagnostics.alerts[-3:]:  # Show last 3
                print(f"     • {alert}")
        
        if diagnostics.recommendations:
            print("   Recommendations:")
            for rec in diagnostics.recommendations[:3]:  # Show first 3
                print(f"     • {rec}")
        
        print("\n3. Error handler component status:")
        error_handler_status = None
        for component in diagnostics.components:
            if component.name == "Error Handler":
                error_handler_status = component
                break
        
        if error_handler_status:
            print(f"   Status: {error_handler_status.status.value}")
            print(f"   Health Score: {error_handler_status.health_score:.2f}")
            print(f"   Error Count: {error_handler_status.error_count}")
            
            if error_handler_status.issues:
                print("   Issues:")
                for issue in error_handler_status.issues:
                    print(f"     - {issue}")
        
    finally:
        diagnostics_manager.shutdown()


def demo_cli_integration():
    """Demo CLI integration with diagnostics."""
    print("\n=== CLI Integration Demo ===")
    
    print("\n1. CLI Commands Available:")
    commands = [
        ("--status", "Quick system status check"),
        ("--health-check", "Comprehensive health assessment"),
        ("--diagnostics", "Full diagnostics report"),
        ("--system-info", "System information display")
    ]
    
    for cmd, desc in commands:
        print(f"   {cmd}: {desc}")
    
    print("\n2. Example CLI Usage:")
    print("   gunshot-localizer --status")
    print("   gunshot-localizer --health-check")
    print("   gunshot-localizer --diagnostics")
    print("   gunshot-localizer --config config.json --enable-performance-monitoring")
    
    print("\n3. Integration Benefits:")
    benefits = [
        "Real-time system monitoring",
        "Proactive issue detection",
        "Performance trend analysis",
        "Automated health reporting",
        "Integration with error handling",
        "Export capabilities for analysis"
    ]
    
    for benefit in benefits:
        print(f"   ✅ {benefit}")


def main():
    """Main demo function."""
    print("Gunshot Localization Diagnostics & Status Reporting Demo")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_quick_system_status()
        demo_diagnostics_manager()
        demo_performance_monitoring()
        demo_error_integration()
        demo_cli_integration()
        
        print("\n" + "=" * 60)
        print("Diagnostics Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Quick system status checks")
        print("✅ Comprehensive diagnostics reporting")
        print("✅ Real-time performance monitoring")
        print("✅ Microphone connectivity tracking")
        print("✅ Component health assessment")
        print("✅ Error integration and alerting")
        print("✅ Multiple export formats (JSON, text)")
        print("✅ CLI integration with status commands")
        print("✅ Performance trend analysis")
        print("✅ Automated health recommendations")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()