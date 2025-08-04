"""
Runtime diagnostics and status reporting for gunshot localization system.
"""
import time
import threading
import logging
import json
import psutil
import platform
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import numpy as np
from datetime import datetime, timedelta


class SystemStatus(Enum):
    """System status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


class ComponentType(Enum):
    """Types of system components."""
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class ComponentStatus:
    """Status information for a system component."""
    name: str
    component_type: ComponentType
    status: SystemStatus
    health_score: float  # 0.0 to 1.0
    last_check: float
    uptime: float
    error_count: int
    warning_count: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MicrophoneStatus:
    """Status information for individual microphones."""
    microphone_id: int
    is_connected: bool
    signal_quality: float  # 0.0 to 1.0
    noise_level_db: float
    last_signal_time: Optional[float]
    sample_rate: int
    channel_active: bool
    error_count: int
    calibration_status: str
    position: Tuple[float, float, float]
    issues: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Performance metrics for the system."""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_mb: float
    processing_latency_ms: float
    detection_rate_per_second: float
    localization_accuracy: float
    error_rate_per_hour: float
    uptime_hours: float


@dataclass
class SystemDiagnostics:
    """Complete system diagnostics information."""
    timestamp: float
    overall_status: SystemStatus
    overall_health_score: float
    components: List[ComponentStatus]
    microphones: List[MicrophoneStatus]
    performance: PerformanceMetrics
    alerts: List[str]
    recommendations: List[str]
    system_info: Dict[str, Any]

class DiagnosticsManager:
    """Main diagnostics and status reporting manager."""
    
    def __init__(self, pipeline=None, update_interval: float = 5.0):
        """
        Initialize diagnostics manager.
        
        Args:
            pipeline: Reference to main pipeline
            update_interval: How often to update diagnostics (seconds)
        """
        self.pipeline = pipeline
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        
        # Status tracking
        self.component_statuses: Dict[str, ComponentStatus] = {}
        self.microphone_statuses: Dict[int, MicrophoneStatus] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=100)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = time.time()
        
        # Thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 80.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 85.0
        self.disk_critical_threshold = 95.0
        
        self.logger.info("Diagnostics manager initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous diagnostics monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="DiagnosticsMonitor"
        )
        self.monitor_thread.start()
        self.logger.info("Diagnostics monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop diagnostics monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Diagnostics monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._update_diagnostics()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in diagnostics monitoring: {e}")
                time.sleep(1.0)  # Brief pause before continuing
    
    def _update_diagnostics(self) -> None:
        """Update all diagnostic information."""
        current_time = time.time()
        
        # Update component statuses
        self._update_component_statuses()
        
        # Update microphone statuses
        self._update_microphone_statuses()
        
        # Update performance metrics
        performance = self._collect_performance_metrics()
        self.performance_history.append(performance)
        
        # Check for alerts
        self._check_alerts(performance)
    
    def _update_component_statuses(self) -> None:
        """Update status of all system components."""
        current_time = time.time()
        
        # Audio capture component
        if self.pipeline and hasattr(self.pipeline, 'audio_capture'):
            self._update_audio_capture_status()
        
        # Gunshot detector component
        if self.pipeline and hasattr(self.pipeline, 'gunshot_detector'):
            self._update_detector_status()
        
        # TDoA localizer component
        if self.pipeline and hasattr(self.pipeline, 'tdoa_localizer'):
            self._update_localizer_status()
        
        # Error handler component
        if self.pipeline and hasattr(self.pipeline, 'error_handler'):
            self._update_error_handler_status()
        
        # System resources
        self._update_system_resource_status()
    
    def _update_audio_capture_status(self) -> None:
        """Update audio capture component status."""
        try:
            audio_capture = self.pipeline.audio_capture
            current_time = time.time()
            
            # Determine status based on capture state
            if hasattr(audio_capture, '_capturing') and audio_capture._capturing:
                status = SystemStatus.HEALTHY
                health_score = 1.0
                issues = []
            else:
                status = SystemStatus.WARNING
                health_score = 0.5
                issues = ["Audio capture not active"]
            
            # Get metrics if available
            metrics = {}
            if hasattr(audio_capture, 'get_status'):
                try:
                    audio_status = audio_capture.get_status()
                    metrics.update(audio_status)
                except:
                    pass
            
            self.component_statuses['audio_capture'] = ComponentStatus(
                name="Audio Capture",
                component_type=ComponentType.HARDWARE,
                status=status,
                health_score=health_score,
                last_check=current_time,
                uptime=current_time - self.start_time,
                error_count=0,
                warning_count=len(issues),
                metrics=metrics,
                issues=issues,
                recommendations=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating audio capture status: {e}")
    
    def _update_detector_status(self) -> None:
        """Update gunshot detector component status."""
        try:
            detector = self.pipeline.gunshot_detector
            current_time = time.time()
            
            # Basic health check
            status = SystemStatus.HEALTHY
            health_score = 1.0
            issues = []
            metrics = {}
            
            # Get detector-specific metrics if available
            if hasattr(detector, 'get_detection_stats'):
                try:
                    stats = detector.get_detection_stats()
                    metrics.update(stats)
                except:
                    pass
            
            self.component_statuses['gunshot_detector'] = ComponentStatus(
                name="Gunshot Detector",
                component_type=ComponentType.SOFTWARE,
                status=status,
                health_score=health_score,
                last_check=current_time,
                uptime=current_time - self.start_time,
                error_count=0,
                warning_count=0,
                metrics=metrics,
                issues=issues,
                recommendations=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating detector status: {e}")
    
    def _update_localizer_status(self) -> None:
        """Update TDoA localizer component status."""
        try:
            localizer = self.pipeline.tdoa_localizer
            current_time = time.time()
            
            status = SystemStatus.HEALTHY
            health_score = 1.0
            issues = []
            metrics = {}
            
            # Get localizer-specific metrics if available
            if hasattr(localizer, 'get_localization_stats'):
                try:
                    stats = localizer.get_localization_stats()
                    metrics.update(stats)
                except:
                    pass
            
            self.component_statuses['tdoa_localizer'] = ComponentStatus(
                name="TDoA Localizer",
                component_type=ComponentType.SOFTWARE,
                status=status,
                health_score=health_score,
                last_check=current_time,
                uptime=current_time - self.start_time,
                error_count=0,
                warning_count=0,
                metrics=metrics,
                issues=issues,
                recommendations=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating localizer status: {e}")
    
    def _update_error_handler_status(self) -> None:
        """Update error handler component status."""
        try:
            error_handler = self.pipeline.error_handler
            current_time = time.time()
            
            # Get error handler health
            health_info = error_handler.get_system_health()
            error_stats = error_handler.get_error_statistics()
            
            # Determine status based on error rates
            if health_info['critical_errors'] > 0:
                status = SystemStatus.CRITICAL
                health_score = 0.3
            elif error_stats['recent_errors_1h'] > 10:
                status = SystemStatus.WARNING
                health_score = 0.6
            else:
                status = SystemStatus.HEALTHY
                health_score = health_info['overall_health_score']
            
            issues = []
            if health_info['system_degraded']:
                issues.append("System in degraded mode")
            if health_info['critical_errors'] > 0:
                issues.append(f"{health_info['critical_errors']} critical errors")
            
            self.component_statuses['error_handler'] = ComponentStatus(
                name="Error Handler",
                component_type=ComponentType.SOFTWARE,
                status=status,
                health_score=health_score,
                last_check=current_time,
                uptime=current_time - self.start_time,
                error_count=error_stats['total_errors'],
                warning_count=len(issues),
                metrics={
                    'total_errors': error_stats['total_errors'],
                    'recent_errors': error_stats['recent_errors_1h'],
                    'error_rate': error_stats['error_rate_per_hour']
                },
                issues=issues,
                recommendations=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error updating error handler status: {e}")
    
    def _update_system_resource_status(self) -> None:
        """Update system resource status."""
        try:
            current_time = time.time()
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine overall system status
            status = SystemStatus.HEALTHY
            health_score = 1.0
            issues = []
            recommendations = []
            
            # Check CPU usage
            if cpu_percent > self.cpu_critical_threshold:
                status = SystemStatus.CRITICAL
                health_score = min(health_score, 0.3)
                issues.append(f"Critical CPU usage: {cpu_percent:.1f}%")
                recommendations.append("Reduce system load or upgrade CPU")
            elif cpu_percent > self.cpu_warning_threshold:
                status = SystemStatus.WARNING
                health_score = min(health_score, 0.7)
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                recommendations.append("Monitor CPU usage trends")
            
            # Check memory usage
            memory_percent = memory.percent
            if memory_percent > self.memory_critical_threshold:
                status = SystemStatus.CRITICAL
                health_score = min(health_score, 0.3)
                issues.append(f"Critical memory usage: {memory_percent:.1f}%")
                recommendations.append("Free memory or add more RAM")
            elif memory_percent > self.memory_warning_threshold:
                status = SystemStatus.WARNING
                health_score = min(health_score, 0.7)
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                recommendations.append("Monitor memory usage trends")
            
            # Check disk usage
            disk_percent = disk.percent
            if disk_percent > self.disk_critical_threshold:
                status = SystemStatus.CRITICAL
                health_score = min(health_score, 0.3)
                issues.append(f"Critical disk usage: {disk_percent:.1f}%")
                recommendations.append("Free disk space immediately")
            elif disk_percent > self.disk_warning_threshold:
                status = SystemStatus.WARNING
                health_score = min(health_score, 0.7)
                issues.append(f"High disk usage: {disk_percent:.1f}%")
                recommendations.append("Plan for disk space cleanup")
            
            self.component_statuses['system_resources'] = ComponentStatus(
                name="System Resources",
                component_type=ComponentType.HARDWARE,
                status=status,
                health_score=health_score,
                last_check=current_time,
                uptime=current_time - self.start_time,
                error_count=0,
                warning_count=len(issues),
                metrics={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk.free / (1024**3)
                },
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error updating system resource status: {e}")
    
    def _update_microphone_statuses(self) -> None:
        """Update status of all microphones."""
        try:
            if not self.pipeline or not hasattr(self.pipeline, 'config_manager'):
                return
            
            # Get microphone positions from config
            mic_positions = self.pipeline.config_manager.get_microphone_positions()
            current_time = time.time()
            
            for mic in mic_positions:
                mic_id = mic.id
                
                # Basic microphone status (would be enhanced with real hardware checks)
                self.microphone_statuses[mic_id] = MicrophoneStatus(
                    microphone_id=mic_id,
                    is_connected=True,  # Would check actual hardware
                    signal_quality=0.8 + np.random.random() * 0.2,  # Simulated
                    noise_level_db=-30.0 + np.random.random() * 10,  # Simulated
                    last_signal_time=current_time,
                    sample_rate=48000,
                    channel_active=True,
                    error_count=0,
                    calibration_status="calibrated",
                    position=(mic.x, mic.y, mic.z),
                    issues=[]
                )
                
        except Exception as e:
            self.logger.error(f"Error updating microphone statuses: {e}")
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            current_time = time.time()
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O (simplified)
            network_io = 0.0
            try:
                net_io = psutil.net_io_counters()
                network_io = (net_io.bytes_sent + net_io.bytes_recv) / (1024**2)
            except:
                pass
            
            # Pipeline metrics
            processing_latency = 0.0
            detection_rate = 0.0
            localization_accuracy = 0.0
            error_rate = 0.0
            
            if self.pipeline:
                try:
                    pipeline_metrics = self.pipeline.get_metrics()
                    processing_latency = pipeline_metrics.average_latency * 1000  # Convert to ms
                    detection_rate = pipeline_metrics.detection_rate
                    localization_accuracy = pipeline_metrics.localization_accuracy
                    
                    # Calculate error rate
                    if hasattr(self.pipeline, 'error_handler'):
                        error_stats = self.pipeline.error_handler.get_error_statistics()
                        error_rate = error_stats.get('error_rate_per_hour', 0.0)
                except:
                    pass
            
            return PerformanceMetrics(
                timestamp=current_time,
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory.used / (1024**2),
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_io_mb=network_io,
                processing_latency_ms=processing_latency,
                detection_rate_per_second=detection_rate,
                localization_accuracy=localization_accuracy,
                error_rate_per_hour=error_rate,
                uptime_hours=(current_time - self.start_time) / 3600
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_io_mb=0.0,
                processing_latency_ms=0.0,
                detection_rate_per_second=0.0,
                localization_accuracy=0.0,
                error_rate_per_hour=0.0,
                uptime_hours=0.0
            )
    
    def _check_alerts(self, performance: PerformanceMetrics) -> None:
        """Check for alert conditions."""
        alerts = []
        
        # Performance-based alerts
        if performance.cpu_usage_percent > self.cpu_critical_threshold:
            alerts.append(f"CRITICAL: CPU usage at {performance.cpu_usage_percent:.1f}%")
        
        if performance.memory_usage_percent > self.memory_critical_threshold:
            alerts.append(f"CRITICAL: Memory usage at {performance.memory_usage_percent:.1f}%")
        
        if performance.processing_latency_ms > 500:  # 500ms threshold
            alerts.append(f"WARNING: High processing latency: {performance.processing_latency_ms:.1f}ms")
        
        if performance.error_rate_per_hour > 50:  # 50 errors per hour threshold
            alerts.append(f"WARNING: High error rate: {performance.error_rate_per_hour:.1f}/hour")
        
        # Component-based alerts
        for component in self.component_statuses.values():
            if component.status == SystemStatus.CRITICAL:
                alerts.append(f"CRITICAL: {component.name} is in critical state")
            elif component.status == SystemStatus.WARNING and component.health_score < 0.5:
                alerts.append(f"WARNING: {component.name} health score low: {component.health_score:.2f}")
        
        # Store alerts
        for alert in alerts:
            self.alert_history.append({
                'timestamp': time.time(),
                'message': alert,
                'level': 'CRITICAL' if 'CRITICAL' in alert else 'WARNING'
            })
    
    def get_system_diagnostics(self) -> SystemDiagnostics:
        """Get complete system diagnostics."""
        current_time = time.time()
        
        # Calculate overall status and health
        if not self.component_statuses:
            overall_status = SystemStatus.OFFLINE
            overall_health_score = 0.0
        else:
            # Determine worst status
            statuses = [comp.status for comp in self.component_statuses.values()]
            if SystemStatus.CRITICAL in statuses:
                overall_status = SystemStatus.CRITICAL
            elif SystemStatus.WARNING in statuses:
                overall_status = SystemStatus.WARNING
            else:
                overall_status = SystemStatus.HEALTHY
            
            # Calculate average health score
            health_scores = [comp.health_score for comp in self.component_statuses.values()]
            overall_health_score = sum(health_scores) / len(health_scores) if health_scores else 0.0
        
        # Get recent alerts
        recent_alerts = [
            alert['message'] for alert in list(self.alert_history)[-10:]
        ]
        
        # Generate recommendations
        recommendations = []
        for component in self.component_statuses.values():
            recommendations.extend(component.recommendations)
        
        # Get latest performance metrics
        latest_performance = self.performance_history[-1] if self.performance_history else PerformanceMetrics(
            timestamp=current_time,
            cpu_usage_percent=0.0,
            memory_usage_mb=0.0,
            memory_usage_percent=0.0,
            disk_usage_percent=0.0,
            network_io_mb=0.0,
            processing_latency_ms=0.0,
            detection_rate_per_second=0.0,
            localization_accuracy=0.0,
            error_rate_per_hour=0.0,
            uptime_hours=0.0
        )
        
        # System information
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'uptime_hours': (current_time - self.start_time) / 3600
        }
        
        return SystemDiagnostics(
            timestamp=current_time,
            overall_status=overall_status,
            overall_health_score=overall_health_score,
            components=list(self.component_statuses.values()),
            microphones=list(self.microphone_statuses.values()),
            performance=latest_performance,
            alerts=recent_alerts,
            recommendations=list(set(recommendations)),  # Remove duplicates
            system_info=system_info
        )
    
    def get_microphone_connectivity_report(self) -> Dict[str, Any]:
        """Get detailed microphone connectivity report."""
        report = {
            'timestamp': time.time(),
            'total_microphones': len(self.microphone_statuses),
            'connected_microphones': 0,
            'disconnected_microphones': 0,
            'average_signal_quality': 0.0,
            'microphones': []
        }
        
        if not self.microphone_statuses:
            return report
        
        signal_qualities = []
        for mic_status in self.microphone_statuses.values():
            if mic_status.is_connected:
                report['connected_microphones'] += 1
                signal_qualities.append(mic_status.signal_quality)
            else:
                report['disconnected_microphones'] += 1
            
            report['microphones'].append({
                'id': mic_status.microphone_id,
                'connected': mic_status.is_connected,
                'signal_quality': mic_status.signal_quality,
                'noise_level_db': mic_status.noise_level_db,
                'position': mic_status.position,
                'issues': mic_status.issues
            })
        
        if signal_qualities:
            report['average_signal_quality'] = sum(signal_qualities) / len(signal_qualities)
        
        return report
    
    def get_performance_summary(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get performance summary for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [
            m for m in self.performance_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No performance data available for the specified period'}
        
        # Calculate averages and trends
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        memory_values = [m.memory_usage_percent for m in recent_metrics]
        latency_values = [m.processing_latency_ms for m in recent_metrics]
        accuracy_values = [m.localization_accuracy for m in recent_metrics]
        
        return {
            'time_period_hours': hours,
            'data_points': len(recent_metrics),
            'cpu_usage': {
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory_usage': {
                'average': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'processing_latency_ms': {
                'average': sum(latency_values) / len(latency_values),
                'min': min(latency_values),
                'max': max(latency_values)
            },
            'localization_accuracy': {
                'average': sum(accuracy_values) / len(accuracy_values),
                'min': min(accuracy_values),
                'max': max(accuracy_values)
            },
            'latest_metrics': asdict(recent_metrics[-1])
        }
    
    def export_diagnostics_report(self, format_type: str = 'json') -> str:
        """Export comprehensive diagnostics report."""
        diagnostics = self.get_system_diagnostics()
        
        if format_type.lower() == 'json':
            return json.dumps(asdict(diagnostics), indent=2, default=str)
        elif format_type.lower() == 'text':
            return self._format_diagnostics_text(diagnostics)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _format_diagnostics_text(self, diagnostics: SystemDiagnostics) -> str:
        """Format diagnostics as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("GUNSHOT LOCALIZATION SYSTEM DIAGNOSTICS REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.fromtimestamp(diagnostics.timestamp)}")
        lines.append(f"Overall Status: {diagnostics.overall_status.value.upper()}")
        lines.append(f"Overall Health Score: {diagnostics.overall_health_score:.2f}/1.00")
        lines.append("")
        
        # System Information
        lines.append("SYSTEM INFORMATION:")
        lines.append("-" * 20)
        for key, value in diagnostics.system_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Component Status
        lines.append("COMPONENT STATUS:")
        lines.append("-" * 20)
        for component in diagnostics.components:
            status_symbol = {
                SystemStatus.HEALTHY: "✓",
                SystemStatus.WARNING: "⚠",
                SystemStatus.CRITICAL: "✗",
                SystemStatus.OFFLINE: "○"
            }.get(component.status, "?")
            
            lines.append(f"  {status_symbol} {component.name}: {component.status.value.upper()} "
                        f"(Health: {component.health_score:.2f})")
            
            if component.issues:
                for issue in component.issues:
                    lines.append(f"    - {issue}")
        lines.append("")
        
        # Performance Metrics
        lines.append("PERFORMANCE METRICS:")
        lines.append("-" * 20)
        perf = diagnostics.performance
        lines.append(f"  CPU Usage: {perf.cpu_usage_percent:.1f}%")
        lines.append(f"  Memory Usage: {perf.memory_usage_percent:.1f}% ({perf.memory_usage_mb:.0f} MB)")
        lines.append(f"  Disk Usage: {perf.disk_usage_percent:.1f}%")
        lines.append(f"  Processing Latency: {perf.processing_latency_ms:.1f} ms")
        lines.append(f"  Detection Rate: {perf.detection_rate_per_second:.2f} /sec")
        lines.append(f"  Localization Accuracy: {perf.localization_accuracy:.1%}")
        lines.append(f"  Uptime: {perf.uptime_hours:.1f} hours")
        lines.append("")
        
        # Microphone Status
        if diagnostics.microphones:
            lines.append("MICROPHONE STATUS:")
            lines.append("-" * 20)
            for mic in diagnostics.microphones:
                status_symbol = "✓" if mic.is_connected else "✗"
                lines.append(f"  {status_symbol} Microphone {mic.microphone_id}: "
                           f"Quality {mic.signal_quality:.2f}, "
                           f"Noise {mic.noise_level_db:.1f} dB")
        lines.append("")
        
        # Alerts
        if diagnostics.alerts:
            lines.append("RECENT ALERTS:")
            lines.append("-" * 20)
            for alert in diagnostics.alerts:
                lines.append(f"  • {alert}")
        lines.append("")
        
        # Recommendations
        if diagnostics.recommendations:
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 20)
            for rec in diagnostics.recommendations:
                lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def shutdown(self) -> None:
        """Shutdown diagnostics manager."""
        self.stop_monitoring()
        self.logger.info("Diagnostics manager shutdown complete")


# Convenience functions
def create_diagnostics_manager(pipeline=None) -> DiagnosticsManager:
    """Create diagnostics manager with default settings."""
    return DiagnosticsManager(pipeline=pipeline)


def get_quick_system_status(pipeline=None) -> Dict[str, Any]:
    """Get quick system status without full diagnostics."""
    try:
        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        status = {
            'timestamp': time.time(),
            'system_healthy': True,
            'cpu_usage_percent': cpu_percent,
            'memory_usage_percent': memory.percent,
            'issues': []
        }
        
        # Check for critical issues
        if cpu_percent > 95:
            status['system_healthy'] = False
            status['issues'].append(f"Critical CPU usage: {cpu_percent:.1f}%")
        
        if memory.percent > 95:
            status['system_healthy'] = False
            status['issues'].append(f"Critical memory usage: {memory.percent:.1f}%")
        
        # Pipeline status
        if pipeline:
            try:
                pipeline_status = pipeline.get_system_status()
                status['pipeline_state'] = pipeline_status.get('pipeline_state', 'unknown')
                status['pipeline_running'] = pipeline_status.get('is_running', False)
            except:
                status['pipeline_state'] = 'error'
                status['pipeline_running'] = False
        
        return status
        
    except Exception as e:
        return {
            'timestamp': time.time(),
            'system_healthy': False,
            'error': str(e),
            'issues': ['Failed to collect system status']
        }