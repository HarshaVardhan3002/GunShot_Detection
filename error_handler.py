"""
Comprehensive error handling and recovery system for gunshot localization.
"""
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import traceback
from abc import ABC, abstractmethod


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, system continues normally
    MEDIUM = "medium"     # Moderate issues, some degradation
    HIGH = "high"         # Serious issues, significant degradation
    CRITICAL = "critical" # System failure, requires restart


class ErrorCategory(Enum):
    """Categories of errors."""
    HARDWARE = "hardware"           # Microphone, audio device issues
    SOFTWARE = "software"           # Algorithm, processing errors
    CONFIGURATION = "configuration" # Config file, parameter issues
    NETWORK = "network"             # Network connectivity issues
    RESOURCE = "resource"           # Memory, CPU, disk issues
    DATA = "data"                   # Invalid data, corruption
    TIMEOUT = "timeout"             # Operation timeouts
    UNKNOWN = "unknown"             # Unclassified errors


class RecoveryAction(Enum):
    """Types of recovery actions."""
    IGNORE = "ignore"                    # Log and continue
    RETRY = "retry"                      # Retry the operation
    FALLBACK = "fallback"                # Use fallback method
    RESTART_COMPONENT = "restart_component" # Restart specific component
    RESTART_SYSTEM = "restart_system"    # Restart entire system
    GRACEFUL_DEGRADATION = "graceful_degradation" # Reduce functionality
    ALERT_OPERATOR = "alert_operator"    # Notify human operator


@dataclass
class ErrorEvent:
    """Container for error event information."""
    timestamp: float
    error_type: str
    error_message: str
    component: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_action: Optional[RecoveryAction] = None


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_name: str
    is_healthy: bool = True
    last_error_time: Optional[float] = None
    error_count: int = 0
    consecutive_errors: int = 0
    last_successful_operation: Optional[float] = None
    degraded_mode: bool = False
    restart_count: int = 0
    health_score: float = 1.0  # 0.0 to 1.0


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration."""
    error_patterns: List[str]
    severity_threshold: ErrorSeverity
    max_retries: int
    retry_delay: float
    recovery_action: RecoveryAction
    timeout: float
    fallback_enabled: bool = True
    auto_restart: bool = False


class ErrorRecoveryHandler(ABC):
    """Abstract base class for error recovery handlers."""
    
    @abstractmethod
    def can_handle(self, error_event: ErrorEvent) -> bool:
        """Check if this handler can handle the error."""
        pass
    
    @abstractmethod
    def handle_error(self, error_event: ErrorEvent) -> bool:
        """Handle the error and return success status."""
        pass
    
    @abstractmethod
    def get_recovery_time_estimate(self, error_event: ErrorEvent) -> float:
        """Estimate recovery time in seconds."""
        pass


class MicrophoneFailureHandler(ErrorRecoveryHandler):
    """Handler for microphone and audio device failures."""
    
    def __init__(self, min_microphones: int = 4):
        """
        Initialize microphone failure handler.
        
        Args:
            min_microphones: Minimum number of microphones needed
        """
        self.min_microphones = min_microphones
        self.failed_microphones: Set[int] = set()
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, error_event: ErrorEvent) -> bool:
        """Check if this is a microphone-related error."""
        return (error_event.category == ErrorCategory.HARDWARE and
                any(keyword in error_event.error_message.lower() 
                    for keyword in ['microphone', 'audio', 'device', 'channel']))
    
    def handle_error(self, error_event: ErrorEvent) -> bool:
        """Handle microphone failure."""
        try:
            # Extract microphone ID from context
            mic_id = error_event.context.get('microphone_id')
            if mic_id is not None:
                self.failed_microphones.add(mic_id)
                self.logger.warning(f"Microphone {mic_id} marked as failed")
            
            # Check if we still have enough microphones
            total_mics = error_event.context.get('total_microphones', 8)
            working_mics = total_mics - len(self.failed_microphones)
            
            if working_mics >= self.min_microphones:
                self.logger.info(f"Continuing with {working_mics} working microphones")
                return True
            else:
                self.logger.error(f"Insufficient microphones: {working_mics} < {self.min_microphones}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in microphone failure handler: {e}")
            return False
    
    def get_recovery_time_estimate(self, error_event: ErrorEvent) -> float:
        """Estimate recovery time for microphone issues."""
        return 1.0  # Immediate fallback to remaining microphones


class ProcessingTimeoutHandler(ErrorRecoveryHandler):
    """Handler for processing timeout errors."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        """
        Initialize timeout handler.
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, error_event: ErrorEvent) -> bool:
        """Check if this is a timeout error."""
        return (error_event.category == ErrorCategory.TIMEOUT or
                'timeout' in error_event.error_message.lower())
    
    def handle_error(self, error_event: ErrorEvent) -> bool:
        """Handle timeout error with retry logic."""
        operation_id = f"{error_event.component}_{error_event.error_type}"
        
        if self.retry_counts[operation_id] < self.max_retries:
            self.retry_counts[operation_id] += 1
            self.logger.info(f"Retrying {operation_id} (attempt {self.retry_counts[operation_id]})")
            
            # Wait before retry
            time.sleep(self.retry_delay)
            return True
        else:
            self.logger.error(f"Max retries exceeded for {operation_id}")
            self.retry_counts[operation_id] = 0  # Reset for future
            return False
    
    def get_recovery_time_estimate(self, error_event: ErrorEvent) -> float:
        """Estimate recovery time for timeout errors."""
        return self.retry_delay * self.max_retries


class ResourceExhaustionHandler(ErrorRecoveryHandler):
    """Handler for resource exhaustion errors."""
    
    def __init__(self, memory_threshold: float = 0.9, cpu_threshold: float = 0.95):
        """
        Initialize resource handler.
        
        Args:
            memory_threshold: Memory usage threshold (0.0-1.0)
            cpu_threshold: CPU usage threshold (0.0-1.0)
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.logger = logging.getLogger(__name__)
    
    def can_handle(self, error_event: ErrorEvent) -> bool:
        """Check if this is a resource exhaustion error."""
        return (error_event.category == ErrorCategory.RESOURCE or
                any(keyword in error_event.error_message.lower()
                    for keyword in ['memory', 'cpu', 'disk', 'resource']))
    
    def handle_error(self, error_event: ErrorEvent) -> bool:
        """Handle resource exhaustion."""
        try:
            # Get current resource usage
            import psutil
            process = psutil.Process()
            
            memory_percent = process.memory_percent()
            cpu_percent = process.cpu_percent()
            
            self.logger.warning(f"Resource usage: Memory {memory_percent:.1f}%, CPU {cpu_percent:.1f}%")
            
            # Trigger garbage collection
            import gc
            gc.collect()
            
            # Reduce buffer sizes if possible
            if hasattr(error_event.context, 'reduce_buffers'):
                error_event.context['reduce_buffers']()
                self.logger.info("Reduced buffer sizes to free memory")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in resource handler: {e}")
            return False
    
    def get_recovery_time_estimate(self, error_event: ErrorEvent) -> float:
        """Estimate recovery time for resource issues."""
        return 2.0  # Time for garbage collection and buffer reduction


class ErrorHandlingSystem:
    """Main error handling and recovery system."""
    
    def __init__(self, max_error_history: int = 1000):
        """
        Initialize error handling system.
        
        Args:
            max_error_history: Maximum number of errors to keep in history
        """
        self.logger = logging.getLogger(__name__)
        self.max_error_history = max_error_history
        
        # Error tracking
        self.error_history: deque = deque(maxlen=max_error_history)
        self.component_health: Dict[str, ComponentHealth] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_errors: Dict[str, float] = {}
        
        # Recovery handlers
        self.recovery_handlers: List[ErrorRecoveryHandler] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # System state
        self.system_degraded = False
        self.critical_errors = 0
        self.last_health_check = time.time()
        self.health_check_interval = 30.0  # seconds
        
        # Threading
        self.lock = threading.Lock()
        self.health_monitor_thread = None
        self.monitoring_active = False
        
        # Initialize default handlers
        self._initialize_default_handlers()
        self._initialize_default_strategies()
        
        self.logger.info("Error handling system initialized")
    
    def _initialize_default_handlers(self) -> None:
        """Initialize default error recovery handlers."""
        self.recovery_handlers = [
            MicrophoneFailureHandler(min_microphones=4),
            ProcessingTimeoutHandler(max_retries=3, retry_delay=0.1),
            ResourceExhaustionHandler()
        ]
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies."""
        self.recovery_strategies = {
            'microphone_failure': RecoveryStrategy(
                error_patterns=['microphone', 'audio device', 'channel'],
                severity_threshold=ErrorSeverity.MEDIUM,
                max_retries=1,
                retry_delay=1.0,
                recovery_action=RecoveryAction.GRACEFUL_DEGRADATION,
                timeout=5.0,
                fallback_enabled=True
            ),
            'processing_timeout': RecoveryStrategy(
                error_patterns=['timeout', 'processing time'],
                severity_threshold=ErrorSeverity.MEDIUM,
                max_retries=3,
                retry_delay=0.1,
                recovery_action=RecoveryAction.RETRY,
                timeout=10.0,
                fallback_enabled=True
            ),
            'memory_exhaustion': RecoveryStrategy(
                error_patterns=['memory', 'out of memory'],
                severity_threshold=ErrorSeverity.HIGH,
                max_retries=1,
                retry_delay=2.0,
                recovery_action=RecoveryAction.GRACEFUL_DEGRADATION,
                timeout=15.0,
                fallback_enabled=True
            ),
            'critical_failure': RecoveryStrategy(
                error_patterns=['critical', 'fatal', 'system'],
                severity_threshold=ErrorSeverity.CRITICAL,
                max_retries=1,
                retry_delay=5.0,
                recovery_action=RecoveryAction.RESTART_SYSTEM,
                timeout=30.0,
                fallback_enabled=False,
                auto_restart=True
            )
        }
    
    def register_component(self, component_name: str) -> None:
        """Register a component for health monitoring."""
        with self.lock:
            if component_name not in self.component_health:
                self.component_health[component_name] = ComponentHealth(component_name)
                self.logger.info(f"Registered component for monitoring: {component_name}")
    
    def report_error(self, component: str, error: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.UNKNOWN) -> bool:
        """
        Report an error and attempt recovery.
        
        Args:
            component: Component where error occurred
            error: The exception that occurred
            context: Additional context information
            severity: Error severity level
            category: Error category
            
        Returns:
            True if error was handled successfully, False otherwise
        """
        try:
            # Create error event
            error_event = ErrorEvent(
                timestamp=time.time(),
                error_type=type(error).__name__,
                error_message=str(error),
                component=component,
                severity=severity,
                category=category,
                context=context or {},
                exception=error,
                stack_trace=traceback.format_exc()
            )
            
            # Log the error
            self._log_error(error_event)
            
            # Update component health
            self._update_component_health(component, False)
            
            # Store in history
            with self.lock:
                self.error_history.append(error_event)
                self.error_counts[f"{component}_{error_event.error_type}"] += 1
                self.last_errors[component] = error_event.timestamp
            
            # Attempt recovery
            recovery_successful = self._attempt_recovery(error_event)
            
            # Update error event with recovery status
            error_event.recovery_attempted = True
            error_event.recovery_successful = recovery_successful
            
            # Check if system degradation is needed
            if not recovery_successful and severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self._handle_system_degradation(error_event)
            
            return recovery_successful
            
        except Exception as e:
            self.logger.error(f"Error in error handling system: {e}")
            return False
    
    def _log_error(self, error_event: ErrorEvent) -> None:
        """Log error event with appropriate level."""
        log_message = f"{error_event.component}: {error_event.error_message}"
        
        if error_event.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=error_event.exception)
        elif error_event.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=error_event.exception)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _update_component_health(self, component: str, success: bool) -> None:
        """Update component health status."""
        with self.lock:
            if component not in self.component_health:
                self.register_component(component)
            
            health = self.component_health[component]
            current_time = time.time()
            
            if success:
                health.last_successful_operation = current_time
                health.consecutive_errors = 0
                health.is_healthy = True
                health.health_score = min(1.0, health.health_score + 0.1)
            else:
                health.last_error_time = current_time
                health.error_count += 1
                health.consecutive_errors += 1
                health.health_score = max(0.0, health.health_score - 0.2)
                
                # Mark as unhealthy if too many consecutive errors
                if health.consecutive_errors >= 5:
                    health.is_healthy = False
                    health.degraded_mode = True
    
    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover from the error."""
        try:
            # Find appropriate recovery handler
            for handler in self.recovery_handlers:
                if handler.can_handle(error_event):
                    self.logger.info(f"Attempting recovery with {type(handler).__name__}")
                    
                    recovery_successful = handler.handle_error(error_event)
                    error_event.recovery_action = RecoveryAction.RETRY  # Simplified
                    
                    if recovery_successful:
                        self.logger.info(f"Recovery successful for {error_event.component}")
                        self._update_component_health(error_event.component, True)
                        return True
                    else:
                        self.logger.warning(f"Recovery failed for {error_event.component}")
            
            # If no specific handler, try generic recovery
            return self._generic_recovery(error_event)
            
        except Exception as e:
            self.logger.error(f"Error during recovery attempt: {e}")
            return False
    
    def _generic_recovery(self, error_event: ErrorEvent) -> bool:
        """Generic recovery for unhandled errors."""
        # For low severity errors, just log and continue
        if error_event.severity == ErrorSeverity.LOW:
            return True
        
        # For medium severity, try to continue with degraded functionality
        if error_event.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Continuing with degraded functionality due to {error_event.error_type}")
            return True
        
        # For high/critical severity, more aggressive recovery needed
        return False
    
    def _handle_system_degradation(self, error_event: ErrorEvent) -> None:
        """Handle system-wide degradation."""
        with self.lock:
            if not self.system_degraded:
                self.system_degraded = True
                self.logger.warning("System entering degraded mode")
            
            if error_event.severity == ErrorSeverity.CRITICAL:
                self.critical_errors += 1
                if self.critical_errors >= 3:
                    self.logger.critical("Multiple critical errors detected - system restart may be needed")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        with self.lock:
            current_time = time.time()
            
            # Calculate overall health score
            if self.component_health:
                overall_health = sum(h.health_score for h in self.component_health.values()) / len(self.component_health)
            else:
                overall_health = 1.0
            
            # Recent error rate (last 5 minutes)
            recent_errors = [e for e in self.error_history 
                           if current_time - e.timestamp < 300]
            
            return {
                'overall_health_score': overall_health,
                'system_degraded': self.system_degraded,
                'critical_errors': self.critical_errors,
                'total_errors': len(self.error_history),
                'recent_errors': len(recent_errors),
                'component_health': {name: {
                    'is_healthy': health.is_healthy,
                    'health_score': health.health_score,
                    'error_count': health.error_count,
                    'consecutive_errors': health.consecutive_errors,
                    'degraded_mode': health.degraded_mode,
                    'restart_count': health.restart_count
                } for name, health in self.component_health.items()},
                'last_health_check': self.last_health_check
            }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        with self.lock:
            current_time = time.time()
            
            # Error counts by category
            category_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            component_counts = defaultdict(int)
            
            for error in self.error_history:
                category_counts[error.category.value] += 1
                severity_counts[error.severity.value] += 1
                component_counts[error.component] += 1
            
            # Recent error trends (last hour)
            recent_errors = [e for e in self.error_history 
                           if current_time - e.timestamp < 3600]
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors_1h': len(recent_errors),
                'error_rate_per_hour': len(recent_errors),
                'errors_by_category': dict(category_counts),
                'errors_by_severity': dict(severity_counts),
                'errors_by_component': dict(component_counts),
                'most_problematic_component': max(component_counts.items(), 
                                                key=lambda x: x[1])[0] if component_counts else None
            }
    
    def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.health_monitor_thread = threading.Thread(
                target=self._health_monitor_loop,
                name="ErrorHandlerHealthMonitor"
            )
            self.health_monitor_thread.start()
            self.logger.info("Health monitoring started")
    
    def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        self.monitoring_active = False
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self.monitoring_active:
            try:
                self._perform_health_check()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(5.0)  # Brief pause before continuing
    
    def _perform_health_check(self) -> None:
        """Perform periodic health check."""
        current_time = time.time()
        
        with self.lock:
            self.last_health_check = current_time
            
            # Check for components that haven't reported in a while
            for name, health in self.component_health.items():
                if (health.last_successful_operation and 
                    current_time - health.last_successful_operation > 60.0):  # 1 minute
                    if health.is_healthy:
                        self.logger.warning(f"Component {name} hasn't reported success in 60s")
                        health.health_score = max(0.0, health.health_score - 0.1)
            
            # Check for error rate spikes
            recent_errors = [e for e in self.error_history 
                           if current_time - e.timestamp < 300]  # 5 minutes
            
            if len(recent_errors) > 10:  # More than 10 errors in 5 minutes
                self.logger.warning(f"High error rate detected: {len(recent_errors)} errors in 5 minutes")
    
    def reset_error_history(self) -> None:
        """Reset error history and statistics."""
        with self.lock:
            self.error_history.clear()
            self.error_counts.clear()
            self.last_errors.clear()
            self.critical_errors = 0
            self.system_degraded = False
            
            # Reset component health
            for health in self.component_health.values():
                health.error_count = 0
                health.consecutive_errors = 0
                health.is_healthy = True
                health.degraded_mode = False
                health.health_score = 1.0
            
            self.logger.info("Error history and statistics reset")
    
    def shutdown(self) -> None:
        """Shutdown error handling system."""
        self.stop_health_monitoring()
        
        # Log final statistics
        stats = self.get_error_statistics()
        health = self.get_system_health()
        
        self.logger.info(f"Error handling system shutdown. Final stats: {stats['total_errors']} total errors")
        self.logger.info(f"Final health score: {health['overall_health_score']:.2f}")


# Convenience functions
def create_error_handler() -> ErrorHandlingSystem:
    """Create error handling system with default configuration."""
    return ErrorHandlingSystem()


def handle_component_error(error_handler: ErrorHandlingSystem, 
                         component: str, error: Exception,
                         context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to handle component errors.
    
    Args:
        error_handler: Error handling system instance
        component: Component name
        error: Exception that occurred
        context: Additional context
        
    Returns:
        True if error was handled successfully
    """
    # Determine severity based on error type
    if isinstance(error, (MemoryError, SystemError)):
        severity = ErrorSeverity.CRITICAL
        category = ErrorCategory.RESOURCE
    elif isinstance(error, TimeoutError):
        severity = ErrorSeverity.MEDIUM
        category = ErrorCategory.TIMEOUT
    elif isinstance(error, (IOError, OSError)):
        severity = ErrorSeverity.HIGH
        category = ErrorCategory.HARDWARE
    elif isinstance(error, (ValueError, TypeError)):
        severity = ErrorSeverity.MEDIUM
        category = ErrorCategory.DATA
    else:
        severity = ErrorSeverity.MEDIUM
        category = ErrorCategory.SOFTWARE
    
    return error_handler.report_error(
        component=component,
        error=error,
        context=context,
        severity=severity,
        category=category
    )