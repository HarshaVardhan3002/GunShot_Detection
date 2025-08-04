"""
Structured logging system for gunshot localization system.
"""
import logging
import logging.handlers
import json
import time
import os
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path


class LogEventType(Enum):
    """Types of log events."""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    DETECTION = "detection"
    LOCALIZATION = "localization"
    PERFORMANCE = "performance"
    ERROR = "error"
    WARNING = "warning"
    DEBUG = "debug"
    COMPONENT_STATUS = "component_status"
    CONFIGURATION = "configuration"


@dataclass
class StructuredLogEntry:
    """Structured log entry with metadata."""
    timestamp: str
    event_type: LogEventType
    level: str
    message: str
    component: str
    data: Dict[str, Any]
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    process_id: Optional[int] = None


class JSONLogFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize JSON formatter.
        
        Args:
            session_id: Optional session identifier
        """
        super().__init__()
        self.session_id = session_id or f"session_{int(time.time())}"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract structured data if available
        structured_data = getattr(record, 'structured_data', {})
        event_type = getattr(record, 'event_type', LogEventType.DEBUG.value)
        component = getattr(record, 'component', record.name)
        
        # Create structured entry
        entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'event_type': event_type,
            'level': record.levelname,
            'message': record.getMessage(),
            'component': component,
            'data': structured_data,
            'session_id': self.session_id,
            'thread_id': str(threading.current_thread().ident),
            'process_id': os.getpid()
        }
        
        # Add exception info if present
        if record.exc_info:
            entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(entry, default=str)


class PerformanceLogFormatter(logging.Formatter):
    """Specialized formatter for performance metrics."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format performance log record."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        
        # Extract performance data
        perf_data = getattr(record, 'structured_data', {})
        
        # Format key metrics
        latency = perf_data.get('latency_ms', 0)
        accuracy = perf_data.get('accuracy', 0)
        throughput = perf_data.get('throughput', 0)
        
        return f"{timestamp} | PERF | Latency: {latency:.1f}ms | Accuracy: {accuracy:.1%} | Throughput: {throughput:.2f}/s"


class StructuredLogger:
    """Main structured logging system."""
    
    def __init__(self, name: str, log_dir: str = "logs", 
                 session_id: Optional[str] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_performance_log: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            session_id: Session identifier
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_performance_log: Enable separate performance log
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_performance_log = enable_performance_log
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self.main_logger = logging.getLogger(f"{name}.main")
        self.performance_logger = logging.getLogger(f"{name}.performance")
        self.error_logger = logging.getLogger(f"{name}.error")
        
        # Set levels
        self.main_logger.setLevel(logging.DEBUG)
        self.performance_logger.setLevel(logging.INFO)
        self.error_logger.setLevel(logging.WARNING)
        
        # Setup handlers
        self._setup_handlers()
        
        # Statistics
        self.log_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        self.log_system_start()
    
    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        # Clear existing handlers
        for logger in [self.main_logger, self.performance_logger, self.error_logger]:
            logger.handlers.clear()
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.main_logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file:
            # Main log file (JSON format)
            main_log_file = self.log_dir / f"gunshot_localizer_{self.session_id}.jsonl"
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_file, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            main_handler.setLevel(logging.DEBUG)
            main_handler.setFormatter(JSONLogFormatter(self.session_id))
            self.main_logger.addHandler(main_handler)
            
            # Error log file
            error_log_file = self.log_dir / f"errors_{self.session_id}.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file, maxBytes=self.max_file_size, backupCount=self.backup_count
            )
            error_handler.setLevel(logging.WARNING)
            error_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.error_logger.addHandler(error_handler)
            
            # Performance log file
            if self.enable_performance_log:
                perf_log_file = self.log_dir / f"performance_{self.session_id}.log"
                perf_handler = logging.handlers.RotatingFileHandler(
                    perf_log_file, maxBytes=self.max_file_size, backupCount=self.backup_count
                )
                perf_handler.setLevel(logging.INFO)
                perf_handler.setFormatter(PerformanceLogFormatter())
                self.performance_logger.addHandler(perf_handler)
    
    def _log_structured(self, logger: logging.Logger, level: int, 
                       event_type: LogEventType, message: str,
                       component: str, data: Dict[str, Any]) -> None:
        """Log structured entry."""
        # Create log record
        record = logger.makeRecord(
            logger.name, level, "", 0, message, (), None
        )
        
        # Add structured data
        record.structured_data = data
        record.event_type = event_type.value
        record.component = component
        
        # Log the record
        logger.handle(record)
        
        # Update statistics
        self.log_count += 1
        self.last_log_time = time.time()
    
    def log_system_start(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Log system startup."""
        data = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'configuration': config or {}
        }
        self._log_structured(
            self.main_logger, logging.INFO, LogEventType.SYSTEM_START,
            "Gunshot localization system started", "system", data
        )
    
    def log_system_stop(self, uptime: float, final_stats: Dict[str, Any]) -> None:
        """Log system shutdown."""
        data = {
            'session_id': self.session_id,
            'uptime_seconds': uptime,
            'final_statistics': final_stats,
            'total_logs': self.log_count
        }
        self._log_structured(
            self.main_logger, logging.INFO, LogEventType.SYSTEM_STOP,
            "Gunshot localization system stopped", "system", data
        )
    
    def log_detection(self, confidence: float, detection_method: str,
                     processing_time: float, channels: List[int],
                     metadata: Dict[str, Any]) -> None:
        """Log gunshot detection event."""
        data = {
            'confidence': confidence,
            'detection_method': detection_method,
            'processing_time_ms': processing_time * 1000,
            'channels_detected': channels,
            'metadata': metadata
        }
        self._log_structured(
            self.main_logger, logging.INFO, LogEventType.DETECTION,
            f"Gunshot detected with confidence {confidence:.3f}", "detector", data
        )
    
    def log_localization(self, x: float, y: float, z: float, confidence: float,
                        error: float, processing_time: float, channels_used: List[int],
                        method: str, metadata: Dict[str, Any]) -> None:
        """Log gunshot localization result."""
        data = {
            'position': {'x': x, 'y': y, 'z': z},
            'confidence': confidence,
            'error_meters': error,
            'processing_time_ms': processing_time * 1000,
            'channels_used': channels_used,
            'localization_method': method,
            'metadata': metadata
        }
        self._log_structured(
            self.main_logger, logging.INFO, LogEventType.LOCALIZATION,
            f"Gunshot localized at ({x:.2f}, {y:.2f}, {z:.2f})", "localizer", data
        )
    
    def log_performance(self, latency_ms: float, accuracy: float, throughput: float,
                       resource_usage: Dict[str, Any], component_times: Dict[str, float]) -> None:
        """Log performance metrics."""
        data = {
            'latency_ms': latency_ms,
            'accuracy': accuracy,
            'throughput': throughput,
            'resource_usage': resource_usage,
            'component_processing_times': component_times
        }
        self._log_structured(
            self.performance_logger, logging.INFO, LogEventType.PERFORMANCE,
            f"Performance: {latency_ms:.1f}ms latency, {accuracy:.1%} accuracy", 
            "performance", data
        )
    
    def log_component_status(self, component: str, status: str, 
                           metrics: Dict[str, Any]) -> None:
        """Log component status update."""
        data = {
            'component_name': component,
            'status': status,
            'metrics': metrics,
            'timestamp': time.time()
        }
        self._log_structured(
            self.main_logger, logging.INFO, LogEventType.COMPONENT_STATUS,
            f"Component {component} status: {status}", component, data
        )
    
    def log_error(self, error: str, context: str, component: str,
                 exception: Optional[Exception] = None, 
                 additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Log error event."""
        data = {
            'error_message': error,
            'context': context,
            'component': component,
            'additional_data': additional_data or {}
        }
        
        if exception:
            data['exception_type'] = type(exception).__name__
            data['exception_details'] = str(exception)
        
        self._log_structured(
            self.error_logger, logging.ERROR, LogEventType.ERROR,
            f"Error in {component}: {error}", component, data
        )
    
    def log_warning(self, warning: str, component: str, 
                   additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Log warning event."""
        data = {
            'warning_message': warning,
            'component': component,
            'additional_data': additional_data or {}
        }
        self._log_structured(
            self.main_logger, logging.WARNING, LogEventType.WARNING,
            f"Warning in {component}: {warning}", component, data
        )
    
    def log_configuration(self, config: Dict[str, Any], component: str) -> None:
        """Log configuration changes."""
        data = {
            'configuration': config,
            'component': component,
            'timestamp': time.time()
        }
        self._log_structured(
            self.main_logger, logging.INFO, LogEventType.CONFIGURATION,
            f"Configuration updated for {component}", component, data
        )
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        uptime = time.time() - self.start_time
        return {
            'session_id': self.session_id,
            'total_logs': self.log_count,
            'uptime_seconds': uptime,
            'logs_per_second': self.log_count / uptime if uptime > 0 else 0,
            'last_log_time': self.last_log_time,
            'log_directory': str(self.log_dir),
            'handlers_enabled': {
                'console': self.enable_console,
                'file': self.enable_file,
                'performance': self.enable_performance_log
            }
        }
    
    def flush_logs(self) -> None:
        """Flush all log handlers."""
        for logger in [self.main_logger, self.performance_logger, self.error_logger]:
            for handler in logger.handlers:
                handler.flush()
    
    def close(self) -> None:
        """Close logger and cleanup resources."""
        # Log system stop
        uptime = time.time() - self.start_time
        final_stats = self.get_log_statistics()
        self.log_system_stop(uptime, final_stats)
        
        # Flush and close handlers
        for logger in [self.main_logger, self.performance_logger, self.error_logger]:
            for handler in logger.handlers:
                handler.flush()
                handler.close()
            logger.handlers.clear()


class PerformanceTracker:
    """Performance tracking and logging utility."""
    
    def __init__(self, logger: StructuredLogger):
        """
        Initialize performance tracker.
        
        Args:
            logger: Structured logger instance
        """
        self.logger = logger
        self.metrics = {}
        self.start_times = {}
        self.component_times = {}
        self.lock = threading.Lock()
    
    def start_timing(self, operation: str) -> None:
        """Start timing an operation."""
        with self.lock:
            self.start_times[operation] = time.time()
    
    def end_timing(self, operation: str) -> float:
        """End timing an operation and return duration."""
        with self.lock:
            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                self.component_times[operation] = duration
                del self.start_times[operation]
                return duration
            return 0.0
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a performance metric."""
        with self.lock:
            self.metrics[name] = value
    
    def log_performance_summary(self, accuracy: float, throughput: float) -> None:
        """Log comprehensive performance summary."""
        with self.lock:
            total_latency = sum(self.component_times.values()) * 1000  # Convert to ms
            
            # Get resource usage (simplified)
            import psutil
            process = psutil.Process()
            resource_usage = {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'threads': process.num_threads()
            }
            
            self.logger.log_performance(
                latency_ms=total_latency,
                accuracy=accuracy,
                throughput=throughput,
                resource_usage=resource_usage,
                component_times={k: v*1000 for k, v in self.component_times.items()}
            )
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.component_times.clear()
            self.start_times.clear()


# Convenience functions
def create_structured_logger(name: str = "gunshot_localizer", 
                           log_dir: str = "logs") -> StructuredLogger:
    """Create a structured logger with default settings."""
    return StructuredLogger(
        name=name,
        log_dir=log_dir,
        enable_console=True,
        enable_file=True,
        enable_performance_log=True
    )


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> StructuredLogger:
    """
    Setup logging system for gunshot localization.
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
        
    Returns:
        Configured structured logger
    """
    # Set root logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    
    # Create structured logger
    logger = create_structured_logger("gunshot_localizer", log_dir)
    
    return logger