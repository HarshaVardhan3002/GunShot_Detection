"""
Real-time output formatting for gunshot localization system.
"""
import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading
from collections import deque
import sys
import os


class OutputFormat(Enum):
    """Output format types."""
    CONSOLE = "console"
    JSON = "json"
    CSV = "csv"
    STRUCTURED = "structured"


class LogLevel(Enum):
    """Custom log levels for gunshot events."""
    DETECTION = "DETECTION"
    LOCALIZATION = "LOCALIZATION"
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    ERROR = "ERROR"


@dataclass
class FormattedOutput:
    """Container for formatted output data."""
    timestamp: str
    event_type: str
    message: str
    data: Dict[str, Any]
    format_type: OutputFormat


class ConsoleFormatter:
    """Console output formatter with colors and formatting."""
    
    # ANSI color codes
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'reset': '\033[0m'
    }
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize console formatter.
        
        Args:
            use_colors: Whether to use ANSI colors in output
        """
        self.use_colors = use_colors and self._supports_color()
        self.logger = logging.getLogger(__name__)
    
    def _supports_color(self) -> bool:
        """Check if terminal supports ANSI colors."""
        return (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
            os.environ.get('TERM') != 'dumb'
        )
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def format_detection(self, timestamp: str, confidence: float, 
                        detection_method: str, processing_time: float) -> str:
        """Format gunshot detection output."""
        confidence_color = 'green' if confidence > 0.8 else 'yellow' if confidence > 0.5 else 'red'
        confidence_str = self._colorize(f"{confidence:.3f}", confidence_color)
        
        time_color = 'green' if processing_time < 0.01 else 'yellow' if processing_time < 0.05 else 'red'
        time_str = self._colorize(f"{processing_time*1000:.1f}ms", time_color)
        
        return (
            f"{self._colorize('[DETECTION]', 'cyan')} "
            f"{timestamp} | "
            f"Confidence: {confidence_str} | "
            f"Method: {detection_method} | "
            f"Time: {time_str}"
        )
    
    def format_localization(self, timestamp: str, x: float, y: float, z: float,
                          confidence: float, error: float, processing_time: float,
                          channels_used: int) -> str:
        """Format gunshot localization output."""
        # Color code based on confidence
        conf_color = 'green' if confidence > 0.7 else 'yellow' if confidence > 0.4 else 'red'
        confidence_str = self._colorize(f"{confidence:.3f}", conf_color)
        
        # Color code based on error
        error_color = 'green' if error < 0.5 else 'yellow' if error < 2.0 else 'red'
        error_str = self._colorize(f"{error:.2f}m", error_color)
        
        # Color code based on processing time
        time_color = 'green' if processing_time < 0.05 else 'yellow' if processing_time < 0.1 else 'red'
        time_str = self._colorize(f"{processing_time*1000:.1f}ms", time_color)
        
        # Format coordinates
        coords = self._colorize(f"({x:6.2f}, {y:6.2f}, {z:6.2f})", 'white')
        
        return (
            f"{self._colorize('[LOCATION]', 'magenta')} "
            f"{timestamp} | "
            f"Position: {coords} | "
            f"Confidence: {confidence_str} | "
            f"Error: {error_str} | "
            f"Channels: {channels_used} | "
            f"Time: {time_str}"
        )
    
    def format_system_status(self, timestamp: str, status: Dict[str, Any]) -> str:
        """Format system status output."""
        state = status.get('pipeline_state', 'unknown')
        state_color = {
            'running': 'green',
            'stopped': 'red',
            'starting': 'yellow',
            'stopping': 'yellow',
            'error': 'red'
        }.get(state, 'white')
        
        state_str = self._colorize(state.upper(), state_color)
        
        metrics = status.get('metrics', {})
        detections = metrics.get('total_detections', 0)
        accuracy = metrics.get('localization_accuracy', 0.0)
        avg_time = metrics.get('average_processing_time', 0.0)
        
        accuracy_color = 'green' if accuracy > 0.8 else 'yellow' if accuracy > 0.5 else 'red'
        accuracy_str = self._colorize(f"{accuracy:.1%}", accuracy_color)
        
        return (
            f"{self._colorize('[SYSTEM]', 'blue')} "
            f"{timestamp} | "
            f"State: {state_str} | "
            f"Detections: {detections} | "
            f"Accuracy: {accuracy_str} | "
            f"Avg Time: {avg_time*1000:.1f}ms"
        )
    
    def format_performance(self, timestamp: str, metrics: Dict[str, Any]) -> str:
        """Format performance metrics output."""
        latency = metrics.get('average_latency', 0.0)
        detection_rate = metrics.get('detection_rate', 0.0)
        uptime = metrics.get('uptime', 0.0)
        
        latency_color = 'green' if latency < 0.1 else 'yellow' if latency < 0.5 else 'red'
        latency_str = self._colorize(f"{latency*1000:.1f}ms", latency_color)
        
        rate_str = self._colorize(f"{detection_rate:.2f}/s", 'cyan')
        uptime_str = self._colorize(f"{uptime:.0f}s", 'white')
        
        return (
            f"{self._colorize('[PERF]', 'yellow')} "
            f"{timestamp} | "
            f"Latency: {latency_str} | "
            f"Rate: {rate_str} | "
            f"Uptime: {uptime_str}"
        )
    
    def format_error(self, timestamp: str, error: str, context: str) -> str:
        """Format error output."""
        return (
            f"{self._colorize('[ERROR]', 'red')} "
            f"{timestamp} | "
            f"Context: {self._colorize(context, 'yellow')} | "
            f"Error: {self._colorize(error, 'red')}"
        )


class JSONFormatter:
    """JSON output formatter for structured logging."""
    
    def __init__(self, pretty_print: bool = False):
        """
        Initialize JSON formatter.
        
        Args:
            pretty_print: Whether to format JSON with indentation
        """
        self.pretty_print = pretty_print
        self.logger = logging.getLogger(__name__)
    
    def format_detection(self, timestamp: str, confidence: float,
                        detection_method: str, processing_time: float) -> str:
        """Format detection as JSON."""
        data = {
            "event_type": "detection",
            "timestamp": timestamp,
            "confidence": confidence,
            "detection_method": detection_method,
            "processing_time_ms": processing_time * 1000
        }
        return self._format_json(data)
    
    def format_localization(self, timestamp: str, x: float, y: float, z: float,
                          confidence: float, error: float, processing_time: float,
                          channels_used: int) -> str:
        """Format localization as JSON."""
        data = {
            "event_type": "localization",
            "timestamp": timestamp,
            "position": {"x": x, "y": y, "z": z},
            "confidence": confidence,
            "error_meters": error,
            "processing_time_ms": processing_time * 1000,
            "channels_used": channels_used
        }
        return self._format_json(data)
    
    def format_system_status(self, timestamp: str, status: Dict[str, Any]) -> str:
        """Format system status as JSON."""
        data = {
            "event_type": "system_status",
            "timestamp": timestamp,
            "status": status
        }
        return self._format_json(data)
    
    def format_performance(self, timestamp: str, metrics: Dict[str, Any]) -> str:
        """Format performance metrics as JSON."""
        data = {
            "event_type": "performance",
            "timestamp": timestamp,
            "metrics": metrics
        }
        return self._format_json(data)
    
    def format_error(self, timestamp: str, error: str, context: str) -> str:
        """Format error as JSON."""
        data = {
            "event_type": "error",
            "timestamp": timestamp,
            "error": error,
            "context": context
        }
        return self._format_json(data)
    
    def _format_json(self, data: Dict[str, Any]) -> str:
        """Format data as JSON string."""
        if self.pretty_print:
            return json.dumps(data, indent=2, default=str)
        return json.dumps(data, default=str)


class CSVFormatter:
    """CSV output formatter for data analysis."""
    
    def __init__(self):
        """Initialize CSV formatter."""
        self.logger = logging.getLogger(__name__)
        self.headers_written = set()
    
    def format_detection(self, timestamp: str, confidence: float,
                        detection_method: str, processing_time: float) -> str:
        """Format detection as CSV."""
        return f"{timestamp},detection,{confidence},{detection_method},{processing_time*1000:.3f}"
    
    def format_localization(self, timestamp: str, x: float, y: float, z: float,
                          confidence: float, error: float, processing_time: float,
                          channels_used: int) -> str:
        """Format localization as CSV."""
        return (f"{timestamp},localization,{x:.3f},{y:.3f},{z:.3f},"
                f"{confidence:.3f},{error:.3f},{processing_time*1000:.3f},{channels_used}")
    
    def get_detection_header(self) -> str:
        """Get CSV header for detection events."""
        return "timestamp,event_type,confidence,detection_method,processing_time_ms"
    
    def get_localization_header(self) -> str:
        """Get CSV header for localization events."""
        return ("timestamp,event_type,x,y,z,confidence,error_meters,"
                "processing_time_ms,channels_used")


class RealTimeOutputManager:
    """Manager for real-time output formatting and display."""
    
    def __init__(self, output_format: OutputFormat = OutputFormat.CONSOLE,
                 enable_file_output: bool = False, output_file: Optional[str] = None,
                 buffer_size: int = 1000):
        """
        Initialize output manager.
        
        Args:
            output_format: Primary output format
            enable_file_output: Whether to write to file
            output_file: Output file path (if file output enabled)
            buffer_size: Size of output buffer for performance
        """
        self.output_format = output_format
        self.enable_file_output = enable_file_output
        self.output_file = output_file
        self.buffer_size = buffer_size
        
        # Initialize formatters
        self.console_formatter = ConsoleFormatter()
        self.json_formatter = JSONFormatter()
        self.csv_formatter = CSVFormatter()
        
        # Output buffer and threading
        self.output_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.file_handle = None
        
        # Statistics
        self.output_count = 0
        self.last_output_time = time.time()
        
        self.logger = logging.getLogger(__name__)
        
        # Open file if needed
        if self.enable_file_output and self.output_file:
            self._open_output_file()
    
    def _open_output_file(self) -> None:
        """Open output file for writing."""
        try:
            self.file_handle = open(self.output_file, 'a', encoding='utf-8')
            self.logger.info(f"Output file opened: {self.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to open output file {self.output_file}: {e}")
            self.enable_file_output = False
    
    def _write_output(self, formatted_output: str) -> None:
        """Write formatted output to console and/or file."""
        # Write to console
        print(formatted_output, flush=True)
        
        # Write to file if enabled
        if self.enable_file_output and self.file_handle:
            try:
                self.file_handle.write(formatted_output + '\n')
                self.file_handle.flush()
            except Exception as e:
                self.logger.error(f"Failed to write to output file: {e}")
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def output_detection(self, confidence: float, detection_method: str,
                        processing_time: float) -> None:
        """Output gunshot detection event."""
        timestamp = self._get_timestamp()
        
        if self.output_format == OutputFormat.CONSOLE:
            formatted = self.console_formatter.format_detection(
                timestamp, confidence, detection_method, processing_time
            )
        elif self.output_format == OutputFormat.JSON:
            formatted = self.json_formatter.format_detection(
                timestamp, confidence, detection_method, processing_time
            )
        elif self.output_format == OutputFormat.CSV:
            formatted = self.csv_formatter.format_detection(
                timestamp, confidence, detection_method, processing_time
            )
        else:
            formatted = f"DETECTION: {timestamp} | Confidence: {confidence:.3f}"
        
        self._write_output(formatted)
        self.output_count += 1
        self.last_output_time = time.time()
    
    def output_localization(self, x: float, y: float, z: float, confidence: float,
                          error: float, processing_time: float, channels_used: int) -> None:
        """Output gunshot localization result."""
        timestamp = self._get_timestamp()
        
        if self.output_format == OutputFormat.CONSOLE:
            formatted = self.console_formatter.format_localization(
                timestamp, x, y, z, confidence, error, processing_time, channels_used
            )
        elif self.output_format == OutputFormat.JSON:
            formatted = self.json_formatter.format_localization(
                timestamp, x, y, z, confidence, error, processing_time, channels_used
            )
        elif self.output_format == OutputFormat.CSV:
            formatted = self.csv_formatter.format_localization(
                timestamp, x, y, z, confidence, error, processing_time, channels_used
            )
        else:
            formatted = f"LOCATION: {timestamp} | Position: ({x:.2f}, {y:.2f}, {z:.2f})"
        
        self._write_output(formatted)
        self.output_count += 1
        self.last_output_time = time.time()
    
    def output_system_status(self, status: Dict[str, Any]) -> None:
        """Output system status information."""
        timestamp = self._get_timestamp()
        
        if self.output_format == OutputFormat.CONSOLE:
            formatted = self.console_formatter.format_system_status(timestamp, status)
        elif self.output_format == OutputFormat.JSON:
            formatted = self.json_formatter.format_system_status(timestamp, status)
        else:
            state = status.get('pipeline_state', 'unknown')
            formatted = f"SYSTEM: {timestamp} | State: {state}"
        
        self._write_output(formatted)
        self.output_count += 1
    
    def output_performance(self, metrics: Dict[str, Any]) -> None:
        """Output performance metrics."""
        timestamp = self._get_timestamp()
        
        if self.output_format == OutputFormat.CONSOLE:
            formatted = self.console_formatter.format_performance(timestamp, metrics)
        elif self.output_format == OutputFormat.JSON:
            formatted = self.json_formatter.format_performance(timestamp, metrics)
        else:
            latency = metrics.get('average_latency', 0.0)
            formatted = f"PERFORMANCE: {timestamp} | Latency: {latency*1000:.1f}ms"
        
        self._write_output(formatted)
        self.output_count += 1
    
    def output_error(self, error: str, context: str) -> None:
        """Output error information."""
        timestamp = self._get_timestamp()
        
        if self.output_format == OutputFormat.CONSOLE:
            formatted = self.console_formatter.format_error(timestamp, error, context)
        elif self.output_format == OutputFormat.JSON:
            formatted = self.json_formatter.format_error(timestamp, error, context)
        else:
            formatted = f"ERROR: {timestamp} | {context}: {error}"
        
        self._write_output(formatted)
        self.output_count += 1
    
    def get_output_stats(self) -> Dict[str, Any]:
        """Get output statistics."""
        return {
            'total_outputs': self.output_count,
            'last_output_time': self.last_output_time,
            'output_format': self.output_format.value,
            'file_output_enabled': self.enable_file_output,
            'output_file': self.output_file,
            'buffer_size': self.buffer_size
        }
    
    def close(self) -> None:
        """Close output manager and cleanup resources."""
        if self.file_handle:
            try:
                self.file_handle.close()
                self.logger.info("Output file closed")
            except Exception as e:
                self.logger.error(f"Error closing output file: {e}")
        
        self.logger.info(f"Output manager closed. Total outputs: {self.output_count}")


# Convenience functions for quick output formatting
def format_detection_console(confidence: float, method: str, time_ms: float) -> str:
    """Quick console format for detection."""
    formatter = ConsoleFormatter()
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return formatter.format_detection(timestamp, confidence, method, time_ms/1000)


def format_localization_console(x: float, y: float, confidence: float, 
                               error: float, time_ms: float) -> str:
    """Quick console format for localization."""
    formatter = ConsoleFormatter()
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return formatter.format_localization(timestamp, x, y, 0.0, confidence, 
                                       error, time_ms/1000, 8)


def create_output_manager(format_type: str = "console", 
                         output_file: Optional[str] = None) -> RealTimeOutputManager:
    """
    Create output manager with specified format.
    
    Args:
        format_type: Output format ("console", "json", "csv")
        output_file: Optional output file path
        
    Returns:
        Configured output manager
    """
    format_map = {
        "console": OutputFormat.CONSOLE,
        "json": OutputFormat.JSON,
        "csv": OutputFormat.CSV,
        "structured": OutputFormat.STRUCTURED
    }
    
    output_format = format_map.get(format_type.lower(), OutputFormat.CONSOLE)
    enable_file = output_file is not None
    
    return RealTimeOutputManager(
        output_format=output_format,
        enable_file_output=enable_file,
        output_file=output_file
    )