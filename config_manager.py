"""
Configuration management for gunshot localization system.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import json
import logging
from pathlib import Path


@dataclass
class MicrophonePosition:
    """Microphone position data."""
    id: int
    x: float
    y: float
    z: float = 0.0


@dataclass
class SystemConfig:
    """System configuration parameters."""
    sample_rate: int = 48000
    sound_speed: float = 343.0
    detection_threshold_db: float = -20.0
    buffer_duration: float = 2.0
    min_confidence: float = 0.7


class ConfigurationManagerInterface(ABC):
    """Abstract interface for configuration management."""
    
    @abstractmethod
    def load_config(self, config_path: str) -> bool:
        """Load configuration from file."""
        pass
    
    @abstractmethod
    def get_microphone_positions(self) -> List[MicrophonePosition]:
        """Get microphone position data."""
        pass
    
    @abstractmethod
    def get_system_config(self) -> SystemConfig:
        """Get system configuration parameters."""
        pass
    
    @abstractmethod
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate configuration and return errors if any."""
        pass


class ConfigurationManager(ConfigurationManagerInterface):
    """Concrete implementation of configuration management."""
    
    def __init__(self):
        """Initialize configuration manager with defaults."""
        self.logger = logging.getLogger(__name__)
        self._system_config = SystemConfig()
        self._config_loaded = False
        
        # Default microphone positions (well-distributed array)
        self._default_positions = [
            MicrophonePosition(1, 0.0, 0.0, 0.0),      # Southwest corner
            MicrophonePosition(2, 10.0, 0.0, 0.0),     # Southeast corner
            MicrophonePosition(3, 10.0, 10.0, 0.0),    # Northeast corner
            MicrophonePosition(4, 0.0, 10.0, 0.0),     # Northwest corner
            MicrophonePosition(5, 5.0, 0.0, 0.0),      # South edge center
            MicrophonePosition(6, 10.0, 5.0, 0.0),     # East edge center
            MicrophonePosition(7, 5.0, 10.0, 0.0),     # North edge center
            MicrophonePosition(8, 0.0, 5.0, 0.0)       # West edge center
        ]
        
        # Initialize with defaults
        self._microphone_positions = self._default_positions.copy()
    
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                self.logger.warning(f"Configuration file not found: {config_path}")
                self._load_defaults()
                return False
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Load microphone positions
            if 'microphones' in config_data:
                self._microphone_positions = []
                for mic_data in config_data['microphones']:
                    mic_pos = MicrophonePosition(
                        id=mic_data['id'],
                        x=float(mic_data['x']),
                        y=float(mic_data['y']),
                        z=float(mic_data.get('z', 0.0))
                    )
                    self._microphone_positions.append(mic_pos)
            else:
                self.logger.warning("No microphone positions found in config, using defaults")
                self._microphone_positions = self._default_positions.copy()
            
            # Load system configuration
            if 'system' in config_data:
                system_data = config_data['system']
                self._system_config = SystemConfig(
                    sample_rate=int(system_data.get('sample_rate', 48000)),
                    sound_speed=float(system_data.get('sound_speed', 343.0)),
                    detection_threshold_db=float(system_data.get('detection_threshold_db', -20.0)),
                    buffer_duration=float(system_data.get('buffer_duration', 2.0)),
                    min_confidence=float(system_data.get('min_confidence', 0.7))
                )
            
            self._config_loaded = True
            self.logger.info(f"Configuration loaded successfully from {config_path}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            self._load_defaults()
            return False
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self._load_defaults()
            return False
    
    def _load_defaults(self):
        """Load default configuration values."""
        self._microphone_positions = self._default_positions.copy()
        self._system_config = SystemConfig()
        self._config_loaded = False
        self.logger.info("Using default configuration values")
    
    def get_microphone_positions(self) -> List[MicrophonePosition]:
        """Get microphone position data."""
        return self._microphone_positions.copy()
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration parameters."""
        return self._system_config
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration and return errors if any.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check microphone count
        if len(self._microphone_positions) != 8:
            errors.append(f"Expected 8 microphones, found {len(self._microphone_positions)}")
        
        # Check for duplicate microphone IDs
        mic_ids = [mic.id for mic in self._microphone_positions]
        if len(set(mic_ids)) != len(mic_ids):
            errors.append("Duplicate microphone IDs found")
        
        # Check microphone ID range
        for mic in self._microphone_positions:
            if mic.id < 1 or mic.id > 8:
                errors.append(f"Microphone ID {mic.id} out of range (1-8)")
        
        # Check for duplicate positions
        position_tuples = [(mic.x, mic.y, mic.z) for mic in self._microphone_positions]
        if len(set(position_tuples)) != len(position_tuples):
            errors.append("Duplicate microphone positions found")
        
        # Validate geometric constraints for triangulation
        geometric_errors = self._validate_geometric_constraints()
        errors.extend(geometric_errors)
        
        # Check system parameters
        if self._system_config.sample_rate <= 0:
            errors.append("Sample rate must be positive")
        
        if self._system_config.sound_speed <= 0:
            errors.append("Sound speed must be positive")
        
        if self._system_config.buffer_duration <= 0:
            errors.append("Buffer duration must be positive")
        
        if not (0.0 <= self._system_config.min_confidence <= 1.0):
            errors.append("Minimum confidence must be between 0.0 and 1.0")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_geometric_constraints(self) -> List[str]:
        """
        Validate geometric constraints for triangulation feasibility.
        
        Returns:
            List of error messages
        """
        errors = []
        positions = [(mic.x, mic.y) for mic in self._microphone_positions]
        
        if len(positions) < 4:
            errors.append("At least 4 microphones required for 2D triangulation")
            return errors
        
        # Check for collinear positions
        if self._check_collinear_positions(positions):
            errors.append("Microphones are too collinear for accurate triangulation")
        
        # Check minimum distances between microphones
        min_distance_errors = self._check_minimum_distances(positions)
        errors.extend(min_distance_errors)
        
        # Check array geometry for good triangulation
        geometry_errors = self._check_array_geometry(positions)
        errors.extend(geometry_errors)
        
        return errors
    
    def _check_collinear_positions(self, positions: List[Tuple[float, float]]) -> bool:
        """
        Check if microphone positions are too collinear.
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            True if positions are too collinear
        """
        if len(positions) < 3:
            return True
        
        # Check multiple combinations of 3 points to ensure good geometry
        collinear_count = 0
        total_combinations = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                for k in range(j + 1, len(positions)):
                    p1, p2, p3 = positions[i], positions[j], positions[k]
                    area = abs((p1[0] * (p2[1] - p3[1]) + 
                               p2[0] * (p3[1] - p1[1]) + 
                               p3[0] * (p1[1] - p2[1])) / 2.0)
                    
                    total_combinations += 1
                    if area < 0.1:  # Minimum area threshold
                        collinear_count += 1
        
        # If more than 50% of combinations are collinear, geometry is poor
        return collinear_count / total_combinations > 0.5
    
    def _check_minimum_distances(self, positions: List[Tuple[float, float]]) -> List[str]:
        """
        Check minimum distances between microphones.
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            List of error messages
        """
        errors = []
        min_distance = 0.5  # Minimum 0.5 meters between microphones
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                p1, p2 = positions[i], positions[j]
                distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                
                if distance < min_distance:
                    mic1_id = self._microphone_positions[i].id
                    mic2_id = self._microphone_positions[j].id
                    errors.append(
                        f"Microphones {mic1_id} and {mic2_id} are too close "
                        f"({distance:.2f}m < {min_distance}m minimum)"
                    )
        
        return errors
    
    def _check_array_geometry(self, positions: List[Tuple[float, float]]) -> List[str]:
        """
        Check array geometry for good triangulation performance.
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Calculate array span (max distance between any two microphones)
        max_distance = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                p1, p2 = positions[i], positions[j]
                distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                max_distance = max(max_distance, distance)
        
        # Check if array is too small (affects accuracy at long range)
        if max_distance < 5.0:
            errors.append(
                f"Microphone array span is small ({max_distance:.1f}m). "
                "Consider increasing spacing for better long-range accuracy."
            )
        
        # Check if array is excessively large (affects synchronization)
        if max_distance > 100.0:
            errors.append(
                f"Microphone array span is very large ({max_distance:.1f}m). "
                "This may cause synchronization issues."
            )
        
        # Check for good perimeter coverage
        perimeter_errors = self._check_perimeter_coverage(positions)
        errors.extend(perimeter_errors)
        
        return errors
    
    def _check_perimeter_coverage(self, positions: List[Tuple[float, float]]) -> List[str]:
        """
        Check if microphones provide good perimeter coverage.
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Calculate centroid
        centroid_x = sum(p[0] for p in positions) / len(positions)
        centroid_y = sum(p[1] for p in positions) / len(positions)
        
        # Calculate angles from centroid to each microphone
        import math
        angles = []
        for x, y in positions:
            angle = math.atan2(y - centroid_y, x - centroid_x)
            angles.append(angle)
        
        # Sort angles and check for large gaps
        angles.sort()
        
        # Check gaps between consecutive angles
        max_gap = 0
        for i in range(len(angles)):
            next_i = (i + 1) % len(angles)
            gap = angles[next_i] - angles[i]
            if gap < 0:
                gap += 2 * math.pi  # Handle wrap-around
            max_gap = max(max_gap, gap)
        
        # If largest gap is more than 90 degrees, coverage may be poor
        if max_gap > math.pi / 2:
            errors.append(
                f"Large gap in microphone coverage ({math.degrees(max_gap):.1f}°). "
                "Consider redistributing microphones for better angular coverage."
            )
        
        return errors
    
    def is_config_loaded(self) -> bool:
        """Check if configuration was loaded from file."""
        return self._config_loaded
    
    def generate_config_template(self, output_path: str, array_type: str = "default") -> bool:
        """
        Generate a configuration template file.
        
        Args:
            output_path: Path where to save the template
            array_type: Type of array layout ("default", "square", "circular", "linear")
            
        Returns:
            True if template generated successfully
        """
        try:
            template_positions = self._generate_array_positions(array_type)
            
            config_template = {
                "microphones": [
                    {
                        "id": pos.id,
                        "x": pos.x,
                        "y": pos.y,
                        "z": pos.z
                    } for pos in template_positions
                ],
                "system": {
                    "sample_rate": 48000,
                    "sound_speed": 343.0,
                    "detection_threshold_db": -20.0,
                    "buffer_duration": 2.0,
                    "min_confidence": 0.7
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(config_template, f, indent=2)
            
            self.logger.info(f"Configuration template generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating template: {e}")
            return False
    
    def _generate_array_positions(self, array_type: str) -> List[MicrophonePosition]:
        """
        Generate microphone positions for different array types.
        
        Args:
            array_type: Type of array layout
            
        Returns:
            List of microphone positions
        """
        import math
        
        if array_type == "circular":
            # Circular array with 8 microphones
            radius = 8.0
            positions = []
            for i in range(8):
                angle = 2 * math.pi * i / 8
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions.append(MicrophonePosition(i + 1, x, y, 0.0))
            return positions
            
        elif array_type == "linear":
            # Linear array (not recommended but included for completeness)
            positions = []
            for i in range(8):
                x = float(i * 2)  # 2m spacing
                positions.append(MicrophonePosition(i + 1, x, 0.0, 0.0))
            return positions
            
        elif array_type == "square":
            # Square perimeter with internal points
            return [
                MicrophonePosition(1, 0.0, 0.0, 0.0),    # Corner
                MicrophonePosition(2, 12.0, 0.0, 0.0),   # Corner
                MicrophonePosition(3, 12.0, 12.0, 0.0),  # Corner
                MicrophonePosition(4, 0.0, 12.0, 0.0),   # Corner
                MicrophonePosition(5, 6.0, 0.0, 0.0),    # Edge center
                MicrophonePosition(6, 12.0, 6.0, 0.0),   # Edge center
                MicrophonePosition(7, 6.0, 12.0, 0.0),   # Edge center
                MicrophonePosition(8, 0.0, 6.0, 0.0)     # Edge center
            ]
        
        else:  # default
            return self._default_positions.copy()