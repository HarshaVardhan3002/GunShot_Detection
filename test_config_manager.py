"""
Unit tests for ConfigurationManager.
"""
import json
import tempfile
import unittest
from pathlib import Path

from config_manager import ConfigurationManager, MicrophonePosition, SystemConfig


class TestConfigurationManager(unittest.TestCase):
    """Test cases for ConfigurationManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager()
    
    def test_default_initialization(self):
        """Test default initialization without loading config file."""
        positions = self.config_manager.get_microphone_positions()
        system_config = self.config_manager.get_system_config()
        
        # Check default microphone count
        self.assertEqual(len(positions), 8)
        
        # Check default system config
        self.assertEqual(system_config.sample_rate, 48000)
        self.assertEqual(system_config.sound_speed, 343.0)
        self.assertEqual(system_config.detection_threshold_db, -20.0)
        
        # Check that config is not marked as loaded
        self.assertFalse(self.config_manager.is_config_loaded())
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_data = {
            "microphones": [
                {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                {"id": 2, "x": 5.0, "y": 0.0, "z": 0.0},
                {"id": 3, "x": 10.0, "y": 0.0, "z": 0.0},
                {"id": 4, "x": 0.0, "y": 5.0, "z": 0.0},
                {"id": 5, "x": 5.0, "y": 5.0, "z": 0.0},
                {"id": 6, "x": 10.0, "y": 5.0, "z": 0.0},
                {"id": 7, "x": 0.0, "y": 10.0, "z": 0.0},
                {"id": 8, "x": 5.0, "y": 10.0, "z": 0.0}
            ],
            "system": {
                "sample_rate": 44100,
                "sound_speed": 340.0,
                "detection_threshold_db": -25.0,
                "buffer_duration": 3.0,
                "min_confidence": 0.8
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load configuration
            result = self.config_manager.load_config(config_path)
            self.assertTrue(result)
            self.assertTrue(self.config_manager.is_config_loaded())
            
            # Check loaded microphone positions
            positions = self.config_manager.get_microphone_positions()
            self.assertEqual(len(positions), 8)
            self.assertEqual(positions[0].id, 1)
            self.assertEqual(positions[0].x, 0.0)
            self.assertEqual(positions[1].x, 5.0)
            
            # Check loaded system config
            system_config = self.config_manager.get_system_config()
            self.assertEqual(system_config.sample_rate, 44100)
            self.assertEqual(system_config.sound_speed, 340.0)
            self.assertEqual(system_config.detection_threshold_db, -25.0)
            
        finally:
            Path(config_path).unlink()
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        result = self.config_manager.load_config("nonexistent.json")
        self.assertFalse(result)
        self.assertFalse(self.config_manager.is_config_loaded())
        
        # Should fall back to defaults
        positions = self.config_manager.get_microphone_positions()
        self.assertEqual(len(positions), 8)
    
    def test_load_invalid_json(self):
        """Test loading an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name
        
        try:
            result = self.config_manager.load_config(config_path)
            self.assertFalse(result)
            self.assertFalse(self.config_manager.is_config_loaded())
            
        finally:
            Path(config_path).unlink()
    
    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        is_valid, errors = self.config_manager.validate_config()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_invalid_microphone_count(self):
        """Test validation with wrong number of microphones."""
        # Manually set invalid microphone count
        self.config_manager._microphone_positions = [
            MicrophonePosition(1, 0.0, 0.0, 0.0),
            MicrophonePosition(2, 1.0, 0.0, 0.0)
        ]
        
        is_valid, errors = self.config_manager.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("Expected 8 microphones, found 2", errors)
    
    def test_validate_duplicate_microphone_ids(self):
        """Test validation with duplicate microphone IDs."""
        # Create positions with duplicate IDs
        positions = []
        for i in range(8):
            positions.append(MicrophonePosition(1, float(i), 0.0, 0.0))  # All ID = 1
        
        self.config_manager._microphone_positions = positions
        
        is_valid, errors = self.config_manager.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("Duplicate microphone IDs found", errors)
    
    def test_validate_invalid_system_params(self):
        """Test validation with invalid system parameters."""
        # Set invalid system config
        self.config_manager._system_config = SystemConfig(
            sample_rate=-1000,  # Invalid
            sound_speed=-100,   # Invalid
            buffer_duration=-1, # Invalid
            min_confidence=1.5  # Invalid
        )
        
        is_valid, errors = self.config_manager.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("Sample rate must be positive", errors)
        self.assertIn("Sound speed must be positive", errors)
        self.assertIn("Buffer duration must be positive", errors)
        self.assertIn("Minimum confidence must be between 0.0 and 1.0", errors)
    
    def test_validate_collinear_positions(self):
        """Test validation with collinear microphone positions."""
        # Create collinear positions (all on a line)
        positions = []
        for i in range(8):
            positions.append(MicrophonePosition(i+1, float(i), 0.0, 0.0))
        
        self.config_manager._microphone_positions = positions
        
        is_valid, errors = self.config_manager.validate_config()
        self.assertFalse(is_valid)
        self.assertTrue(any("collinear" in error for error in errors))
    
    def test_validate_minimum_distances(self):
        """Test validation with microphones too close together."""
        # Create positions with some microphones too close
        positions = [
            MicrophonePosition(1, 0.0, 0.0, 0.0),
            MicrophonePosition(2, 0.1, 0.0, 0.0),  # Too close to mic 1
            MicrophonePosition(3, 5.0, 0.0, 0.0),
            MicrophonePosition(4, 0.0, 5.0, 0.0),
            MicrophonePosition(5, 5.0, 5.0, 0.0),
            MicrophonePosition(6, 2.5, 2.5, 0.0),
            MicrophonePosition(7, 7.5, 2.5, 0.0),
            MicrophonePosition(8, 7.5, 7.5, 0.0)
        ]
        
        self.config_manager._microphone_positions = positions
        
        is_valid, errors = self.config_manager.validate_config()
        self.assertFalse(is_valid)
        self.assertTrue(any("too close" in error for error in errors))
    
    def test_validate_duplicate_positions(self):
        """Test validation with duplicate microphone positions."""
        # Create positions with duplicates
        positions = [
            MicrophonePosition(1, 0.0, 0.0, 0.0),
            MicrophonePosition(2, 0.0, 0.0, 0.0),  # Duplicate position
            MicrophonePosition(3, 5.0, 0.0, 0.0),
            MicrophonePosition(4, 0.0, 5.0, 0.0),
            MicrophonePosition(5, 5.0, 5.0, 0.0),
            MicrophonePosition(6, 2.5, 2.5, 0.0),
            MicrophonePosition(7, 7.5, 2.5, 0.0),
            MicrophonePosition(8, 7.5, 7.5, 0.0)
        ]
        
        self.config_manager._microphone_positions = positions
        
        is_valid, errors = self.config_manager.validate_config()
        self.assertFalse(is_valid)
        self.assertTrue(any("Duplicate microphone positions" in error for error in errors))
    
    def test_generate_config_template(self):
        """Test configuration template generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            template_path = f.name
        
        try:
            # Generate default template
            result = self.config_manager.generate_config_template(template_path, "default")
            self.assertTrue(result)
            
            # Load and validate generated template
            with open(template_path, 'r') as f:
                template_data = json.load(f)
            
            self.assertIn("microphones", template_data)
            self.assertIn("system", template_data)
            self.assertEqual(len(template_data["microphones"]), 8)
            
        finally:
            Path(template_path).unlink()
    
    def test_generate_circular_array_template(self):
        """Test circular array template generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            template_path = f.name
        
        try:
            # Generate circular template
            result = self.config_manager.generate_config_template(template_path, "circular")
            self.assertTrue(result)
            
            # Load generated template and create new config manager to test it
            test_config = ConfigurationManager()
            load_result = test_config.load_config(template_path)
            self.assertTrue(load_result)
            
            # Validate the circular array
            is_valid, errors = test_config.validate_config()
            self.assertTrue(is_valid, f"Circular array validation failed: {errors}")
            
        finally:
            Path(template_path).unlink()
    
    def test_partial_config_loading(self):
        """Test loading configuration with missing sections."""
        # Config with only microphones, no system section
        config_data = {
            "microphones": [
                {"id": i+1, "x": float(i), "y": 0.0} for i in range(8)
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            result = self.config_manager.load_config(config_path)
            self.assertTrue(result)
            
            # Should use default system config
            system_config = self.config_manager.get_system_config()
            self.assertEqual(system_config.sample_rate, 48000)  # Default
            
        finally:
            Path(config_path).unlink()


if __name__ == '__main__':
    unittest.main()