"""
Unit tests for configuration manager.
"""
import unittest
import tempfile
import json
import os
from unittest.mock import patch, mock_open
from config_manager import (
    ConfigurationManager, MicrophonePosition, SystemConfig,
    ValidationError
)


class TestMicrophonePosition(unittest.TestCase):
    """Test cases for MicrophonePosition class."""
    
    def test_microphone_position_creation(self):
        """Test microphone position creation."""
        mic = MicrophonePosition(id=1, x=1.0, y=2.0, z=3.0)
        
        self.assertEqual(mic.id, 1)
        self.assertEqual(mic.x, 1.0)
        self.assertEqual(mic.y, 2.0)
        self.assertEqual(mic.z, 3.0)
    
    def test_microphone_position_distance(self):
        """Test distance calculation between microphones."""
        mic1 = MicrophonePosition(id=1, x=0.0, y=0.0, z=0.0)
        mic2 = MicrophonePosition(id=2, x=3.0, y=4.0, z=0.0)
        
        distance = mic1.distance_to(mic2)
        self.assertAlmostEqual(distance, 5.0, places=5)  # 3-4-5 triangle
    
    def test_microphone_position_equality(self):
        """Test microphone position equality."""
        mic1 = MicrophonePosition(id=1, x=1.0, y=2.0, z=3.0)
        mic2 = MicrophonePosition(id=1, x=1.0, y=2.0, z=3.0)
        mic3 = MicrophonePosition(id=2, x=1.0, y=2.0, z=3.0)
        
        self.assertEqual(mic1, mic2)
        self.assertNotEqual(mic1, mic3)


class TestSystemConfig(unittest.TestCase):
    """Test cases for SystemConfig class."""
    
    def test_system_config_creation(self):
        """Test system configuration creation."""
        config = SystemConfig(
            sample_rate=48000,
            sound_speed=343.0,
            detection_threshold_db=-20.0,
            buffer_duration=1.0,
            min_confidence=0.7
        )
        
        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.sound_speed, 343.0)
        self.assertEqual(config.detection_threshold_db, -20.0)
        self.assertEqual(config.buffer_duration, 1.0)
        self.assertEqual(config.min_confidence, 0.7)
    
    def test_system_config_defaults(self):
        """Test system configuration with default values."""
        config = SystemConfig()
        
        self.assertEqual(config.sample_rate, 48000)
        self.assertEqual(config.sound_speed, 343.0)
        self.assertEqual(config.detection_threshold_db, -20.0)
        self.assertEqual(config.buffer_duration, 2.0)
        self.assertEqual(config.min_confidence, 0.7)


class TestConfigurationManager(unittest.TestCase):
    """Test cases for ConfigurationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigurationManager()
        
        # Create test configuration data
        self.test_config_data = {
            "microphones": [
                {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0},
                {"id": 3, "x": 0.0, "y": 1.0, "z": 0.0},
                {"id": 4, "x": 1.0, "y": 1.0, "z": 0.0},
                {"id": 5, "x": 0.5, "y": 0.5, "z": 1.0},
                {"id": 6, "x": 1.5, "y": 0.5, "z": 0.0},
                {"id": 7, "x": 0.5, "y": 1.5, "z": 0.0},
                {"id": 8, "x": 1.5, "y": 1.5, "z": 0.0}
            ],
            "system": {
                "sample_rate": 44100,
                "sound_speed": 340.0,
                "detection_threshold_db": -25.0,
                "buffer_duration": 1.5,
                "min_confidence": 0.8
            }
        }
    
    def test_initialization(self):
        """Test configuration manager initialization."""
        self.assertIsNotNone(self.config_manager)
        self.assertFalse(self.config_manager.is_config_loaded())
        self.assertEqual(len(self.config_manager.get_microphone_positions()), 8)  # Default positions
    
    def test_load_config_success(self):
        """Test successful configuration loading."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config_data, f)
            temp_path = f.name
        
        try:
            # Load configuration
            success = self.config_manager.load_config(temp_path)
            
            self.assertTrue(success)
            self.assertTrue(self.config_manager.is_config_loaded())
            
            # Check microphone positions
            positions = self.config_manager.get_microphone_positions()
            self.assertEqual(len(positions), 8)
            self.assertEqual(positions[0].id, 1)
            self.assertEqual(positions[0].x, 0.0)
            
            # Check system configuration
            system_config = self.config_manager.get_system_config()
            self.assertEqual(system_config.sample_rate, 44100)
            self.assertEqual(system_config.sound_speed, 340.0)
            self.assertEqual(system_config.detection_threshold_db, -25.0)
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with non-existent file."""
        success = self.config_manager.load_config("nonexistent_file.json")
        
        self.assertFalse(success)
        self.assertFalse(self.config_manager.is_config_loaded())
    
    def test_load_config_invalid_json(self):
        """Test configuration loading with invalid JSON."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_path = f.name
        
        try:
            success = self.config_manager.load_config(temp_path)
            
            self.assertFalse(success)
            self.assertFalse(self.config_manager.is_config_loaded())
            
        finally:
            os.unlink(temp_path)
    
    def test_load_config_missing_microphones(self):
        """Test configuration loading with missing microphones section."""
        config_data = {
            "system": {
                "sample_rate": 48000,
                "sound_speed": 343.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            success = self.config_manager.load_config(temp_path)
            
            self.assertTrue(success)  # Should succeed with defaults
            
            # Should use default microphone positions
            positions = self.config_manager.get_microphone_positions()
            self.assertEqual(len(positions), 8)
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        # Load valid configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            is_valid, errors = self.config_manager.validate_config()
            
            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_wrong_microphone_count(self):
        """Test configuration validation with wrong microphone count."""
        # Create config with only 4 microphones
        config_data = {
            "microphones": [
                {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0},
                {"id": 3, "x": 0.0, "y": 1.0, "z": 0.0},
                {"id": 4, "x": 1.0, "y": 1.0, "z": 0.0}
            ],
            "system": {"sample_rate": 48000}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            is_valid, errors = self.config_manager.validate_config()
            
            self.assertFalse(is_valid)
            self.assertGreater(len(errors), 0)
            self.assertTrue(any("Expected 8 microphones" in error for error in errors))
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_duplicate_ids(self):
        """Test configuration validation with duplicate microphone IDs."""
        # Create config with duplicate IDs
        config_data = self.test_config_data.copy()
        config_data["microphones"][1]["id"] = 1  # Duplicate ID
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            is_valid, errors = self.config_manager.validate_config()
            
            self.assertFalse(is_valid)
            self.assertTrue(any("Duplicate microphone IDs" in error for error in errors))
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_duplicate_positions(self):
        """Test configuration validation with duplicate positions."""
        # Create config with duplicate positions
        config_data = self.test_config_data.copy()
        config_data["microphones"][1]["x"] = 0.0
        config_data["microphones"][1]["y"] = 0.0
        config_data["microphones"][1]["z"] = 0.0
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            is_valid, errors = self.config_manager.validate_config()
            
            self.assertFalse(is_valid)
            self.assertTrue(any("Duplicate microphone positions" in error for error in errors))
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_invalid_system_params(self):
        """Test configuration validation with invalid system parameters."""
        # Create config with invalid system parameters
        config_data = self.test_config_data.copy()
        config_data["system"]["sample_rate"] = -1000  # Invalid
        config_data["system"]["sound_speed"] = 0  # Invalid
        config_data["system"]["min_confidence"] = 1.5  # Invalid (> 1.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            is_valid, errors = self.config_manager.validate_config()
            
            self.assertFalse(is_valid)
            self.assertTrue(any("Sample rate must be positive" in error for error in errors))
            self.assertTrue(any("Sound speed must be positive" in error for error in errors))
            self.assertTrue(any("Minimum confidence must be between 0.0 and 1.0" in error for error in errors))
            
        finally:
            os.unlink(temp_path)
    
    def test_get_microphone_by_id(self):
        """Test getting microphone by ID."""
        # Load configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            
            # Test existing microphone
            mic = self.config_manager.get_microphone_by_id(1)
            self.assertIsNotNone(mic)
            self.assertEqual(mic.id, 1)
            self.assertEqual(mic.x, 0.0)
            
            # Test non-existent microphone
            mic = self.config_manager.get_microphone_by_id(99)
            self.assertIsNone(mic)
            
        finally:
            os.unlink(temp_path)
    
    def test_get_microphone_pairs(self):
        """Test getting microphone pairs."""
        pairs = self.config_manager.get_microphone_pairs()
        
        # Should have 28 pairs for 8 microphones (8*7/2)
        self.assertEqual(len(pairs), 28)
        
        # Check that all pairs are unique
        pair_set = set()
        for mic1, mic2 in pairs:
            pair_key = tuple(sorted([mic1.id, mic2.id]))
            self.assertNotIn(pair_key, pair_set)
            pair_set.add(pair_key)
    
    def test_export_config(self):
        """Test configuration export."""
        # Load configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            
            # Export configuration
            exported_config = self.config_manager.export_config()
            
            self.assertIn('microphones', exported_config)
            self.assertIn('system', exported_config)
            self.assertEqual(len(exported_config['microphones']), 8)
            
            # Check that exported data matches loaded data
            self.assertEqual(exported_config['system']['sample_rate'], 44100)
            
        finally:
            os.unlink(temp_path)
    
    def test_update_microphone_position(self):
        """Test updating microphone position."""
        # Load configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            
            # Update microphone position
            success = self.config_manager.update_microphone_position(1, 5.0, 6.0, 7.0)
            self.assertTrue(success)
            
            # Verify update
            mic = self.config_manager.get_microphone_by_id(1)
            self.assertEqual(mic.x, 5.0)
            self.assertEqual(mic.y, 6.0)
            self.assertEqual(mic.z, 7.0)
            
            # Test updating non-existent microphone
            success = self.config_manager.update_microphone_position(99, 1.0, 2.0, 3.0)
            self.assertFalse(success)
            
        finally:
            os.unlink(temp_path)
    
    def test_update_system_config(self):
        """Test updating system configuration."""
        # Load configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_config_data, f)
            temp_path = f.name
        
        try:
            self.config_manager.load_config(temp_path)
            
            # Update system configuration
            updates = {
                'sample_rate': 96000,
                'sound_speed': 350.0,
                'min_confidence': 0.9
            }
            
            self.config_manager.update_system_config(**updates)
            
            # Verify updates
            system_config = self.config_manager.get_system_config()
            self.assertEqual(system_config.sample_rate, 96000)
            self.assertEqual(system_config.sound_speed, 350.0)
            self.assertEqual(system_config.min_confidence, 0.9)
            
        finally:
            os.unlink(temp_path)


def create_test_suite():
    """Create test suite for configuration manager."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestMicrophonePosition))
    suite.addTest(unittest.makeSuite(TestSystemConfig))
    suite.addTest(unittest.makeSuite(TestConfigurationManager))
    
    return suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")