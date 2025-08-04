"""
Tests for diagnostics and status reporting system.
"""
import unittest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from diagnostics import (
    DiagnosticsManager, SystemStatus, ComponentType, ComponentStatus,
    MicrophoneStatus, PerformanceMetrics, SystemDiagnostics,
    get_quick_system_status
)


class TestDiagnosticsManager(unittest.TestCase):
    """Test cases for diagnostics manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_pipeline = Mock()
        self.diagnostics_manager = DiagnosticsManager(
            pipeline=self.mock_pipeline,
            update_interval=0.1  # Fast updates for testing
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.diagnostics_manager.shutdown()
    
    def test_initialization(self):
        """Test diagnostics manager initialization."""
        self.assertIsNotNone(self.diagnostics_manager)
        self.assertEqual(self.diagnostics_manager.pipeline, self.mock_pipeline)
        self.assertFalse(self.diagnostics_manager.monitoring_active)
        self.assertEqual(len(self.diagnostics_manager.component_statuses), 0)
        self.assertEqual(len(self.diagnostics_manager.microphone_statuses), 0)
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop."""
        # Start monitoring
        self.diagnostics_manager.start_monitoring()
        self.assertTrue(self.diagnostics_manager.monitoring_active)
        self.assertIsNotNone(self.diagnostics_manager.monitor_thread)
        
        # Brief pause to let monitoring run
        time.sleep(0.2)
        
        # Stop monitoring
        self.diagnostics_manager.stop_monitoring()
        self.assertFalse(self.diagnostics_manager.monitoring_active)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_performance_metrics_collection(self, mock_disk, mock_memory, mock_cpu):
        """Test performance metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(
            used=2*1024**3,  # 2GB
            percent=50.0
        )
        mock_disk.return_value = Mock(percent=30.0)
        
        # Mock pipeline metrics
        mock_pipeline_metrics = Mock()
        mock_pipeline_metrics.average_latency = 0.025
        mock_pipeline_metrics.detection_rate = 5.0
        mock_pipeline_metrics.localization_accuracy = 0.85
        self.mock_pipeline.get_metrics.return_value = mock_pipeline_metrics
        
        # Collect metrics
        metrics = self.diagnostics_manager._collect_performance_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.cpu_usage_percent, 45.0)
        self.assertEqual(metrics.memory_usage_percent, 50.0)
        self.assertEqual(metrics.disk_usage_percent, 30.0)
        self.assertEqual(metrics.processing_latency_ms, 25.0)  # Converted to ms
        self.assertEqual(metrics.detection_rate_per_second, 5.0)
        self.assertEqual(metrics.localization_accuracy, 0.85)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_system_resource_status_update(self, mock_disk, mock_memory, mock_cpu):
        """Test system resource status updates."""
        # Mock normal resource usage
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(
            percent=50.0,
            available=4*1024**3  # 4GB available
        )
        mock_disk.return_value = Mock(
            percent=30.0,
            free=100*1024**3  # 100GB free
        )
        
        # Update system resource status
        self.diagnostics_manager._update_system_resource_status()
        
        # Check component status
        self.assertIn('system_resources', self.diagnostics_manager.component_statuses)
        status = self.diagnostics_manager.component_statuses['system_resources']
        
        self.assertEqual(status.status, SystemStatus.HEALTHY)
        self.assertEqual(status.component_type, ComponentType.HARDWARE)
        self.assertGreater(status.health_score, 0.9)
        self.assertEqual(len(status.issues), 0)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_system_resource_critical_status(self, mock_disk, mock_memory, mock_cpu):
        """Test system resource critical status detection."""
        # Mock critical resource usage
        mock_cpu.return_value = 98.0  # Critical CPU
        mock_memory.return_value = Mock(
            percent=97.0,  # Critical memory
            available=100*1024**2  # 100MB available
        )
        mock_disk.return_value = Mock(
            percent=98.0,  # Critical disk
            free=1*1024**3  # 1GB free
        )
        
        # Update system resource status
        self.diagnostics_manager._update_system_resource_status()
        
        # Check component status
        status = self.diagnostics_manager.component_statuses['system_resources']
        
        self.assertEqual(status.status, SystemStatus.CRITICAL)
        self.assertLess(status.health_score, 0.5)
        self.assertGreater(len(status.issues), 0)
        self.assertGreater(len(status.recommendations), 0)
    
    def test_microphone_status_update(self):
        """Test microphone status updates."""
        # Mock config manager with microphone positions
        mock_mic_positions = [
            Mock(id=1, x=0.0, y=0.0, z=0.0),
            Mock(id=2, x=1.0, y=0.0, z=0.0),
            Mock(id=3, x=0.0, y=1.0, z=0.0),
            Mock(id=4, x=1.0, y=1.0, z=0.0)
        ]
        
        mock_config_manager = Mock()
        mock_config_manager.get_microphone_positions.return_value = mock_mic_positions
        self.mock_pipeline.config_manager = mock_config_manager
        
        # Update microphone statuses
        self.diagnostics_manager._update_microphone_statuses()
        
        # Check microphone statuses
        self.assertEqual(len(self.diagnostics_manager.microphone_statuses), 4)
        
        for mic_id in [1, 2, 3, 4]:
            self.assertIn(mic_id, self.diagnostics_manager.microphone_statuses)
            mic_status = self.diagnostics_manager.microphone_statuses[mic_id]
            self.assertIsInstance(mic_status, MicrophoneStatus)
            self.assertEqual(mic_status.microphone_id, mic_id)
            self.assertTrue(mic_status.is_connected)
            self.assertGreater(mic_status.signal_quality, 0.0)
    
    def test_system_diagnostics_generation(self):
        """Test system diagnostics generation."""
        # Add some mock component statuses
        self.diagnostics_manager.component_statuses['test_component'] = ComponentStatus(
            name="Test Component",
            component_type=ComponentType.SOFTWARE,
            status=SystemStatus.HEALTHY,
            health_score=0.9,
            last_check=time.time(),
            uptime=3600.0,
            error_count=0,
            warning_count=0,
            metrics={'test_metric': 42},
            issues=[],
            recommendations=[]
        )
        
        # Add some mock microphone statuses
        self.diagnostics_manager.microphone_statuses[1] = MicrophoneStatus(
            microphone_id=1,
            is_connected=True,
            signal_quality=0.85,
            noise_level_db=-25.0,
            last_signal_time=time.time(),
            sample_rate=48000,
            channel_active=True,
            error_count=0,
            calibration_status="calibrated",
            position=(0.0, 0.0, 0.0),
            issues=[]
        )
        
        # Add some performance metrics
        test_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage_percent=45.0,
            memory_usage_mb=2048.0,
            memory_usage_percent=50.0,
            disk_usage_percent=30.0,
            network_io_mb=10.0,
            processing_latency_ms=25.0,
            detection_rate_per_second=5.0,
            localization_accuracy=0.85,
            error_rate_per_hour=2.0,
            uptime_hours=1.0
        )
        self.diagnostics_manager.performance_history.append(test_metrics)
        
        # Generate diagnostics
        diagnostics = self.diagnostics_manager.get_system_diagnostics()
        
        # Verify diagnostics structure
        self.assertIsInstance(diagnostics, SystemDiagnostics)
        self.assertEqual(diagnostics.overall_status, SystemStatus.HEALTHY)
        self.assertGreater(diagnostics.overall_health_score, 0.8)
        self.assertEqual(len(diagnostics.components), 1)
        self.assertEqual(len(diagnostics.microphones), 1)
        self.assertIsInstance(diagnostics.performance, PerformanceMetrics)
        self.assertIsInstance(diagnostics.system_info, dict)
    
    def test_microphone_connectivity_report(self):
        """Test microphone connectivity report generation."""
        # Add mock microphone statuses
        for i in range(1, 5):
            self.diagnostics_manager.microphone_statuses[i] = MicrophoneStatus(
                microphone_id=i,
                is_connected=i <= 3,  # First 3 connected, 4th disconnected
                signal_quality=0.8 + i * 0.05,
                noise_level_db=-30.0 + i,
                last_signal_time=time.time(),
                sample_rate=48000,
                channel_active=i <= 3,
                error_count=0 if i <= 3 else 1,
                calibration_status="calibrated" if i <= 3 else "error",
                position=(i * 1.0, 0.0, 0.0),
                issues=[] if i <= 3 else ["Connection lost"]
            )
        
        # Generate report
        report = self.diagnostics_manager.get_microphone_connectivity_report()
        
        # Verify report
        self.assertEqual(report['total_microphones'], 4)
        self.assertEqual(report['connected_microphones'], 3)
        self.assertEqual(report['disconnected_microphones'], 1)
        self.assertGreater(report['average_signal_quality'], 0.8)
        self.assertEqual(len(report['microphones']), 4)
        
        # Check individual microphone data
        for mic_data in report['microphones']:
            self.assertIn('id', mic_data)
            self.assertIn('connected', mic_data)
            self.assertIn('signal_quality', mic_data)
            self.assertIn('position', mic_data)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add mock performance data
        current_time = time.time()
        for i in range(10):
            metrics = PerformanceMetrics(
                timestamp=current_time - (i * 60),  # 1 minute intervals
                cpu_usage_percent=40.0 + i * 2,
                memory_usage_mb=2000.0 + i * 50,
                memory_usage_percent=45.0 + i * 2,
                disk_usage_percent=30.0,
                network_io_mb=10.0,
                processing_latency_ms=20.0 + i,
                detection_rate_per_second=5.0,
                localization_accuracy=0.8 + i * 0.01,
                error_rate_per_hour=1.0,
                uptime_hours=1.0 + i * 0.1
            )
            self.diagnostics_manager.performance_history.append(metrics)
        
        # Generate summary
        summary = self.diagnostics_manager.get_performance_summary(hours=1.0)
        
        # Verify summary
        self.assertIn('time_period_hours', summary)
        self.assertIn('data_points', summary)
        self.assertIn('cpu_usage', summary)
        self.assertIn('memory_usage', summary)
        self.assertIn('processing_latency_ms', summary)
        self.assertIn('localization_accuracy', summary)
        
        # Check that averages are calculated
        self.assertIn('average', summary['cpu_usage'])
        self.assertIn('min', summary['cpu_usage'])
        self.assertIn('max', summary['cpu_usage'])
    
    def test_diagnostics_export(self):
        """Test diagnostics export functionality."""
        # Add some test data
        self.diagnostics_manager.component_statuses['test'] = ComponentStatus(
            name="Test",
            component_type=ComponentType.SOFTWARE,
            status=SystemStatus.HEALTHY,
            health_score=1.0,
            last_check=time.time(),
            uptime=3600.0,
            error_count=0,
            warning_count=0
        )
        
        # Test JSON export
        json_report = self.diagnostics_manager.export_diagnostics_report('json')
        self.assertIsInstance(json_report, str)
        
        # Verify it's valid JSON
        parsed_json = json.loads(json_report)
        self.assertIn('timestamp', parsed_json)
        self.assertIn('overall_status', parsed_json)
        self.assertIn('components', parsed_json)
        
        # Test text export
        text_report = self.diagnostics_manager.export_diagnostics_report('text')
        self.assertIsInstance(text_report, str)
        self.assertIn('DIAGNOSTICS REPORT', text_report)
        self.assertIn('COMPONENT STATUS', text_report)


class TestQuickSystemStatus(unittest.TestCase):
    """Test cases for quick system status function."""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_quick_status_healthy(self, mock_memory, mock_cpu):
        """Test quick status with healthy system."""
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(percent=50.0)
        
        status = get_quick_system_status()
        
        self.assertTrue(status['system_healthy'])
        self.assertEqual(status['cpu_usage_percent'], 45.0)
        self.assertEqual(status['memory_usage_percent'], 50.0)
        self.assertEqual(len(status['issues']), 0)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_quick_status_critical(self, mock_memory, mock_cpu):
        """Test quick status with critical system issues."""
        mock_cpu.return_value = 98.0  # Critical CPU
        mock_memory.return_value = Mock(percent=97.0)  # Critical memory
        
        status = get_quick_system_status()
        
        self.assertFalse(status['system_healthy'])
        self.assertEqual(status['cpu_usage_percent'], 98.0)
        self.assertEqual(status['memory_usage_percent'], 97.0)
        self.assertGreater(len(status['issues']), 0)
        
        # Check that critical issues are detected
        issues_text = ' '.join(status['issues'])
        self.assertIn('Critical CPU usage', issues_text)
        self.assertIn('Critical memory usage', issues_text)
    
    def test_quick_status_with_pipeline(self):
        """Test quick status with pipeline information."""
        mock_pipeline = Mock()
        mock_pipeline.get_system_status.return_value = {
            'pipeline_state': 'running',
            'is_running': True
        }
        
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory', return_value=Mock(percent=50.0)):
            
            status = get_quick_system_status(pipeline=mock_pipeline)
            
            self.assertEqual(status['pipeline_state'], 'running')
            self.assertTrue(status['pipeline_running'])


def create_test_suite():
    """Create test suite for diagnostics system."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDiagnosticsManager))
    suite.addTest(unittest.makeSuite(TestQuickSystemStatus))
    
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