"""
Unit tests for environment manager functionality
"""

import unittest
import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.environment_manager import (
    EnvironmentManager,
    ModelEnvironmentContext,
    with_environment,
    run_model_in_environment,
    setup_model_environment
)


class TestEnvironmentManager(unittest.TestCase):
    """Test suite for EnvironmentManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = EnvironmentManager()
        self.test_env_name = "test_env"
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up after tests"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @patch('subprocess.run')
    def test_find_conda_path(self, mock_run):
        """Test finding conda installation path"""
        # Mock successful conda path finding
        mock_run.return_value = MagicMock(
            stdout="/usr/local/miniconda3/bin/conda\n",
            returncode=0
        )
        
        manager = EnvironmentManager()
        self.assertIsNotNone(manager.base_conda_path)
    
    @patch('subprocess.run')
    def test_check_environment_exists(self, mock_run):
        """Test checking if environment exists"""
        # Mock conda env list output
        env_list = {
            "envs": [
                "/usr/local/miniconda3/envs/test_env",
                "/usr/local/miniconda3/envs/another_env"
            ]
        }
        
        mock_run.return_value = MagicMock(
            stdout=json.dumps(env_list),
            returncode=0
        )
        
        exists = self.manager.check_environment_exists("test_env")
        self.assertTrue(exists)
        
        exists = self.manager.check_environment_exists("nonexistent_env")
        self.assertFalse(exists)
    
    @patch('subprocess.run')
    def test_activate_environment(self, mock_run):
        """Test environment activation"""
        # Mock environment variables from activated environment
        mock_env = {
            "PATH": "/conda/envs/test_env/bin:/usr/bin",
            "CONDA_DEFAULT_ENV": "test_env",
            "CONDA_PREFIX": "/conda/envs/test_env"
        }
        
        mock_run.return_value = MagicMock(
            stdout=json.dumps(mock_env),
            returncode=0
        )
        
        env_vars = self.manager.activate_environment("test_env")
        
        self.assertEqual(self.manager.current_env, "test_env")
        self.assertIn("CONDA_DEFAULT_ENV", env_vars)
        self.assertEqual(env_vars["CONDA_DEFAULT_ENV"], "test_env")
    
    def test_deactivate_environment(self):
        """Test environment deactivation"""
        # Set a current environment
        self.manager.current_env = "test_env"
        
        # Deactivate
        self.manager.deactivate_environment()
        
        self.assertIsNone(self.manager.current_env)
    
    @patch('subprocess.run')
    def test_run_in_environment(self, mock_run):
        """Test running command in environment"""
        mock_run.return_value = MagicMock(
            stdout="Hello from test_env",
            returncode=0
        )
        
        result = self.manager.run_in_environment(
            "test_env",
            "echo 'Hello from test_env'"
        )
        
        self.assertEqual(result.stdout, "Hello from test_env")
        self.assertEqual(result.returncode, 0)
    
    def test_get_python_executable(self):
        """Test getting Python executable path"""
        with patch('os.path.exists', return_value=True):
            python_path = self.manager.get_python_executable("test_env")
            self.assertTrue(python_path.endswith("python"))
            self.assertIn("test_env", python_path)
    
    @patch('subprocess.run')
    def test_install_requirements(self, mock_run):
        """Test installing requirements in environment"""
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
            f.write("numpy==1.21.0\npandas==1.3.0")
            f.flush()
            
            self.manager.install_requirements("test_env", f.name)
            
            # Check that pip install was called
            mock_run.assert_called()


class TestModelEnvironmentContext(unittest.TestCase):
    """Test suite for ModelEnvironmentContext"""
    
    def test_context_manager_with_env(self):
        """Test context manager with environment specified"""
        config = {'conda_env': 'test_env'}
        
        with patch.object(EnvironmentManager, 'activate_environment') as mock_activate:
            with patch.object(EnvironmentManager, 'deactivate_environment') as mock_deactivate:
                with ModelEnvironmentContext(config) as ctx:
                    self.assertTrue(ctx.activated)
                
                mock_activate.assert_called_once_with('test_env')
                mock_deactivate.assert_called_once()
    
    def test_context_manager_no_env(self):
        """Test context manager without environment specified"""
        config = {}
        
        with patch.object(EnvironmentManager, 'activate_environment') as mock_activate:
            with patch.object(EnvironmentManager, 'deactivate_environment') as mock_deactivate:
                with ModelEnvironmentContext(config) as ctx:
                    self.assertFalse(ctx.activated)
                
                mock_activate.assert_not_called()
                mock_deactivate.assert_not_called()


class TestEnvironmentDecorator(unittest.TestCase):
    """Test suite for environment decorator"""
    
    @patch.object(EnvironmentManager, 'activate_environment')
    @patch.object(EnvironmentManager, 'deactivate_environment')
    def test_with_environment_decorator(self, mock_deactivate, mock_activate):
        """Test the with_environment decorator"""
        
        @with_environment('test_env')
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        
        self.assertEqual(result, 5)
        mock_activate.assert_called_once_with('test_env')
        mock_deactivate.assert_called_once()


class TestRunModelInEnvironment(unittest.TestCase):
    """Test suite for run_model_in_environment function"""
    
    def test_run_with_environment(self):
        """Test running model function with environment"""
        config = {'conda_env': 'test_env'}
        
        def model_function(x):
            return x * 2
        
        with patch.object(EnvironmentManager, 'activate_environment'):
            with patch.object(EnvironmentManager, 'deactivate_environment'):
                result = run_model_in_environment(config, model_function, 5)
                self.assertEqual(result, 10)
    
    def test_run_without_environment(self):
        """Test running model function without environment"""
        config = {}
        
        def model_function(x):
            return x * 2
        
        result = run_model_in_environment(config, model_function, 5)
        self.assertEqual(result, 10)


class TestSetupModelEnvironment(unittest.TestCase):
    """Test suite for setup_model_environment function"""
    
    @patch.object(EnvironmentManager, 'check_environment_exists')
    @patch.object(EnvironmentManager, 'run_in_environment')
    @patch.object(EnvironmentManager, 'install_requirements')
    def test_setup_new_environment(self, mock_install, mock_run, mock_exists):
        """Test setting up a new environment"""
        mock_exists.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        
        requirements = {
            'python': '3.9',
            'numpy': '1.21.0',
            'pandas': '1.3.0'
        }
        
        setup_model_environment('test_env', requirements)
        
        # Check that conda create was called
        mock_run.assert_called()
        # Check that requirements were installed
        mock_install.assert_called()
    
    @patch.object(EnvironmentManager, 'check_environment_exists')
    def test_setup_existing_environment(self, mock_exists):
        """Test setup when environment already exists"""
        mock_exists.return_value = True
        
        requirements = {'python': '3.9'}
        
        # Should not raise error, just log that env exists
        setup_model_environment('test_env', requirements)
        
        mock_exists.assert_called_once_with('test_env')


class TestIntegrationWithModels(unittest.TestCase):
    """Integration tests with model classes"""
    
    def test_model_with_env_methods(self):
        """Test that models have environment-aware methods"""
        from src.models.base_model import BaseDNAModel
        
        # Create a mock model class
        class MockModel(BaseDNAModel):
            def load_pretrained(self, path=None):
                return "loaded"
            
            def get_embeddings(self, sequences):
                return "embeddings"
            
            def predict(self, sequences):
                return {"predictions": [0, 1]}
            
            def fine_tune(self, train_dataset, val_dataset=None, **kwargs):
                return {"loss": 0.5}
            
            def save_model(self, path):
                pass
            
            def load_model(self, path):
                pass
        
        config = {
            'conda_env': 'test_env',
            'use_env_manager': True
        }
        
        model = MockModel(config)
        
        # Check that environment-aware methods exist
        self.assertTrue(hasattr(model, 'load_pretrained_with_env'))
        self.assertTrue(hasattr(model, 'get_embeddings_with_env'))
        self.assertTrue(hasattr(model, 'predict_with_env'))
        self.assertTrue(hasattr(model, 'fine_tune_with_env'))
        
        # Test with environment management disabled
        config['use_env_manager'] = False
        model = MockModel(config)
        
        with patch.object(EnvironmentManager, 'activate_environment') as mock_activate:
            result = model.predict_with_env(["ATCG"])
            # Should not activate environment when disabled
            mock_activate.assert_not_called()
            self.assertEqual(result["predictions"], [0, 1])


if __name__ == '__main__':
    unittest.main()