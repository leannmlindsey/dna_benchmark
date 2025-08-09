"""
Environment Manager for handling conda environment switching
"""

import subprocess
import os
import sys
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
import json
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages conda environment activation and deactivation for models"""
    
    def __init__(self, base_conda_path: Optional[str] = None):
        """
        Initialize the environment manager
        
        Args:
            base_conda_path: Path to conda installation (if not in PATH)
        """
        self.base_conda_path = base_conda_path or self._find_conda_path()
        self.current_env = None
        self.original_env = os.environ.copy()
        
    def _find_conda_path(self) -> str:
        """Find conda installation path"""
        try:
            result = subprocess.run(
                ["which", "conda"],
                capture_output=True,
                text=True,
                check=True
            )
            conda_path = Path(result.stdout.strip()).parent.parent
            return str(conda_path)
        except subprocess.CalledProcessError:
            # Common conda installation paths
            common_paths = [
                "/usr/local/anaconda3",
                "/opt/anaconda3",
                "/home/anaconda3",
                os.path.expanduser("~/anaconda3"),
                os.path.expanduser("~/miniconda3"),
                "/usr/local/miniconda3",
                "/opt/miniconda3",
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    return path
            
            raise RuntimeError("Could not find conda installation")
    
    def activate_environment(self, env_name: str) -> Dict[str, str]:
        """
        Activate a conda environment
        
        Args:
            env_name: Name of the conda environment to activate
            
        Returns:
            Dictionary of environment variables for the activated environment
        """
        if self.current_env == env_name:
            logger.info(f"Environment {env_name} is already active")
            return os.environ.copy()
        
        logger.info(f"Activating conda environment: {env_name}")
        
        # Get environment variables for the conda environment
        conda_sh = os.path.join(self.base_conda_path, "etc", "profile.d", "conda.sh")
        
        # Create a script to source conda and get environment variables
        script = f"""
        source {conda_sh}
        conda activate {env_name}
        python -c "import os, json; print(json.dumps(dict(os.environ)))"
        """
        
        try:
            result = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                check=True
            )
            
            env_vars = json.loads(result.stdout)
            
            # Update current process environment
            os.environ.clear()
            os.environ.update(env_vars)
            
            self.current_env = env_name
            
            logger.info(f"Successfully activated environment: {env_name}")
            return env_vars
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to activate environment {env_name}: {e.stderr}")
            raise RuntimeError(f"Could not activate conda environment: {env_name}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse environment variables: {e}")
            raise RuntimeError(f"Could not parse environment variables for: {env_name}")
    
    def deactivate_environment(self):
        """Deactivate current conda environment and restore original"""
        if self.current_env is None:
            logger.info("No environment to deactivate")
            return
        
        logger.info(f"Deactivating environment: {self.current_env}")
        
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        
        self.current_env = None
        logger.info("Restored original environment")
    
    def run_in_environment(self, env_name: str, command: str, **kwargs) -> subprocess.CompletedProcess:
        """
        Run a command in a specific conda environment
        
        Args:
            env_name: Name of the conda environment
            command: Command to run
            **kwargs: Additional arguments for subprocess.run
            
        Returns:
            CompletedProcess instance with the result
        """
        conda_sh = os.path.join(self.base_conda_path, "etc", "profile.d", "conda.sh")
        
        # Create full command with conda activation
        full_command = f"""
        source {conda_sh}
        conda activate {env_name}
        {command}
        """
        
        logger.info(f"Running command in {env_name}: {command}")
        
        return subprocess.run(
            ["bash", "-c", full_command],
            capture_output=True,
            text=True,
            **kwargs
        )
    
    def get_python_executable(self, env_name: str) -> str:
        """
        Get the path to Python executable for a specific environment
        
        Args:
            env_name: Name of the conda environment
            
        Returns:
            Path to Python executable
        """
        env_path = os.path.join(self.base_conda_path, "envs", env_name)
        python_path = os.path.join(env_path, "bin", "python")
        
        if not os.path.exists(python_path):
            # Try alternative location
            python_path = os.path.join(env_path, "python")
        
        if not os.path.exists(python_path):
            raise RuntimeError(f"Python executable not found for environment: {env_name}")
        
        return python_path
    
    def check_environment_exists(self, env_name: str) -> bool:
        """
        Check if a conda environment exists
        
        Args:
            env_name: Name of the conda environment
            
        Returns:
            True if environment exists, False otherwise
        """
        result = self.run_in_environment(
            "base",
            "conda env list --json",
            check=False
        )
        
        if result.returncode != 0:
            return False
        
        try:
            env_data = json.loads(result.stdout)
            env_names = [os.path.basename(env) for env in env_data.get("envs", [])]
            return env_name in env_names
        except json.JSONDecodeError:
            return False
    
    def install_requirements(self, env_name: str, requirements_file: str):
        """
        Install requirements in a specific environment
        
        Args:
            env_name: Name of the conda environment
            requirements_file: Path to requirements file
        """
        logger.info(f"Installing requirements in {env_name} from {requirements_file}")
        
        result = self.run_in_environment(
            env_name,
            f"pip install -r {requirements_file}",
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to install requirements: {result.stderr}")
            raise RuntimeError(f"Could not install requirements in {env_name}")
        
        logger.info(f"Successfully installed requirements in {env_name}")


def with_environment(env_name: str):
    """
    Decorator to run a function in a specific conda environment
    
    Args:
        env_name: Name of the conda environment
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get environment manager
            manager = EnvironmentManager()
            
            # Check if we need to switch environments
            if manager.current_env != env_name:
                # Activate environment
                manager.activate_environment(env_name)
                
                try:
                    # Run function
                    result = func(*args, **kwargs)
                finally:
                    # Deactivate environment
                    manager.deactivate_environment()
                
                return result
            else:
                # Already in correct environment
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ModelEnvironmentContext:
    """Context manager for model-specific conda environments"""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize context manager
        
        Args:
            model_config: Model configuration dictionary with 'conda_env' key
        """
        self.env_name = model_config.get('conda_env')
        self.manager = EnvironmentManager()
        self.activated = False
    
    def __enter__(self):
        """Activate the environment"""
        if self.env_name:
            self.manager.activate_environment(self.env_name)
            self.activated = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deactivate the environment"""
        if self.activated:
            self.manager.deactivate_environment()
        return False


def run_model_in_environment(model_config: Dict[str, Any], 
                            model_function: Callable,
                            *args, **kwargs) -> Any:
    """
    Run a model function in its designated conda environment
    
    Args:
        model_config: Model configuration with 'conda_env' key
        model_function: Function to run
        *args, **kwargs: Arguments for the model function
        
    Returns:
        Result from the model function
    """
    env_name = model_config.get('conda_env')
    
    if not env_name:
        # No specific environment, run directly
        return model_function(*args, **kwargs)
    
    # Use context manager to handle environment
    with ModelEnvironmentContext(model_config):
        return model_function(*args, **kwargs)


def setup_model_environment(env_name: str, requirements: Dict[str, str]):
    """
    Set up a conda environment for a specific model
    
    Args:
        env_name: Name for the conda environment
        requirements: Dictionary of package requirements
    """
    manager = EnvironmentManager()
    
    # Check if environment exists
    if manager.check_environment_exists(env_name):
        logger.info(f"Environment {env_name} already exists")
        return
    
    logger.info(f"Creating conda environment: {env_name}")
    
    # Create environment with Python
    python_version = requirements.get('python', '3.9')
    create_cmd = f"conda create -n {env_name} python={python_version} -y"
    
    result = manager.run_in_environment("base", create_cmd, check=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create environment {env_name}: {result.stderr}")
    
    # Install packages
    packages = []
    for package, version in requirements.items():
        if package != 'python':
            if version:
                packages.append(f"{package}=={version}")
            else:
                packages.append(package)
    
    if packages:
        # Create temporary requirements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('\n'.join(packages))
            temp_requirements = f.name
        
        try:
            manager.install_requirements(env_name, temp_requirements)
        finally:
            os.unlink(temp_requirements)
    
    logger.info(f"Successfully set up environment: {env_name}")