#!/usr/bin/env python
"""
Script to set up conda environments for all DNA language models
"""

import argparse
import yaml
import sys
import os
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.environment_manager import setup_model_environment, EnvironmentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model-specific requirements
MODEL_REQUIREMENTS = {
    'dnabert1_env': {
        'python': '3.8',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
        'biopython': '1.79',
    },
    
    'dnabert2_env': {
        'python': '3.9',
        'torch': '2.0.0',
        'transformers': '4.35.0',
        'triton': '',  # DNABERT2 specific
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
    },
    
    'nucleotide_transformer_env': {
        'python': '3.9',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'numpy': '1.21.0',
        'einops': '0.6.0',  # NT specific
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
    },
    
    'prokbert_env': {
        'python': '3.9',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
        'tokenizers': '0.13.0',  # ProkBERT specific
    },
    
    'grover_env': {
        'python': '3.8',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
        'sentencepiece': '0.1.99',  # GROVER specific
    },
    
    'gena_lm_env': {
        'python': '3.9',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
    },
    
    'inherit_env': {
        'python': '3.8',
        'torch': '2.0.0',
        'transformers': '4.30.0',
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
        'biopython': '1.79',
    },
    
    'hyenadna_env': {
        'python': '3.9',
        'torch': '2.1.0',  # HyenaDNA needs newer torch
        'transformers': '4.35.0',
        'numpy': '1.21.0',
        'einops': '0.7.0',  # HyenaDNA specific
        'flash-attn': '',  # Optional but recommended
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
    },
    
    'evo_env': {
        'python': '3.10',  # EVO needs Python 3.10+
        'torch': '2.1.0',
        'transformers': '4.35.0',
        'numpy': '1.21.0',
        'einops': '0.7.0',
        'flash-attn': '',  # EVO specific
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
    },
    
    'caduceus_env': {
        'python': '3.10',  # Caduceus needs Python 3.10+
        'torch': '2.1.0',
        'transformers': '4.35.0',
        'numpy': '1.21.0',
        'mamba-ssm': '',  # Caduceus specific
        'causal-conv1d': '',  # Caduceus specific
        'pandas': '1.3.0',
        'scikit-learn': '1.0.0',
    },
}


def load_model_config(config_path: str) -> dict:
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_environment(env_name: str, requirements: dict, force: bool = False):
    """
    Set up a single conda environment
    
    Args:
        env_name: Name of the environment
        requirements: Dictionary of requirements
        force: Force recreation of environment
    """
    manager = EnvironmentManager()
    
    # Check if environment exists
    if manager.check_environment_exists(env_name):
        if not force:
            logger.info(f"Environment {env_name} already exists. Use --force to recreate.")
            return
        else:
            logger.info(f"Removing existing environment {env_name}")
            # Remove existing environment
            result = manager.run_in_environment(
                "base",
                f"conda env remove -n {env_name} -y",
                check=False
            )
            if result.returncode != 0:
                logger.warning(f"Could not remove environment {env_name}: {result.stderr}")
    
    # Set up the environment
    try:
        setup_model_environment(env_name, requirements)
        logger.info(f"Successfully set up environment: {env_name}")
    except Exception as e:
        logger.error(f"Failed to set up environment {env_name}: {e}")
        raise


def setup_all_environments(config_path: str, force: bool = False):
    """
    Set up all model environments
    
    Args:
        config_path: Path to models.yaml configuration
        force: Force recreation of environments
    """
    # Load model configuration
    config = load_model_config(config_path)
    
    # Get unique environments from config
    environments = set()
    for model_name, model_config in config['models'].items():
        env_name = model_config.get('conda_env')
        if env_name:
            environments.add(env_name)
    
    logger.info(f"Found {len(environments)} unique environments to set up")
    
    # Set up each environment
    success_count = 0
    failed_envs = []
    
    for env_name in environments:
        if env_name in MODEL_REQUIREMENTS:
            logger.info(f"\nSetting up environment: {env_name}")
            try:
                setup_environment(env_name, MODEL_REQUIREMENTS[env_name], force)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to set up {env_name}: {e}")
                failed_envs.append(env_name)
        else:
            logger.warning(f"No requirements defined for environment: {env_name}")
            failed_envs.append(env_name)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Setup complete: {success_count}/{len(environments)} environments created")
    
    if failed_envs:
        logger.warning(f"Failed environments: {', '.join(failed_envs)}")
        return 1
    
    return 0


def verify_environments(config_path: str):
    """
    Verify that all required environments exist and have correct packages
    
    Args:
        config_path: Path to models.yaml configuration
    """
    manager = EnvironmentManager()
    config = load_model_config(config_path)
    
    # Get unique environments
    environments = set()
    for model_name, model_config in config['models'].items():
        env_name = model_config.get('conda_env')
        if env_name:
            environments.add(env_name)
    
    logger.info(f"Verifying {len(environments)} environments...")
    
    all_good = True
    for env_name in environments:
        if manager.check_environment_exists(env_name):
            logger.info(f"✓ {env_name} exists")
            
            # Check for key packages
            if env_name in MODEL_REQUIREMENTS:
                result = manager.run_in_environment(
                    env_name,
                    "python -c 'import torch, transformers; print(f\"torch: {torch.__version__}, transformers: {transformers.__version__}\")'",
                    check=False
                )
                
                if result.returncode == 0:
                    logger.info(f"  Packages: {result.stdout.strip()}")
                else:
                    logger.warning(f"  Could not verify packages: {result.stderr}")
                    all_good = False
        else:
            logger.error(f"✗ {env_name} does not exist")
            all_good = False
    
    if all_good:
        logger.info("\nAll environments verified successfully!")
        return 0
    else:
        logger.error("\nSome environments are missing or incomplete")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Set up conda environments for DNA models')
    parser.add_argument(
        '--config',
        type=str,
        default='config/models.yaml',
        help='Path to models configuration file'
    )
    parser.add_argument(
        '--env',
        type=str,
        help='Set up only a specific environment'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreation of existing environments'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify that environments exist and are properly configured'
    )
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    try:
        if args.verify:
            # Verify environments
            return verify_environments(str(config_path))
        elif args.env:
            # Set up specific environment
            if args.env in MODEL_REQUIREMENTS:
                setup_environment(args.env, MODEL_REQUIREMENTS[args.env], args.force)
                return 0
            else:
                logger.error(f"Unknown environment: {args.env}")
                logger.info(f"Available environments: {', '.join(MODEL_REQUIREMENTS.keys())}")
                return 1
        else:
            # Set up all environments
            return setup_all_environments(str(config_path), args.force)
    
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())