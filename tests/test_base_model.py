"""
Base test class with common test methods for all DNA models
"""

import unittest
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
import tempfile
import os
import logging
from unittest.mock import patch, MagicMock
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDNAModelTest(unittest.TestCase, ABC):
    """
    Abstract base test class that all model-specific test classes should inherit from.
    Provides common test methods that every model must pass.
    """
    
    @abstractmethod
    def get_model_class(self) -> Type:
        """Return the model class to test"""
        pass
    
    @abstractmethod
    def get_test_config(self) -> Dict:
        """Return test configuration for the model"""
        pass
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_class = self.get_model_class()
        self.config = self.get_test_config()
        self.test_sequences = [
            "ATCGATCGATCGATCG",
            "GCTAGCTAGCTAGCTA",
            "AAAAAAAAAAAAAAAA",
            "TTTTTTTTTTTTTTTT",
            "ATCGNNNNATCGATCG"
        ]
        self.long_sequence = "ATCG" * 1000  # 4000 bp sequence
        
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any temporary files or resources
        pass
    
    # ============ Core Initialization Tests ============
    
    def test_model_initialization_default(self):
        """Test model initialization with default configuration"""
        model = self.model_class(self.config)
        
        # Check basic attributes
        self.assertIsNotNone(model)
        self.assertEqual(model.config, self.config)
        self.assertEqual(model.max_length, self.config.get('max_length', 512))
        self.assertEqual(model.num_labels, self.config.get('num_labels', 2))
        
        # Check device assignment
        if torch.cuda.is_available():
            self.assertEqual(model.device.type, 'cuda')
        else:
            self.assertEqual(model.device.type, 'cpu')
    
    def test_model_initialization_custom_config(self):
        """Test model initialization with custom configuration"""
        custom_config = self.config.copy()
        custom_config['max_length'] = 256
        custom_config['num_labels'] = 3
        custom_config['device'] = 'cpu'
        
        model = self.model_class(custom_config)
        
        self.assertEqual(model.max_length, 256)
        self.assertEqual(model.num_labels, 3)
        self.assertEqual(str(model.device), 'cpu')
    
    def test_model_inheritance(self):
        """Test that model properly inherits from BaseDNAModel"""
        from src.models.base_model import BaseDNAModel
        
        model = self.model_class(self.config)
        self.assertIsInstance(model, BaseDNAModel)
        
        # Check that all required methods are implemented
        required_methods = [
            'load_pretrained', 'get_embeddings', 'predict',
            'fine_tune', 'save_model', 'load_model'
        ]
        
        for method_name in required_methods:
            self.assertTrue(
                hasattr(model, method_name),
                f"Model missing required method: {method_name}"
            )
    
    def test_model_repr(self):
        """Test model string representation"""
        model = self.model_class(self.config)
        repr_str = repr(model)
        
        self.assertIn(self.model_class.__name__, repr_str)
        self.assertIn(str(model.max_length), repr_str)
        self.assertIn(str(model.num_labels), repr_str)
    
    # ============ Configuration Tests ============
    
    def test_conda_environment_configuration(self):
        """Test that conda environment is properly configured"""
        model = self.model_class(self.config)
        
        if 'conda_env' in self.config:
            self.assertEqual(model.conda_env, self.config['conda_env'])
            self.assertIsNotNone(model.conda_env)
    
    def test_tokenizer_configuration(self):
        """Test that tokenizer type is properly configured"""
        if 'tokenizer' in self.config:
            model = self.model_class(self.config)
            
            valid_tokenizers = ['kmer', 'bpe', '6mer', 'lca', 'character', 'byte']
            self.assertIn(
                self.config['tokenizer'], 
                valid_tokenizers,
                f"Invalid tokenizer type: {self.config['tokenizer']}"
            )
    
    def test_model_specific_parameters(self):
        """Test model-specific configuration parameters"""
        model = self.model_class(self.config)
        
        # Check k-mer models have k value
        if self.config.get('tokenizer') == 'kmer':
            self.assertIn('kmer', self.config)
            self.assertIn(self.config['kmer'], [3, 4, 5, 6])
        
        # Check LCA models have shift parameter
        if self.config.get('tokenizer') == 'lca':
            self.assertIn('shift', self.config)
    
    # ============ Preprocessing Tests ============
    
    def test_sequence_validation_valid(self):
        """Test sequence validation with valid sequences"""
        model = self.model_class(self.config)
        
        for seq in self.test_sequences:
            self.assertTrue(
                model.validate_sequence(seq),
                f"Valid sequence incorrectly marked as invalid: {seq}"
            )
    
    def test_sequence_validation_invalid(self):
        """Test sequence validation with invalid sequences"""
        model = self.model_class(self.config)
        
        invalid_sequences = [
            "",  # Empty
            "ATCG123",  # Contains numbers
            "ATCG!@#",  # Contains special characters
            "Hello World",  # Not DNA
        ]
        
        for seq in invalid_sequences:
            self.assertFalse(
                model.validate_sequence(seq),
                f"Invalid sequence incorrectly marked as valid: {seq}"
            )
    
    def test_sequence_preprocessing(self):
        """Test sequence preprocessing functionality"""
        model = self.model_class(self.config)
        
        # Test uppercase conversion
        seq = "atcgatcg"
        processed = model.preprocess_sequence(seq)
        self.assertEqual(processed, "ATCGATCG")
        
        # Test invalid character replacement
        seq = "ATCGXYZ"
        processed = model.preprocess_sequence(seq)
        self.assertEqual(processed, "ATCGNNN")
        
        # Test mixed case and invalid
        seq = "aTcGxYz"
        processed = model.preprocess_sequence(seq)
        self.assertEqual(processed, "ATCGNNN")
    
    def test_sequence_stats(self):
        """Test sequence statistics calculation"""
        model = self.model_class(self.config)
        
        stats = model.get_sequence_stats(self.test_sequences)
        
        # Check required fields
        required_fields = [
            'num_sequences', 'total_length', 'mean_length',
            'min_length', 'max_length', 'base_counts',
            'gc_content', 'sequences_too_long'
        ]
        
        for field in required_fields:
            self.assertIn(field, stats, f"Missing stats field: {field}")
        
        # Validate stats values
        self.assertEqual(stats['num_sequences'], len(self.test_sequences))
        self.assertGreaterEqual(stats['gc_content'], 0.0)
        self.assertLessEqual(stats['gc_content'], 1.0)
    
    # ============ Sequence Chunking Tests ============
    
    def test_sequence_chunking_needed(self):
        """Test chunking for sequences longer than max_length"""
        config = self.config.copy()
        config['max_length'] = 100
        model = self.model_class(config)
        
        long_seq = "A" * 250
        chunks, counts = model.split_long_sequences(long_seq, overlap=20)
        
        # Check that sequence was chunked
        self.assertGreater(len(chunks), 1)
        self.assertEqual(counts[0], len(chunks))
        
        # Check chunk lengths
        for chunk in chunks[:-1]:  # All but last chunk
            self.assertEqual(len(chunk), 100)
    
    def test_sequence_chunking_not_needed(self):
        """Test that short sequences are not chunked"""
        model = self.model_class(self.config)
        
        short_seq = "ATCGATCG"
        chunks, counts = model.split_long_sequences(short_seq, overlap=2)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_seq)
        self.assertEqual(counts[0], 1)
    
    def test_chunk_aggregation_mean(self):
        """Test mean aggregation of chunk predictions"""
        model = self.model_class(self.config)
        
        # Mock predictions for 3 chunks from 2 sequences
        predictions = np.array([
            [0.8, 0.2],  # Seq 1, chunk 1
            [0.6, 0.4],  # Seq 2, chunk 1
            [0.7, 0.3],  # Seq 2, chunk 2
        ])
        chunk_counts = [1, 2]
        
        aggregated = model.aggregate_chunk_predictions(
            predictions, chunk_counts, method='mean'
        )
        
        self.assertEqual(len(aggregated), 2)
        # First sequence unchanged (1 chunk)
        np.testing.assert_array_almost_equal(aggregated[0], [0.8, 0.2])
        # Second sequence averaged (2 chunks)
        np.testing.assert_array_almost_equal(aggregated[1], [0.65, 0.35])
    
    # ============ Model Info Tests ============
    
    def test_get_model_info(self):
        """Test model information retrieval"""
        model = self.model_class(self.config)
        info = model.get_model_info()
        
        required_fields = [
            'model_name', 'model_class', 'max_length',
            'num_labels', 'device', 'config'
        ]
        
        for field in required_fields:
            self.assertIn(field, info, f"Missing info field: {field}")
        
        self.assertEqual(info['model_class'], self.model_class.__name__)
        self.assertEqual(info['config'], self.config)
    
    # ============ Device Management Tests ============
    
    def test_device_movement(self):
        """Test moving model between devices"""
        model = self.model_class(self.config)
        
        # Move to CPU
        model.to('cpu')
        self.assertEqual(str(model.device), 'cpu')
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            model.to('cuda')
            self.assertEqual(model.device.type, 'cuda')
    
    def test_eval_train_modes(self):
        """Test switching between eval and train modes"""
        model = self.model_class(self.config)
        
        # Test eval mode
        model.eval()
        if model.model is not None:
            self.assertFalse(model.model.training)
        
        # Test train mode
        model.train()
        if model.model is not None:
            self.assertTrue(model.model.training)
    
    # ============ Environment Management Tests ============
    
    def test_environment_aware_methods(self):
        """Test that environment-aware methods exist"""
        model = self.model_class(self.config)
        
        env_methods = [
            'load_pretrained_with_env',
            'get_embeddings_with_env',
            'predict_with_env',
            'fine_tune_with_env'
        ]
        
        for method_name in env_methods:
            self.assertTrue(
                hasattr(model, method_name),
                f"Missing environment-aware method: {method_name}"
            )
    
    @patch('src.utils.environment_manager.EnvironmentManager')
    def test_environment_switching_disabled(self, mock_env_manager):
        """Test that environment switching can be disabled"""
        config = self.config.copy()
        config['use_env_manager'] = False
        
        model = self.model_class(config)
        
        # Mock predict method
        with patch.object(model, 'predict', return_value={'labels': [0, 1]}):
            result = model.predict_with_env(["ATCG"])
            
            # Environment manager should not be called when disabled
            mock_env_manager.assert_not_called()
            self.assertEqual(result['labels'], [0, 1])


class ModelTestGenerator:
    """Helper class to generate test classes for each model"""
    
    @staticmethod
    def create_test_class(model_name: str, model_class: Type, test_config: Dict) -> Type:
        """
        Create a test class for a specific model
        
        Args:
            model_name: Name of the model
            model_class: Model class to test
            test_config: Test configuration for the model
            
        Returns:
            Test class for the model
        """
        class_name = f"Test{model_name}Core"
        
        def get_model_class(self):
            return model_class
        
        def get_test_config(self):
            return test_config
        
        # Create test class dynamically
        test_class = type(
            class_name,
            (BaseDNAModelTest,),
            {
                'get_model_class': get_model_class,
                'get_test_config': get_test_config,
            }
        )
        
        return test_class