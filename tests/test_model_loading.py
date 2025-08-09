"""
Unit tests for verifying that all DNA language models can be loaded
"""

import unittest
import torch
import warnings
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.base_model import BaseDNAModel
from src.models.dnabert1 import DNABert1Model
from src.models.dnabert2 import DNABert2Model
from src.models.nucleotide_transformer import NucleotideTransformerModel
from src.models.prokbert import ProkBERTModel
from src.models.grover import GroverModel
from src.models.gena_lm import GenaLMModel
from src.models.inherit import INHERITModel
from src.models.hyenadna import HyenaDNAModel
from src.models.evo import EVOModel


class TestModelLoading(unittest.TestCase):
    """Test suite for model loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Suppress transformers warnings during testing
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Basic configuration for all models
        self.base_config = {
            'num_labels': 2,
            'max_length': 512,
            'batch_size': 8,
            'device': 'cpu',  # Use CPU for testing
            'use_env_manager': False  # Disable environment switching for tests
        }
    
    def test_dnabert1_initialization(self):
        """Test DNABERT1 model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'zhihan1996/DNA_bert_6',
            'kmer': 6,
            'tokenizer': 'kmer'
        })
        
        model = DNABert1Model(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.kmer, 6)
        self.assertEqual(model.max_length, 512)
    
    def test_dnabert2_initialization(self):
        """Test DNABERT2 model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'zhihan1996/DNABERT-2-117M',
            'tokenizer': 'bpe'
        })
        
        model = DNABert2Model(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.max_length, 512)
    
    def test_nucleotide_transformer_initialization(self):
        """Test Nucleotide Transformer model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
            'tokenizer': '6mer',
            'max_length': 1000
        })
        
        model = NucleotideTransformerModel(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.max_length, 1000)
    
    def test_prokbert_initialization(self):
        """Test ProkBERT model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'neuralbioinfo/prokbert-mini',
            'tokenizer': 'lca',
            'kmer': 6,
            'shift': 1
        })
        
        model = ProkBERTModel(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.kmer, 6)
        self.assertEqual(model.shift, 1)
    
    def test_grover_initialization(self):
        """Test GROVER model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'PoetschLab/GROVER',
            'tokenizer': 'bpe',
            'max_length': 510,
            'bpe_vocab_size': 5000
        })
        
        model = GroverModel(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.max_length, 510)
    
    def test_gena_lm_initialization(self):
        """Test GENA-LM model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'AIRI-Institute/gena-lm-bert-base-t2t',
            'tokenizer': 'bpe',
            'max_length': 4500
        })
        
        model = GenaLMModel(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.max_length, 4500)
    
    def test_inherit_initialization(self):
        """Test INHERIT model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'zhihan1996/DNA_bert_6',
            'tokenizer': 'kmer',
            'kmer': 6
        })
        
        model = INHERITModel(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.kmer, 6)
    
    def test_hyenadna_initialization(self):
        """Test HyenaDNA model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'LongSafari/hyenadna-medium-450k-seqlen',
            'tokenizer': 'character',
            'max_length': 450000
        })
        
        model = HyenaDNAModel(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.max_length, 450000)
    
    def test_evo_initialization(self):
        """Test EVO model initialization"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'togethercomputer/evo-1-8k-base',
            'tokenizer': 'byte',
            'max_length': 8192,
            'revision': '1.1_fix'
        })
        
        model = EVOModel(config)
        self.assertIsInstance(model, BaseDNAModel)
        self.assertEqual(model.max_length, 8192)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_model_loading_with_mock(self, mock_model, mock_tokenizer):
        """Test model loading with mocked HuggingFace models"""
        # Mock the HuggingFace model and tokenizer
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'test-model',
            'tokenizer': 'bpe'
        })
        
        # Test with DNABERT2 as example
        model = DNABert2Model(config)
        model.load_pretrained()
        
        # Verify that the model loading was attempted
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.tokenizer)
    
    def test_model_device_handling(self):
        """Test that models handle device assignment correctly"""
        config = self.base_config.copy()
        config.update({
            'pretrained_path': 'test-model',
            'device': 'cpu'
        })
        
        # Test with multiple model types
        models = [
            DNABert1Model(config),
            DNABert2Model(config),
            NucleotideTransformerModel(config)
        ]
        
        for model in models:
            self.assertEqual(str(model.device), 'cpu')
            
            # Test moving to different device (if CUDA available)
            if torch.cuda.is_available():
                model.to('cuda')
                self.assertEqual(str(model.device), 'cuda:0')
    
    def test_sequence_preprocessing(self):
        """Test sequence preprocessing functionality"""
        config = self.base_config.copy()
        model = DNABert1Model(config)
        
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
    
    def test_sequence_validation(self):
        """Test sequence validation functionality"""
        config = self.base_config.copy()
        model = DNABert1Model(config)
        
        # Valid sequences
        self.assertTrue(model.validate_sequence("ATCGATCG"))
        self.assertTrue(model.validate_sequence("ATCGNNNN"))
        self.assertTrue(model.validate_sequence("atcgatcg"))  # lowercase
        
        # Invalid sequences
        self.assertFalse(model.validate_sequence(""))
        self.assertFalse(model.validate_sequence("ATCG123"))
        self.assertFalse(model.validate_sequence("ATCG!@#"))
    
    def test_sequence_chunking(self):
        """Test long sequence chunking functionality"""
        config = self.base_config.copy()
        config['max_length'] = 10
        model = DNABert1Model(config)
        
        # Test sequence that needs chunking
        long_seq = "A" * 25
        chunks, counts = model.split_long_sequences(long_seq, overlap=2)
        
        # Should create 3 chunks with overlap
        self.assertEqual(len(chunks), 3)
        self.assertEqual(counts[0], 3)
        
        # Test sequence that doesn't need chunking
        short_seq = "ATCGATCG"
        chunks, counts = model.split_long_sequences(short_seq, overlap=2)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_seq)
    
    def test_model_info(self):
        """Test model information retrieval"""
        config = self.base_config.copy()
        config['pretrained_path'] = 'test-model'
        
        model = DNABert1Model(config)
        info = model.get_model_info()
        
        self.assertIn('model_name', info)
        self.assertIn('model_class', info)
        self.assertIn('max_length', info)
        self.assertIn('num_labels', info)
        self.assertIn('device', info)
        self.assertEqual(info['model_class'], 'DNABert1Model')
        self.assertEqual(info['max_length'], 512)
    
    def tearDown(self):
        """Clean up after tests"""
        warnings.resetwarnings()


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model functionality"""
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gpu_compatibility(self):
        """Test that models work with GPU if available"""
        config = {
            'num_labels': 2,
            'max_length': 512,
            'pretrained_path': 'test-model',
            'device': 'cuda'
        }
        
        model = DNABert1Model(config)
        self.assertEqual(model.device.type, 'cuda')
    
    def test_all_models_inherit_base(self):
        """Verify all models inherit from BaseDNAModel"""
        models = [
            DNABert1Model,
            DNABert2Model,
            NucleotideTransformerModel,
            ProkBERTModel,
            GroverModel,
            GenaLMModel,
            INHERITModel,
            HyenaDNAModel,
            EVOModel
        ]
        
        for model_class in models:
            self.assertTrue(issubclass(model_class, BaseDNAModel))


if __name__ == '__main__':
    unittest.main()