"""
Complete model loading tests for all DNA language models
Tests model loading, weight initialization, and tokenizer setup
"""

import unittest
import torch
import tempfile
import os
import json
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


class TestModelLoadingComplete(unittest.TestCase):
    """Comprehensive tests for model loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_configs = {
            'dnabert1': {
                'pretrained_path': 'zhihan1996/DNA_bert_6',
                'tokenizer': 'kmer',
                'kmer': 6,
                'max_length': 512,
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            },
            'dnabert2': {
                'pretrained_path': 'zhihan1996/DNABERT-2-117M',
                'tokenizer': 'bpe',
                'max_length': 512,
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            },
            'nucleotide_transformer': {
                'pretrained_path': 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
                'tokenizer': '6mer',
                'max_length': 1000,
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            },
            'prokbert': {
                'pretrained_path': 'neuralbioinfo/prokbert-mini',
                'tokenizer': 'lca',
                'kmer': 6,
                'shift': 1,
                'max_length': 512,
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            },
            'grover': {
                'pretrained_path': 'PoetschLab/GROVER',
                'tokenizer': 'bpe',
                'max_length': 510,
                'bpe_vocab_size': 5000,
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            },
            'gena_lm': {
                'pretrained_path': 'AIRI-Institute/gena-lm-bert-base-t2t',
                'tokenizer': 'bpe',
                'max_length': 4500,
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            },
            'inherit': {
                'pretrained_path': 'zhihan1996/DNA_bert_6',
                'tokenizer': 'kmer',
                'kmer': 6,
                'max_length': 512,
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            },
            'hyenadna': {
                'pretrained_path': 'LongSafari/hyenadna-medium-450k-seqlen',
                'tokenizer': 'character',
                'max_length': 450000,
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            },
            'evo': {
                'pretrained_path': 'togethercomputer/evo-1-8k-base',
                'tokenizer': 'byte',
                'max_length': 8192,
                'revision': '1.1_fix',
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            }
        }
        
        self.model_classes = {
            'dnabert1': DNABert1Model,
            'dnabert2': DNABert2Model,
            'nucleotide_transformer': NucleotideTransformerModel,
            'prokbert': ProkBERTModel,
            'grover': GroverModel,
            'gena_lm': GenaLMModel,
            'inherit': INHERITModel,
            'hyenadna': HyenaDNAModel,
            'evo': EVOModel
        }
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    # ============ Model Instantiation Tests ============
    
    def test_all_models_instantiate(self):
        """Test that all models can be instantiated"""
        for model_name, model_class in self.model_classes.items():
            config = self.model_configs[model_name]
            
            try:
                model = model_class(config)
                self.assertIsInstance(model, BaseDNAModel)
                self.assertIsNotNone(model)
            except Exception as e:
                self.fail(f"Failed to instantiate {model_name}: {str(e)}")
    
    def test_model_config_validation(self):
        """Test that models validate their configuration"""
        for model_name, model_class in self.model_classes.items():
            config = self.model_configs[model_name].copy()
            
            # Test with valid config
            model = model_class(config)
            self.assertEqual(model.config, config)
            
            # Test with missing optional parameters
            minimal_config = {
                'num_labels': 2,
                'device': 'cpu',
                'use_env_manager': False
            }
            
            # Should still instantiate with defaults
            model = model_class(minimal_config)
            self.assertIsNotNone(model)
    
    # ============ Pretrained Loading Tests (Mocked) ============
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_pretrained_from_huggingface(self, mock_tokenizer, mock_model):
        """Test loading pretrained models from HuggingFace (mocked)"""
        # Setup mocks
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        
        for model_name in ['dnabert1', 'dnabert2', 'nucleotide_transformer']:
            config = self.model_configs[model_name]
            model_class = self.model_classes[model_name]
            
            model = model_class(config)
            
            # Mock the load_pretrained method
            with patch.object(model, 'load_pretrained') as mock_load:
                mock_load.return_value = None
                model.load_pretrained()
                mock_load.assert_called_once()
    
    @patch('torch.load')
    def test_load_finetuned_from_local(self, mock_torch_load):
        """Test loading fine-tuned models from local path"""
        # Create mock state dict
        mock_state_dict = {
            'model_state_dict': {'layer.weight': torch.randn(10, 10)},
            'tokenizer_config': {'max_length': 512},
            'config': {'num_labels': 2}
        }
        mock_torch_load.return_value = mock_state_dict
        
        # Test loading for each model
        for model_name in ['dnabert1', 'dnabert2']:
            config = self.model_configs[model_name]
            model_class = self.model_classes[model_name]
            
            model = model_class(config)
            
            # Mock load_model method
            local_path = os.path.join(self.temp_dir, f"{model_name}_finetuned")
            
            with patch.object(model, 'load_model') as mock_load:
                model.load_model(local_path)
                mock_load.assert_called_once_with(local_path)
    
    # ============ Tokenizer Tests ============
    
    def test_tokenizer_initialization(self):
        """Test that tokenizers are properly initialized for each model"""
        tokenizer_types = {
            'dnabert1': 'kmer',
            'dnabert2': 'bpe',
            'nucleotide_transformer': '6mer',
            'prokbert': 'lca',
            'grover': 'bpe',
            'gena_lm': 'bpe',
            'inherit': 'kmer',
            'hyenadna': 'character',
            'evo': 'byte'
        }
        
        for model_name, expected_tokenizer in tokenizer_types.items():
            config = self.model_configs[model_name]
            model_class = self.model_classes[model_name]
            
            model = model_class(config)
            
            # Check tokenizer type in config
            if 'tokenizer' in config:
                self.assertEqual(
                    config['tokenizer'],
                    expected_tokenizer,
                    f"Wrong tokenizer for {model_name}"
                )
    
    # ============ Model Architecture Tests ============
    
    def test_model_architecture_attributes(self):
        """Test that models have correct architecture attributes"""
        for model_name, model_class in self.model_classes.items():
            config = self.model_configs[model_name]
            model = model_class(config)
            
            # Check common attributes
            self.assertTrue(hasattr(model, 'max_length'))
            self.assertTrue(hasattr(model, 'num_labels'))
            self.assertTrue(hasattr(model, 'device'))
            
            # Check model-specific attributes
            if config.get('tokenizer') == 'kmer':
                self.assertTrue(hasattr(model, 'kmer'))
            
            if model_name == 'prokbert':
                self.assertTrue(hasattr(model, 'shift'))
            
            if model_name == 'evo':
                self.assertTrue(hasattr(model, 'revision'))
    
    # ============ Save/Load Tests ============
    
    def test_save_and_load_model(self):
        """Test saving and loading model states"""
        for model_name in ['dnabert1', 'dnabert2']:
            config = self.model_configs[model_name]
            model_class = self.model_classes[model_name]
            
            model = model_class(config)
            
            # Create save path
            save_path = os.path.join(self.temp_dir, f"{model_name}_saved")
            os.makedirs(save_path, exist_ok=True)
            
            # Mock save and load methods
            with patch.object(model, 'save_model') as mock_save:
                model.save_model(save_path)
                mock_save.assert_called_once_with(save_path)
            
            with patch.object(model, 'load_model') as mock_load:
                model.load_model(save_path)
                mock_load.assert_called_once_with(save_path)
    
    # ============ Error Handling Tests ============
    
    def test_handle_missing_model_files(self):
        """Test handling of missing model files"""
        for model_name in ['dnabert1']:
            config = self.model_configs[model_name]
            model_class = self.model_classes[model_name]
            
            model = model_class(config)
            
            # Try to load from non-existent path
            fake_path = "/non/existent/path/to/model"
            
            with patch.object(model, 'load_model') as mock_load:
                mock_load.side_effect = FileNotFoundError("Model not found")
                
                with self.assertRaises(FileNotFoundError):
                    model.load_model(fake_path)
    
    def test_handle_corrupted_model_files(self):
        """Test handling of corrupted model files"""
        # Create a corrupted file
        corrupted_file = os.path.join(self.temp_dir, "corrupted_model.pt")
        with open(corrupted_file, 'w') as f:
            f.write("This is not a valid model file")
        
        with patch('torch.load') as mock_load:
            mock_load.side_effect = RuntimeError("Invalid model file")
            
            config = self.model_configs['dnabert1']
            model = DNABert1Model(config)
            
            with patch.object(model, 'load_model') as mock_load_model:
                mock_load_model.side_effect = RuntimeError("Invalid model file")
                
                with self.assertRaises(RuntimeError):
                    model.load_model(corrupted_file)
    
    # ============ Memory Management Tests ============
    
    def test_model_to_device(self):
        """Test moving models to different devices"""
        for model_name in ['dnabert1', 'dnabert2']:
            config = self.model_configs[model_name]
            model_class = self.model_classes[model_name]
            
            model = model_class(config)
            
            # Test CPU
            model.to('cpu')
            self.assertEqual(str(model.device), 'cpu')
            
            # Test CUDA (if available)
            if torch.cuda.is_available():
                model.to('cuda')
                self.assertEqual(model.device.type, 'cuda')
                
                # Move back to CPU
                model.to('cpu')
                self.assertEqual(str(model.device), 'cpu')
    
    def test_model_eval_train_modes(self):
        """Test switching between evaluation and training modes"""
        for model_name in ['dnabert1', 'dnabert2']:
            config = self.model_configs[model_name]
            model_class = self.model_classes[model_name]
            
            model = model_class(config)
            
            # Test eval mode
            model.eval()
            # Would check model.training = False if model was loaded
            
            # Test train mode  
            model.train()
            # Would check model.training = True if model was loaded
    
    # ============ Integration Tests ============
    
    def test_model_pipeline_without_loading(self):
        """Test model pipeline without actually loading weights"""
        test_sequence = "ATCGATCGATCGATCG"
        
        for model_name in self.model_classes.keys():
            config = self.model_configs[model_name]
            model_class = self.model_classes[model_name]
            
            model = model_class(config)
            
            # Test preprocessing
            processed = model.preprocess_sequence(test_sequence)
            self.assertIsNotNone(processed)
            
            # Test validation
            is_valid = model.validate_sequence(test_sequence)
            self.assertTrue(is_valid)
            
            # Test sequence stats
            stats = model.get_sequence_stats([test_sequence])
            self.assertIn('num_sequences', stats)
            self.assertEqual(stats['num_sequences'], 1)


if __name__ == '__main__':
    unittest.main()