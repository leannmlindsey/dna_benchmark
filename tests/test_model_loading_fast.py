"""
Fast unit tests for model loading that don't require downloading actual models
These tests verify the model classes can be instantiated and basic functionality works
"""

import unittest
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_model import BaseDNAModel


class TestModelLoadingFast(unittest.TestCase):
    """Fast tests that don't require downloading models"""
    
    @classmethod
    def setUpClass(cls):
        """Load test configuration"""
        config_path = Path(__file__).parent / 'test_config.yaml'
        with open(config_path, 'r') as f:
            cls.test_config = yaml.safe_load(f)
    
    def test_all_models_defined_in_config(self):
        """Verify all expected models are defined in config"""
        expected_models = [
            'dnabert1', 'dnabert2', 'nucleotide_transformer',
            'prokbert', 'grover', 'gena_lm', 'inherit',
            'hyenadna', 'evo'
        ]
        
        for model_name in expected_models:
            self.assertIn(model_name, self.test_config['test_models'],
                         f"Model {model_name} not found in test config")
    
    def test_model_configs_have_required_fields(self):
        """Verify each model config has required fields"""
        required_fields = ['class_name', 'pretrained_path', 'max_length', 'num_labels']
        
        for model_name, config in self.test_config['test_models'].items():
            for field in required_fields:
                self.assertIn(field, config,
                             f"Model {model_name} missing required field: {field}")
    
    def test_model_max_lengths(self):
        """Verify model max lengths are reasonable"""
        for model_name, config in self.test_config['test_models'].items():
            max_length = config['max_length']
            self.assertIsInstance(max_length, int)
            self.assertGreater(max_length, 0)
            self.assertLessEqual(max_length, 1000000,  # 1M max
                               f"Model {model_name} has unreasonably large max_length")
    
    def test_tokenizer_types(self):
        """Verify tokenizer types are valid"""
        valid_tokenizers = ['kmer', 'bpe', '6mer', 'lca', 'character', 'byte']
        
        for model_name, config in self.test_config['test_models'].items():
            if 'tokenizer' in config:
                self.assertIn(config['tokenizer'], valid_tokenizers,
                             f"Model {model_name} has invalid tokenizer: {config['tokenizer']}")
    
    def test_kmer_models_have_k_value(self):
        """Verify k-mer models have k value specified"""
        kmer_models = ['dnabert1', 'prokbert', 'inherit']
        
        for model_name in kmer_models:
            config = self.test_config['test_models'][model_name]
            self.assertIn('kmer', config,
                         f"K-mer model {model_name} missing 'kmer' parameter")
            self.assertIn(config['kmer'], [3, 4, 5, 6],
                         f"Model {model_name} has invalid k-mer value: {config['kmer']}")
    
    def test_test_sequences_valid(self):
        """Verify test sequences are properly formatted"""
        sequences = self.test_config['test_sequences']
        
        # Check valid sequences
        for seq in sequences['valid']:
            self.assertIsInstance(seq, str)
            self.assertGreater(len(seq), 0)
            # Check only contains valid DNA characters
            valid_chars = set('ATCGN')
            self.assertTrue(all(c in valid_chars for c in seq.upper()))
        
        # Check invalid sequences are actually invalid
        for seq in sequences['invalid']:
            if seq:  # Skip empty string test case
                # Should contain invalid characters
                valid_chars = set('ATCGN')
                has_invalid = not all(c in valid_chars for c in seq.upper())
                self.assertTrue(has_invalid or seq == "",
                              f"Invalid test sequence seems valid: {seq}")
        
        # Check long sequences are actually long
        for seq in sequences['long']:
            self.assertGreater(len(seq), 500,
                             "Long test sequence is not long enough")


class TestModelImports(unittest.TestCase):
    """Test that model modules can be imported"""
    
    def test_import_base_model(self):
        """Test importing base model"""
        try:
            from src.models.base_model import BaseDNAModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import BaseDNAModel: {e}")
    
    def test_import_dnabert1(self):
        """Test importing DNABERT1 model"""
        try:
            from src.models.dnabert1 import DNABert1Model
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import DNABert1Model: {e}")
    
    def test_import_dnabert2(self):
        """Test importing DNABERT2 model"""
        try:
            from src.models.dnabert2 import DNABert2Model
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import DNABert2Model: {e}")
    
    def test_import_nucleotide_transformer(self):
        """Test importing Nucleotide Transformer model"""
        try:
            from src.models.nucleotide_transformer import NucleotideTransformerModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import NucleotideTransformerModel: {e}")
    
    def test_import_prokbert(self):
        """Test importing ProkBERT model"""
        try:
            from src.models.prokbert import ProkBERTModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ProkBERTModel: {e}")
    
    def test_import_grover(self):
        """Test importing GROVER model"""
        try:
            from src.models.grover import GroverModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import GroverModel: {e}")
    
    def test_import_gena_lm(self):
        """Test importing GENA-LM model"""
        try:
            from src.models.gena_lm import GenaLMModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import GenaLMModel: {e}")
    
    def test_import_inherit(self):
        """Test importing INHERIT model"""
        try:
            from src.models.inherit import INHERITModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import INHERITModel: {e}")
    
    def test_import_hyenadna(self):
        """Test importing HyenaDNA model"""
        try:
            from src.models.hyenadna import HyenaDNAModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import HyenaDNAModel: {e}")
    
    def test_import_evo(self):
        """Test importing EVO model"""
        try:
            from src.models.evo import EVOModel
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import EVOModel: {e}")


class TestBaseModelFunctionality(unittest.TestCase):
    """Test base model functionality without requiring actual models"""
    
    def setUp(self):
        """Create a mock model class for testing"""
        from src.models.base_model import BaseDNAModel
        
        class MockDNAModel(BaseDNAModel):
            def load_pretrained(self, path=None):
                pass
            
            def get_embeddings(self, sequences):
                import torch
                import numpy as np
                if isinstance(sequences, str):
                    sequences = [sequences]
                return torch.randn(len(sequences), 768)
            
            def predict(self, sequences):
                import numpy as np
                if isinstance(sequences, str):
                    sequences = [sequences]
                n = len(sequences)
                return {
                    'labels': np.random.randint(0, 2, n),
                    'probabilities': np.random.rand(n, 2)
                }
            
            def fine_tune(self, train_dataset, val_dataset=None, **kwargs):
                return {'loss': 0.5, 'accuracy': 0.85}
            
            def save_model(self, path):
                pass
            
            def load_model(self, path):
                pass
        
        self.MockModel = MockDNAModel
    
    def test_sequence_stats(self):
        """Test sequence statistics calculation"""
        model = self.MockModel({'max_length': 100})
        
        sequences = ["ATCGATCG", "GCGCGCGC", "AAAATTTT"]
        stats = model.get_sequence_stats(sequences)
        
        self.assertEqual(stats['num_sequences'], 3)
        self.assertEqual(stats['total_length'], 24)
        self.assertEqual(stats['mean_length'], 8.0)
        self.assertIn('gc_content', stats)
        self.assertIn('base_counts', stats)
    
    def test_sequence_validation(self):
        """Test sequence validation"""
        model = self.MockModel({'max_length': 100})
        
        # Valid sequences
        self.assertTrue(model.validate_sequence("ATCG"))
        self.assertTrue(model.validate_sequence("atcg"))
        self.assertTrue(model.validate_sequence("ATCGN"))
        
        # Invalid sequences
        self.assertFalse(model.validate_sequence(""))
        self.assertFalse(model.validate_sequence("ATCG123"))
    
    def test_sequence_preprocessing(self):
        """Test sequence preprocessing"""
        model = self.MockModel({'max_length': 100})
        
        # Test uppercase conversion
        self.assertEqual(model.preprocess_sequence("atcg"), "ATCG")
        
        # Test invalid character replacement
        self.assertEqual(model.preprocess_sequence("ATCGXYZ"), "ATCGNNN")
    
    def test_chunk_aggregation(self):
        """Test chunk prediction aggregation"""
        import numpy as np
        model = self.MockModel({'max_length': 100})
        
        # Test mean aggregation
        predictions = np.array([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3]])
        chunk_counts = [1, 2]
        
        aggregated = model.aggregate_chunk_predictions(
            predictions, chunk_counts, method='mean'
        )
        
        self.assertEqual(len(aggregated), 2)


if __name__ == '__main__':
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoadingFast))
    suite.addTests(loader.loadTestsFromTestCase(TestModelImports))
    suite.addTests(loader.loadTestsFromTestCase(TestBaseModelFunctionality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)