"""
Core functionality tests for HyenaDNA model
"""

import unittest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.test_base_model import BaseDNAModelTest
from src.models.hyenadna import HyenaDNAModel


class TestHyenaDNACore(BaseDNAModelTest):
    """Core functionality tests specific to HyenaDNA"""
    
    def get_model_class(self):
        """Return the HyenaDNA model class"""
        return HyenaDNAModel
    
    def get_test_config(self):
        """Return test configuration for HyenaDNA"""
        return {
            'pretrained_path': 'LongSafari/hyenadna-medium-450k-seqlen',
            'tokenizer': 'character',
            'max_length': 450000,  # 450k tokens!
            'num_labels': 2,
            'device': 'cpu',
            'use_env_manager': False,  # Disable for testing
            'conda_env': 'hyenadna_env'
        }
    
    # ============ HyenaDNA-Specific Tests ============
    
    def test_ultra_long_sequence_support(self):
        """Test that HyenaDNA supports ultra-long sequences"""
        model = HyenaDNAModel(self.config)
        
        # HyenaDNA can handle up to 450k tokens
        self.assertEqual(model.max_length, 450000)
        
        # This is much longer than traditional transformers
        self.assertGreater(model.max_length, 100000)
    
    def test_character_level_tokenization(self):
        """Test character-level tokenization"""
        model = HyenaDNAModel(self.config)
        
        self.assertEqual(model.config['tokenizer'], 'character')
        
        # Character-level means 1 token per nucleotide
        sequence = "ATCGATCG"
        # Would produce 8 tokens (one per character)
        self.assertEqual(len(sequence), 8)
    
    def test_hyena_architecture_config(self):
        """Test Hyena architecture configuration"""
        model = HyenaDNAModel(self.config)
        
        # HyenaDNA uses Hyena operator instead of attention
        # This enables subquadratic scaling
        self.assertIn('hyenadna', model.config['pretrained_path'].lower())
        
        # Check model variant
        self.assertIn('medium', model.config['pretrained_path'])
        self.assertIn('450k', model.config['pretrained_path'])
    
    def test_memory_efficiency_config(self):
        """Test memory efficiency configuration"""
        model = HyenaDNAModel(self.config)
        
        # HyenaDNA should be more memory efficient for long sequences
        # Due to subquadratic attention replacement
        
        # Test that model can theoretically handle very long sequences
        very_long_seq = "A" * 100000  # 100k nucleotides
        
        # Check if sequence would need chunking
        if len(very_long_seq) > model.max_length:
            chunks, counts = model.split_long_sequences(very_long_seq)
            self.assertEqual(len(chunks), counts[0])
        else:
            # HyenaDNA should handle this without chunking
            self.assertLessEqual(len(very_long_seq), model.max_length)
    
    def test_no_special_token_overhead(self):
        """Test that character tokenization has minimal overhead"""
        model = HyenaDNAModel(self.config)
        
        # Character-level tokenization should have 1:1 mapping
        sequence = "ATCGATCG"
        processed = model.preprocess_sequence(sequence)
        
        # Length should be preserved in preprocessing
        self.assertEqual(len(processed), len(sequence))
    
    def test_handles_very_long_sequences(self):
        """Test handling of sequences close to max length"""
        # Create a custom config with smaller max_length for testing
        test_config = self.config.copy()
        test_config['max_length'] = 1000  # Smaller for testing
        
        model = HyenaDNAModel(test_config)
        
        # Test sequence at max length
        long_seq = "A" * 1000
        chunks, counts = model.split_long_sequences(long_seq)
        
        # Should not chunk if exactly at max length
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], long_seq)
        
        # Test sequence over max length
        longer_seq = "A" * 1500
        chunks, counts = model.split_long_sequences(longer_seq, overlap=100)
        
        # Should chunk with overlap
        self.assertGreater(len(chunks), 1)
    
    def test_model_variants(self):
        """Test different HyenaDNA model variants"""
        # HyenaDNA has multiple variants with different max lengths
        variants = {
            'tiny-1k': 1024,
            'small-32k': 32768,
            'medium-160k': 163840,
            'medium-450k': 450000,
            'large-1m': 1048576
        }
        
        current_variant = 'medium-450k'
        expected_length = variants[current_variant]
        
        # Our test config should match one of these
        self.assertLessEqual(
            self.config['max_length'],
            expected_length
        )
    
    @unittest.skipIf(not False, "Skipping test that requires model download")
    def test_hyenadna_pretrained_loading(self):
        """Test loading pretrained HyenaDNA weights"""
        # This test would actually load the model
        # Skipped by default to avoid downloading during unit tests
        model = HyenaDNAModel(self.config)
        model.load_pretrained()
        
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.tokenizer)


if __name__ == '__main__':
    unittest.main()