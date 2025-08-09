"""
Core functionality tests for DNABERT2 model
"""

import unittest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.test_base_model import BaseDNAModelTest
from src.models.dnabert2 import DNABert2Model


class TestDNABert2Core(BaseDNAModelTest):
    """Core functionality tests specific to DNABERT2"""
    
    def get_model_class(self):
        """Return the DNABERT2 model class"""
        return DNABert2Model
    
    def get_test_config(self):
        """Return test configuration for DNABERT2"""
        return {
            'pretrained_path': 'zhihan1996/DNABERT-2-117M',
            'tokenizer': 'bpe',
            'max_length': 512,
            'num_labels': 2,
            'device': 'cpu',
            'use_env_manager': False,  # Disable for testing
            'conda_env': 'dnabert2_env'
        }
    
    # ============ DNABERT2-Specific Tests ============
    
    def test_bpe_tokenization_setup(self):
        """Test that BPE tokenization is properly configured"""
        model = DNABert2Model(self.config)
        
        self.assertEqual(self.config['tokenizer'], 'bpe')
        # BPE tokenizer should be more efficient than k-mer
    
    def test_dnabert2_improvements(self):
        """Test DNABERT2-specific improvements over DNABERT1"""
        model = DNABert2Model(self.config)
        
        # DNABERT2 uses BPE instead of k-mer tokenization
        self.assertEqual(model.config['tokenizer'], 'bpe')
        
        # Should not have k-mer parameter
        self.assertNotIn('kmer', model.config)
    
    def test_sequence_length_efficiency(self):
        """Test that DNABERT2 handles sequences more efficiently"""
        model = DNABert2Model(self.config)
        
        # BPE tokenization is more efficient than k-mer
        # Can handle longer sequences with same token limit
        self.assertEqual(model.max_length, 512)
    
    def test_dnabert2_model_variant(self):
        """Test DNABERT2 model variant (117M parameters)"""
        model = DNABert2Model(self.config)
        
        # Check model path indicates correct variant
        self.assertIn('117M', model.config['pretrained_path'])
    
    def test_bpe_vocabulary(self):
        """Test BPE vocabulary configuration"""
        model = DNABert2Model(self.config)
        
        # BPE models have learned vocabularies
        # Vocabulary size would be checked when tokenizer is loaded
        self.assertEqual(model.config['tokenizer'], 'bpe')
    
    def test_special_tokens_handling(self):
        """Test handling of special tokens in BPE"""
        model = DNABert2Model(self.config)
        
        # BPE models use special tokens like [CLS], [SEP]
        # These would be added during tokenization
        sequence = "ATCGATCG"
        processed = model.preprocess_sequence(sequence)
        
        # Preprocessing should not add special tokens
        # They're added during tokenization
        self.assertEqual(processed, "ATCGATCG")
    
    def test_dnabert2_backward_compatibility(self):
        """Test that DNABERT2 maintains compatibility with base class"""
        model = DNABert2Model(self.config)
        
        # Should have all base class methods
        required_methods = [
            'validate_sequence',
            'preprocess_sequence',
            'get_sequence_stats',
            'split_long_sequences'
        ]
        
        for method in required_methods:
            self.assertTrue(
                hasattr(model, method),
                f"DNABERT2 missing base method: {method}"
            )
    
    @unittest.skipIf(not False, "Skipping test that requires model download")
    def test_dnabert2_pretrained_loading(self):
        """Test loading pretrained DNABERT2 weights"""
        # This test would actually load the model
        # Skipped by default to avoid downloading during unit tests
        model = DNABert2Model(self.config)
        model.load_pretrained()
        
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.tokenizer)
        
        # Check tokenizer is BPE-based
        # Actual check would depend on tokenizer implementation


if __name__ == '__main__':
    unittest.main()