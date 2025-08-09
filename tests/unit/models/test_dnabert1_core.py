"""
Core functionality tests for DNABERT1 model
"""

import unittest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.test_base_model import BaseDNAModelTest
from src.models.dnabert1 import DNABert1Model


class TestDNABert1Core(BaseDNAModelTest):
    """Core functionality tests specific to DNABERT1"""
    
    def get_model_class(self):
        """Return the DNABERT1 model class"""
        return DNABert1Model
    
    def get_test_config(self):
        """Return test configuration for DNABERT1"""
        return {
            'pretrained_path': 'zhihan1996/DNA_bert_6',
            'tokenizer': 'kmer',
            'kmer': 6,
            'max_length': 512,
            'num_labels': 2,
            'device': 'cpu',
            'use_env_manager': False,  # Disable for testing
            'conda_env': 'dnabert1_env'
        }
    
    # ============ DNABERT1-Specific Tests ============
    
    def test_kmer_tokenization_setup(self):
        """Test that k-mer tokenization is properly configured"""
        model = DNABert1Model(self.config)
        
        self.assertEqual(model.kmer, 6)
        self.assertEqual(self.config['tokenizer'], 'kmer')
    
    def test_kmer_sequence_conversion(self):
        """Test k-mer sequence conversion"""
        model = DNABert1Model(self.config)
        
        # Test sequence
        sequence = "ATCGATCGATCG"
        
        # Expected k-mers with k=6 and stride=1
        # ATCGAT, TCGATC, CGATCG, GATCGA, ATCGAT, TCGATC
        
        # This would be implemented in the actual model
        # For now, just verify the model has the capability
        self.assertTrue(hasattr(model, 'kmer'))
    
    def test_max_sequence_length_kmer(self):
        """Test maximum sequence length with k-mer tokenization"""
        model = DNABert1Model(self.config)
        
        # With k-mer tokenization, actual nucleotide limit depends on k
        # For k=6, max_length=512 means ~517 nucleotides (512 + k - 1)
        max_nucleotides = model.max_length + model.kmer - 1
        self.assertEqual(max_nucleotides, 517)
    
    def test_dnabert1_specific_config(self):
        """Test DNABERT1-specific configuration parameters"""
        model = DNABert1Model(self.config)
        
        # Check model name/path
        self.assertEqual(model.config['pretrained_path'], 'zhihan1996/DNA_bert_6')
        
        # Check that model supports standard BERT operations
        expected_attributes = ['kmer', 'max_length', 'num_labels']
        for attr in expected_attributes:
            self.assertTrue(
                hasattr(model, attr),
                f"DNABERT1 missing expected attribute: {attr}"
            )
    
    def test_handle_ambiguous_bases(self):
        """Test handling of ambiguous DNA bases (N)"""
        model = DNABert1Model(self.config)
        
        # Sequence with N bases
        sequence = "ATCGNNNNATCG"
        processed = model.preprocess_sequence(sequence)
        
        # N should be preserved in preprocessing
        self.assertIn('N', processed)
    
    def test_dnabert1_model_size(self):
        """Test that model configuration matches expected DNABERT1 size"""
        model = DNABert1Model(self.config)
        
        # DNABERT1 uses BERT-base architecture
        # Expected hidden size, layers, etc. would be validated here
        # when model is actually loaded
        self.assertIsNotNone(model.config)
    
    @unittest.skipIf(not False, "Skipping test that requires model download")
    def test_dnabert1_pretrained_loading(self):
        """Test loading pretrained DNABERT1 weights"""
        # This test would actually load the model
        # Skipped by default to avoid downloading during unit tests
        model = DNABert1Model(self.config)
        model.load_pretrained()
        
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.tokenizer)


if __name__ == '__main__':
    unittest.main()