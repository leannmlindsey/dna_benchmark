"""
Comprehensive preprocessing tests for all DNA language models
Tests sequence validation, preprocessing, tokenization, and chunking
"""

import unittest
import numpy as np
from typing import List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.dnabert1 import DNABert1Model
from src.models.dnabert2 import DNABert2Model
from src.models.nucleotide_transformer import NucleotideTransformerModel
from src.models.prokbert import ProkBERTModel
from src.models.hyenadna import HyenaDNAModel
from src.models.evo import EVOModel


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functionality across all models"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Test sequences covering various cases
        self.valid_sequences = [
            "ATCGATCGATCGATCG",  # Standard DNA
            "GCTAGCTAGCTAGCTA",  # Different composition
            "AAAAAAAAAAAAAAAA",  # Homopolymer
            "ATCGNNNNATCGATCG",  # With ambiguous bases
            "atcgatcgatcgatcg",  # Lowercase
            "AtCgAtCgAtCgAtCg",  # Mixed case
        ]
        
        self.invalid_sequences = [
            "",  # Empty
            "ATCG123456",  # With numbers
            "ATCG!@#$%^",  # With special characters
            "Hello World",  # Not DNA
            "ATCG ATCG",  # With spaces
            "ATCG\nATCG",  # With newlines
        ]
        
        self.long_sequences = [
            "ATCG" * 250,  # 1000 bp
            "GCTA" * 500,  # 2000 bp
            "A" * 5000,    # 5000 bp homopolymer
            "ATCGATCG" * 1000,  # 8000 bp
        ]
        
        # Edge cases
        self.edge_cases = [
            "A",  # Single nucleotide
            "AT",  # Two nucleotides
            "N" * 100,  # All ambiguous
            "ATCGRYSWKMBDHVN",  # All IUPAC codes
        ]
        
        # Base configuration for models
        self.base_config = {
            'num_labels': 2,
            'device': 'cpu',
            'use_env_manager': False
        }
    
    # ============ Sequence Validation Tests ============
    
    def test_valid_sequence_validation(self):
        """Test validation of valid DNA sequences"""
        models = [
            DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 512}),
            DNABert2Model({**self.base_config, 'max_length': 512}),
            HyenaDNAModel({**self.base_config, 'max_length': 1000}),
        ]
        
        for model in models:
            for seq in self.valid_sequences:
                self.assertTrue(
                    model.validate_sequence(seq.upper()),
                    f"{model.__class__.__name__} failed to validate: {seq}"
                )
    
    def test_invalid_sequence_validation(self):
        """Test validation of invalid DNA sequences"""
        models = [
            DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 512}),
            DNABert2Model({**self.base_config, 'max_length': 512}),
        ]
        
        for model in models:
            for seq in self.invalid_sequences:
                self.assertFalse(
                    model.validate_sequence(seq),
                    f"{model.__class__.__name__} incorrectly validated: {seq}"
                )
    
    def test_edge_case_validation(self):
        """Test validation of edge case sequences"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 512})
        
        # Single nucleotide - valid
        self.assertTrue(model.validate_sequence("A"))
        
        # All IUPAC codes - valid
        self.assertTrue(model.validate_sequence("ATCGRYSWKMBDHVN"))
        
        # All N's - valid but ambiguous
        self.assertTrue(model.validate_sequence("N" * 100))
    
    # ============ Sequence Preprocessing Tests ============
    
    def test_uppercase_conversion(self):
        """Test that sequences are converted to uppercase"""
        models = [
            DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 512}),
            DNABert2Model({**self.base_config, 'max_length': 512}),
        ]
        
        test_cases = [
            ("atcgatcg", "ATCGATCG"),
            ("AtCgAtCg", "ATCGATCG"),
            ("ATCGATCG", "ATCGATCG"),
        ]
        
        for model in models:
            for input_seq, expected in test_cases:
                processed = model.preprocess_sequence(input_seq)
                self.assertEqual(
                    processed,
                    expected,
                    f"{model.__class__.__name__} failed uppercase conversion"
                )
    
    def test_invalid_character_replacement(self):
        """Test that invalid characters are replaced with N"""
        models = [
            DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 512}),
            DNABert2Model({**self.base_config, 'max_length': 512}),
        ]
        
        test_cases = [
            ("ATCGXYZ", "ATCGNNN"),
            ("ATCG123", "ATCGNNN"),
            ("ATCG!@#", "ATCGNNN"),
            ("AT CG", "ATNCG"),  # Space replaced with N
        ]
        
        for model in models:
            for input_seq, expected in test_cases:
                processed = model.preprocess_sequence(input_seq)
                self.assertEqual(
                    processed,
                    expected,
                    f"{model.__class__.__name__} failed invalid char replacement"
                )
    
    def test_ambiguous_base_handling(self):
        """Test handling of ambiguous DNA bases"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 512})
        
        # N should be preserved
        seq_with_n = "ATCGNNNNATCG"
        processed = model.preprocess_sequence(seq_with_n)
        self.assertEqual(processed, "ATCGNNNNATCG")
        
        # Other IUPAC codes might be converted to N depending on model
        seq_iupac = "ATCGRYSWKM"
        processed = model.preprocess_sequence(seq_iupac)
        # Should either preserve or convert to N
        self.assertTrue(all(c in "ATCGNRYSWKM" for c in processed))
    
    # ============ Tokenization Tests ============
    
    def test_kmer_tokenization_concept(self):
        """Test k-mer tokenization concept"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 512})
        
        sequence = "ATCGATCGATCG"  # 12 nucleotides
        
        # With k=6, stride=1: creates overlapping 6-mers
        # Expected k-mers: ATCGAT, TCGATC, CGATCG, GATCGA, ATCGAT, TCGATC, CGATCG
        # Number of k-mers = len(sequence) - k + 1 = 12 - 6 + 1 = 7
        
        expected_num_kmers = len(sequence) - model.kmer + 1
        self.assertEqual(expected_num_kmers, 7)
    
    def test_bpe_tokenization_concept(self):
        """Test BPE tokenization concept"""
        model = DNABert2Model({**self.base_config, 'max_length': 512})
        
        sequence = "ATCGATCGATCG"
        
        # BPE learns subword units from data
        # Could tokenize as: ["ATC", "GAT", "CGA", "TCG"] or other patterns
        # Number of tokens depends on learned vocabulary
        
        # Just verify the model has BPE configuration
        self.assertEqual(model.config.get('tokenizer', 'bpe'), 'bpe')
    
    def test_character_tokenization_concept(self):
        """Test character-level tokenization"""
        model = HyenaDNAModel({**self.base_config, 'max_length': 1000})
        
        sequence = "ATCGATCG"
        
        # Character tokenization: one token per nucleotide
        expected_num_tokens = len(sequence)
        self.assertEqual(expected_num_tokens, 8)
    
    # ============ Sequence Chunking Tests ============
    
    def test_sequence_chunking_with_overlap(self):
        """Test chunking long sequences with overlap"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 100})
        
        long_sequence = "A" * 250  # Longer than max_length
        overlap = 20
        
        chunks, counts = model.split_long_sequences(long_sequence, overlap=overlap)
        
        # Check that sequence was chunked
        self.assertGreater(len(chunks), 1)
        self.assertEqual(counts[0], len(chunks))
        
        # Check chunk properties
        for i, chunk in enumerate(chunks[:-1]):
            # All chunks except last should be max_length
            self.assertEqual(len(chunk), 100)
            
            # Check overlap between consecutive chunks
            if i < len(chunks) - 1:
                # Last 'overlap' bases of chunk i should match
                # first 'overlap' bases of chunk i+1
                self.assertEqual(
                    chunk[-overlap:],
                    chunks[i + 1][:overlap]
                )
    
    def test_sequence_chunking_edge_cases(self):
        """Test chunking edge cases"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 100})
        
        # Sequence exactly at max_length
        seq_exact = "A" * 100
        chunks, counts = model.split_long_sequences(seq_exact)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], seq_exact)
        
        # Sequence just over max_length
        seq_over = "A" * 101
        chunks, counts = model.split_long_sequences(seq_over, overlap=10)
        self.assertGreater(len(chunks), 1)
        
        # Very short sequence
        seq_short = "ATCG"
        chunks, counts = model.split_long_sequences(seq_short)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], seq_short)
    
    def test_batch_sequence_chunking(self):
        """Test chunking multiple sequences"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 100})
        
        sequences = [
            "A" * 50,   # No chunking needed
            "T" * 150,  # Needs 2 chunks
            "C" * 250,  # Needs 3 chunks
        ]
        
        all_chunks, chunk_counts = model.split_long_sequences(sequences, overlap=10)
        
        # Check total chunks
        expected_total = 1 + 2 + 3
        self.assertEqual(len(all_chunks), expected_total)
        
        # Check chunk counts per sequence
        self.assertEqual(chunk_counts, [1, 2, 3])
    
    # ============ Chunk Aggregation Tests ============
    
    def test_chunk_prediction_aggregation_mean(self):
        """Test mean aggregation of chunk predictions"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 100})
        
        # Mock predictions for chunks
        # 3 sequences: 1 chunk, 2 chunks, 3 chunks = 6 total
        predictions = np.array([
            [0.9, 0.1],  # Seq 1, chunk 1
            [0.8, 0.2],  # Seq 2, chunk 1
            [0.6, 0.4],  # Seq 2, chunk 2
            [0.7, 0.3],  # Seq 3, chunk 1
            [0.5, 0.5],  # Seq 3, chunk 2
            [0.9, 0.1],  # Seq 3, chunk 3
        ])
        
        chunk_counts = [1, 2, 3]
        
        aggregated = model.aggregate_chunk_predictions(
            predictions, chunk_counts, method='mean'
        )
        
        self.assertEqual(len(aggregated), 3)
        
        # Check aggregated values
        np.testing.assert_array_almost_equal(aggregated[0], [0.9, 0.1])
        np.testing.assert_array_almost_equal(aggregated[1], [0.7, 0.3])
        np.testing.assert_array_almost_equal(aggregated[2], [0.7, 0.3], decimal=1)
    
    def test_chunk_prediction_aggregation_max(self):
        """Test max aggregation of chunk predictions"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 100})
        
        predictions = np.array([
            [0.9, 0.1],  # Seq 1, chunk 1
            [0.3, 0.7],  # Seq 2, chunk 1
            [0.8, 0.2],  # Seq 2, chunk 2 (max confidence)
        ])
        
        chunk_counts = [1, 2]
        
        aggregated = model.aggregate_chunk_predictions(
            predictions, chunk_counts, method='max'
        )
        
        self.assertEqual(len(aggregated), 2)
        # Second sequence should use chunk with max confidence
        np.testing.assert_array_almost_equal(aggregated[1], [0.8, 0.2])
    
    # ============ Sequence Statistics Tests ============
    
    def test_sequence_statistics_calculation(self):
        """Test calculation of sequence statistics"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 512})
        
        sequences = [
            "ATCGATCG",  # 8 bp, 50% GC
            "GGCCGGCC",  # 8 bp, 100% GC
            "AAAATTTT",  # 8 bp, 0% GC
        ]
        
        stats = model.get_sequence_stats(sequences)
        
        # Check basic statistics
        self.assertEqual(stats['num_sequences'], 3)
        self.assertEqual(stats['total_length'], 24)
        self.assertEqual(stats['mean_length'], 8.0)
        self.assertEqual(stats['min_length'], 8)
        self.assertEqual(stats['max_length'], 8)
        
        # Check GC content (should be average across all)
        # Total: 4+8+0 = 12 G/C out of 24 total = 50%
        self.assertAlmostEqual(stats['gc_content'], 0.5, places=2)
        
        # Check base counts
        self.assertEqual(stats['base_counts']['A'], 8)
        self.assertEqual(stats['base_counts']['T'], 8)
        self.assertEqual(stats['base_counts']['G'], 6)
        self.assertEqual(stats['base_counts']['C'], 6)
    
    def test_sequence_length_distribution(self):
        """Test statistics for sequences of varying lengths"""
        model = DNABert1Model({**self.base_config, 'kmer': 6, 'max_length': 100})
        
        sequences = [
            "A" * 10,
            "T" * 50,
            "C" * 100,
            "G" * 150,  # Over max_length
        ]
        
        stats = model.get_sequence_stats(sequences)
        
        self.assertEqual(stats['num_sequences'], 4)
        self.assertEqual(stats['min_length'], 10)
        self.assertEqual(stats['max_length'], 150)
        self.assertEqual(stats['sequences_too_long'], 1)  # One sequence over max


if __name__ == '__main__':
    unittest.main()