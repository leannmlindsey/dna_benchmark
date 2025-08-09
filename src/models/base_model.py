"""
Base DNA Model Class
Abstract base class that defines the interface for all DNA language models
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class BaseDNAModel(ABC):
    """Abstract base class for DNA language models"""
    
    def __init__(self, config: Dict):
        """
        Initialize the DNA model
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Common configuration parameters
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        self.model_name = config.get('model_name', self.__class__.__name__)
        
    @abstractmethod
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """
        Load pretrained model weights
        
        Args:
            path: Path to model weights or HuggingFace model name
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """
        Get embeddings for DNA sequences
        
        Args:
            sequences: Single sequence or list of sequences
            
        Returns:
            Tensor of shape (batch_size, hidden_size) containing embeddings
        """
        pass
    
    @abstractmethod
    def predict(self, sequences: Union[str, List[str]]) -> Dict:
        """
        Make predictions on DNA sequences
        
        Args:
            sequences: Single sequence or list of sequences
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        pass
    
    @abstractmethod
    def fine_tune(self, train_dataset, val_dataset=None, **kwargs) -> Dict:
        """
        Fine-tune the model on a downstream task
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the fine-tuned model
        
        Args:
            path: Directory to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a fine-tuned model
        
        Args:
            path: Directory containing the saved model
        """
        pass
    
    def preprocess_sequence(self, sequence: str) -> str:
        """
        Preprocess a DNA sequence (default implementation)
        
        Args:
            sequence: Raw DNA sequence
            
        Returns:
            Preprocessed sequence
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Replace invalid characters with N
        valid_bases = set('ATCGN')
        sequence = ''.join([base if base in valid_bases else 'N' for base in sequence])
        
        return sequence
    
    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate a DNA sequence
        
        Args:
            sequence: DNA sequence to validate
            
        Returns:
            True if sequence is valid, False otherwise
        """
        if not sequence:
            return False
        
        # Check if sequence contains only valid DNA bases
        valid_bases = set('ATCGNRYSWKMBDHV')  # Include degenerate bases
        return all(base.upper() in valid_bases for base in sequence)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'model_class': self.__class__.__name__,
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'device': str(self.device),
            'config': self.config
        }
    
    def get_sequence_stats(self, sequences: Union[str, List[str]]) -> Dict:
        """
        Get statistics about input sequences
        
        Args:
            sequences: Single sequence or list of sequences
            
        Returns:
            Dictionary containing sequence statistics
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        lengths = [len(seq) for seq in sequences]
        
        # Count nucleotides
        base_counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0, 'N': 0, 'Other': 0}
        total_bases = 0
        
        for seq in sequences:
            seq = seq.upper()
            for base in seq:
                if base in base_counts:
                    base_counts[base] += 1
                else:
                    base_counts['Other'] += 1
                total_bases += 1
        
        # Calculate GC content
        gc_count = base_counts['G'] + base_counts['C']
        gc_content = gc_count / total_bases if total_bases > 0 else 0
        
        return {
            'num_sequences': len(sequences),
            'total_length': sum(lengths),
            'mean_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'base_counts': base_counts,
            'gc_content': gc_content,
            'sequences_too_long': sum(1 for length in lengths if length > self.max_length)
        }
    
    def split_long_sequences(self, sequences: Union[str, List[str]], 
                           overlap: int = 100) -> Tuple[List[str], List[int]]:
        """
        Split sequences that are longer than max_length into smaller chunks
        
        Args:
            sequences: Single sequence or list of sequences
            overlap: Number of bases to overlap between chunks
            
        Returns:
            Tuple of (chunked_sequences, chunk_counts_per_sequence)
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        
        chunked_sequences = []
        chunk_counts = []
        
        for seq in sequences:
            if len(seq) <= self.max_length:
                chunked_sequences.append(seq)
                chunk_counts.append(1)
            else:
                # Split into overlapping chunks
                chunks = []
                step = self.max_length - overlap
                
                for i in range(0, len(seq), step):
                    chunk = seq[i:i + self.max_length]
                    if len(chunk) >= 100:  # Only keep chunks with sufficient length
                        chunks.append(chunk)
                
                chunked_sequences.extend(chunks)
                chunk_counts.append(len(chunks))
        
        return chunked_sequences, chunk_counts
    
    def aggregate_chunk_predictions(self, predictions: np.ndarray, 
                                  chunk_counts: List[int],
                                  method: str = 'mean') -> np.ndarray:
        """
        Aggregate predictions from sequence chunks
        
        Args:
            predictions: Array of predictions from chunks
            chunk_counts: Number of chunks per original sequence
            method: Aggregation method ('mean', 'max', 'vote')
            
        Returns:
            Aggregated predictions for original sequences
        """
        aggregated = []
        start_idx = 0
        
        for count in chunk_counts:
            if count == 1:
                aggregated.append(predictions[start_idx])
            else:
                chunk_preds = predictions[start_idx:start_idx + count]
                
                if method == 'mean':
                    if len(chunk_preds.shape) > 1:  # Probabilities
                        agg_pred = np.mean(chunk_preds, axis=0)
                    else:  # Classes
                        # Convert to probabilities and average
                        num_classes = len(np.unique(predictions))
                        probs = np.eye(num_classes)[chunk_preds]
                        agg_pred = np.argmax(np.mean(probs, axis=0))
                elif method == 'max':
                    if len(chunk_preds.shape) > 1:
                        agg_pred = chunk_preds[np.argmax(np.max(chunk_preds, axis=1))]
                    else:
                        agg_pred = chunk_preds[0]  # First chunk
                elif method == 'vote':
                    if len(chunk_preds.shape) > 1:
                        # Vote on argmax
                        votes = np.argmax(chunk_preds, axis=1)
                        agg_pred = np.bincount(votes).argmax()
                    else:
                        agg_pred = np.bincount(chunk_preds).argmax()
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")
                
                aggregated.append(agg_pred)
            
            start_idx += count
        
        return np.array(aggregated)
    
    def to(self, device):
        """Move model to device"""
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode"""
        if self.model is not None:
            self.model.train()
        return self
    
    def __repr__(self):
        return f"{self.__class__.__name__}(max_length={self.max_length}, num_labels={self.num_labels})"

