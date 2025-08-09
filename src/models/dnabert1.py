"""
DNABERT1 Model Implementation
Based on: https://github.com/jerryji1993/DNABERT
Paper: DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .base_model import BaseDNAModel

logger = logging.getLogger(__name__)


class DNABERT1Tokenizer:
    """Custom tokenizer for DNABERT1 with k-mer tokenization"""
    
    def __init__(self, kmer: int = 6):
        self.kmer = kmer
        self.vocab = self._create_vocab()
        self.vocab_size = len(self.vocab)
        
    def _create_vocab(self):
        """Create vocabulary for k-mer tokenization"""
        bases = ['A', 'T', 'C', 'G', 'N']
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        
        # Generate all possible k-mers
        def generate_kmers(k):
            if k == 1:
                return bases
            else:
                prev_kmers = generate_kmers(k-1)
                return [prev + base for prev in prev_kmers for base in bases]
        
        vocab.extend(generate_kmers(self.kmer))
        return {token: idx for idx, token in enumerate(vocab)}
    
    def tokenize(self, sequence: str) -> List[str]:
        """Convert DNA sequence to k-mer tokens"""
        sequence = sequence.upper().replace('N', 'N')  # Handle degenerate bases
        
        if len(sequence) < self.kmer:
            return ['[UNK]']
        
        tokens = []
        for i in range(len(sequence) - self.kmer + 1):
            kmer = sequence[i:i + self.kmer]
            # Replace any unknown characters with N
            kmer = ''.join([c if c in 'ATCGN' else 'N' for c in kmer])
            tokens.append(kmer)
        
        return tokens
    
    def encode(self, sequence: str, max_length: int = 512, add_special_tokens: bool = True, 
               return_tensors: str = None, padding: bool = True, truncation: bool = True) -> Dict:
        """Encode sequence to token IDs"""
        tokens = self.tokenize(sequence)
        
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        
        # Handle padding and truncation
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.vocab['[SEP]']]
        
        if padding and len(token_ids) < max_length:
            pad_length = max_length - len(token_ids)
            token_ids.extend([self.vocab['[PAD]']] * pad_length)
        
        attention_mask = [1 if token_id != self.vocab['[PAD]'] else 0 for token_id in token_ids]
        
        result = {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
        
        if return_tensors == 'pt':
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result


class DNABert1Model(BaseDNAModel):
    """DNABERT1 implementation for DNA sequence analysis"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.kmer = config.get('kmer', 6)
        self.model_name = config.get('pretrained_path', 'zhihan1996/DNA_bert_6')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        
        # Initialize tokenizer
        self.tokenizer = DNABERT1Tokenizer(kmer=self.kmer)
        
        # Model will be loaded in load_pretrained
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained DNABERT1 model"""
        model_path = path or self.model_name
        
        try:
            # Try to load from HuggingFace
            if self.num_labels > 1:  # Classification task
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=self.num_labels,
                    trust_remote_code=True
                )
            else:  # Feature extraction
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            
            # Try to use HF tokenizer first, fallback to custom
            try:
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                logger.warning("Could not load HF tokenizer, using custom k-mer tokenizer")
                self.hf_tokenizer = None
                
        except Exception as e:
            logger.warning(f"Could not load pretrained model from {model_path}: {e}")
            # Initialize model from scratch
            from transformers import BertConfig, BertForSequenceClassification, BertModel
            
            config = BertConfig(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512,
                num_labels=self.num_labels
            )
            
            if self.num_labels > 1:
                self.model = BertForSequenceClassification(config)
            else:
                self.model = BertModel(config)
        
        self.model.to(self.device)
        logger.info(f"DNABERT1 model loaded with {self.kmer}-mer tokenization")
    
    def _tokenize_sequence(self, sequence: str) -> Dict:
        """Tokenize DNA sequence using appropriate tokenizer"""
        if self.hf_tokenizer is not None:
            # Use HuggingFace tokenizer if available
            return self.hf_tokenizer(
                sequence,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
        else:
            # Use custom k-mer tokenizer
            return self.tokenizer.encode(
                sequence,
                max_length=self.max_length,
                return_tensors='pt'
            )
    
    def get_embeddings(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """Get sequence embeddings"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for sequence in sequences:
                # Tokenize sequence
                inputs = self._tokenize_sequence(sequence)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                if hasattr(self.model, 'bert'):  # Classification model
                    outputs = self.model.bert(**inputs)
                else:  # Base model
                    outputs = self.model(**inputs)
                
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                embeddings.append(cls_embedding.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def predict(self, sequences: Union[str, List[str]]) -> Dict:
        """Make predictions on sequences"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        if self.num_labels <= 1:
            raise ValueError("Model not configured for classification. Use get_embeddings instead.")
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for sequence in sequences:
                # Tokenize sequence
                inputs = self._tokenize_sequence(sequence)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
                
                predictions.append(pred.cpu().numpy())
                probabilities.append(probs.cpu().numpy())
        
        return {
            'predictions': np.concatenate(predictions),
            'probabilities': np.concatenate(probabilities, axis=0)
        }
    
    def fine_tune(self, train_dataset, val_dataset=None, **kwargs) -> Dict:
        """Fine-tune the model on downstream task"""
        from transformers import TrainingArguments, Trainer
        import torch.optim as optim
        
        # Set model to training mode
        self.model.train()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './dnabert1_finetuned'),
            num_train_epochs=kwargs.get('num_epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', 16),
            per_device_eval_batch_size=kwargs.get('eval_batch_size', 16),
            warmup_steps=kwargs.get('warmup_steps', 500),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir=kwargs.get('logging_dir', './logs'),
            logging_steps=kwargs.get('logging_steps', 100),
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            learning_rate=kwargs.get('learning_rate', 5e-5),
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.hf_tokenizer if self.hf_tokenizer else None,
        )
        
        # Train
        trainer.train()
        
        # Return training metrics
        return trainer.state.log_history
    
    def save_model(self, path: str) -> None:
        """Save the fine-tuned model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        if self.hf_tokenizer:
            self.hf_tokenizer.save_pretrained(path)
        
        # Save custom tokenizer config
        import json
        config = {
            'kmer': self.kmer,
            'vocab_size': self.tokenizer.vocab_size,
            'max_length': self.max_length
        }
        with open(Path(path) / 'dnabert1_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model"""
        if self.num_labels > 1:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
        
        try:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(path)
        except:
            pass  # Use custom tokenizer
        
        self.model.to(self.device)
