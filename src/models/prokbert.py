"""
prokBERT Model Implementation
Based on: https://github.com/nbrg-ppcu/prokbert
Paper: ProkBERT family: genomic language models for microbiome applications
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .base_model import BaseDNAModel

logger = logging.getLogger(__name__)


class ProkBERTModel(BaseDNAModel):
    """prokBERT implementation for microbiome sequence analysis"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('pretrained_path', 'neuralbioinfo/prokbert-mini')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        self.kmer = config.get('kmer', 6)
        self.shift = config.get('shift', 1)
        
        # Model and tokenizer will be loaded in load_pretrained
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained prokBERT model"""
        model_path = path or self.model_name
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Load model
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
                
        except Exception as e:
            logger.error(f"Could not load pretrained prokBERT model from {model_path}: {e}")
            # Try alternative approach using direct import
            try:
                self._load_prokbert_direct(model_path)
            except Exception as e2:
                logger.error(f"Alternative loading also failed: {e2}")
                raise e
        
        self.model.to(self.device)
        logger.info(f"prokBERT model loaded from {model_path}")
    
    def _load_prokbert_direct(self, model_path: str) -> None:
        """Direct loading approach for prokBERT"""
        try:
            # Try loading with prokbert package if available
            import prokbert
            from prokbert.prokbert_seqloader import SequenceDataset
            
            # Create custom tokenizer for prokBERT
            self.tokenizer = self._create_prokbert_tokenizer()
            
            # Load model directly
            if self.num_labels > 1:
                from transformers import BertForSequenceClassification, BertConfig
                config = BertConfig.from_pretrained(model_path)
                config.num_labels = self.num_labels
                self.model = BertForSequenceClassification(config)
            else:
                self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                
        except ImportError:
            logger.warning("prokbert package not available, using fallback approach")
            self._create_fallback_model()
    
    def _create_prokbert_tokenizer(self):
        """Create LCA tokenizer for prokBERT"""
        class LCATokenizer:
            def __init__(self, kmer=6, shift=1):
                self.kmer = kmer
                self.shift = shift
                self.vocab = self._create_vocab()
                
            def _create_vocab(self):
                """Create vocabulary with LCA approach"""
                bases = ['A', 'T', 'C', 'G', 'N']
                vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
                
                # Generate k-mers
                def generate_kmers(k):
                    if k == 1:
                        return bases
                    else:
                        prev_kmers = generate_kmers(k-1)
                        return [prev + base for prev in prev_kmers for base in bases]
                
                kmers = generate_kmers(self.kmer)
                for i, kmer in enumerate(kmers):
                    vocab[kmer] = len(vocab)
                    
                return vocab
            
            def tokenize(self, sequence):
                """Tokenize with LCA approach"""
                sequence = sequence.upper()
                tokens = []
                
                for i in range(0, len(sequence) - self.kmer + 1, self.shift):
                    kmer = sequence[i:i + self.kmer]
                    # Handle degenerate bases
                    kmer = ''.join([c if c in 'ATCGN' else 'N' for c in kmer])
                    tokens.append(kmer)
                
                return tokens
            
            def __call__(self, sequences, max_length=512, padding=True, truncation=True, return_tensors='pt'):
                if isinstance(sequences, str):
                    sequences = [sequences]
                
                all_input_ids = []
                all_attention_masks = []
                
                for sequence in sequences:
                    tokens = ['[CLS]'] + self.tokenize(sequence) + ['[SEP]']
                    input_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
                    
                    # Truncation
                    if truncation and len(input_ids) > max_length:
                        input_ids = input_ids[:max_length-1] + [self.vocab['[SEP]']]
                    
                    # Padding
                    attention_mask = [1] * len(input_ids)
                    if padding and len(input_ids) < max_length:
                        pad_length = max_length - len(input_ids)
                        input_ids.extend([self.vocab['[PAD]']] * pad_length)
                        attention_mask.extend([0] * pad_length)
                    
                    all_input_ids.append(input_ids)
                    all_attention_masks.append(attention_mask)
                
                result = {
                    'input_ids': all_input_ids,
                    'attention_mask': all_attention_masks
                }
                
                if return_tensors == 'pt':
                    result = {k: torch.tensor(v) for k, v in result.items()}
                
                return result
        
        return LCATokenizer(self.kmer, self.shift)
    
    def _create_fallback_model(self):
        """Create fallback model if direct loading fails"""
        from transformers import BertConfig, BertModel, BertForSequenceClassification
        
        # Create config
        config = BertConfig(
            vocab_size=len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else 5000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            num_labels=self.num_labels if self.num_labels > 1 else None
        )
        
        # Create model
        if self.num_labels > 1:
            self.model = BertForSequenceClassification(config)
        else:
            self.model = BertModel(config)
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for prokBERT"""
        # Convert to uppercase and handle degenerate bases
        sequence = sequence.upper()
        valid_bases = set('ATCGN')
        sequence = ''.join([base if base in valid_bases else 'N' for base in sequence])
        return sequence
    
    def _tokenize_sequences(self, sequences: Union[str, List[str]]) -> Dict:
        """Tokenize DNA sequences"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Preprocess sequences
        processed_sequences = [self._preprocess_sequence(seq) for seq in sequences]
        
        # Tokenize
        return self.tokenizer(
            processed_sequences,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
    
    def get_embeddings(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """Get sequence embeddings"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        self.model.eval()
        embeddings = []
        
        # Process in batches
        batch_size = 8
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                # Tokenize batch
                inputs = self._tokenize_sequences(batch_sequences)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                if hasattr(self.model, 'bert'):  # Classification model
                    outputs = self.model.bert(**inputs)
                else:  # Base model
                    outputs = self.model(**inputs)
                
                # Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def predict(self, sequences: Union[str, List[str]]) -> Dict:
        """Make predictions on sequences"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        if self.num_labels <= 1:
            raise ValueError("Model not configured for classification. Use get_embeddings instead.")
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        batch_size = 8
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                # Tokenize batch
                inputs = self._tokenize_sequences(batch_sequences)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_predictions.append(preds.cpu().numpy())
                all_probabilities.append(probs.cpu().numpy())
        
        return {
            'predictions': np.concatenate(all_predictions),
            'probabilities': np.concatenate(all_probabilities, axis=0)
        }
    
    def fine_tune(self, train_dataset, val_dataset=None, **kwargs) -> Dict:
        """Fine-tune the model on downstream task"""
        from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
        
        # Set model to training mode
        self.model.train()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './prokbert_finetuned'),
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
            fp16=torch.cuda.is_available(),
        )
        
        # Custom compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted'),
                'precision': precision_score(labels, predictions, average='weighted'),
                'recall': recall_score(labels, predictions, average='weighted'),
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None,
        )
        
        # Train
        train_result = trainer.train()
        
        return {
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'train_loss': train_result.metrics['train_loss'],
            'log_history': trainer.state.log_history
        }
    
    def save_model(self, path: str) -> None:
        """Save the fine-tuned model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        
        # Save tokenizer if it's HF tokenizer
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(path)
        
        # Save additional config
        import json
        config = {
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'kmer': self.kmer,
            'shift': self.shift,
            'model_type': 'prokbert'
        }
        with open(Path(path) / 'prokbert_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model"""
        # Load config
        config_path = Path(path) / 'prokbert_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.kmer = config.get('kmer', 6)
                self.shift = config.get('shift', 1)
                self.num_labels = config.get('num_labels', 2)
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except:
            self.tokenizer = self._create_prokbert_tokenizer()
        
        # Load model
        if self.num_labels > 1:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
        
        self.model.to(self.device)
