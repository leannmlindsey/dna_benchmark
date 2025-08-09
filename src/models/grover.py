"""
GROVER Model Implementation
Based on: https://zenodo.org/records/13135894 (tutorials and weights)
Paper: DNA language model GROVER learns sequence context in the human genome
HuggingFace: PoetschLab/GROVER
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .base_model import BaseDNAModel

logger = logging.getLogger(__name__)


class GroverModel(BaseDNAModel):
    """GROVER implementation for human genome sequence analysis"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('pretrained_path', 'PoetschLab/GROVER')
        self.max_length = config.get('max_length', 510)  # GROVER uses 510 tokens
        self.num_labels = config.get('num_labels', 2)
        
        # Model and tokenizer will be loaded in load_pretrained
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GROVER specific parameters
        self.bpe_vocab_size = config.get('bpe_vocab_size', 5000)  # BPE vocabulary size
        
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained GROVER model"""
        model_path = path or self.model_name
        
        try:
            # Load tokenizer - GROVER uses custom BPE tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Load model
            if self.num_labels > 1:  # Classification task
                # Load base model and add classification head
                base_model = AutoModelForMaskedLM.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                self.model = GroverForSequenceClassification(
                    base_model, 
                    num_labels=self.num_labels
                )
            else:  # Feature extraction
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
        except Exception as e:
            logger.error(f"Could not load pretrained GROVER model from {model_path}: {e}")
            # Try fallback approach
            try:
                self._create_fallback_grover()
            except Exception as e2:
                logger.error(f"Fallback approach also failed: {e2}")
                raise e
        
        self.model.to(self.device)
        logger.info(f"GROVER model loaded from {model_path}")
    
    def _create_fallback_grover(self):
        """Create fallback GROVER model if direct loading fails"""
        # Create custom BPE tokenizer for GROVER
        self.tokenizer = self._create_grover_tokenizer()
        
        # Create BERT-like model with GROVER configuration
        from transformers import BertConfig, BertModel, BertForMaskedLM, BertForSequenceClassification
        
        config = BertConfig(
            vocab_size=self.bpe_vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=1,
            num_labels=self.num_labels if self.num_labels > 1 else None
        )
        
        if self.num_labels > 1:
            self.model = BertForSequenceClassification(config)
        else:
            self.model = BertForMaskedLM(config)
    
    def _create_grover_tokenizer(self):
        """Create BPE tokenizer for GROVER"""
        class GroverBPETokenizer:
            def __init__(self, vocab_size=5000):
                self.vocab_size = vocab_size
                self.vocab = self._create_bpe_vocab()
                
            def _create_bpe_vocab(self):
                """Create BPE vocabulary optimized for human genome"""
                # Start with special tokens
                vocab = {
                    '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
                }
                
                # Add single nucleotides
                bases = ['A', 'T', 'C', 'G', 'N']
                for base in bases:
                    vocab[base] = len(vocab)
                
                # Add common 2-mers, 3-mers, etc. based on frequency in human genome
                # This is a simplified version - real GROVER uses learned BPE
                common_patterns = [
                    'AT', 'TA', 'GC', 'CG', 'AA', 'TT', 'CC', 'GG',
                    'ATA', 'TAT', 'GCG', 'CGC', 'AAA', 'TTT', 'CCC', 'GGG',
                    'ATAT', 'TATA', 'GCGC', 'CGCG', 'AAAA', 'TTTT', 'CCCC', 'GGGG'
                ]
                
                for pattern in common_patterns:
                    if len(vocab) < self.vocab_size:
                        vocab[pattern] = len(vocab)
                
                # Fill remaining vocabulary with random k-mers
                import itertools
                for k in range(2, 8):  # 2-mer to 7-mer
                    if len(vocab) >= self.vocab_size:
                        break
                    for kmer in itertools.product('ATCG', repeat=k):
                        if len(vocab) >= self.vocab_size:
                            break
                        kmer_str = ''.join(kmer)
                        if kmer_str not in vocab:
                            vocab[kmer_str] = len(vocab)
                
                return vocab
            
            def _bpe_tokenize(self, sequence):
                """Simple BPE tokenization"""
                tokens = []
                i = 0
                while i < len(sequence):
                    # Try to find longest match in vocabulary
                    found = False
                    for length in range(min(10, len(sequence) - i), 0, -1):
                        candidate = sequence[i:i+length]
                        if candidate in self.vocab:
                            tokens.append(candidate)
                            i += length
                            found = True
                            break
                    
                    if not found:
                        # Fall back to single character or UNK
                        if sequence[i] in self.vocab:
                            tokens.append(sequence[i])
                        else:
                            tokens.append('[UNK]')
                        i += 1
                
                return tokens
            
            def __call__(self, sequences, max_length=510, padding=True, truncation=True, return_tensors='pt'):
                if isinstance(sequences, str):
                    sequences = [sequences]
                
                all_input_ids = []
                all_attention_masks = []
                
                for sequence in sequences:
                    # Preprocess sequence
                    sequence = sequence.upper()
                    sequence = ''.join([c if c in 'ATCGN' else 'N' for c in sequence])
                    
                    # Tokenize with BPE
                    tokens = ['[CLS]'] + self._bpe_tokenize(sequence) + ['[SEP]']
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
        
        return GroverBPETokenizer(self.bpe_vocab_size)
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for GROVER"""
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
        
        # Tokenize with BPE tokenizer
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
                if hasattr(self.model, 'bert'):  # BERT-based model
                    outputs = self.model.bert(**inputs)
                elif hasattr(self.model, 'base_model'):  # Custom classification wrapper
                    outputs = self.model.base_model.bert(**inputs)
                else:  # Base model
                    outputs = self.model(**inputs, output_hidden_states=True)
                    outputs = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
                
                # Use [CLS] token embedding
                if hasattr(outputs, 'last_hidden_state'):
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    cls_embeddings = outputs[:, 0, :]
                    
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
        from transformers import TrainingArguments, Trainer
        
        # Set model to training mode
        self.model.train()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './grover_finetuned'),
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
            'bpe_vocab_size': self.bpe_vocab_size,
            'model_type': 'grover'
        }
        with open(Path(path) / 'grover_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model"""
        # Load config
        config_path = Path(path) / 'grover_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.bpe_vocab_size = config.get('bpe_vocab_size', 5000)
                self.num_labels = config.get('num_labels', 2)
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except:
            self.tokenizer = self._create_grover_tokenizer()
        
        # Load model
        if self.num_labels > 1:
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    path, trust_remote_code=True
                )
            except:
                # Load base model and add classification head
                base_model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
                self.model = GroverForSequenceClassification(
                    base_model, 
                    num_labels=self.num_labels
                )
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
        
        self.model.to(self.device)


class GroverForSequenceClassification(nn.Module):
    """Wrapper to add classification head to GROVER"""
    
    def __init__(self, base_model, num_labels: int):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        
        # Add classification head
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, **inputs):
        # Get outputs from base model
        if hasattr(self.base_model, 'bert'):
            outputs = self.base_model.bert(**inputs)
        else:
            outputs = self.base_model(**inputs, output_hidden_states=True)
            if hasattr(outputs, 'hidden_states'):
                outputs = type('Outputs', (), {'last_hidden_state': outputs.hidden_states[-1]})()
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Apply classification head
        logits = self.classifier(pooled_output)
        
        return type('Outputs', (), {'logits': logits})()
    
    def save_pretrained(self, path: str):
        """Save the model"""
        self.base_model.save_pretrained(path)
        
        # Save classification head separately
        torch.save(self.classifier.state_dict(), Path(path) / 'classifier_head.pt')
        
        # Save config
        import json
        config = {'num_labels': self.num_labels}
        with open(Path(path) / 'classification_config.json', 'w') as f:
            json.dump(config, f)
