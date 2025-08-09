"""
GENA-LM Model Implementation
Based on: https://github.com/AIRI-Institute/GENA_LM
Paper: GENA-LM: a family of open-source foundational DNA language models for long sequences
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


class GenaLMModel(BaseDNAModel):
    """GENA-LM implementation for long DNA sequence analysis"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('pretrained_path', 'AIRI-Institute/gena-lm-bert-base-t2t')
        self.max_length = config.get('max_length', 4500)  # GENA-LM supports long sequences
        self.num_labels = config.get('num_labels', 2)
        
        # GENA-LM specific parameters
        self.model_variant = config.get('model_variant', 'bert-base')  # bert-base, bigbird-base, etc.
        self.use_sparse_attention = config.get('use_sparse_attention', False)
        
        # Model and tokenizer will be loaded in load_pretrained
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained GENA-LM model"""
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
            logger.error(f"Could not load pretrained GENA-LM model from {model_path}: {e}")
            # Try fallback approach with manual loading
            try:
                self._load_gena_lm_direct(model_path)
            except Exception as e2:
                logger.error(f"Direct loading also failed: {e2}")
                raise e
        
        self.model.to(self.device)
        logger.info(f"GENA-LM model loaded from {model_path}")
    
    def _load_gena_lm_direct(self, model_path: str) -> None:
        """Direct loading approach for GENA-LM"""
        try:
            # Try to use the custom GENA-LM configuration
            from transformers import BertConfig, BertModel, BertForSequenceClassification
            
            # Create GENA-LM specific tokenizer
            self.tokenizer = self._create_gena_lm_tokenizer()
            
            # Load model configuration
            config = BertConfig(
                vocab_size=self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 4106,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=self.max_length,
                type_vocab_size=1,
                num_labels=self.num_labels if self.num_labels > 1 else None
            )
            
            # Create model
            if self.num_labels > 1:
                self.model = BertForSequenceClassification(config)
            else:
                self.model = BertModel(config)
                
        except Exception as e:
            logger.error(f"Direct GENA-LM loading failed: {e}")
            raise
    
    def _create_gena_lm_tokenizer(self):
        """Create BPE tokenizer for GENA-LM"""
        class GenaLMTokenizer:
            def __init__(self, max_length=4500):
                self.max_length = max_length
                self.vocab = self._create_bpe_vocab()
                self.vocab_size = len(self.vocab)
                
            def _create_bpe_vocab(self):
                """Create BPE vocabulary for GENA-LM"""
                # GENA-LM uses BPE with special tokens
                vocab = {
                    '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4
                }
                
                # Add single nucleotides
                bases = ['A', 'T', 'C', 'G', 'N']
                for base in bases:
                    vocab[base] = len(vocab)
                
                # Add common patterns learned from T2T genome
                # This is a simplified version - real GENA-LM uses learned BPE
                import itertools
                
                # Add 2-mers to 9-mers (median token length is 9 bp)
                for k in range(2, 10):
                    for kmer in itertools.product('ATCG', repeat=k):
                        kmer_str = ''.join(kmer)
                        if len(vocab) < 4100:  # Keep vocab size reasonable
                            vocab[kmer_str] = len(vocab)
                        else:
                            break
                    if len(vocab) >= 4100:
                        break
                
                return vocab
            
            def _simple_bpe(self, sequence):
                """Simple BPE tokenization"""
                tokens = []
                i = 0
                while i < len(sequence):
                    # Try to find longest match in vocabulary
                    found = False
                    for length in range(min(9, len(sequence) - i), 0, -1):  # Max 9-mer
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
            
            def __call__(self, sequences, max_length=None, padding=True, truncation=True, return_tensors='pt'):
                if isinstance(sequences, str):
                    sequences = [sequences]
                
                if max_length is None:
                    max_length = self.max_length
                
                all_input_ids = []
                all_attention_masks = []
                
                for sequence in sequences:
                    # Preprocess sequence
                    sequence = sequence.upper()
                    sequence = ''.join([c if c in 'ATCGN' else 'N' for c in sequence])
                    
                    # Tokenize with BPE
                    tokens = ['[CLS]'] + self._simple_bpe(sequence) + ['[SEP]']
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
        
        return GenaLMTokenizer(self.max_length)
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for GENA-LM"""
        # Convert to uppercase and handle degenerate bases
        sequence = sequence.upper()
        valid_bases = set('ATCGN')
        sequence = ''.join([base if base in valid_bases else 'N' for base in sequence])
        return sequence
    
    def _tokenize_sequences(self, sequences: Union[str, List[str]], max_length: Optional[int] = None) -> Dict:
        """Tokenize DNA sequences"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Preprocess sequences
        processed_sequences = [self._preprocess_sequence(seq) for seq in sequences]
        
        # Use custom max_length if provided
        actual_max_length = max_length or self.max_length
        
        # Tokenize with BPE tokenizer
        return self.tokenizer(
            processed_sequences,
            max_length=actual_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
    
    def get_embeddings(self, sequences: Union[str, List[str]], layer: int = -1) -> torch.Tensor:
        """Get sequence embeddings from specified layer"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        self.model.eval()
        embeddings = []
        
        # Process in batches to handle memory constraints (GENA-LM can be large)
        batch_size = 4 if self.max_length > 1000 else 8
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                # Tokenize batch - may need to use shorter sequences for memory
                try:
                    inputs = self._tokenize_sequences(batch_sequences)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Try with shorter sequences
                        logger.warning("Memory issue, trying with shorter sequences")
                        inputs = self._tokenize_sequences(batch_sequences, max_length=min(1000, self.max_length))
                    else:
                        raise e
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                if hasattr(self.model, 'bert'):  # Classification model
                    outputs = self.model.bert(**inputs, output_hidden_states=True)
                else:  # Base model
                    outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get embeddings from specified layer
                if hasattr(outputs, 'hidden_states'):
                    hidden_states = outputs.hidden_states[layer]
                else:
                    hidden_states = outputs.last_hidden_state
                
                # Use [CLS] token embedding
                cls_embeddings = hidden_states[:, 0, :]
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
        batch_size = 4 if self.max_length > 1000 else 8
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                # Tokenize batch
                try:
                    inputs = self._tokenize_sequences(batch_sequences)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        inputs = self._tokenize_sequences(batch_sequences, max_length=min(1000, self.max_length))
                    else:
                        raise e
                
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
        
        # Adjust batch size based on sequence length
        default_batch_size = 4 if self.max_length > 1000 else 8
        
        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer if hasattr(self.tokenizer, 'pad_token') else None,
            padding=True,
            return_tensors='pt'
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './gena_lm_finetuned'),
            num_train_epochs=kwargs.get('num_epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', default_batch_size),
            per_device_eval_batch_size=kwargs.get('eval_batch_size', default_batch_size),
            warmup_steps=kwargs.get('warmup_steps', 500),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir=kwargs.get('logging_dir', './logs'),
            logging_steps=kwargs.get('logging_steps', 100),
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            learning_rate=kwargs.get('learning_rate', 3e-5),
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 2),
            dataloader_drop_last=True,  # Help with memory consistency
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
            data_collator=data_collator,
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
            'model_variant': self.model_variant,
            'use_sparse_attention': self.use_sparse_attention,
            'model_type': 'gena_lm'
        }
        with open(Path(path) / 'gena_lm_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model"""
        # Load config
        config_path = Path(path) / 'gena_lm_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.max_length = config.get('max_length', 4500)
                self.num_labels = config.get('num_labels', 2)
                self.model_variant = config.get('model_variant', 'bert-base')
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except:
            self.tokenizer = self._create_gena_lm_tokenizer()
        
        # Load model
        if self.num_labels > 1:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
        
        self.model.to(self.device)
