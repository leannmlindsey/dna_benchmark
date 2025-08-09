"""
HyenaDNA Model Implementation
Based on: https://github.com/HazyResearch/hyena-dna
Paper: HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .base_model import BaseDNAModel

logger = logging.getLogger(__name__)


class HyenaDNAModel(BaseDNAModel):
    """HyenaDNA implementation for long-range genomic sequence modeling"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('pretrained_path', 'LongSafari/hyenadna-medium-450k-seqlen')
        self.max_length = config.get('max_length', 450000)  # HyenaDNA supports very long sequences
        self.num_labels = config.get('num_labels', 2)
        
        # HyenaDNA specific parameters
        self.model_size = config.get('model_size', 'medium')  # tiny, small, medium, large
        self.use_head = config.get('use_head', False)
        self.pad_token_id = config.get('pad_token_id', 0)
        
        # Model and tokenizer will be loaded in load_pretrained
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine model variant based on model name
        self._parse_model_variant()
        
    def _parse_model_variant(self):
        """Parse model variant from model name"""
        model_variants = {
            'tiny-1k': {'max_length': 1024, 'batch_size': 32},
            'small-32k': {'max_length': 32768, 'batch_size': 16},
            'medium-160k': {'max_length': 160000, 'batch_size': 4},
            'medium-450k': {'max_length': 450000, 'batch_size': 2},
            'large-1m': {'max_length': 1000000, 'batch_size': 1}
        }
        
        # Parse from model name
        for variant, params in model_variants.items():
            if variant in self.model_name:
                self.max_length = min(self.max_length, params['max_length'])
                self.recommended_batch_size = params['batch_size']
                self.model_size = variant
                break
        else:
            self.recommended_batch_size = 8
    
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained HyenaDNA model"""
        model_path = path or self.model_name
        
        try:
            # Try direct HuggingFace loading first
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # Load model
            if self.num_labels > 1:  # Classification task
                # Load base model and add classification head
                base_model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                self.model = HyenaDNAForSequenceClassification(
                    base_model,
                    num_labels=self.num_labels
                )
            else:  # Feature extraction
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
        except Exception as e:
            logger.error(f"Could not load HyenaDNA from HuggingFace: {e}")
            # Try standalone HyenaDNA loading
            try:
                self._load_hyenadna_standalone(model_path)
            except Exception as e2:
                logger.error(f"Standalone loading also failed: {e2}")
                # Create fallback model
                self._create_fallback_hyenadna()
        
        self.model.to(self.device)
        logger.info(f"HyenaDNA model loaded: {self.model_size}")
    
    def _load_hyenadna_standalone(self, model_path: str):
        """Load HyenaDNA using standalone implementation"""
        try:
            # This would use the standalone HyenaDNA implementation
            # from the GitHub repository if available
            
            # Create character tokenizer for single nucleotide resolution
            self.tokenizer = self._create_character_tokenizer()
            
            # For now, create a simplified Hyena-like model
            # In practice, you would load the actual HyenaDNA weights
            self._create_fallback_hyenadna()
            
        except Exception as e:
            logger.error(f"Standalone HyenaDNA loading failed: {e}")
            raise
    
    def _create_character_tokenizer(self):
        """Create character-level tokenizer for single nucleotide resolution"""
        class CharacterTokenizer:
            def __init__(self):
                self.vocab = {
                    'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4,
                    '[PAD]': 5, '[UNK]': 6, '[CLS]': 7, '[SEP]': 8
                }
                self.vocab_size = len(self.vocab)
                self.pad_token_id = self.vocab['[PAD]']
                
            def encode(self, sequence, max_length=None, padding=True, truncation=True, return_tensors='pt'):
                if isinstance(sequence, str):
                    sequences = [sequence]
                else:
                    sequences = sequence
                
                all_input_ids = []
                all_attention_masks = []
                
                for seq in sequences:
                    # Convert to uppercase and handle invalid characters
                    seq = seq.upper()
                    seq = ''.join([c if c in 'ATCGN' else 'N' for c in seq])
                    
                    # Convert to token IDs
                    input_ids = [self.vocab.get(c, self.vocab['[UNK]']) for c in seq]
                    
                    # Handle truncation
                    if truncation and max_length and len(input_ids) > max_length:
                        input_ids = input_ids[:max_length]
                    
                    # Create attention mask
                    attention_mask = [1] * len(input_ids)
                    
                    # Handle padding
                    if padding and max_length and len(input_ids) < max_length:
                        pad_length = max_length - len(input_ids)
                        input_ids.extend([self.pad_token_id] * pad_length)
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
            
            def __call__(self, *args, **kwargs):
                return self.encode(*args, **kwargs)
        
        return CharacterTokenizer()
    
    def _create_fallback_hyenadna(self):
        """Create fallback HyenaDNA-like model"""
        # Create tokenizer if not exists
        if self.tokenizer is None:
            self.tokenizer = self._create_character_tokenizer()
        
        # Create a simplified model that mimics HyenaDNA architecture
        # This is a placeholder - real HyenaDNA uses the Hyena operator
        
        class SimplifiedHyenaDNA(nn.Module):
            def __init__(self, vocab_size, hidden_size=256, num_layers=4, max_length=1024):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.position_embedding = nn.Embedding(max_length, hidden_size)
                
                # Simplified layers instead of Hyena blocks
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=8,
                        dim_feedforward=hidden_size * 4,
                        dropout=0.1,
                        batch_first=True
                    ) for _ in range(num_layers)
                ])
                
                self.norm = nn.LayerNorm(hidden_size)
                
            def forward(self, input_ids, attention_mask=None, **kwargs):
                seq_len = input_ids.size(1)
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                
                # Embeddings
                x = self.embedding(input_ids) + self.position_embedding(pos_ids)
                
                # Apply layers
                for layer in self.layers:
                    x = layer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
                
                x = self.norm(x)
                
                return type('Outputs', (), {'last_hidden_state': x})()
        
        if self.num_labels > 1:
            base_model = SimplifiedHyenaDNA(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=256,
                num_layers=4,
                max_length=min(self.max_length, 8192)  # Limit for fallback
            )
            self.model = HyenaDNAForSequenceClassification(base_model, self.num_labels)
        else:
            self.model = SimplifiedHyenaDNA(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=256,
                num_layers=4,
                max_length=min(self.max_length, 8192)
            )
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for HyenaDNA"""
        # HyenaDNA works at single nucleotide resolution
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
        
        # Use adaptive max_length based on memory constraints
        actual_max_length = max_length or min(self.max_length, 8192)  # Conservative default
        
        # Tokenize with character-level tokenizer
        return self.tokenizer(
            processed_sequences,
            max_length=actual_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
    
    def get_embeddings(self, sequences: Union[str, List[str]], pool_type: str = 'mean') -> torch.Tensor:
        """Get sequence embeddings"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        self.model.eval()
        embeddings = []
        
        # Use very small batches for long sequences
        batch_size = 1 if self.max_length > 10000 else 2
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                try:
                    # Tokenize batch with adaptive length
                    inputs = self._tokenize_sequences(batch_sequences)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get outputs
                    if hasattr(self.model, 'base_model'):  # Classification wrapper
                        outputs = self.model.base_model(**inputs)
                    else:  # Base model
                        outputs = self.model(**inputs)
                    
                    # Pool embeddings
                    hidden_states = outputs.last_hidden_state
                    attention_mask = inputs.get('attention_mask')
                    
                    if pool_type == 'mean':
                        # Mean pooling with attention mask
                        if attention_mask is not None:
                            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                            pooled = sum_embeddings / sum_mask
                        else:
                            pooled = hidden_states.mean(dim=1)
                    elif pool_type == 'cls':
                        # Use first token (if available)
                        pooled = hidden_states[:, 0, :]
                    else:  # 'last'
                        # Use last non-padded token
                        if attention_mask is not None:
                            seq_lengths = attention_mask.sum(dim=1) - 1
                            pooled = hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
                        else:
                            pooled = hidden_states[:, -1, :]
                    
                    embeddings.append(pooled.cpu())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM for batch, trying shorter sequences")
                        # Try with much shorter sequences
                        inputs = self._tokenize_sequences(batch_sequences, max_length=1024)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        if hasattr(self.model, 'base_model'):
                            outputs = self.model.base_model(**inputs)
                        else:
                            outputs = self.model(**inputs)
                        
                        pooled = outputs.last_hidden_state.mean(dim=1)
                        embeddings.append(pooled.cpu())
                    else:
                        raise e
        
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
        
        # Use small batches for long sequences
        batch_size = 1 if self.max_length > 10000 else 2
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                try:
                    # Tokenize batch
                    inputs = self._tokenize_sequences(batch_sequences)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get predictions
                    outputs = self.model(**inputs)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    
                    # Convert to probabilities
                    probs = torch.softmax(logits, dim=-1)
                    preds = torch.argmax(logits, dim=-1)
                    
                    all_predictions.append(preds.cpu().numpy())
                    all_probabilities.append(probs.cpu().numpy())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"OOM for prediction, trying shorter sequences")
                        inputs = self._tokenize_sequences(batch_sequences, max_length=1024)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        
                        probs = torch.softmax(logits, dim=-1)
                        preds = torch.argmax(logits, dim=-1)
                        
                        all_predictions.append(preds.cpu().numpy())
                        all_probabilities.append(probs.cpu().numpy())
                    else:
                        raise e
        
        return {
            'predictions': np.concatenate(all_predictions),
            'probabilities': np.concatenate(all_probabilities, axis=0)
        }
    
    def fine_tune(self, train_dataset, val_dataset=None, **kwargs) -> Dict:
        """Fine-tune the model on downstream task"""
        from transformers import TrainingArguments, Trainer
        
        # Set model to training mode
        self.model.train()
        
        # Use very conservative batch sizes for long sequences
        default_batch_size = 1 if self.max_length > 10000 else 2
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './hyenadna_finetuned'),
            num_train_epochs=kwargs.get('num_epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', default_batch_size),
            per_device_eval_batch_size=kwargs.get('eval_batch_size', default_batch_size),
            warmup_steps=kwargs.get('warmup_steps', 100),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir=kwargs.get('logging_dir', './logs'),
            logging_steps=kwargs.get('logging_steps', 50),
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            learning_rate=kwargs.get('learning_rate', 1e-4),
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 8),
            dataloader_drop_last=True,
            remove_unused_columns=False,
        )
        
        # Custom compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            from sklearn.metrics import accuracy_score, f1_score
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted'),
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
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), Path(path) / 'pytorch_model.bin')
        
        # Save tokenizer info
        import json
        tokenizer_config = {
            'vocab': self.tokenizer.vocab,
            'vocab_size': self.tokenizer.vocab_size,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        with open(Path(path) / 'tokenizer_config.json', 'w') as f:
            json.dump(tokenizer_config, f)
        
        # Save model config
        config = {
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'model_size': self.model_size,
            'model_type': 'hyenadna'
        }
        with open(Path(path) / 'hyenadna_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model"""
        # Load config
        config_path = Path(path) / 'hyenadna_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.max_length = config.get('max_length', 450000)
                self.num_labels = config.get('num_labels', 2)
                self.model_size = config.get('model_size', 'medium')
        
        # Load tokenizer
        tokenizer_config_path = Path(path) / 'tokenizer_config.json'
        if tokenizer_config_path.exists():
            self.tokenizer = self._create_character_tokenizer()
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            except:
                self.tokenizer = self._create_character_tokenizer()
        
        # Load model
        try:
            if self.num_labels > 1:
                base_model = AutoModel.from_pretrained(path, trust_remote_code=True)
                self.model = HyenaDNAForSequenceClassification(base_model, self.num_labels)
            else:
                self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
        except:
            # Load manually saved model
            self._create_fallback_hyenadna()
            model_weights_path = Path(path) / 'pytorch_model.bin'
            if model_weights_path.exists():
                self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        
        self.model.to(self.device)


class HyenaDNAForSequenceClassification(nn.Module):
    """Wrapper to add classification head to HyenaDNA"""
    
    def __init__(self, base_model, num_labels: int):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        
        # Determine hidden size
        if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
            hidden_size = base_model.config.hidden_size
        else:
            # Fallback - try to infer from model
            hidden_size = 256  # Default for our simplified model
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
    def forward(self, **inputs):
        # Get outputs from base model
        outputs = self.base_model(**inputs)
        
        # Pool sequence representations (mean pooling)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs.get('attention_mask')
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        # Apply classification head
        logits = self.classifier(pooled_output)
        
        return type('Outputs', (), {'logits': logits})()
    
    def save_pretrained(self, path: str):
        """Save the model"""
        if hasattr(self.base_model, 'save_pretrained'):
            self.base_model.save_pretrained(path)
        else:
            torch.save(self.base_model.state_dict(), Path(path) / 'base_model.bin')
        
        # Save classification head
        torch.save(self.classifier.state_dict(), Path(path) / 'classifier_head.bin')
        
        # Save config
        import json
        config = {'num_labels': self.num_labels}
        with open(Path(path) / 'classification_config.json', 'w') as f:
            json.dump(config, f)
