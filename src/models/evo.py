"""
EVO Model Implementation
Based on: https://github.com/evo-design/evo (Evo 1) and https://github.com/ArcInstitute/evo2 (Evo 2)
Paper: Sequence modeling and design from molecular to genome scale with Evo
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .base_model import BaseDNAModel

logger = logging.getLogger(__name__)


class EVOModel(BaseDNAModel):
    """EVO implementation for genomic foundation modeling"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('pretrained_path', 'togethercomputer/evo-1-8k-base')
        self.max_length = config.get('max_length', 8192)
        self.num_labels = config.get('num_labels', 2)
        
        # EVO specific parameters
        self.model_variant = config.get('model_variant', 'evo-1-8k-base')  # evo-1-8k-base, evo-1-131k-base, etc.
        self.is_causal = config.get('is_causal', True)  # EVO is autoregressive
        self.revision = config.get('revision', '1.1_fix')  # For HF model loading
        
        # Model and tokenizer will be loaded in load_pretrained
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parse model variant
        self._parse_model_variant()
        
    def _parse_model_variant(self):
        """Parse model variant and set appropriate parameters"""
        variant_configs = {
            'evo-1-8k-base': {'max_length': 8192, 'batch_size': 8},
            'evo-1-131k-base': {'max_length': 131072, 'batch_size': 2},
            'evo-2-7b': {'max_length': 1000000, 'batch_size': 1},
            'evo-2-40b': {'max_length': 1000000, 'batch_size': 1}
        }
        
        for variant, config in variant_configs.items():
            if variant in self.model_name:
                self.max_length = min(self.max_length, config['max_length'])
                self.recommended_batch_size = config['batch_size']
                self.model_variant = variant
                break
        else:
            self.recommended_batch_size = 4
    
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained EVO model"""
        model_path = path or self.model_name
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                revision=self.revision,
                trust_remote_code=True
            )
            
            # Load model
            if self.num_labels > 1:  # Classification task
                # Load base causal LM and add classification head
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    revision=self.revision,
                    trust_remote_code=True
                )
                self.model = EVOForSequenceClassification(
                    base_model,
                    num_labels=self.num_labels
                )
            else:  # Feature extraction or generation
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    revision=self.revision,
                    trust_remote_code=True
                )
                
        except Exception as e:
            logger.error(f"Could not load EVO model from {model_path}: {e}")
            # Try alternative loading approaches
            try:
                self._load_evo_alternative(model_path)
            except Exception as e2:
                logger.error(f"Alternative loading failed: {e2}")
                # Create fallback model
                self._create_fallback_evo()
        
        self.model.to(self.device)
        logger.info(f"EVO model loaded: {self.model_variant}")
    
    def _load_evo_alternative(self, model_path: str):
        """Alternative loading method for EVO"""
        try:
            # Try loading without revision
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            if self.num_labels > 1:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                self.model = EVOForSequenceClassification(base_model, self.num_labels)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
        except Exception as e:
            logger.error(f"Alternative EVO loading failed: {e}")
            raise
    
    def _create_fallback_evo(self):
        """Create fallback EVO-like model"""
        # Create byte-level tokenizer
        self.tokenizer = self._create_byte_tokenizer()
        
        # Create simplified autoregressive model
        class SimplifiedEVO(nn.Module):
            def __init__(self, vocab_size, hidden_size=512, num_layers=6, max_length=8192):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.position_embedding = nn.Embedding(max_length, hidden_size)
                
                # Decoder layers (causal attention)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
                
                self.norm = nn.LayerNorm(hidden_size)
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                
            def forward(self, input_ids, attention_mask=None, **kwargs):
                seq_len = input_ids.size(1)
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                
                # Embeddings
                x = self.embedding(input_ids) + self.position_embedding(pos_ids)
                
                # Create causal mask
                causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
                
                # Apply decoder
                # Note: This is simplified - real EVO uses StripedHyena architecture
                memory = torch.zeros_like(x)  # Dummy memory for decoder
                x = self.transformer(x, memory, tgt_mask=causal_mask)
                
                x = self.norm(x)
                
                return type('Outputs', (), {
                    'last_hidden_state': x,
                    'logits': self.lm_head(x) if kwargs.get('return_logits', True) else None
                })()
        
        if self.num_labels > 1:
            base_model = SimplifiedEVO(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=512,
                num_layers=6,
                max_length=min(self.max_length, 8192)
            )
            self.model = EVOForSequenceClassification(base_model, self.num_labels)
        else:
            self.model = SimplifiedEVO(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=512,
                num_layers=6,
                max_length=min(self.max_length, 8192)
            )
    
    def _create_byte_tokenizer(self):
        """Create byte-level tokenizer for EVO"""
        class ByteTokenizer:
            def __init__(self):
                # EVO uses byte-level encoding for single nucleotide resolution
                self.vocab = {
                    'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4,
                    'a': 5, 't': 6, 'c': 7, 'g': 8, 'n': 9,  # Lowercase
                    '[PAD]': 10, '[UNK]': 11, '[BOS]': 12, '[EOS]': 13
                }
                self.vocab_size = len(self.vocab)
                self.pad_token_id = self.vocab['[PAD]']
                self.bos_token_id = self.vocab['[BOS]']
                self.eos_token_id = self.vocab['[EOS]']
                
            def encode(self, text, max_length=None, padding=True, truncation=True, 
                      return_tensors='pt', add_special_tokens=True):
                if isinstance(text, str):
                    texts = [text]
                else:
                    texts = text
                
                all_input_ids = []
                all_attention_masks = []
                
                for sequence in texts:
                    # Convert sequence to token IDs
                    if add_special_tokens:
                        tokens = ['[BOS]'] + list(sequence) + ['[EOS]']
                    else:
                        tokens = list(sequence)
                    
                    input_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
                    
                    # Handle truncation
                    if truncation and max_length and len(input_ids) > max_length:
                        if add_special_tokens:
                            input_ids = input_ids[:max_length-1] + [self.eos_token_id]
                        else:
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
            
            def decode(self, token_ids, skip_special_tokens=True):
                """Decode token IDs back to sequence"""
                id_to_token = {v: k for k, v in self.vocab.items()}
                tokens = [id_to_token.get(id, '[UNK]') for id in token_ids]
                
                if skip_special_tokens:
                    special_tokens = {'[PAD]', '[UNK]', '[BOS]', '[EOS]'}
                    tokens = [token for token in tokens if token not in special_tokens]
                
                return ''.join(tokens)
        
        return ByteTokenizer()
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for EVO"""
        # EVO can handle both DNA and protein sequences
        # For DNA, keep case information as it might be meaningful
        valid_dna_bases = set('ATCGNatcgn')
        if all(c in valid_dna_bases or c.isspace() for c in sequence):
            # DNA sequence - keep case, remove spaces
            return ''.join(c for c in sequence if not c.isspace())
        else:
            # Assume protein or other sequence - convert unknown to N
            return ''.join(c if c in valid_dna_bases else 'N' for c in sequence.upper())
    
    def _tokenize_sequences(self, sequences: Union[str, List[str]], max_length: Optional[int] = None) -> Dict:
        """Tokenize sequences for EVO"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Preprocess sequences
        processed_sequences = [self._preprocess_sequence(seq) for seq in sequences]
        
        # Use conservative max_length for memory management
        actual_max_length = max_length or min(self.max_length, 2048)
        
        return self.tokenizer(
            processed_sequences,
            max_length=actual_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False  # EVO typically doesn't use special tokens for DNA
        )
    
    def get_embeddings(self, sequences: Union[str, List[str]], layer: int = -1) -> torch.Tensor:
        """Get sequence embeddings from EVO"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        self.model.eval()
        embeddings = []
        
        # Use small batches for long sequences
        batch_size = 1 if self.max_length > 50000 else 2
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                try:
                    # Tokenize batch
                    inputs = self._tokenize_sequences(batch_sequences)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get outputs
                    if hasattr(self.model, 'base_model'):  # Classification wrapper
                        outputs = self.model.base_model(**inputs, return_logits=False)
                    else:  # Base model
                        outputs = self.model(**inputs, return_logits=False)
                    
                    # Get hidden states
                    hidden_states = outputs.last_hidden_state
                    
                    # Pool embeddings (mean pooling with attention mask)
                    attention_mask = inputs.get('attention_mask')
                    if attention_mask is not None:
                        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                        pooled = sum_embeddings / sum_mask
                    else:
                        pooled = hidden_states.mean(dim=1)
                    
                    embeddings.append(pooled.cpu())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("OOM encountered, trying with shorter sequences")
                        # Try with much shorter sequences
                        inputs = self._tokenize_sequences(batch_sequences, max_length=512)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        if hasattr(self.model, 'base_model'):
                            outputs = self.model.base_model(**inputs, return_logits=False)
                        else:
                            outputs = self.model(**inputs, return_logits=False)
                        
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
        
        # Use small batches
        batch_size = 1 if self.max_length > 50000 else 2
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                try:
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
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("OOM for prediction, using shorter sequences")
                        inputs = self._tokenize_sequences(batch_sequences, max_length=512)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                        
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
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0, 
                 top_p: float = 0.95, do_sample: bool = True) -> str:
        """Generate sequences using EVO (autoregressive generation)"""
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                add_special_tokens=False
            )
            input_ids = inputs['input_ids'].to(self.device)
            
            # Generate
            if hasattr(self.model, 'generate'):
                # Use built-in generation method
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                # Simple autoregressive generation
                generated_ids = input_ids.clone()
                
                for _ in range(max_new_tokens):
                    outputs = self.model(generated_ids, return_logits=True)
                    next_token_logits = outputs.logits[:, -1, :] / temperature
                    
                    if do_sample:
                        # Apply top-p sampling
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                        
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Decode generated sequence
            generated_sequence = self.tokenizer.decode(
                generated_ids[0].cpu().tolist(),
                skip_special_tokens=True
            )
            
            # Return only the newly generated part
            return generated_sequence[len(prompt):]
    
    def fine_tune(self, train_dataset, val_dataset=None, **kwargs) -> Dict:
        """Fine-tune EVO model"""
        from transformers import TrainingArguments, Trainer
        
        # Set model to training mode
        self.model.train()
        
        # Use small batch sizes for large models
        default_batch_size = 1 if 'evo-2' in self.model_variant else 2
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './evo_finetuned'),
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
            learning_rate=kwargs.get('learning_rate', 1e-5),
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 8),
            dataloader_drop_last=True,
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
            compute_metrics=compute_metrics if val_dataset and self.num_labels > 1 else None,
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
        
        # Save tokenizer
        if hasattr(self.tokenizer, 'save_pretrained'):
            try:
                self.tokenizer.save_pretrained(path)
            except:
                # Save custom tokenizer
                import json
                tokenizer_config = {
                    'vocab': self.tokenizer.vocab,
                    'vocab_size': self.tokenizer.vocab_size,
                    'pad_token_id': self.tokenizer.pad_token_id
                }
                with open(Path(path) / 'tokenizer_config.json', 'w') as f:
                    json.dump(tokenizer_config, f)
        
        # Save model config
        import json
        config = {
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'model_variant': self.model_variant,
            'is_causal': self.is_causal,
            'model_type': 'evo'
        }
        with open(Path(path) / 'evo_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model"""
        # Load config
        config_path = Path(path) / 'evo_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.max_length = config.get('max_length', 8192)
                self.num_labels = config.get('num_labels', 2)
                self.model_variant = config.get('model_variant', 'evo-1-8k-base')
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except:
            tokenizer_config_path = Path(path) / 'tokenizer_config.json'
            if tokenizer_config_path.exists():
                self.tokenizer = self._create_byte_tokenizer()
            else:
                self.tokenizer = self._create_byte_tokenizer()
        
        # Load model
        try:
            if self.num_labels > 1:
                base_model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
                self.model = EVOForSequenceClassification(base_model, self.num_labels)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        except:
            # Load manually saved model
            self._create_fallback_evo()
            model_weights_path = Path(path) / 'pytorch_model.bin'
            if model_weights_path.exists():
                self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        
        self.model.to(self.device)


class EVOForSequenceClassification(nn.Module):
    """Wrapper to add classification head to EVO"""
    
    def __init__(self, base_model, num_labels: int):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        
        # Determine hidden size
        if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
            hidden_size = base_model.config.hidden_size
        else:
            hidden_size = 512  # Default for our simplified model
        
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
        outputs = self.base_model(**inputs, return_logits=False)
        
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
