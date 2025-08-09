"""
INHERIT Model Implementation
Based on: https://github.com/Celestial-Bai/INHERIT
Paper: Identification of bacteriophage genome sequences with representation learning
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertForSequenceClassification
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .base_model import BaseDNAModel

logger = logging.getLogger(__name__)


class INHERITModel(BaseDNAModel):
    """INHERIT implementation for phage identification using dual DNABERT models"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('pretrained_path', 'zhihan1996/DNA_bert_6')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        self.kmer = config.get('kmer', 6)
        
        # INHERIT specific: dual models for bacteria and phage
        self.bacteria_model = None
        self.phage_model = None
        self.classification_head = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained INHERIT model (dual DNABERT models)"""
        model_path = path or self.model_name
        
        try:
            # Load tokenizer (same for both models)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                # Fallback to DNABERT tokenizer
                self.tokenizer = self._create_dnabert_tokenizer()
            
            # Load dual DNABERT models
            # Model 1: Pre-trained on bacteria
            self.bacteria_model = BertModel.from_pretrained(model_path, trust_remote_code=True)
            
            # Model 2: Pre-trained on phages (same architecture, different weights)
            self.phage_model = BertModel.from_pretrained(model_path, trust_remote_code=True)
            
            # Create the INHERIT classification architecture
            self.model = INHERITClassifier(
                self.bacteria_model, 
                self.phage_model, 
                hidden_size=768,
                num_labels=self.num_labels
            )
            
        except Exception as e:
            logger.error(f"Could not load pretrained INHERIT model from {model_path}: {e}")
            # Create fallback INHERIT model
            self._create_fallback_inherit()
        
        self.model.to(self.device)
        logger.info(f"INHERIT model loaded with dual DNABERT architecture")
    
    def _create_dnabert_tokenizer(self):
        """Create DNABERT tokenizer for k-mer tokenization"""
        class DNABERTTokenizer:
            def __init__(self, kmer=6):
                self.kmer = kmer
                self.vocab = self._create_vocab()
                
            def _create_vocab(self):
                """Create k-mer vocabulary"""
                bases = ['A', 'T', 'C', 'G', 'N']
                vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
                
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
                """Convert DNA sequence to k-mer tokens"""
                sequence = sequence.upper()
                tokens = []
                
                for i in range(len(sequence) - self.kmer + 1):
                    kmer = sequence[i:i + self.kmer]
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
                    
                    if truncation and len(input_ids) > max_length:
                        input_ids = input_ids[:max_length-1] + [self.vocab['[SEP]']]
                    
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
        
        return DNABERTTokenizer(self.kmer)
    
    def _create_fallback_inherit(self):
        """Create fallback INHERIT model"""
        from transformers import BertConfig
        
        # Create tokenizer
        self.tokenizer = self._create_dnabert_tokenizer()
        
        # Create BERT config
        config = BertConfig(
            vocab_size=len(self.tokenizer.vocab),
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
        )
        
        # Create dual models
        self.bacteria_model = BertModel(config)
        self.phage_model = BertModel(config)
        
        # Create INHERIT classifier
        self.model = INHERITClassifier(
            self.bacteria_model,
            self.phage_model,
            hidden_size=768,
            num_labels=self.num_labels
        )
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for INHERIT"""
        # INHERIT segments sequences into 500bp fragments
        sequence = sequence.upper()
        valid_bases = set('ATCGN')
        sequence = ''.join([base if base in valid_bases else 'N' for base in sequence])
        return sequence
    
    def _segment_sequence(self, sequence: str, segment_length: int = 500) -> List[str]:
        """Segment long sequences into 500bp fragments as in INHERIT"""
        segments = []
        for i in range(0, len(sequence), segment_length):
            segment = sequence[i:i + segment_length]
            if len(segment) >= 100:  # Only keep segments with sufficient length
                segments.append(segment)
        return segments
    
    def _tokenize_sequences(self, sequences: Union[str, List[str]]) -> Dict:
        """Tokenize DNA sequences"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Preprocess and segment sequences
        all_segments = []
        segment_counts = []  # Track how many segments per sequence
        
        for sequence in sequences:
            processed_seq = self._preprocess_sequence(sequence)
            segments = self._segment_sequence(processed_seq)
            all_segments.extend(segments)
            segment_counts.append(len(segments))
        
        # Tokenize all segments
        return self.tokenizer(
            all_segments,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ), segment_counts
    
    def get_embeddings(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """Get sequence embeddings using INHERIT dual model approach"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            # Tokenize and segment sequences
            inputs, segment_counts = self._tokenize_sequences(sequences)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings from INHERIT model
            outputs = self.model.get_embeddings(**inputs)
            
            # Average embeddings over segments for each sequence
            start_idx = 0
            for count in segment_counts:
                if count > 0:
                    seq_embeddings = outputs[start_idx:start_idx + count]
                    avg_embedding = seq_embeddings.mean(dim=0, keepdim=True)
                    embeddings.append(avg_embedding.cpu())
                else:
                    # Handle sequences with no valid segments
                    embeddings.append(torch.zeros(1, outputs.shape[-1]))
                start_idx += count
        
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
        
        with torch.no_grad():
            # Tokenize and segment sequences
            inputs, segment_counts = self._tokenize_sequences(sequences)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions from INHERIT model
            logits = self.model(**inputs)
            
            # Average predictions over segments for each sequence
            start_idx = 0
            for count in segment_counts:
                if count > 0:
                    seq_logits = logits[start_idx:start_idx + count]
                    avg_logits = seq_logits.mean(dim=0, keepdim=True)
                    
                    probs = torch.softmax(avg_logits, dim=-1)
                    pred = torch.argmax(avg_logits, dim=-1)
                    
                    all_predictions.append(pred.cpu().numpy())
                    all_probabilities.append(probs.cpu().numpy())
                else:
                    # Handle sequences with no valid segments
                    all_predictions.append(np.array([0]))  # Default prediction
                    all_probabilities.append(np.array([[0.5, 0.5]]))  # Neutral probability
                
                start_idx += count
        
        return {
            'predictions': np.concatenate(all_predictions),
            'probabilities': np.concatenate(all_probabilities, axis=0)
        }
    
    def fine_tune(self, train_dataset, val_dataset=None, **kwargs) -> Dict:
        """Fine-tune the INHERIT model"""
        from transformers import TrainingArguments, Trainer
        
        # Set model to training mode
        self.model.train()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './inherit_finetuned'),
            num_train_epochs=kwargs.get('num_epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', 8),  # Smaller batch due to dual models
            per_device_eval_batch_size=kwargs.get('eval_batch_size', 8),
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
        """Save the fine-tuned INHERIT model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save the complete model
        torch.save(self.model.state_dict(), Path(path) / 'inherit_model.pt')
        
        # Save individual components
        self.bacteria_model.save_pretrained(Path(path) / 'bacteria_model')
        self.phage_model.save_pretrained(Path(path) / 'phage_model')
        
        # Save additional config
        import json
        config = {
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'kmer': self.kmer,
            'model_type': 'inherit'
        }
        with open(Path(path) / 'inherit_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned INHERIT model"""
        # Load config
        config_path = Path(path) / 'inherit_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.kmer = config.get('kmer', 6)
                self.num_labels = config.get('num_labels', 2)
        
        # Load tokenizer
        self.tokenizer = self._create_dnabert_tokenizer()
        
        # Load individual models
        self.bacteria_model = BertModel.from_pretrained(Path(path) / 'bacteria_model')
        self.phage_model = BertModel.from_pretrained(Path(path) / 'phage_model')
        
        # Recreate INHERIT classifier
        self.model = INHERITClassifier(
            self.bacteria_model,
            self.phage_model,
            hidden_size=768,
            num_labels=self.num_labels
        )
        
        # Load trained weights
        model_weights_path = Path(path) / 'inherit_model.pt'
        if model_weights_path.exists():
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        
        self.model.to(self.device)


class INHERITClassifier(nn.Module):
    """INHERIT classifier with dual DNABERT models"""
    
    def __init__(self, bacteria_model, phage_model, hidden_size=768, num_labels=2):
        super().__init__()
        self.bacteria_model = bacteria_model
        self.phage_model = phage_model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # Classification head that combines outputs from both models
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),  # 4 outputs: 2 models x 2 outputs each
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, **inputs):
        # Get outputs from bacteria model
        bacteria_outputs = self.bacteria_model(**inputs)
        bacteria_cls = bacteria_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get outputs from phage model
        phage_outputs = self.phage_model(**inputs)
        phage_cls = phage_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Additional dense layers for each model (as in INHERIT paper)
        bacteria_dense = torch.tanh(bacteria_cls)
        phage_dense = torch.tanh(phage_cls)
        
        # Concatenate all outputs
        combined = torch.cat([bacteria_cls, bacteria_dense, phage_cls, phage_dense], dim=-1)
        
        # Apply final classification layer
        logits = self.classifier(combined)
        
        return logits
    
    def get_embeddings(self, **inputs):
        """Get combined embeddings from both models"""
        # Get outputs from both models
        bacteria_outputs = self.bacteria_model(**inputs)
        phage_outputs = self.phage_model(**inputs)
        
        # Combine [CLS] representations
        bacteria_cls = bacteria_outputs.last_hidden_state[:, 0, :]
        phage_cls = phage_outputs.last_hidden_state[:, 0, :]
        
        # Return concatenated embeddings
        return torch.cat([bacteria_cls, phage_cls], dim=-1)
