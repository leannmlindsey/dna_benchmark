"""
DNABERT2 Model Implementation
Based on: https://github.com/MAGICS-LAB/DNABERT_2
Paper: DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome
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


class DNABert2Model(BaseDNAModel):
    """DNABERT2 implementation for DNA sequence analysis"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('pretrained_path', 'zhihan1996/DNABERT-2-117M')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        
        # Model and tokenizer will be loaded in load_pretrained
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained DNABERT2 model"""
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
            logger.error(f"Could not load pretrained DNABERT2 model from {model_path}: {e}")
            raise
        
        self.model.to(self.device)
        logger.info(f"DNABERT2 model loaded from {model_path}")
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for DNABERT2"""
        # Convert to uppercase and handle degenerate bases
        sequence = sequence.upper()
        
        # Replace degenerate bases with N
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
        
        # Process in batches to handle memory constraints
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
                
                # Use [CLS] token embedding (first token)
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
        
        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors='pt'
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './dnabert2_finetuned'),
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
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            dataloader_drop_last=False,
            remove_unused_columns=False,
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
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if val_dataset else None,
        )
        
        # Train
        train_result = trainer.train()
        
        # Return training metrics
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
        self.tokenizer.save_pretrained(path)
        
        # Save additional config
        import json
        config = {
            'max_length': self.max_length,
            'num_labels': self.num_labels,
            'model_type': 'dnabert2'
        }
        with open(Path(path) / 'dnabert2_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model"""
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        if self.num_labels > 1:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, trust_remote_code=True
            )
        else:
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
        
        self.model.to(self.device)
    
    def evaluate_model(self, test_dataset, **kwargs) -> Dict:
        """Evaluate model on test dataset"""
        from transformers import Trainer, DataCollatorWithPadding
        
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors='pt'
        )
        
        # Create a dummy trainer for evaluation
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Evaluate
        eval_results = trainer.evaluate(test_dataset)
        
        return eval_results
    
    def get_attention_weights(self, sequence: str, layer: int = -1) -> Dict:
        """Get attention weights for interpretability"""
        self.model.eval()
        
        with torch.no_grad():
            inputs = self._tokenize_sequences(sequence)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get outputs with attention weights
            outputs = self.model(**inputs, output_attentions=True)
            
            # Extract attention weights from specified layer
            attention_weights = outputs.attentions[layer]  # Shape: [batch, heads, seq_len, seq_len]
            
            # Decode tokens for visualization
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            return {
                'tokens': tokens,
                'attention_weights': attention_weights.cpu().numpy(),
                'input_ids': inputs['input_ids'].cpu().numpy()
            }
