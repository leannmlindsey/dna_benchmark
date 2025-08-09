"""
Nucleotide Transformer Model Implementation
Based on: https://github.com/instadeepai/nucleotide-transformer
Paper: The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .base_model import BaseDNAModel

logger = logging.getLogger(__name__)


class NucleotideTransformerModel(BaseDNAModel):
    """Nucleotide Transformer implementation for DNA sequence analysis"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('pretrained_path', 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species')
        self.max_length = config.get('max_length', 1000)  # NT supports up to 1000 tokens
        self.num_labels = config.get('num_labels', 2)
        
        # Model and tokenizer will be loaded in load_pretrained
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Nucleotide Transformer specific parameters
        self.padding_side = config.get('padding_side', 'left')  # NT typically uses left padding
        
    def load_pretrained(self, path: Optional[str] = None) -> None:
        """Load pretrained Nucleotide Transformer model"""
        model_path = path or self.model_name
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                padding_side=self.padding_side
            )
            
            # Load model - NT typically uses AutoModelForMaskedLM
            if self.num_labels > 1:  # Classification task - need custom head
                # Load base model and add classification head
                base_model = AutoModelForMaskedLM.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                # Create classification model
                self.model = NucleotideTransformerForSequenceClassification(
                    base_model, 
                    num_labels=self.num_labels
                )
            else:  # Feature extraction
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
        except Exception as e:
            logger.error(f"Could not load pretrained Nucleotide Transformer from {model_path}: {e}")
            raise
        
        self.model.to(self.device)
        logger.info(f"Nucleotide Transformer loaded from {model_path}")
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess DNA sequence for Nucleotide Transformer"""
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Replace degenerate bases with N
        valid_bases = set('ATCGN')
        sequence = ''.join([base if base in valid_bases else 'N' for base in sequence])
        
        # Add spaces between nucleotides for 6-mer tokenization
        # This helps the BPE tokenizer work better
        spaced_sequence = ' '.join(sequence)
        
        return spaced_sequence
    
    def _tokenize_sequences(self, sequences: Union[str, List[str]]) -> Dict:
        """Tokenize DNA sequences using 6-mer tokenizer"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        # Preprocess sequences
        processed_sequences = [self._preprocess_sequence(seq) for seq in sequences]
        
        # Tokenize with 6-mer BPE tokenizer
        return self.tokenizer(
            processed_sequences,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
    
    def get_embeddings(self, sequences: Union[str, List[str]], layer: int = -1) -> torch.Tensor:
        """Get sequence embeddings from specified layer"""
        if isinstance(sequences, str):
            sequences = [sequences]
        
        self.model.eval()
        embeddings = []
        
        # Process in batches to handle memory constraints
        batch_size = 4  # NT models are large, use smaller batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
                # Tokenize batch
                inputs = self._tokenize_sequences(batch_sequences)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get outputs with hidden states
                if hasattr(self.model, 'roberta'):  # NT is RoBERTa-based
                    outputs = self.model.roberta(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer]  # Get specified layer
                elif hasattr(self.model, 'base_model'):  # Custom classification model
                    outputs = self.model.base_model.roberta(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer]
                else:
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[layer]
                
                # Use [CLS] token embedding (first token)
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
        batch_size = 4
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            with torch.no_grad():
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
            output_dir=kwargs.get('output_dir', './nucleotide_transformer_finetuned'),
            num_train_epochs=kwargs.get('num_epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', 4),  # Smaller batch size for large model
            per_device_eval_batch_size=kwargs.get('eval_batch_size', 4),
            warmup_steps=kwargs.get('warmup_steps', 500),
            weight_decay=kwargs.get('weight_decay', 0.01),
            logging_dir=kwargs.get('logging_dir', './logs'),
            logging_steps=kwargs.get('logging_steps', 100),
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            learning_rate=kwargs.get('learning_rate', 1e-5),  # Lower LR for large pretrained model
            fp16=torch.cuda.is_available(),
            dataloader_drop_last=False,
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 4),
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
            'model_type': 'nucleotide_transformer',
            'padding_side': self.padding_side
        }
        with open(Path(path) / 'nt_config.json', 'w') as f:
            json.dump(config, f)
    
    def load_model(self, path: str) -> None:
        """Load a fine-tuned model"""
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        # Try to detect if it's a classification model
        config_path = Path(path) / 'nt_config.json'
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.num_labels = config.get('num_labels', 2)
        
        if self.num_labels > 1:
            # Load base model and add classification head
            base_model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
            self.model = NucleotideTransformerForSequenceClassification(
                base_model, 
                num_labels=self.num_labels
            )
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True)
        
        self.model.to(self.device)


class NucleotideTransformerForSequenceClassification(nn.Module):
    """Wrapper to add classification head to Nucleotide Transformer"""
    
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
        outputs = self.base_model.roberta(**inputs)
        
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
