"""
DNA Language Model Registry
Centralized registry for all supported DNA language models
"""

from .base_model import BaseDNAModel
from .dnabert1 import DNABert1Model
from .dnabert2 import DNABert2Model
from .nucleotide_transformer import NucleotideTransformerModel
from .prokbert import ProkBERTModel
from .grover import GroverModel
from .gena_lm import GenaLMModel
from .inherit import INHERITModel
from .hyenadna import HyenaDNAModel
from .evo import EVOModel
from .caduceus import CaduceusModel

# Model registry for easy lookup
MODEL_REGISTRY = {
    # DNABERT variants
    "dnabert1": DNABert1Model,
    "dnabert": DNABert1Model,  # Alias
    "dnabert2": DNABert2Model,
    
    # Nucleotide Transformer
    "nucleotide_transformer": NucleotideTransformerModel,
    "nt": NucleotideTransformerModel,  # Alias
    
    # Prokaryotic-focused models
    "prokbert": ProkBERTModel,
    
    # Human genome-focused models
    "grover": GroverModel,
    
    # Long sequence models
    "gena_lm": GenaLMModel,
    "gena-lm": GenaLMModel,  # Alias
    "genalm": GenaLMModel,   # Alias
    
    # Phage identification models
    "inherit": INHERITModel,
    
    # Long-range models
    "hyenadna": HyenaDNAModel,
    "hyena_dna": HyenaDNAModel,  # Alias
    "hyena-dna": HyenaDNAModel,  # Alias
    
    # Generative models
    "evo": EVOModel,
    "evo1": EVOModel,    # Alias
    "evo2": EVOModel,    # Alias (Evo 2 would use same interface)
    
    # Bidirectional equivariant models
    "caduceus": CaduceusModel,
}

# Default model configurations
DEFAULT_CONFIGS = {
    "dnabert1": {
        "pretrained_path": "zhihan1996/DNA_bert_6",
        "tokenizer": "kmer",
        "kmer": 6,
        "max_length": 512,
    },
    
    "dnabert2": {
        "pretrained_path": "zhihan1996/DNABERT-2-117M",
        "tokenizer": "bpe",
        "max_length": 512,
    },
    
    "nucleotide_transformer": {
        "pretrained_path": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "tokenizer": "6mer",
        "max_length": 1000,
    },
    
    "prokbert": {
        "pretrained_path": "neuralbioinfo/prokbert-mini",
        "tokenizer": "lca",
        "kmer": 6,
        "shift": 1,
        "max_length": 512,
    },
    
    "grover": {
        "pretrained_path": "PoetschLab/GROVER",
        "tokenizer": "bpe",
        "max_length": 510,
        "bpe_vocab_size": 5000,
    },
    
    "gena_lm": {
        "pretrained_path": "AIRI-Institute/gena-lm-bert-base-t2t",
        "tokenizer": "bpe",
        "max_length": 4500,
    },
    
    "inherit": {
        "pretrained_path": "zhihan1996/DNA_bert_6",
        "tokenizer": "kmer",
        "kmer": 6,
        "max_length": 512,
    },
    
    "hyenadna": {
        "pretrained_path": "LongSafari/hyenadna-medium-450k-seqlen",
        "tokenizer": "character",
        "max_length": 450000,
    },
    
    "evo": {
        "pretrained_path": "togethercomputer/evo-1-8k-base",
        "tokenizer": "byte",
        "max_length": 8192,
        "revision": "1.1_fix",
    },
    
    "caduceus": {
        "pretrained_path": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        "tokenizer": "character",
        "max_length": 131072,
        "model_variant": "ph",
        "rc_equivariance": True,
        "bidirectional": True,
    },
}

def get_model_class(model_name: str):
    """Get model class by name"""
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return MODEL_REGISTRY[model_name]

def get_default_config(model_name: str):
    """Get default configuration for a model"""
    if model_name not in DEFAULT_CONFIGS:
        available_configs = list(DEFAULT_CONFIGS.keys())
        raise ValueError(f"Default config for '{model_name}' not found. Available configs: {available_configs}")
    
    return DEFAULT_CONFIGS[model_name].copy()

def create_model(model_name: str, config: dict = None):
    """Factory function to create a model instance"""
    model_class = get_model_class(model_name)
    
    if config is None:
        config = get_default_config(model_name)
    else:
        # Merge with default config
        default_config = get_default_config(model_name)
        default_config.update(config)
        config = default_config
    
    return model_class(config)

def list_available_models():
    """List all available models"""
    return list(MODEL_REGISTRY.keys())

def get_model_info():
    """Get information about all available models"""
    info = {}
    for model_name, model_class in MODEL_REGISTRY.items():
        config = DEFAULT_CONFIGS.get(model_name, {})
        info[model_name] = {
            "class": model_class.__name__,
            "pretrained_path": config.get("pretrained_path", "N/A"),
            "tokenizer": config.get("tokenizer", "N/A"),
            "max_length": config.get("max_length", "N/A"),
            "description": model_class.__doc__.split('\n')[0] if model_class.__doc__ else "No description"
        }
    return info

# Export main components
__all__ = [
    'BaseDNAModel',
    'DNABert1Model',
    'DNABert2Model', 
    'NucleotideTransformerModel',
    'ProkBERTModel',
    'GroverModel',
    'GenaLMModel',
    'INHERITModel',
    'HyenaDNAModel',
    'EVOModel',
    'CaduceusModel',
    'MODEL_REGISTRY',
    'DEFAULT_CONFIGS',
    'get_model_class',
    'get_default_config',
    'create_model',
    'list_available_models',
    'get_model_info'
]
