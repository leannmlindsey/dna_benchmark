# DNA Language Model Benchmarking Framework

A comprehensive benchmarking framework for evaluating and comparing state-of-the-art DNA language models on prophage identification and other genomic classification tasks.

## Overview

This framework provides a unified interface for benchmarking multiple DNA language models, including DNABERT, Nucleotide Transformer, HyenaDNA, and others. It's specifically designed for evaluating model performance on prophage identification but can be extended to other genomic sequence classification tasks.

## Features

- **10+ Pre-trained DNA Language Models**: Support for DNABERT1/2, Nucleotide Transformer, ProkBERT, GROVER, GENA-LM, INHERIT, HyenaDNA, EVO, and Caduceus
- **Unified Interface**: Abstract base class providing consistent API across all models
- **Flexible Tokenization**: Support for k-mer, BPE, character-level, and specialized tokenization strategies
- **Long Sequence Handling**: Automatic sequence chunking and aggregation for models with different max lengths (512 to 1M+ tokens)
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1, precision, recall, AUROC, and AUPRC
- **Cross-validation Support**: Built-in k-fold cross-validation with stratification
- **SLURM Integration**: Ready-to-use templates for HPC cluster deployment

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dna-benchmark.git
cd dna-benchmark

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- CUDA-capable GPU (recommended)

## Project Structure

```
dna_benchmark/
├── config/                 # Configuration files
│   ├── benchmark.yaml      # Benchmark settings
│   └── models.yaml         # Model configurations
├── data/                   # Data directories
│   ├── raw/               # Raw sequence data
│   ├── processed/         # Processed datasets
│   └── splits/            # Train/val/test splits
├── src/
│   └── models/            # Model implementations
│       ├── base_model.py  # Abstract base class
│       ├── dnabert1.py    # DNABERT v1
│       ├── dnabert2.py    # DNABERT v2
│       ├── nucleotide_transformer.py
│       ├── prokbert.py
│       ├── grover.py
│       ├── gena_lm.py
│       ├── inherit.py
│       ├── hyenadna.py
│       ├── evo.py
│       └── README.md      # Model details
├── results/               # Output directories
│   ├── metrics/          # Evaluation metrics
│   ├── runs/             # Training logs
│   └── visualizations/   # Plots and figures
├── slurm/                # SLURM job templates
├── scripts/              # Utility scripts
└── tests/                # Unit tests
```

## Supported Models

| Model | Architecture | Tokenizer | Max Length | HuggingFace ID |
|-------|--------------|-----------|------------|----------------|
| DNABERT1 | BERT | k-mer (k=6) | 512 | zhihan1996/DNA_bert_6 |
| DNABERT2 | BERT | BPE | 512 | zhihan1996/DNABERT-2-117M |
| Nucleotide Transformer | RoBERTa | 6-mer | 1000 | InstaDeepAI/nucleotide-transformer-v2-500m-multi-species |
| ProkBERT | BERT | LCA (k=6) | 512 | neuralbioinfo/prokbert-mini |
| GROVER | BERT | BPE | 510 | PoetschLab/GROVER |
| GENA-LM | BERT | BPE | 4500 | AIRI-Institute/gena-lm-bert-base-t2t |
| INHERIT | Dual-BERT | k-mer | 512 | Custom (bacteria+phage) |
| HyenaDNA | Hyena | Character | 450k | LongSafari/hyenadna-medium-450k-seqlen |
| EVO | StripedHyena | Byte | 8192 | togethercomputer/evo-1-8k-base |
| Caduceus | BiMamba | Character | 131k | kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16 |

## Quick Start

### 1. Load a Model

```python
from src.models.dnabert2 import DNABert2Model

# Initialize model with config
config = {
    'pretrained_path': 'zhihan1996/DNABERT-2-117M',
    'num_labels': 2,
    'max_length': 512
}

model = DNABert2Model(config)
model.load_pretrained()
```

### 2. Get Embeddings

```python
# Single sequence
sequence = "ATCGATCGATCGATCG"
embeddings = model.get_embeddings(sequence)

# Multiple sequences
sequences = ["ATCGATCG", "GCTAGCTA", "TTTTAAAA"]
embeddings = model.get_embeddings(sequences)
```

### 3. Make Predictions

```python
# Predict prophage vs non-prophage
predictions = model.predict(sequences)
print(predictions['labels'])  # Predicted classes
print(predictions['probabilities'])  # Class probabilities
```

### 4. Fine-tune on Custom Data

```python
# Fine-tune model on your dataset
metrics = model.fine_tune(
    train_dataset=train_data,
    val_dataset=val_data,
    epochs=10,
    batch_size=16,
    learning_rate=2e-5
)

# Save fine-tuned model
model.save_model("path/to/save/model")
```

## Configuration

### Benchmark Configuration (`config/benchmark.yaml`)

```yaml
datasets:
  prophage_bench_v1:
    path: "data/processed/prophage_bench_v1"
    splits:
      train: 0.7
      val: 0.15
      test: 0.15
    sequence_length: 1000
    overlap: 200
    
evaluation:
  metrics:
    - accuracy
    - f1
    - precision
    - recall
    - auroc
    - auprc
    - mcc
```

### Model Configuration (`config/models.yaml`)

Each model has specific configuration parameters:

```yaml
models:
  dnabert2:
    class_name: "DNABert2Model"
    pretrained_path: "zhihan1996/DNABERT-2-117M"
    tokenizer: "bpe"
    max_length: 512
```

## Running Benchmarks

### Local Execution

```bash
# Run benchmark for a specific model
python scripts/run_benchmark.py --model dnabert2 --dataset prophage_bench_v1

# Run benchmark for all models
python scripts/run_benchmark.py --model all --dataset prophage_bench_v1
```

### SLURM Cluster Execution

```bash
# Submit job to SLURM
sbatch slurm/templates/benchmark_job.sh
```

## Dataset Format

The framework expects datasets in the following format:

```
data/processed/dataset_name/
├── train.csv
├── val.csv
└── test.csv
```

Each CSV file should contain:
- `sequence`: DNA sequence (string)
- `label`: Class label (integer)
- `id`: Unique identifier (optional)

## Evaluation Metrics

The framework computes the following metrics:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUROC**: Area under the ROC curve
- **AUPRC**: Area under the Precision-Recall curve
- **MCC**: Matthews Correlation Coefficient - a balanced measure for binary classification

## Advanced Features

### Sequence Chunking

For models with limited sequence length, the framework automatically chunks long sequences:

```python
# Sequences longer than max_length are split with overlap
chunked_seqs, chunk_counts = model.split_long_sequences(
    sequences, 
    overlap=100
)

# Predictions are aggregated across chunks
predictions = model.aggregate_chunk_predictions(
    chunk_predictions,
    chunk_counts,
    method='mean'  # or 'max', 'vote'
)
```

### Custom Model Integration

To add a new model, extend the `BaseDNAModel` class:

```python
from src.models.base_model import BaseDNAModel

class CustomDNAModel(BaseDNAModel):
    def load_pretrained(self, path=None):
        # Load model weights
        pass
    
    def get_embeddings(self, sequences):
        # Generate embeddings
        pass
    
    def predict(self, sequences):
        # Make predictions
        pass
    
    def fine_tune(self, train_dataset, val_dataset=None, **kwargs):
        # Fine-tune model
        pass
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{dna_benchmark,
  title = {DNA Language Model Benchmarking Framework},
  author = {LeAnn Lindsey},
  year = {2024},
  url = {https://github.com/leannmlindsey/dna_benchmark}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This framework builds upon the following excellent DNA language models. Please cite the original papers and repositories if you use these models in your research:

- **DNABERT**: [GitHub Repository](https://github.com/jerryji1993/DNABERT) | [HuggingFace](https://huggingface.co/zhihan1996/DNA_bert_6)
- **DNABERT-2**: [GitHub Repository](https://github.com/MAGICS-LAB/DNABERT_2) | [HuggingFace](https://huggingface.co/zhihan1996/DNABERT-2-117M)
- **Nucleotide Transformer**: [GitHub Repository](https://github.com/instadeepai/nucleotide-transformer) | [HuggingFace](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species)
- **ProkBERT**: [GitHub Repository](https://github.com/nbrg-ppcu/prokbert) | [HuggingFace](https://huggingface.co/neuralbioinfo/prokbert-mini)
- **GROVER**: [Zenodo](https://zenodo.org/records/13135894) | [HuggingFace](https://huggingface.co/PoetschLab/GROVER)
- **GENA-LM**: [GitHub Repository](https://github.com/AIRI-Institute/GENA_LM) | [HuggingFace](https://huggingface.co/AIRI-Institute/gena-lm-bert-base-t2t)
- **INHERIT**: [GitHub Repository](https://github.com/Celestial-Bai/INHERIT)
- **HyenaDNA**: [GitHub Repository](https://github.com/HazyResearch/hyena-dna) | [HuggingFace](https://huggingface.co/LongSafari/hyenadna-medium-450k-seqlen)
- **EVO**: [GitHub Repository (Evo-1)](https://github.com/evo-design/evo) | [GitHub Repository (Evo-2)](https://github.com/ArcInstitute/evo2) | [HuggingFace](https://huggingface.co/togethercomputer/evo-1-8k-base)
- **Caduceus**: [GitHub Repository](https://github.com/kuleshov-group/caduceus) | [HuggingFace](https://huggingface.co/kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16)

**Important**: When using any of these models, please ensure you cite the original authors' work as specified in their respective repositories.

## Contact

For questions and support, please open an issue on GitHub or contact leann.lindsey@nih.gov.