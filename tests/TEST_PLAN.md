# Comprehensive Unit Test Plan for DNA Language Models

## Overview
This document outlines the comprehensive testing strategy for ensuring all generic functions work correctly for each of the 10 DNA language models in the benchmarking framework.

## Test Categories

### 1. Core Functionality Tests
Each model must pass these fundamental tests to ensure basic operations work correctly.

#### 1.1 Model Initialization Tests
- **Test Files**: `test_[model_name]_core.py` for each model
- **Coverage**:
  - ✓ Model instantiation with default config
  - ✓ Model instantiation with custom config
  - ✓ Proper inheritance from BaseDNAModel
  - ✓ Correct attribute initialization (max_length, num_labels, device)
  - ✓ Environment configuration loading

#### 1.2 Model Loading Tests
- **Test Files**: `test_model_loading_complete.py`
- **Coverage**:
  - ✓ Load pretrained weights from HuggingFace
  - ✓ Load fine-tuned weights from local path
  - ✓ Handle missing model files gracefully
  - ✓ Verify model architecture after loading
  - ✓ Check tokenizer initialization

#### 1.3 Preprocessing Tests
- **Test Files**: `test_preprocessing.py`
- **Coverage**:
  - ✓ Sequence validation (valid/invalid characters)
  - ✓ Sequence preprocessing (uppercase, N replacement)
  - ✓ Tokenization for each model type (k-mer, BPE, character, etc.)
  - ✓ Sequence length handling
  - ✓ Batch preprocessing

### 2. Inference Tests
Tests for model prediction and embedding generation.

#### 2.1 Embedding Generation Tests
- **Test Files**: `test_embeddings.py`
- **Coverage**:
  - ✓ Single sequence embedding
  - ✓ Batch sequence embeddings
  - ✓ Embedding dimension validation
  - ✓ Handling sequences of different lengths
  - ✓ Memory efficiency for large batches

#### 2.2 Prediction Tests
- **Test Files**: `test_predictions.py`
- **Coverage**:
  - ✓ Binary classification predictions
  - ✓ Multi-class predictions (if applicable)
  - ✓ Probability scores validation (sum to 1, range [0,1])
  - ✓ Batch prediction consistency
  - ✓ Output format validation

### 3. Long Sequence Handling Tests
Special tests for models that handle very long sequences.

#### 3.1 Sequence Chunking Tests
- **Test Files**: `test_sequence_chunking.py`
- **Coverage**:
  - ✓ Correct chunking with specified overlap
  - ✓ No data loss during chunking
  - ✓ Chunk count calculation
  - ✓ Edge cases (sequences exactly at max_length)
  - ✓ Very short sequences that don't need chunking

#### 3.2 Chunk Aggregation Tests
- **Test Files**: `test_chunk_aggregation.py`
- **Coverage**:
  - ✓ Mean aggregation method
  - ✓ Max aggregation method
  - ✓ Voting aggregation method
  - ✓ Consistency of aggregated predictions
  - ✓ Performance with different chunk counts

### 4. Model-Specific Tests
Tests for unique features of individual models.

#### 4.1 DNABERT1/2 Tests
- K-mer tokenization correctness
- BPE tokenization for DNABERT2

#### 4.2 Nucleotide Transformer Tests
- 6-mer tokenization validation
- Multi-species model handling

#### 4.3 ProkBERT Tests
- LCA tokenization with shift parameter
- Prokaryotic sequence optimization

#### 4.4 HyenaDNA Tests
- Ultra-long sequence handling (up to 450k)
- Memory management for long sequences

#### 4.5 EVO Tests
- Autoregressive generation capabilities
- Byte-level tokenization

#### 4.6 Caduceus Tests
- Reverse complement equivariance
- Bidirectional processing

### 5. Fine-tuning Tests
Tests for model training and adaptation.

#### 5.1 Fine-tuning Workflow Tests
- **Test Files**: `test_fine_tuning.py`
- **Coverage**:
  - ✓ Training loop initialization
  - ✓ Loss calculation
  - ✓ Gradient updates
  - ✓ Validation during training
  - ✓ Early stopping
  - ✓ Model checkpointing

#### 5.2 Data Handling Tests
- **Test Files**: `test_data_handling.py`
- **Coverage**:
  - ✓ Dataset loading
  - ✓ Train/val/test splitting
  - ✓ Data augmentation (if applicable)
  - ✓ Batch creation
  - ✓ Label encoding

### 6. Evaluation Metrics Tests
Tests for all evaluation metrics.

#### 6.1 Metric Calculation Tests
- **Test Files**: `test_metrics.py`
- **Coverage**:
  - ✓ Accuracy calculation
  - ✓ F1 score (binary and weighted)
  - ✓ Precision and Recall
  - ✓ AUROC (with edge cases)
  - ✓ AUPRC
  - ✓ MCC (Matthews Correlation Coefficient)
  - ✓ Confusion matrix generation

### 7. Integration Tests
End-to-end tests combining multiple components.

#### 7.1 Complete Pipeline Tests
- **Test Files**: `test_pipeline_integration.py`
- **Coverage**:
  - ✓ Load model → preprocess → predict → evaluate
  - ✓ Cross-validation workflow
  - ✓ Multiple model comparison
  - ✓ Results aggregation and reporting

#### 7.2 Benchmark Execution Tests
- **Test Files**: `test_benchmark_execution.py`
- **Coverage**:
  - ✓ Run benchmark on small test dataset
  - ✓ Generate performance reports
  - ✓ Save and load results
  - ✓ Visualization generation

### 8. Performance and Resource Tests
Tests for computational efficiency.

#### 8.1 Memory Usage Tests
- **Test Files**: `test_memory_usage.py`
- **Coverage**:
  - ✓ Memory consumption during inference
  - ✓ Memory leaks detection
  - ✓ GPU memory management
  - ✓ Batch size optimization

#### 8.2 Speed Benchmarks
- **Test Files**: `test_speed_benchmarks.py`
- **Coverage**:
  - ✓ Inference time per sequence
  - ✓ Batch processing speed
  - ✓ Model loading time
  - ✓ Preprocessing overhead

### 9. Error Handling Tests
Tests for robustness and error recovery.

#### 9.1 Input Validation Tests
- **Test Files**: `test_input_validation.py`
- **Coverage**:
  - ✓ Empty sequences
  - ✓ Invalid characters
  - ✓ Extremely long sequences
  - ✓ Malformed batch inputs
  - ✓ Missing configuration parameters

#### 9.2 Recovery Tests
- **Test Files**: `test_error_recovery.py`
- **Coverage**:
  - ✓ Out of memory handling
  - ✓ Model file corruption
  - ✓ Network failures during download
  - ✓ Environment activation failures

### 10. Environment Management Tests
Tests for conda environment handling.

#### 10.1 Environment Switching Tests
- **Test Files**: `test_environment_switching.py`
- **Coverage**:
  - ✓ Correct environment activation
  - ✓ Environment isolation
  - ✓ Dependency resolution
  - ✓ Fallback mechanisms

## Test Implementation Structure

```
tests/
├── fixtures/
│   ├── __init__.py
│   ├── mock_data.py          # Generate mock DNA sequences
│   ├── mock_models.py        # Mock model objects
│   └── test_sequences.fasta  # Real test sequences
│
├── unit/
│   ├── models/
│   │   ├── test_dnabert1_core.py
│   │   ├── test_dnabert2_core.py
│   │   ├── test_nucleotide_transformer_core.py
│   │   ├── test_prokbert_core.py
│   │   ├── test_grover_core.py
│   │   ├── test_gena_lm_core.py
│   │   ├── test_inherit_core.py
│   │   ├── test_hyenadna_core.py
│   │   ├── test_evo_core.py
│   │   └── test_caduceus_core.py
│   │
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_predictions.py
│   ├── test_sequence_chunking.py
│   ├── test_chunk_aggregation.py
│   ├── test_fine_tuning.py
│   ├── test_data_handling.py
│   ├── test_metrics.py
│   ├── test_input_validation.py
│   └── test_error_recovery.py
│
├── integration/
│   ├── test_pipeline_integration.py
│   ├── test_benchmark_execution.py
│   ├── test_model_comparison.py
│   └── test_environment_switching.py
│
├── performance/
│   ├── test_memory_usage.py
│   ├── test_speed_benchmarks.py
│   └── test_scalability.py
│
├── conftest.py               # Pytest configuration
├── test_base_model.py        # Base test class
└── run_all_tests.sh          # Script to run all tests
```

## Test Data Requirements

### Mock Data Generation
```python
# fixtures/mock_data.py
def generate_test_sequences(n=100, length_range=(100, 1000)):
    """Generate random DNA sequences for testing"""
    
def generate_prophage_dataset(n_samples=1000):
    """Generate mock prophage classification dataset"""
    
def generate_long_sequences(n=10, length=10000):
    """Generate long sequences for chunking tests"""
```

### Test Sequence Categories
1. **Valid sequences**: Standard ATCG sequences
2. **Sequences with N**: Contains ambiguous bases
3. **Edge cases**: Empty, single base, maximum length
4. **Invalid sequences**: Contains numbers, special characters
5. **Real sequences**: From actual prophage/bacterial genomes

## Test Execution Strategy

### Local Testing (No GPU)
```bash
# Run fast structural tests
pytest tests/unit/test_preprocessing.py
pytest tests/unit/test_sequence_chunking.py
pytest tests/unit/test_metrics.py -v

# Run with mocked models
pytest tests/ -m "not requires_gpu" --mock-models
```

### SLURM Testing (With GPU)
```bash
#!/bin/bash
#SBATCH --job-name=dna_model_tests
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G

# Load modules
module load cuda/11.8
module load python/3.9

# Run all tests
pytest tests/ --gpu --verbose --tb=short

# Run specific model tests
pytest tests/unit/models/test_dnabert2_core.py --gpu

# Run integration tests
pytest tests/integration/ --gpu --benchmark
```

### Continuous Integration
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run structural tests
        run: pytest tests/ -m "not requires_gpu"
  
  gpu-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Run GPU tests
        run: pytest tests/ --gpu
```

## Success Criteria

### Coverage Requirements
- Minimum 80% code coverage for core functionality
- 100% coverage for critical paths (preprocessing, prediction)
- All models must pass base functionality tests

### Performance Benchmarks
- Inference time < 1s for 100 sequences (batch)
- Memory usage < 8GB for standard operations
- No memory leaks over extended runs

### Reliability Metrics
- All tests pass on 3 consecutive runs
- Error handling prevents crashes
- Graceful degradation when resources limited

## Test Prioritization

### Phase 1: Critical (Week 1)
1. Model initialization and loading
2. Basic preprocessing
3. Simple predictions
4. Core metrics calculation

### Phase 2: Important (Week 2)
1. Sequence chunking
2. Fine-tuning workflow
3. Batch processing
4. Environment management

### Phase 3: Comprehensive (Week 3)
1. Edge cases and error handling
2. Performance benchmarks
3. Integration tests
4. Model comparison

## Monitoring and Reporting

### Test Reports
- Generate HTML coverage reports
- Create performance comparison tables
- Log failing tests with full context
- Track test execution time trends

### Automated Alerts
- Notify on test failures
- Alert on performance regressions
- Report coverage drops
- Flag new untested code

## Maintenance Plan

### Regular Updates
- Add tests for new models
- Update mock data quarterly
- Refresh performance baselines
- Review and refactor test code

### Documentation
- Maintain test documentation
- Document known issues
- Keep troubleshooting guide updated
- Record benchmark results

This comprehensive test plan ensures all generic functions work correctly across all models while maintaining efficiency and reliability.