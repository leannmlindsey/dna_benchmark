# DNA Language Model Repository Information

## Model Repositories Found:

### 1. DNABERT1
- **Repository**: https://github.com/jerryji1993/DNABERT  
- **HuggingFace**: zhihan1996/DNA_bert_6
- **Architecture**: BERT-based with k-mer tokenization (k=3,4,5,6)
- **Tokenizer**: k-mer based
- **Max Length**: 512 nucleotides
- **Status**: Original links to pretrained models expired, but available through HF

### 2. DNABERT2  
- **Repository**: https://github.com/MAGICS-LAB/DNABERT_2
- **HuggingFace**: zhihan1996/DNABERT-2-117M
- **Architecture**: BERT-based with BPE tokenization
- **Tokenizer**: Byte Pair Encoding (BPE)
- **Max Length**: 512 tokens
- **Key Features**: More efficient than DNABERT1, uses BPE

### 3. Nucleotide Transformer
- **Repository**: https://github.com/instadeepai/nucleotide-transformer
- **HuggingFace**: InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
- **Architecture**: RoBERTa-based with 6-mer tokenization
- **Tokenizer**: 6-mer tokenizer (vocab size 4105)
- **Max Length**: 1000 tokens (up to 6000 bp)
- **Variants**: 500M, 2.5B parameters; different training datasets

### 4. prokBERT
- **Repository**: https://github.com/nbrg-ppcu/prokbert
- **HuggingFace**: neuralbioinfo/prokbert-mini
- **Architecture**: BERT-based with Local Context-Aware (LCA) tokenization
- **Tokenizer**: LCA tokenizer (k=6, shift=1)
- **Max Length**: Variable with LCA
- **Focus**: Microbiome and prokaryotic sequences

### 5. GROVER
- **Repository**: https://zenodo.org/records/13135894 (tutorials and weights)
- **HuggingFace**: PoetschLab/GROVER
- **Architecture**: BERT-based with Byte Pair Encoding
- **Tokenizer**: BPE optimized for human genome
- **Max Length**: 510 tokens
- **Focus**: Human genome specific

### 6. GENA-LM
- **Repository**: https://github.com/AIRI-Institute/GENA_LM
- **HuggingFace**: AIRI-Institute/gena-lm-bert-base-t2t
- **Architecture**: BERT-based with BPE tokenization  
- **Tokenizer**: BPE (512 tokens ~ 4500 nucleotides)
- **Max Length**: Up to 36k bp (different variants)
- **Key Features**: Long sequences, trained on T2T genome

### 7. INHERIT
- **Repository**: https://github.com/Celestial-Bai/INHERIT
- **Architecture**: Dual DNABERT models (bacteria + phage pretrained)
- **Tokenizer**: 6-mer DNABERT tokenizer
- **Max Length**: 512 nucleotides (500bp segments)
- **Focus**: Phage identification

### 8. HyenaDNA
- **Repository**: https://github.com/HazyResearch/hyena-dna
- **HuggingFace**: LongSafari/hyenadna-medium-450k-seqlen
- **Architecture**: Hyena (subquadratic attention replacement)
- **Tokenizer**: Single nucleotide (character-level)
- **Max Length**: Up to 1M tokens
- **Variants**: tiny-1k, small-32k, medium-160k, medium-450k, large-1M

### 9. EVO
- **Repository**: https://github.com/evo-design/evo (Evo 1) and https://github.com/ArcInstitute/evo2 (Evo 2)
- **HuggingFace**: togethercomputer/evo-1-8k-base
- **Architecture**: StripedHyena-based autoregressive
- **Tokenizer**: Single nucleotide (byte-level)
- **Max Length**: 8k to 131k (Evo 1), up to 1M (Evo 2)
- **Key Features**: Generative model, multi-modal (DNA/RNA/protein)

### 10. Caduceus
- **Repository**: https://github.com/kuleshov-group/caduceus
- **HuggingFace**: kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
- **Architecture**: BiMamba with RC equivariance
- **Tokenizer**: Single nucleotide
- **Max Length**: Up to 131k
- **Key Features**: Bidirectional, reverse complement equivariant
