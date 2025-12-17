# Antimatter AI: 300M Parameter Autoregressive Language Model

## 1. Overview of the Project

We built a **300M parameter autoregressive language model** from scratch and deployed it on Kaggle using Ollama. The workflow demonstrates a complete end-to-end LLM lifecycle, from raw data processing to production-grade deployment.

### Team Roles & Responsibilities

#### **Member 1: Data & Model Design**
*   **Data Pipeline**: Collected and cleaned a massive text corpus (RefinedWeb + C4 subset), reducing raw data from **45GB** of noisy text to **38GB** of high-quality training tokens.
*   **Tokenization**: Built a custom Byte-Pair Encoding (BPE) tokenizer with a vocabulary size of **50,257** (compatible with GPT-2/3 tokenizers).
*   **Architecture Design**: Designed the decoder-only transformer architecture:
    *   **Layers**: 24
    *   **Hidden Size**: 1024
    *   **Attention Heads**: 16
    *   **FFN Expansion**: 4x (4096 hidden units)
    *   **Context Window**: 2048 tokens
    *   **Activation**: Leaky ReLU (experimental choice for better gradient flow)

#### **Member 2: Model Training**
*   **Implementation**: Written in PyTorch, implementing a modular GPT-style transformer.
*   **Training Loop**:
    *   **Precision**: Mixed Precision (FP16) via `torch.cuda.amp` to reduce memory usage and speed up training.
    *   **Optimization**: AdamW optimizer with cosine learning rate decay and warmup.
    *   **Parallelism**: Utilized `torch.nn.DataParallel` distributing batches across **4x NVIDIA A100 GPUs**.
    *   **Stability**: Implemented gradient clipping (norm 1.0) and gradient accumulation (virtual batch size of 512).
    *   **RAG & Backtracking**: Integrated dynamic weight balancing for rare tokens and a checkpoint backtracking mechanism to recover from loss spikes.

#### **Member 3: Deployment & Kaggle Pipeline**
*   **Orchestration**: Built the Kaggle-specific deployment pipeline using Python to orchestrate the Ollama backend.
*   **GGUF Export**: developed `gguf_export.py` to convert PyTorch `.pt` checkpoints into the efficient GGUF format for inference.
*   **Ollama Integration**: Configured the `Modelfile` for optimal inference parameters (temp 0.7, top-p 0.9).
*   **Chat Interface**: Created a lightweight HTTP/Command-line chat interface for real-time interaction with the trained model.

---

## 2. Project Structure

```
antimatter_project/
│
├── data/
│   ├── download_data.py         # Script to stream/download RefinedWeb (HuggingFace)
│   ├── raw/                     # Original text data (~45GB)
│   └── cleaned/                 # Cleaned dataset (~38GB)
│
├── tokenizer/
│   ├── bpe_tokenizer.json       # Vocabulary + merges (50,257 tokens)
│   └── tokenizer_train.py       # Script to train BPE tokenizer
│
├── model/
│   ├── transformer.py           # GPT-style transformer architecture
│   ├── training_loop.py         # Full PyTorch training pipeline
│   ├── optimizer_config.json    # AdamW hyperparameters + LR schedule
│   └── checkpoints/             # Saved weights (ckpt_*.pt)
│
├── deployment/
│   ├── gguf_export.py           # Converter: PyTorch -> GGUF
│   ├── ollama_pipeline.py       # Kaggle orchestration & server management
│   ├── Modelfile                # Ollama inference configuration
│   └── chat_interface.py        # Interactive chat testing script
│
├── evaluation/
│   ├── test_prompts.txt         # Benchmarking prompts
│   └── evaluation_metrics.py    # Perplexity calculations & coherence checks
│
└── README.md
```

---

## 3. Model Architecture Details

**File**: `model/transformer.py`

The core of Antimatter AI is a decoder-only Transformer. We chose **Leaky ReLU** over the standard GeLU to experiment with sparsity and gradient preservation in deeper layers.

*   **Embedding Layer**:
    *   `token_embeddings`: Maps token IDs to 1024-dim vectors.
    *   `positional_embeddings`: Learned absolute positional encodings (up to 2048 positions).
*   **Transformer Block (x24)**:
    *   **Layer Norm**: Applied before attention and FFN (Pre-LN configuration) for training stability.
    *   **Self-Attention**: Multi-head causal attention (16 heads). Implements a causal mask to prevent peeking at future tokens.
    *   **Feed-Forward Network (FFN)**: Projects 1024 -> 4096 -> 1024 using Leaky ReLU activation.
*   **Output Head**:
    *   Final LayerNorm followed by a linear projection to the vocabulary size (50,257).

---

## 4. Training Pipeline

**File**: `model/training_loop.py`

The training process is designed to be robust on multi-GPU setups.

1.  **Data Loading**: Streams data from `data/cleaned/` using memory mapping to handle the 38GB dataset without RAM overflows.
2.  **Initialization**: Weights initialized with mean 0.0, std 0.02 (textbook GPT intialization).
3.  **Forward Pass**: Computes Cross Entropy Loss.
4.  **Backward Pass (FP16)**:
    *   Scales loss to prevent underflow.
    *   Accumulates gradients over 4 micro-batches.
    *   Clips gradients at 1.0 globally.
5.  **Optimization**:
    *   **AdamW**: Beta1=0.9, Beta2=0.95, Weight Decay=0.1.
    *   **Scheduler**: Linear warmup for first 2000 steps, then Cosine decay to 10% of max LR.

**Backtracking Mechanism**: The loop monitors loss divergence. If loss spikes >3x the moving average, it automatically reloads the last stable checkpoint and reduces learning rate by 0.5x.

---

## 5. Tokenization

**File**: `tokenizer/tokenizer_train.py`

We use a Byte-Pair Encoding (BPE) tokenizer to efficiently represent subwords.
*   **Vocab Size**: 50,257
*   **Special Tokens**:
    *   `[PAD]` (0): Padding for creating batches.
    *   `[BOS]` (1): Beginning of sequence.
    *   `[EOS]` (2): End of sequence.
*   **Training**: Trained on 10GB subset of the cleaned data for optimal coverage.

---

## 6. Deployment & Ollama Integration

**File**: `deployment/ollama_pipeline.py`

This script automates the deployment on Kaggle environments where root access is limited or specific ports are needed.

1.  **Installation**: Downloads and installs the Ollama binary to a local writable path.
2.  **Model Creation**: 
    *   Converts the PyTorch `checkpoint_final.pt` to `antimatter-300m.gguf` using `gguf_export.py`.
    *   Writes a `Modelfile` defining the system prompt and hyperparameters.
    *   Runs `ollama create antimatter -f Modelfile`.
3.  **Serving**: Starts the Ollama server in the background and ensures it binds to the correct port.

**Modelfile Configuration**:
```dockerfile
FROM ./antimatter-300m.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
SYSTEM "You are a helpful AI assistant created by the Antimatter team."
```

---

## 7. Evaluation & Metrics

**File**: `evaluation/evaluation_metrics.py`

Evaluation is continuous during training and final post-training.
*   **Perplexity (PPL)**: The exponentiated average negative log-likelihood. Our target PPL was < 20.
*   **Loss Curves**: Logged to `training_log.csv` (dummy file provided in `model/`).
*   **Coherence Check**: `test_prompts.txt` contains logic puzzles and creative writing prompts used to sanity check the model's reasoning capabilities vs. simple pattern matching.

---

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install torch transformers numpy sentencepiece
    ```
2.  **Train Model** (Multi-GPU recommended):
    ```bash
    python model/training_loop.py
    ```
3.  **Export to GGUF**:
    ```bash
    python deployment/gguf_export.py --checkpoint model/checkpoints/final.pt
    ```
4.  **Run Chat Interface**:
    ```bash
    python deployment/chat_interface.py
    ```
