# Mini-GPT: Build a Language Model from Scratch

An educational implementation of a **character-level, decoder-only Transformer (GPT-style language model)** built with Python and PyTorch. The model is initialized from scratch and trained on the **Tiny Shakespeare** dataset to learn character patterns and generate Shakespeare-inspired text.

This project demonstrates core ideas behind modern GPT-style language models: tokenization, embeddings, causal self-attention, Transformer blocks, next-token prediction, and autoregressive text generation.

> **Important:** This is a small educational language model, not a replica of ChatGPT and not a general-purpose instruction-following chatbot. It learns from a narrow Shakespeare corpus and is most useful for generating text in patterns resembling that dataset.

---

## Architecture

The model follows a decoder-only Transformer workflow. A prompt is encoded as character IDs, passed through token and positional embeddings, processed by causal Transformer blocks, and projected to a probability distribution over the character vocabulary. The model samples a next character and appends it to the context, repeating this process to generate text.

![Mini-GPT decoder-only Transformer architecture](assets/architecture.png)

### Main components

1. **Character-level tokenizer** – maps each unique character to an integer ID and back.
2. **Token and positional embeddings** – represent character identity and position in the sequence.
3. **Causal self-attention** – computes attention using Query, Key, and Value projections while masking future positions.
4. **Multi-head attention** – runs multiple attention heads in parallel.
5. **Feed-forward network** – applies a nonlinear transformation at each sequence position.
6. **Residual connections and layer normalization** – support stable information flow through Transformer blocks.
7. **Language-model head** – projects hidden states to logits over the character vocabulary.
8. **Autoregressive generation** – samples one character at a time and appends it to the running context.

## How this relates to LLMs and Generative AI

Many modern GPT-style language models use a decoder-only Transformer and are trained to predict the next token in a sequence. At generation time, they repeatedly use the prompt and previously generated tokens as context to predict what comes next.

This project demonstrates that central language-modelling mechanism on a much smaller scale:

- It learns statistical patterns from examples in its training corpus.
- It uses causal self-attention to condition predictions on preceding context.
- It generates new text autoregressively, one character at a time.

Production LLMs and modern chatbots add substantial capabilities and engineering beyond this prototype, including much larger and more diverse training corpora, subword tokenizers, vastly larger models, extensive training pipelines, instruction fine-tuning, preference alignment, safety systems, and serving infrastructure. Therefore, this project is an educational demonstration of a foundational GPT-style architecture, not a complete modern chatbot system.

## Dataset

The model is trained on **Tiny Shakespeare**, a compact text corpus assembled from Shakespeare's works. The notebook downloads the text automatically from the original `char-rnn` repository.

**Dataset links:**

- [Tiny Shakespeare — original GitHub dataset](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
- [Direct download — input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
- [Tiny Shakespeare on Hugging Face](https://huggingface.co/datasets/karpathy/tiny_shakespeare)
- [nanoGPT Shakespeare character-level data reference](https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare_char)

The dataset is used as a character sequence. Each distinct character is assigned an integer ID; the model is trained to predict the next character at every position in a sequence.

## What kind of prompts should you use?

Because the model is trained only on Shakespearean text, prompts resembling its training data are likely to produce the most relevant and stylistically consistent continuations.

Example prompts:

```text
ROMEO:
JULIET:
HAMLET:
KING:
To be, or not to be
```

The model does **not** understand these as instructions or character identities in the way a person does. It has learned patterns in the training corpus and generates a likely continuation from the supplied character sequence.

For example, a prompt such as `ROMEO:` may lead to a Shakespeare-like dialogue continuation. A general question such as `Explain machine learning` should not be expected to produce a factual explanation.

## Model configuration

The notebook's default configuration is:

| Setting | Value |
|---|---:|
| Dataset | Tiny Shakespeare |
| Tokenization | Character-level |
| Embedding dimension (`N_EMBD`) | 128 |
| Attention heads (`N_HEADS`) | 4 |
| Transformer layers (`N_LAYERS`) | 3 |
| Context length (`BLOCK_SIZE`) | 128 |
| Batch size (`BATCH_SIZE`) | 32 |
| Dropout (`DROPOUT`) | 0.1 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Default training steps (`MAX_STEPS`) | 1,500 |
| Training / validation split | 90% / 10% |
| Loss | Cross-entropy |
| Framework | PyTorch |

The model uses a character vocabulary derived from the downloaded corpus. Training duration and generation settings can be changed in the notebook.

## Run it in Google Colab

The easiest way to run the project is with Google Colab.

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook**.
3. Upload `notebooks/Mini_GPT_From_Scratch_Colab.ipynb` from this repository.
4. Select **Runtime → Change runtime type** and choose a GPU accelerator if one is available (for example, NVIDIA T4).
5. Run the notebook cells from top to bottom.

The notebook downloads the dataset, builds the tokenizer and model, trains the model, plots the learning curve, generates a sample, and saves a checkpoint.

A GPU is recommended for faster training, but a reduced training run can also be attempted on CPU.

## Generate text

The notebook defines a helper function named `generate_from_prompt()`. After the model has been trained, run a cell such as:

```python
prompt = "ROMEO:"

print(
    generate_from_prompt(
        prompt,
        max_new_tokens=500,
        temperature=0.8
    )
)
```

**Temperature** controls sampling randomness:

- Lower temperature generally makes sampling more conservative.
- Higher temperature generally produces more varied output, but can also make it less coherent.

## Repository structure

```text
Mini-GPT/
├── README.md
├── assets/
│   └── architecture.png
└── notebooks/
    └── Mini_GPT_From_Scratch_Colab.ipynb
```

The notebook creates the dataset and output directories when executed. Model checkpoints and generated samples can be added to the repository separately if desired.

## Learning outcomes

By implementing this project, We can explore:


- Token embeddings and positional embeddings.
- Scaled dot-product attention using Query, Key, and Value.
- Causal masking and why future tokens must be hidden during training.
- Multi-head self-attention.
- Feed-forward networks.
- Residual connections and layer normalization.
- Cross-entropy loss and next-character prediction.
- Autoregressive text generation and temperature-based sampling.
- Training and validation loss monitoring.

## Limitations

- The training corpus is small and domain-specific.
- Character-level tokenization is less efficient than the subword tokenization used by modern LLMs.
- The model has a small number of parameters and a limited context window.
- It is not instruction-tuned and is not designed for general questions, factual answers, or task completion.
- Generated text can be repetitive, grammatically inconsistent, or nonsensical, especially with limited training.

## Acknowledgements

- [Andrej Karpathy's `char-rnn` repository](https://github.com/karpathy/char-rnn) for the Tiny Shakespeare text corpus.
- [PyTorch](https://pytorch.org/) for the deep-learning framework.
- [Google Colab](https://colab.research.google.com/) for the notebook development environment.

---

**The goal of this project is not to build another ChatGPT, but to understand the core mechanics behind GPT-style language models—one Transformer block at a time.**
