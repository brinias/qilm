# QILM: Quantum-Inspired Language Model

### Unlocking the Future of AI with Complex-Valued Neural Networks

---

## 🚀 Introduction

Welcome to **QILM**, a pioneering project that explores the fascinating intersection of classical deep learning and quantum mechanics. QILM is not a quantum computer program, but rather a novel **Quantum-Inspired Language Model** that re-imaginates the fundamental building blocks of neural networks using **complex numbers**. By drawing parallels to quantum states and their evolution, QILM introduces a richer mathematical framework, enabling a deeper understanding and processing of intricate linguistic patterns.

A key innovation in QILM is the **Quantum Compressor**, a specialized component that preprocesses classical token sequences into compressed, quantum-inspired representations. This acts as a learnable "quantum tokenizer" that prepares the input for the main QILM, enhancing efficiency and allowing the model to focus on higher-level semantic interactions.

In an era where AI grapples with the complexities of context, nuance, and emergent intelligence, QILM proposes a paradigm shift. What if, instead of relying solely on real-valued vectors, our AI models could harness the hidden "phase" information inherent in complex numbers, much like quantum systems do? This project is an ambitious step towards answering that question, offering a glimpse into a new frontier for Artificial General Intelligence.

---

## ✨ The Quantum Inspiration: Why Complex Numbers?

The core innovation of QILM lies in its pervasive use of complex numbers (`a + bi`) throughout its architecture, from embeddings to attention mechanisms and feed-forward layers. This design is directly inspired by foundational principles of quantum mechanics:

*   **Quantum States as Complex Vectors:** In quantum mechanics, the state of a system is represented by a complex vector (or wave function) in a Hilbert space. The *magnitude* of the complex components relates to probability amplitudes, while the *phase* encodes crucial information about interference and superposition. QILM posits that natural language, with its inherent ambiguity, polysemy, and context-dependency, could benefit from such rich, multi-faceted representations.

*   **Unitary Transformations & Evolution:** Quantum operations are often unitary transformations, which preserve the "length" or "probability" of quantum states. While QILM's linear layers aren't strictly unitary, their complex-valued nature allows for transformations that operate on both magnitude and phase, potentially enabling a more dynamic and intricate evolution of linguistic representations.

*   **Probability Amplitudes and Inner Products:** A cornerstone of quantum mechanics is the inner product between quantum states, which yields a complex probability amplitude. The square of its magnitude gives the probability of observing a particular state. In QILM's attention mechanism, the dot product between query (`q`) and conjugate of key (`k.conj()`) vectors yields a complex number. Crucially, taking the `.real` part of this complex inner product to compute attention scores is a direct analogy to extracting observable information (like probabilities) from quantum interactions. This mechanism allows the model to compute a "quantum-like" overlap between contextual states.

*   **Phase as Latent Information:** The imaginary component of complex numbers serves as a "hidden" or "phase" dimension. This dimension might encode subtle relationships, temporal dynamics, or contextual nuances that are challenging for purely real-valued models to capture efficiently. It allows for the representation of superposition-like states in the embedding space, where a single token could embody multiple latent meanings or associations simultaneously.

---

## 🔬 Key Architectural Innovations

QILM redefines standard neural network components with a complex-valued approach:

*   **`ComplexEmbedding(vocab_size, embed_dim)`:**
    *   Instead of a single weight matrix, each word in the vocabulary is mapped to *two* trainable parameter matrices: one for the real component and one for the imaginary component. This allows tokens to be embedded directly into a complex vector space.

*   **`ComplexLinear(in_features, out_features, bias=True)`:**
    *   A custom linear layer designed for complex inputs. For `z = x + iy` and complex weights `W = A + iB`, the output `Wz` is calculated as `(Ax - By) + i(Ay + Bx)`. This operation naturally propagates complex values through the network, maintaining the phase information.

*   **`ComplexMultiheadAttention(embed_dim, num_heads)`:**
    *   The core attention mechanism is completely complex-valued. Query, Key, and Value projections are handled by `ComplexLinear` layers.
    *   **The Breakthrough:** Attention scores are computed using `torch.einsum("bhid,bhjd->bhij", q, k.conj()).real`. This means the similarity (overlap) between queries and keys is calculated via a complex inner product, with the *real* part taken as the final attention score before softmax. This directly mirrors how probabilities are derived from quantum amplitudes.
    *   The attention output remains complex, carrying forward both real and imaginary components.

*   **`QILMTransformerBlock(embed_dim, num_heads, ff_dim)`:**
    *   Each block incorporates a `ComplexMultiheadAttention` layer.
    *   **Hybrid Normalization and Activation:** While `ComplexLinear` layers operate on complex numbers, `ComplexLayerNorm` and the `nn.GELU` activation function are applied predominantly to the *real* part of the tensor, with the imaginary part either added back or propagated separately. This design choice implies that the "observable" (real) components are subjected to standard non-linearities and normalization, while the "phase" (imaginary) components carry forward distinct information, potentially enabling unique interactions.
    *   **Rotary Positional Encoding (RoPE):** Applied to complex numbers, RoPE injects positional information by rotating the complex embeddings, ensuring that relative positional information is preserved across the sequence.

*   **`QuantumCompressor(vocab_size, embed_dim, compression_ratio, num_compressor_layers)`:**
    *   This novel component acts as a learnable "quantum tokenizer" or feature extractor, processing classical input sequences before they reach the main QILM.
    *   It takes classical token IDs, embeds them classically (`nn.Embedding`), projects them to complex space (`ComplexLinear`), and then processes them through several `QILMTransformerBlock`s.
    *   A `ComplexPool` layer then *compresses* the sequence length by averaging complex-valued representations over a specified `compression_ratio`. This significantly reduces the sequence length for the subsequent QILM, making it more efficient, especially for longer inputs.
    *   The compressor is trained *jointly* with the QILM, allowing it to learn optimal quantum-inspired representations for effective downstream processing.

*   **`QILM(vocab_size, embed_dim, ...)`:**
    *   The main QILM model now consumes the *compressed quantum tokens* generated by the `QuantumCompressor`.
    *   It stacks multiple `QILMTransformerBlock`s. The final output layer `ComplexLinear` projects the complex-valued output to `vocab_size * prediction_horizon` before its real part is taken for classical token probabilities.

---

## 💡 Potential & Vision

The QILM architecture, now featuring the Quantum Compressor, lays the groundwork for profound advancements:

*   **Efficiency for Long Contexts:** The Quantum Compressor enables the model to handle longer classical input sequences by summarizing them into shorter, information-rich quantum-inspired tokens. This can drastically improve computational efficiency without sacrificing crucial information.
*   **Richer Semantic Understanding:** The added "phase" dimension could allow the model to capture more nuanced relationships between words and concepts, enabling a deeper understanding of polysemy, metaphor, and subtle emotional cues. The compressor's role is to distil these into compact, potent "quantum features."
*   **Enhanced Contextual Grasp:** Complex embeddings might offer a more sophisticated way to represent context, allowing the model to "superpose" different contextual interpretations and dynamically collapse to the most relevant one during attention. The compressor helps pre-process and align these contexts.
*   **Bridging Classical & Quantum Computing:** QILM serves as a conceptual bridge, demonstrating how principles from quantum mechanics can inspire more powerful classical AI algorithms *without* requiring quantum hardware. This could lead to a new class of "quantum-aware" algorithms.
*   **New Research Paradigms:** This project opens up exciting avenues for research into complex-valued neural networks, their mathematical properties, training dynamics, and potential applications beyond language modeling. It challenges the conventional real-valued foundation of deep learning.

---

## 🎯 Features

*   **Quantum-Inspired Architecture:** Full implementation of complex embeddings, linear layers, and attention mechanisms.
*   **Quantum Compressor:** A novel, jointly trained component for intelligent, learned compression of classical sequences into quantum-inspired tokens.
*   **Autoregressive Language Modeling:** Trains to predict the next word(s) in a sequence, a foundational task for language generation, now with multi-token prediction (`prediction_horizon`).
*   **Interactive Chatbot:** A simple command-line interface to interact with the trained QILM model, demonstrating its conversational capabilities.
*   **Modular Design:** Clearly separated components (`QILM`, `QuantumCompressor`, `SimpleTokenizer`, `AutoregressiveDataset`) for easy understanding and extension.
*   **CPU/GPU Agnostic:** Configurable device support for flexible deployment.
*   **Automatic Mixed Precision (AMP) Support:** Utilizes `torch.cuda.amp.GradScaler` for improved training stability and performance on compatible hardware.

---

## ⚙️ Getting Started

Follow these steps to set up and run the QILM project on your local machine.

### Prerequisites

*   Python 3.8+
*   `torch`

You can install `torch` using pip:
```bash
pip install torch
```

### 📂 Project Structure

```
.
├── chat_qilm.py        # Interactive chatbot interface
├── qilm.py             # Core QILM model architecture, tokenizer, and training logic
├── st1000.txt            # Example dialogue dataset for training
├── quantum_compressor_joint.pt   # Saved weights for QuantumCompressor (after training)
├── qilm_model_joint.pt # Saved weights for QILM (after training)
└── tokenizer.pkl       # Saved tokenizer (after training)
```

### 📊 Training Data Format (`st1000.txt`)

The `st1000.txt` file should contain dialogue pairs, formatted as `U: <user_utterance>` and `B: <bot_response>`, like so:

```
U: Hello
B: Hi there! How can I help you?
U: How are you?
B: I am doing well, thank you for asking.
U: Tell me a joke
B: Why don't scientists trust atoms? Because they make up everything!
```

### 🏃 Running the Model

#### 1. Train the QILM Model (with Quantum Compressor)

The `qilm.py` script handles the joint training process for both the `QuantumCompressor` and the `QILM` model.

```bash
python qilm.py
```

This script will:
*   Load the `st1000.txt` corpus.
*   Initialize the `SimpleTokenizer` and `AutoregressiveDataset`.
*   Instantiate both the `QuantumCompressor` and `QILM` models with the specified hyperparameters.
*   **Jointly train** both models for a predefined number of epochs (default 300).
*   Save `quantum_compressor_joint.pt`, `qilm_model_joint.pt`, and `tokenizer.pkl` in the current directory.

You will see epoch-wise loss updates printed to the console. Training will stop early if the loss drops below a certain threshold.

#### 2. Chat with the Trained Model

Once the training is complete and `quantum_compressor_joint.pt`, `qilm_model_joint.pt`, and `tokenizer.pkl` are generated, you can launch the interactive chatbot using `chat_qilm.py`:

```bash
python chat_qilm.py
```

The chatbot will load both trained models and the tokenizer, then prompt you to enter your messages:

```
Models and Tokenizer loaded successfully on cuda!
QILM Chatbot is ready! Type 'exit' to quit.
You: Hello
Bot: Hi there! How can I help you?
You: How are you?
Bot: I am doing well, thank you for asking.
You: exit
Exiting chat...
```

Type `exit` or `quit` to end the chat session.

---



## 🤝 Contributing

We welcome contributions to this groundbreaking project! Whether it's bug fixes, new features, performance improvements, or documentation enhancements, your input is invaluable. Please feel free to open issues or submit pull requests.

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgements

Inspired by the profound principles of quantum mechanics and the relentless pursuit of more intelligent AI systems. Special thanks to the open-source community for providing the tools and knowledge that make such explorations possible.
```