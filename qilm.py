import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import math
import pickle
import os
from torch.cuda.amp import GradScaler 

# --- Utility Functions ---
def load_dialogue_pairs(filename):
    pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith('U:') and lines[i+1].startswith('B:'):
            user = lines[i][2:].strip()
            bot = lines[i+1][2:].strip()
            pairs.append(f"user: {user} bot: {bot}")
            i += 2
        else:
            i += 1
    return pairs

# --- Core Quantum-Inspired Building Blocks ---

class ComplexEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.real = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.imag = nn.Parameter(torch.randn(vocab_size, embed_dim))

    def forward(self, input_ids):
        real = self.real[input_ids]
        imag = self.imag[input_ids]
        return torch.complex(real, imag)

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        real = self.real(x.real) - self.imag(x.imag)
        imag = self.real(x.imag) + self.imag(x.real)
        return torch.complex(real, imag)

def apply_rotary_pos_emb(x):
    """
    Applies Rotary Positional Encoding to a complex tensor.
    Rotates the first half of the embedding dimensions using complex multiplication.
    (a + ib)(cos(theta) + i*sin(theta)) = (a*cos - b*sin) + i(a*sin + b*cos)
    """
    B, T, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be even for rotary encoding."
    half_D = D // 2

    x_real = x.real
    x_imag = x.imag

    x1_real, x2_real = x_real[:, :, :half_D], x_real[:, :, half_D:]
    x1_imag, x2_imag = x_imag[:, :, :half_D], x_imag[:, :, half_D:]

    pos = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half_D, device=x.device).float() / half_D)
    angles = pos * freqs  # [T, half_D]
    cos = torch.cos(angles).unsqueeze(0)  # [1, T, half_D]
    sin = torch.sin(angles).unsqueeze(0)

    rotated_real1 = x1_real * cos - x1_imag * sin
    rotated_imag1 = x1_real * sin + x1_imag * cos

    real = torch.cat([rotated_real1, x2_real], dim=-1)
    imag = torch.cat([rotated_imag1, x2_imag], dim=-1)

    return torch.complex(real, imag)


class ComplexMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.q_proj = ComplexLinear(embed_dim, embed_dim)
        self.k_proj = ComplexLinear(embed_dim, embed_dim)
        self.v_proj = ComplexLinear(embed_dim, embed_dim)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

    def forward(self, x):
        B, T, D = x.size()
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k.conj()).real / math.sqrt(d)
        
        attn_probs = F.softmax(attn_scores, dim=-1)

      
        attn_probs_complex = attn_probs.to(v.dtype)
        
        attn_output = torch.einsum("bhij,bhjd->bhid", attn_probs_complex, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)

class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.real_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.imag_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return torch.complex(self.real_norm(x.real), self.imag_norm(x.imag))

class QILMTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = ComplexMultiheadAttention(embed_dim, num_heads)
        self.norm1 = ComplexLayerNorm(embed_dim)
        self.ff1 = ComplexLinear(embed_dim, ff_dim)
        self.act = nn.GELU()
        self.ff2 = ComplexLinear(ff_dim, embed_dim)
        self.norm2 = ComplexLayerNorm(embed_dim)

    def forward(self, x):
        x_rot = apply_rotary_pos_emb(x)
        
        attn_out = self.attn(x_rot)
        x_norm1 = self.norm1(x_rot + attn_out)

        ff_mid = self.ff1(x_norm1)
        ff_activated = torch.complex(self.act(ff_mid.real), ff_mid.imag)
        ff_out = self.ff2(ff_activated)
        
        x_norm2 = self.norm2(x_norm1 + ff_out)
        return x_norm2


# --- Quantum Compressor (our "Quantum Tokenizer") ---
class ComplexPool(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        B, T, D = x.size()
        
        pad_amount = 0
        if T % self.pool_size != 0:
            pad_amount = self.pool_size - (T % self.pool_size)
            padded_real = F.pad(x.real, (0, 0, 0, pad_amount), "constant", 0)
            padded_imag = F.pad(x.imag, (0, 0, 0, pad_amount), "constant", 0)
            x = torch.complex(padded_real, padded_imag)
            T = x.size(1)

        x_reshaped = x.view(B, T // self.pool_size, self.pool_size, D)
        
        pooled_real = x_reshaped.real.mean(dim=-2)
        pooled_imag = x_reshaped.imag.mean(dim=-2)
        
        return torch.complex(pooled_real, pooled_imag)

class QuantumCompressor(nn.Module):
    def __init__(self, vocab_size, embed_dim, compression_ratio=4, num_compressor_layers=1):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.classical_embedding = nn.Embedding(vocab_size, embed_dim)
        self.to_complex_proj = ComplexLinear(embed_dim, embed_dim)

        self.complex_encoder_blocks = nn.ModuleList([
            QILMTransformerBlock(embed_dim, num_heads=2, ff_dim=embed_dim*2) 
            for _ in range(num_compressor_layers)
        ])
        
        self.compress_pool = ComplexPool(compression_ratio)

        # NOTE: Reconstruction decoder is NOT used in joint training,
        # but kept for potential separate pre-training or diagnostics.
        self.decompress_proj = ComplexLinear(embed_dim, embed_dim * compression_ratio)
        self.reconstruction_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x_real = self.classical_embedding(input_ids)

        x_complex_init = torch.complex(x_real, torch.zeros_like(x_real))
        x_complex = self.to_complex_proj(x_complex_init)

        for block in self.complex_encoder_blocks:
            x_complex = block(x_complex)

        compressed_quantum_tokens = self.compress_pool(x_complex)
        
        return compressed_quantum_tokens

    # NOTE: This method is not called during joint training
    def reconstruct(self, compressed_quantum_tokens, original_sequence_length):
        B, T_comp, D = compressed_quantum_tokens.size()

        decompressed_complex_raw = self.decompress_proj(compressed_quantum_tokens)
        
        reconstructed_complex_full = decompressed_complex_raw.view(B, T_comp * self.compression_ratio, D)

        reconstructed_complex_full = reconstructed_complex_full[:, :original_sequence_length, :]

        reconstructed_logits = self.reconstruction_head(reconstructed_complex_full.real)
        return reconstructed_logits

# --- Main QILM Model (Now consumes QuantumCompressor output) ---
class QILM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, prediction_horizon):
        super().__init__()
        self.blocks = nn.ModuleList([
            QILMTransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.head = ComplexLinear(embed_dim, vocab_size * prediction_horizon) 
        self.vocab_size = vocab_size
        self.prediction_horizon = prediction_horizon

    def forward(self, compressed_quantum_tokens):
        x = compressed_quantum_tokens
        for block in self.blocks:
            x = block(x)
        
        raw_complex_logits = self.head(x[:, -1, :]) 
        
        reshaped_complex_logits = raw_complex_logits.view(-1, self.prediction_horizon, self.vocab_size)
        return reshaped_complex_logits

# --- Quantum-Inspired Loss Function (Now handles multi-token targets) ---
class QuantumInspiredCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1.0, epsilon=1e-8, ignore_index=-100):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.ignore_index = ignore_index # For padding in target sequence

    def forward(self, complex_logits, target):
        complex_logits_flat = complex_logits.view(-1, complex_logits.size(-1))
        target_flat = target.view(-1)

        magnitudes_squared = torch.abs(complex_logits_flat)**2

        scaled_magnitudes = magnitudes_squared / self.temperature + self.epsilon 

        log_probabilities = F.log_softmax(scaled_magnitudes, dim=-1)

        loss = F.nll_loss(log_probabilities, target_flat, ignore_index=self.ignore_index)
        return loss

# --- Tokenizer and Dataset ---
class SimpleTokenizer:
    def __init__(self, texts, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]
        
        counter = Counter()
        for text in texts:
            counter.update(text.split())
            
        self.vocab = {word: i + len(special_tokens) for i, (word, _) in enumerate(counter.items())}
        self.ivocab = {i + len(special_tokens): word for i, (word, _) in enumerate(counter.items())}
        
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.ivocab[i] = token
        
        self.PAD_ID = self.vocab["<pad>"]
        self.UNK_ID = self.vocab["<unk>"]

    def encode(self, text):
        return [self.vocab.get(word, self.UNK_ID) for word in text.split()]

    def decode(self, ids):
        return " ".join([self.ivocab.get(i, "<unk>") for i in ids])

    def __len__(self):
        return len(self.vocab)

class AutoregressiveDataset(Dataset):
    def __init__(self, tokenizer, texts, seq_len=32, prediction_horizon=1):
        self.data = []
        self.prediction_horizon = prediction_horizon
        
        all_ids = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text))
        
        # We need to ensure that when we take a target sequence of `prediction_horizon` tokens,
        # we still have a valid input sequence of `seq_len` tokens preceding it.
        # So loop up to `len(all_ids) - prediction_horizon`
        for i in range(1, len(all_ids) - prediction_horizon + 1): 
            # Input sequence (x)
            x = all_ids[max(0, i - seq_len):i]
            x_padded = [tokenizer.PAD_ID] * (seq_len - len(x)) + x
            
            # Target sequence (y_seq)
            y_seq = all_ids[i : i + prediction_horizon]
            
            # Pad target sequence if not enough tokens (should not happen with loop change)
            # This padding is mostly for cases where all_ids is too short for prediction_horizon
            y_padded = y_seq + [tokenizer.PAD_ID] * (prediction_horizon - len(y_seq))

            self.data.append((torch.tensor(x_padded), torch.tensor(y_padded)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- Training and Generation Functions ---

# Unified training function for joint training
def train_joint(compressor_model, qilm_model, dataloader, optimizer, loss_fn, device, scaler):
    compressor_model.train() # Both models in training mode
    qilm_model.train()

    total_loss = 0
    for x_classical_ids, y_classical_ids_seq in dataloader:
        x_classical_ids, y_classical_ids_seq = x_classical_ids.to(device), y_classical_ids_seq.to(device)

        optimizer.zero_grad()
        
        # Forward pass through the QuantumCompressor
        # Gradients WILL flow through here
        compressed_x = compressor_model(x_classical_ids) 
        
        # Forward pass through QILM
        complex_logits_from_qilm = qilm_model(compressed_x)

        # Calculate loss
        loss = loss_fn(complex_logits_from_qilm, y_classical_ids_seq) 
        
        # Backward pass and optimization
        scaler.scale(loss).backward() # Scale loss for stability
        scaler.step(optimizer)       # Update weights
        scaler.update()              # Update scaler for next iteration

        total_loss += loss.item()
    return total_loss / len(dataloader)

def generate(qilm_model, compressor_model, tokenizer, prompt, device, max_len=64, classical_seq_len=32, 
             temperature=1.0, top_k=10, prediction_horizon=1):
    
    qilm_model.eval()
    compressor_model.eval()

    tokens = tokenizer.encode(prompt)
    generated = tokens[:]
    
    num_prediction_steps = math.ceil(max_len / prediction_horizon)

    with torch.no_grad():
        for _ in range(num_prediction_steps):
            x_classical = generated[-classical_seq_len:]
            x_padded = [tokenizer.PAD_ID] * (classical_seq_len - len(x_classical)) + x_classical
            input_ids_classical = torch.tensor([x_padded]).to(device)

            compressed_input = compressor_model(input_ids_classical)
            complex_logits = qilm_model(compressed_input) # (1, prediction_horizon, vocab_size) complex
            
            for p_idx in range(prediction_horizon):
                current_complex_logits = complex_logits[0, p_idx, :]
                
                magnitudes_squared = torch.abs(current_complex_logits)**2
                probs = F.softmax(magnitudes_squared / temperature, dim=-1)
                
                topk_probs, topk_indices = torch.topk(probs, top_k)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                
                next_token_id = topk_indices[torch.multinomial(topk_probs, 1).item()].item()
                
                if next_token_id in [tokenizer.PAD_ID, tokenizer.UNK_ID]:
                    break # Early exit from inner loop

                generated.append(next_token_id)
                
                # Check if max_len is reached or a stop word is generated
                if len(generated) >= len(tokens) + max_len or \
                   tokenizer.ivocab.get(next_token_id, '') == "user:":
                    break # Early exit from inner loop
            
            # Break outer loop if inner loop already broke
            if next_token_id in [tokenizer.PAD_ID, tokenizer.UNK_ID] or \
               len(generated) >= len(tokens) + max_len or \
               tokenizer.ivocab.get(next_token_id, '') == "user:":
                break


    decoded = tokenizer.decode(generated)
    bot_response_parts = decoded.split("bot:")
    if len(bot_response_parts) > 1:
        bot_response = bot_response_parts[-1].split("user:")[0].strip()
    else:
        bot_response = decoded.replace(prompt, '').strip()

    return bot_response

if __name__ == "__main__":

    EMBED_DIM = 64
    NUM_HEADS = 4
    FF_DIM = EMBED_DIM * 2
    NUM_QILM_LAYERS = 2
    
    COMPRESSION_RATIO = 4 
    NUM_COMPRESSOR_LAYERS = 1
    CLASSICAL_SEQ_LEN = 32 
    
    PREDICTION_HORIZON = 2 

    assert CLASSICAL_SEQ_LEN % COMPRESSION_RATIO == 0, \
        f"Classical sequence length ({CLASSICAL_SEQ_LEN}) must be a multiple of compression ratio ({COMPRESSION_RATIO})."
    
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler() 

    # --- Data Loading and Tokenizer ---
    corpus = load_dialogue_pairs("st1000.txt")
    tokenizer = SimpleTokenizer(corpus) 
    
    VOCAB_SIZE = len(tokenizer)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    # --- Initialize Models ---
    quantum_compressor = QuantumCompressor(VOCAB_SIZE, EMBED_DIM, 
                                           compression_ratio=COMPRESSION_RATIO, 
                                           num_compressor_layers=NUM_COMPRESSOR_LAYERS).to(device)
    
    qilm_model = QILM(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_QILM_LAYERS, 
                      prediction_horizon=PREDICTION_HORIZON).to(device)

    # --- Optimizer and Loss Functions (Now only one optimizer for both models) ---
    # Combine parameters from both models for joint optimization
    joint_optimizer = torch.optim.Adam(list(quantum_compressor.parameters()) + \
                                       list(qilm_model.parameters()), lr=0.001)
    
    # Only one loss function needed for the joint training objective
    joint_loss_fn = QuantumInspiredCrossEntropyLoss(temperature=1.0, 
                                                   ignore_index=tokenizer.PAD_ID).to(device)


    # --- DataLoaders (only need one for the language modeling task) ---
    # The dataset provides (x_classical_ids, y_classical_ids_seq) Default --> batch_size=4
    joint_dataloader = AutoregressiveDataset(tokenizer, corpus, seq_len=CLASSICAL_SEQ_LEN, prediction_horizon=PREDICTION_HORIZON)
    joint_dataloader = DataLoader(joint_dataloader, batch_size=128, shuffle=True, pin_memory=True) 


    # --- Single Phase: Joint Training of Quantum Compressor and QILM ---
    print("\n--- Starting Joint Training of Quantum Compressor and QILM ---")
    TOTAL_JOINT_EPOCHS = 300 
    for epoch in range(TOTAL_JOINT_EPOCHS):
        loss = train_joint(quantum_compressor, qilm_model, joint_dataloader, joint_optimizer, joint_loss_fn, device, scaler)
        print(f"Joint Epoch {epoch+1}: Loss = {loss:.4f}")

        if loss < 0.115: # Adjust stopping condition based on observed loss behavior
            print(f"🛑 Stopping early at epoch {epoch+1}: Loss reached {loss:.4f}")
            break
    
    # Save both models after training
    torch.save(quantum_compressor.state_dict(), "quantum_compressor_joint.pt")
    torch.save(qilm_model.state_dict(), "qilm_model_joint.pt")
    print("Jointly trained models saved!")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved!")

    print("\nQuantum Chatbot Ready! Type 'exit' to stop.")

    # --- Generation Loop ---
    # If you want to load the models back for generation after training (e.g., in a separate run):
    # quantum_compressor.load_state_dict(torch.load("quantum_compressor_joint.pt", map_location=device))
    # qilm_model.load_state_dict(torch.load("qilm_model_joint.pt", map_location=device))
    # with open("tokenizer.pkl", "rb") as f:
    #     tokenizer = pickle.load(f)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        prompt = f"user: {user_input} bot:"
        response = generate(qilm_model, quantum_compressor, tokenizer, prompt, device, 
                            classical_seq_len=CLASSICAL_SEQ_LEN, prediction_horizon=PREDICTION_HORIZON)
        print("Bot:", response)