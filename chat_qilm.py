import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import os

# --- Core Quantum-Inspired Building Blocks ---
# These class definitions MUST be identical to those used during training.
# This makes the chat script self-contained.

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

    B, T, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be even for rotary encoding."
    half_D = D // 2
    x_real, x_imag = x.real, x.imag
    x1_real, x2_real = x_real[:, :, :half_D], x_real[:, :, half_D:]
    x1_imag, x2_imag = x_imag[:, :, :half_D], x_imag[:, :, half_D:]
    pos = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half_D, device=x.device).float() / half_D)
    angles = pos * freqs
    cos = torch.cos(angles).unsqueeze(0)
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

    def reconstruct(self, *args):
        raise NotImplementedError("Reconstruct is for training QuantumCompressor, not inference.")

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

# --- SimpleTokenizer  ---
class SimpleTokenizer:
    def __init__(self, texts=None, special_tokens=None): 
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]
        
        counter = Counter()
        if texts: 
            for text in texts:
                counter.update(text.split())
            
        self.vocab = {}  
        self.ivocab = {}  
        
        # Assign IDs to special tokens first
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.ivocab[i] = token
        

        next_id = len(special_tokens)
        if texts:
            for word, _ in counter.most_common(): 
                if word not in self.vocab: 
                    self.vocab[word] = next_id
                    self.ivocab[next_id] = word
                    next_id += 1
        
        self.PAD_ID = self.vocab["<pad>"]
        self.UNK_ID = self.vocab["<unk>"]

    def encode(self, text):
        return [self.vocab.get(word, self.UNK_ID) for word in text.split()]

    def decode(self, ids):
        return " ".join([self.ivocab.get(i, "<unk>") for i in ids])

    def __len__(self):
        return len(self.vocab)

# === Load Models and Tokenizer ===
# Now loads *both* QuantumCompressor and QILM
def load_models_and_tokenizer(qilm_model_path, compressor_model_path, tokenizer_path, config):
    # Load the tokenizer first to get the correct vocab_size for model initialization
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        
        config["vocab_size"] = len(tokenizer) 
    except FileNotFoundError:
        print(f"Error: Tokenizer file '{tokenizer_path}' not found.")
        print("Please ensure you have run your training script (e.g., qilm_train.py) first to generate this file.")
        exit()

    # Initialize QuantumCompressor
    compressor = QuantumCompressor(
        vocab_size=config["vocab_size"], # Use the loaded vocab_size
        embed_dim=config["embed_dim"],
        compression_ratio=config["compression_ratio"],
        num_compressor_layers=config["num_compressor_layers"]
    )
    compressor.load_state_dict(torch.load(compressor_model_path, map_location=config["device"]))
    compressor.to(config["device"])
    compressor.eval() 

    # Initialize QILM
    qilm = QILM(
        vocab_size=config["vocab_size"], 
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        prediction_horizon=config["prediction_horizon"]
    )
    qilm.load_state_dict(torch.load(qilm_model_path, map_location=config["device"]))
    qilm.to(config["device"])
    qilm.eval() 

    return qilm, compressor, tokenizer


# === Chat Function ===
def chat(qilm_model, compressor_model, tokenizer, device, classical_seq_len=32, max_gen_len=64, 
         temperature=1.0, top_k=10, prediction_horizon=1):
    
    print("QILM Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        prompt = f"user: {user_input} bot:"
        tokens = tokenizer.encode(prompt) 
        generated = tokens[:] 
        
        # Calculate how many full prediction steps are needed for max_gen_len
        num_prediction_steps = math.ceil(max_gen_len / prediction_horizon)

        with torch.no_grad():
            for _ in range(num_prediction_steps):
                # 1. Get the latest classical input sequence
                x_classical = generated[-classical_seq_len:]
                # Pad to fixed classical_seq_len
                x_padded = [tokenizer.PAD_ID] * (classical_seq_len - len(x_classical)) + x_classical
                input_ids_classical = torch.tensor([x_padded]).to(device)

                # 2. Compress the classical input into quantum-inspired tokens
                compressed_input = compressor_model(input_ids_classical)
                
                # 3. Pass compressed input to the QILM
                complex_logits = qilm_model(compressed_input) # (1, prediction_horizon, vocab_size) complex
                
                # 4. Predict and append tokens for the entire horizon
                last_token_was_stop = False # Flag to stop outer loop if an early stop token is hit
                for p_idx in range(prediction_horizon):
                    # Get logits for current token prediction within the horizon
                    current_complex_logits = complex_logits[0, p_idx, :]
                    
                    # Apply Born rule analogy: probability = |amplitude|^2
                    magnitudes_squared = torch.abs(current_complex_logits)**2
                    
                    # Scale by temperature and apply softmax to get probabilities
                    probs = F.softmax(magnitudes_squared / temperature, dim=-1)
                    
                    # Get top_k probabilities and indices for sampling
                    topk_probs, topk_indices = torch.topk(probs, top_k)
                    # Rescale topk_probs to sum to 1 before multinomial
                    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                    
                    # Sample the next token ID
                    next_token_id = topk_indices[torch.multinomial(topk_probs, 1).item()].item()
                    
                    # --- Stop conditions ---
                    if next_token_id in [tokenizer.PAD_ID, tokenizer.UNK_ID]:
                        last_token_was_stop = True
                        break # Break from inner loop
                    
                    generated.append(next_token_id) # Add generated token to the sequence

                    if len(generated) >= len(tokens) + max_gen_len or \
                       tokenizer.ivocab.get(next_token_id, '') == "user:":
                        last_token_was_stop = True
                        break # Break from inner loop
                
                # Break outer loop if inner loop already broke due to stop condition
                if last_token_was_stop:
                    break


        # Decode the full generated sequence and extract bot's response
        decoded = tokenizer.decode(generated)
        bot_response_parts = decoded.split("bot:")
        if len(bot_response_parts) > 1:
            # Take the part after the last "bot:" and before any subsequent "user:"
            bot_response = bot_response_parts[-1].split("user:")[0].strip()
        else:
            # If no "bot:" found, return the full generated text without prompt
            bot_response = decoded.replace(prompt, '').strip()

        print("Bot:", bot_response)


# === Entry Point ===
if __name__ == "__main__":
    # --- Configuration ---
    # These MUST match the configurations used for training!
    config = {
        "embed_dim": 64,             
        "num_heads": 4,              
        "ff_dim": 128,               
        "num_layers": 2,             
        "compression_ratio": 4,      
        "num_compressor_layers": 1,  
        "prediction_horizon": 2,     
        "classical_seq_len": 32,     
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    # Paths to your trained models and tokenizer
    # These names reflect the joint training saves ("_joint.pt")
    qilm_model_path = "qilm_model_joint.pt" 
    compressor_model_path = "quantum_compressor_joint.pt"
    tokenizer_path = "tokenizer.pkl"

    # Load both models and the tokenizer
    qilm_model, compressor_model, tokenizer = load_models_and_tokenizer(
        qilm_model_path, compressor_model_path, tokenizer_path, config
    )
    print(f"Models and Tokenizer loaded successfully on {config['device']}!")

    # Start the chat interface
    # Pass necessary configuration parameters to the chat function
    chat(qilm_model, compressor_model, tokenizer, config["device"], 
         classical_seq_len=config["classical_seq_len"], 
         prediction_horizon=config["prediction_horizon"])