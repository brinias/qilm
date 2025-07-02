import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import math
import pickle


class ComplexEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.real = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.imag = nn.Parameter(torch.randn(vocab_size, embed_dim))

    def forward(self, input_ids):
        real = self.real[input_ids]
        imag = self.imag[input_ids]
        return torch.complex(real, imag)  


# === Quantum-Inspired Complex Embedding ===
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        real = self.real(x.real) - self.imag(x.imag)
        imag = self.real(x.imag) + self.imag(x.real)
        return torch.complex(real, imag)


# === Rotary Positional Encoding ===
def apply_rotary_pos_emb(x):
    B, T, D = x.size()
    assert D % 2 == 0, "Embedding dimension must be even for rotary encoding."
    half_D = D // 2

    x1_real = x.real[:, :, :half_D]
    x1_imag = x.imag[:, :, :half_D]
    x2_real = x.real[:, :, half_D:]
    x2_imag = x.imag[:, :, half_D:]

    pos = torch.arange(T, dtype=torch.float32, device=x.device).unsqueeze(1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half_D, device=x.device).float() / half_D)
    angles = pos * freqs  # [T, half_D]
    cos = torch.cos(angles).unsqueeze(0)  # [1, T, half_D]
    sin = torch.sin(angles).unsqueeze(0)

    rotated_real = x1_real * cos - x1_imag * sin
    rotated_imag = x1_real * sin + x1_imag * cos

    real = torch.cat([rotated_real, x2_real], dim=-1)
    imag = torch.cat([rotated_imag, x2_imag], dim=-1)

    return torch.complex(real, imag)


# === Quantum-Inspired Multihead Attention with Complex Vectors ===
class ComplexMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.q_proj = ComplexLinear(embed_dim, embed_dim)
        self.k_proj = ComplexLinear(embed_dim, embed_dim)
        self.v_proj = ComplexLinear(embed_dim, embed_dim)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, D = x.size()
        H = self.num_heads
        d = D // H

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        attn_scores = torch.einsum("bhid,bhjd->bhij", q, k.conj()).real
        attn_probs = F.softmax(attn_scores / math.sqrt(d), dim=-1)
        attn_probs = attn_probs.to(v.dtype)  # <== 🔧 fix here
        attn_output = torch.einsum("bhij,bhjd->bhid", attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)



# === Transformer Block with Quantum Rotation ===
class QILMTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = ComplexMultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff1 = ComplexLinear(embed_dim, ff_dim)
        self.act = nn.GELU()
        self.ff2 = ComplexLinear(ff_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = apply_rotary_pos_emb(x)
        attn_out = self.attn(x)
        res1 = self.norm1(x.real + attn_out.real)

        ff_mid = self.ff1(attn_out)
        ff_activated = torch.complex(self.act(ff_mid.real), ff_mid.imag)
        ff_out = self.ff2(ff_activated)

        res2 = self.norm2(res1 + ff_out.real)
        return torch.complex(res2, attn_out.imag + ff_out.imag)



# === Full Model ===
class QILM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = ComplexEmbedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            QILMTransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x.real)

# === Tokenizer and Dataset ===
class SimpleTokenizer:
    def __init__(self, texts):
        counter = Counter(" ".join(texts).split())
        self.vocab = {word: i for i, (word, _) in enumerate(counter.items())}
        self.ivocab = {i: w for w, i in self.vocab.items()}

    def encode(self, text):
        return [self.vocab[word] for word in text.split() if word in self.vocab]

    def decode(self, ids):
        return " ".join([self.ivocab[i] for i in ids if i in self.ivocab])

    def __len__(self):
        return len(self.vocab)

class AutoregressiveDataset(Dataset):
    def __init__(self, tokenizer, texts, seq_len=12):
        self.data = []
        for text in texts:
            ids = tokenizer.encode(text)
            for i in range(1, len(ids)):
                x = ids[max(0, i - seq_len):i]
                y = ids[i]
                x = [0] * (seq_len - len(x)) + x
                self.data.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# === Training and Generation ===
def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits = logits[:, -1, :]
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def generate(model, tokenizer, prompt, device, max_len=20, seq_len=12):
    model.eval()
    tokens = tokenizer.encode(prompt)
    generated = tokens[:]
    with torch.no_grad():
        for _ in range(max_len):
            x = tokens[-seq_len:]
            x = [0] * (seq_len - len(x)) + x
            input_ids = torch.tensor([x]).to(device)
            logits = model(input_ids)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            tokens.append(next_token)
            if tokenizer.ivocab.get(next_token, '') == "user:":
                break
    decoded = tokenizer.decode(generated)
    bot_response = decoded.split("bot:")[-1].split("user:")[0].strip()
    return bot_response

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
            i += 1  # skip malformed entries
    return pairs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load from file instead of hardcoded
    corpus = load_dialogue_pairs("test.txt")    # Name of your dataset

    tokenizer = SimpleTokenizer(corpus)
    dataset = AutoregressiveDataset(tokenizer, corpus, seq_len=12)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = QILM(len(tokenizer), embed_dim=32, num_heads=2, ff_dim=64, num_layers=1).to(device) # Depend on the dataset size

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(300):   # Depends on the dataset size
        loss = train(model, dataloader, optimizer, loss_fn, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

        if loss < 0.1265:   # Early stopping
            print(f"🛑 Stopping early at epoch {epoch+1}: Loss reached {loss:.4f}")
            break

    # Save tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Save model weights
    torch.save(model.state_dict(), "qilm.pt")
    print("Model and tokenizer saved!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        prompt = f"user: {user_input} bot:"
        response = generate(model, tokenizer, prompt, device)
        print("Bot:", response)
