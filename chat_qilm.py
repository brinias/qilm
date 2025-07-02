import torch
import torch.nn.functional as F
import pickle
from qilm import QILM  
from tokenizer import SimpleTokenizer  

# === Load Model and Tokenizer ===
def load_model_and_tokenizer(model_path, tokenizer_path, config):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    model = QILM(
        vocab_size=len(tokenizer),
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"]
    )
    model.load_state_dict(torch.load(model_path, map_location=config["device"]))
    model.to(config["device"])
    model.eval()
    return model, tokenizer


# === Chat Function ===
def chat(model, tokenizer, device, seq_len=12, max_len=50):
    print("QILM Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        prompt = f"user: {user_input} bot:"
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
        print("Bot:", bot_response)


# === Entry Point ===
if __name__ == "__main__":
    config = {
        "embed_dim": 16,
        "num_heads": 2,
        "ff_dim": 32,
        "num_layers": 3,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    model_path = "qilm.pt"
    tokenizer_path = "tokenizer.pkl"

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, config)
    chat(model, tokenizer, config["device"])
