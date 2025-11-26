# HW3/test_reward.py
# Evaluate a saved policy (ckpt.pt in --policy_out_dir) with a trained reward model.

import os, argparse, pickle, torch, torch.nn as nn, torch.nn.functional as F

import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
# ---- Reward model (must match your train_reward.py) ----
class RewardGRU(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.emb(x)
        _, hT = self.gru(h)
        return self.head(hT[0]).squeeze(-1)

# ---- helpers ----
def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        m = pickle.load(f)
    return m["itos"], m["stoi"], m["vocab_size"]

def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)

def load_policy(out_dir, device):
    from model import GPTConfig, GPT
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt["model_args"]; model_args["dropout"] = 0.0
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, model_args["block_size"]

@torch.no_grad()
def sample_ids(model, start_id, num_new_tokens, block_size, top_k=50, device="cuda"):
    # minimal top-k sampler
    x = torch.tensor([[start_id]], dtype=torch.long, device=device)
    for _ in range(num_new_tokens):
        x_cond = x[:, -block_size:]
        logits, _ = model(x_cond, x_cond)  # nanoGPT forward returns (logits, loss)
        logits = logits[:, -1, :]
        if top_k is not None:
            v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)))
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, ix, v)
            logits = mask
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_id), dim=1)
    return x[0].tolist()

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_out_dir", required=True, help="folder containing ckpt.pt")
    ap.add_argument("--data_dir", required=True, help="data folder with meta.pkl (same vocab)")
    ap.add_argument("--reward_ckpt", default="HW3/reward_model.pt", help="trained reward model path")
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--gen_len", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    itos, stoi, vocab_size = load_meta(args.data_dir)
    policy, block_size = load_policy(args.policy_out_dir, device)

    blob = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardGRU(blob["vocab_size"]).to(device).eval()
    rm.load_state_dict(blob["state_dict"])
    W = blob["window"]

    results = []
    start_id = stoi.get("\n", 0)
    for _ in range(args.samples):
        ids = sample_ids(policy, start_id, args.gen_len, block_size, args.top_k, device=device)
        text = decode(ids, itos)
        # score last W tokens
        tid = torch.tensor([ids[-W:]], dtype=torch.long, device=device)
        r = rm(tid).item()
        # avoid backslashes inside f-string by precomputing
        preview = text.strip()[:120].replace("\n", " ").replace("\r", " ")
        results.append((r, preview))

    results.sort(key=lambda x: x[0], reverse=True)
    print("\n=== High-reward samples ===")
    for r, t in results[:3]:
        print(f"[{r:.3f}] {t}")
    print("\n=== Low-reward samples ===")
    for r, t in results[-3:]:
        print(f"[{r:.3f}] {t}")

if __name__ == "__main__":
    main()
