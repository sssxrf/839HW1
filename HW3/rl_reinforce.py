# HW3/rl_reinforce.py
import os, sys, argparse, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- import NanoGPT model from repo root (parent of HW3) ---
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from model import GPT, GPTConfig


# --------------------------
# Utilities
# --------------------------
def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    itos = meta["itos"]
    stoi = meta["stoi"]
    vocab_size = meta["vocab_size"]
    return itos, stoi, vocab_size


def load_policy(out_dir, device):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["model_args"]
    model = GPT(GPTConfig(**cfg)).to(device)
    # strip any wrappers like "_orig_mod."
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(state, strict=True)
    model.train()
    return model, cfg["block_size"], cfg["vocab_size"]


# --------------------------
# Reward model (GRU over tokens)
# --------------------------
class RewardNet(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)  # final scalar

    def forward(self, x):  # x: (B, T)
        h, _ = self.gru(self.emb(x))
        r = torch.sigmoid(self.head(h[:, -1, :]))  # (B,1) -> (B,)
        return r.squeeze(-1)


def load_reward_model(reward_ckpt, vocab_size, device):
    ck = torch.load(reward_ckpt, map_location=device)
    # Extract a real state_dict
    if isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck and isinstance(ck["state_dict"], dict):
        sd = ck["state_dict"]
    elif isinstance(ck, dict):
        sd = ck
    else:
        raise RuntimeError("Unrecognized reward checkpoint format")

    # strip possible prefixes
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # Remap 'fc.*' -> 'head.*' if needed
    has_head = any(k.startswith("head.") for k in sd.keys())
    has_fc = any(k.startswith("fc.") for k in sd.keys())
    if has_fc and not has_head:
        sd = {("head." + k.split(".", 1)[1] if k.startswith("fc.") else k): v for k, v in sd.items()}

    # Check embedding size matches vocab
    emb_key = next((k for k in sd.keys() if k.endswith("emb.weight")), None)
    if emb_key is None:
        raise RuntimeError("Reward checkpoint missing 'emb.weight'")
    ck_vocab = sd[emb_key].shape[0]
    if ck_vocab != vocab_size:
        raise AssertionError(f"Reward vocab {ck_vocab} != meta vocab {vocab_size}")

    rm = RewardNet(vocab_size).to(device)
    rm.load_state_dict(sd, strict=True)
    rm.eval()
    print("number of parameters: %.2fM" % (sum(p.numel() for p in rm.parameters())/1e6,))
    return rm


# --------------------------
# Safe top-k + sampling
# --------------------------
def _topk_filter(logits, top_k):
    if top_k is None or top_k <= 0:
        return logits
    k = min(int(top_k), logits.size(-1))
    v, ix = torch.topk(logits, k, dim=1)
    out = logits.new_full(logits.shape, float('-inf'))
    out.scatter_(1, ix, v)
    return out


def gpt_sample_and_logprobs(model, start_ids, max_new_tokens, top_k=50):
    """
    Sample tokens and return:
      ids:       (B, T0 + L)
      logprobs:  (B, L)
      entropies: (B, L)
    NaN/Inf-safe; falls back to uniform if a row degenerates.
    """
    device = next(model.parameters()).device
    ids = start_ids.clone().to(device)
    logprobs = []
    entropies = []

    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -model.config.block_size:], targets=None)  # (B,1,V)
        last = logits[:, -1, :]  # (B, V)

        # sanitize logits first
        last = torch.nan_to_num(last, nan=0.0, posinf=1e4, neginf=-1e4)

        # top-k filter
        last = _topk_filter(last, top_k)

        # compute log-probs and probs
        logp_all = torch.log_softmax(last, dim=-1)          # (B, V)
        probs = torch.exp(logp_all)                          # (B, V)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # fix any degenerate rows: sum<=0 or non-finite
        row_sums = probs.sum(dim=1, keepdim=True)            # (B,1)
        bad = (~torch.isfinite(row_sums)) | (row_sums <= 0)
        if bad.any():
            # fallback to uniform over vocab for those rows
            probs[bad.squeeze(1)] = 1.0 / probs.size(-1)
            row_sums = probs.sum(dim=1, keepdim=True)

        probs = probs / row_sums.clamp_min(1e-12)

        # sample action
        next_id = torch.multinomial(probs, num_samples=1)    # (B,1)
        # logprob of the chosen action
        lp = logp_all.gather(1, next_id)                     # (B,1)
        # entropy
        ent = -(probs * logp_all).sum(dim=1, keepdim=True)   # (B,1)

        ids = torch.cat([ids, next_id], dim=1)
        logprobs.append(lp)
        entropies.append(ent)

    logprobs = torch.cat(logprobs, dim=1)    # (B, L)
    entropies = torch.cat(entropies, dim=1)  # (B, L)
    return ids, logprobs, entropies


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_out_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--reward_ckpt", required=True)
    ap.add_argument("--out", default="out-rlhf")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gen_len", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--entropy_coef", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    use_cuda = (args.device.startswith("cuda") and torch.cuda.is_available())
    device = args.device if use_cuda else "cpu"

    # --- tokenizer meta ---
    itos, stoi, vocab_size = load_meta(args.data_dir)

    # --- policy & frozen reference ---
    policy, block_size, vs_pol = load_policy(args.policy_out_dir, device)
    print("number of parameters: %.2fM" % (policy.get_num_params()/1e6,))
    ref = GPT(policy.config).to(device)
    ref.load_state_dict(policy.state_dict(), strict=True)
    ref.eval()
    print("number of parameters: %.2fM" % (ref.get_num_params()/1e6,))

    assert vocab_size == vs_pol, f"vocab mismatch: meta {vocab_size} vs policy {vs_pol}"

    # --- reward model ---
    rm = load_reward_model(args.reward_ckpt, vocab_size, device)

    # --- optimizer ---
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # BOS prompt
    bos_id = stoi.get("\n", 0)
    bos = torch.full((args.batch_size, 1), bos_id, dtype=torch.long, device=device)

    for step in range(1, args.steps + 1):
        policy.train()
        ref.eval()
        rm.eval()

        # 1) sample from current policy
        seq, logp, Htok = gpt_sample_and_logprobs(policy, bos, args.gen_len, top_k=args.top_k)

        # 2) KL vs reference
        ref_in = seq[:, :-1].contiguous()
        ref_tg = seq[:, 1:].contiguous()
        ref_logits, _ = ref(ref_in, targets=ref_tg)          # (B, T-1, V)
        ref_lp = torch.log_softmax(ref_logits, dim=-1)       # (B, T-1, V)
        taken = ref_tg.unsqueeze(-1)                          # (B, T-1, 1)
        ref_taken_lp = ref_lp.gather(-1, taken).squeeze(-1)   # (B, T-1)

        Tkl = min(logp.size(1), ref_taken_lp.size(1))
        logp_taken = logp[:, :Tkl]
        ref_taken_lp = ref_taken_lp[:, :Tkl].detach()
        kl = (logp_taken - ref_taken_lp).mean(dim=1)          # (B,)

        # 3) rewards (no grad through reward)
        with torch.no_grad():
            R = rm(seq)  # (B,)
        Rn = (R - R.mean()) / (R.std() + 1e-8)

        # 4) total loss
        reinforce_loss = -(Rn.unsqueeze(1) * logp).sum(dim=1).mean()
        kl_loss = args.kl_coef * kl.mean()
        ent_loss = -float(args.entropy_coef) * Htok.mean()
        loss = reinforce_loss + kl_loss + ent_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            print(f"step {step}/{args.steps} | loss {loss.item():.4f} | "
                  f"R {R.mean().item():.4f} | KL {kl.mean().item():.4f} | H {Htok.mean().item():.4f}")

        if step % 200 == 0:
            torch.save({
                "model": policy.state_dict(),
                "model_args": policy.config.__dict__,
            }, os.path.join(args.out, "ckpt.pt"))

    torch.save({
        "model": policy.state_dict(),
        "model_args": policy.config.__dict__,
    }, os.path.join(args.out, "ckpt.pt"))


if __name__ == "__main__":
    main()
