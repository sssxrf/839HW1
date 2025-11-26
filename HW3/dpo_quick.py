import os, sys, time, math, argparse, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# import your NanoGPT model from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from model import GPT, GPTConfig

# ---------------- Utilities ----------------
def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return meta["itos"], meta["stoi"], meta["vocab_size"]

def load_policy(out_dir, device):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    cfg = ckpt["model_args"]
    model = GPT(GPTConfig(**cfg)).to(device)
    state = {k.replace("_orig_mod.",""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    return model, cfg["block_size"], cfg["vocab_size"]

class RewardNet(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)
    def forward(self, x):
        h,_ = self.gru(self.emb(x))
        r = torch.sigmoid(self.head(h[:,-1,:]))
        return r.squeeze(-1)

def load_reward_model(reward_ckpt, vocab_size, device):
    ck = torch.load(reward_ckpt, map_location=device)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    elif isinstance(ck, dict) and "state_dict" in ck and isinstance(ck["state_dict"], dict):
        sd = ck["state_dict"]
    elif isinstance(ck, dict):
        sd = ck
    else:
        raise RuntimeError("Unrecognized reward checkpoint format")

    sd = {k.replace("module.",""): v for k,v in sd.items()}
    # remap fc.* -> head.* if needed
    if any(k.startswith("fc.") for k in sd.keys()) and not any(k.startswith("head.") for k in sd.keys()):
        sd = {("head."+k.split(".",1)[1] if k.startswith("fc.") else k): v for k,v in sd.items()}

    emb_key = next((k for k in sd if k.endswith("emb.weight")), None)
    if emb_key is None:
        raise RuntimeError("Reward ckpt missing emb.weight")
    if sd[emb_key].shape[0] != vocab_size:
        raise AssertionError(f"Reward vocab {sd[emb_key].shape[0]} != meta vocab {vocab_size}")

    rm = RewardNet(vocab_size).to(device)
    rm.load_state_dict(sd, strict=True)
    rm.eval()
    return rm

# safe top-k
def _topk_filter(logits, k):
    if k is None or k <= 0: return logits
    k = min(int(k), logits.size(-1))
    v, ix = torch.topk(logits, k, dim=1)
    out = logits.new_full(logits.shape, float("-inf"))
    out.scatter_(1, ix, v)
    return out

@torch.no_grad()
def sample_safe(model, start_ids, max_new_tokens, top_k=50):
    # returns a sequence tensor that stays on the same device as the model
    model.eval()
    device = next(model.parameters()).device
    ids = start_ids.to(device)
    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -model.config.block_size:])  # model.py returns (B,1,V) at eval
        last = logits[:, -1, :]  # (B,V)

        if top_k is not None:
            v, ix = torch.topk(last, min(top_k, last.size(-1)))
            mask = torch.full_like(last, float('-inf'))
            mask.scatter_(1, ix, v)
            last = mask

        probs = torch.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, 1)  # (B,1) on device
        ids = torch.cat([ids, next_id], dim=1)
    return ids  # keep on device


def seq_logprob_sum(model, seq):
    """
    Sum of token logprobs for seq under model, teacher-forced.
    Avoids model loss inside by ensuring tensors are contiguous so its .view() won't error.
    Returns (B,) logprob sums.
    """
    inp = seq[:, :-1].contiguous()
    tg  = seq[:,  1:].contiguous()
    logits, _ = model(inp, targets=tg)         # (B, T-1, V)
    logp_all = torch.log_softmax(logits, dim=-1)
    taken = tg.unsqueeze(-1)
    tok_logp = logp_all.gather(-1, taken).squeeze(-1)  # (B, T-1)
    return tok_logp.sum(dim=1)

def make_pairs(policy, rm, stoi, n_pairs=48, gen_len=96, top_k=50):
    device = next(policy.parameters()).device
    bos_id = stoi.get("\n", 0)
    pairs = []  # list of (winner_seq, loser_seq)
    with torch.no_grad():
        for _ in range(n_pairs):
            start = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            a = sample_safe(policy.eval(), start, gen_len, top_k=top_k)
            b = sample_safe(policy.eval(), start, gen_len, top_k=top_k)
            Ra = rm(a).item()
            Rb = rm(b).item()
            if Ra >= Rb:
                pairs.append((a.squeeze(0), b.squeeze(0)))
            else:
                pairs.append((b.squeeze(0), a.squeeze(0)))
    return pairs

def dpo_loss(model, ref, winners, losers, beta=0.1):
    """
    winners, losers: (B, T) Long
    L = -log σ( β[(logπθ(Win)-logπθ(Lose)) - (logπref(Win)-logπref(Lose))] )
    """
    with torch.no_grad():
        ref_w = seq_logprob_sum(ref, winners)
        ref_l = seq_logprob_sum(ref, losers)
        ref_delta = ref_w - ref_l
    pol_w = seq_logprob_sum(model, winners)
    pol_l = seq_logprob_sum(model, losers)
    pol_delta = pol_w - pol_l
    s = beta * (pol_delta - ref_delta)
    return F.softplus(-s).mean()  # -log(sigmoid(s)) = softplus(-s)

def batchify_pairs(pairs, bs):
    for i in range(0, len(pairs), bs):
        ws, ls = zip(*pairs[i:i+bs])
        # pad to same length within batch (left pad with BOS id 0 ok)
        maxlen = max(w.size(0) for w in ws + ls)
        def pad_to(x):
            if x.size(0) == maxlen: return x
            pad = x.new_full((maxlen - x.size(0),), 0)
            return torch.cat([x, pad], dim=0)
        W = torch.stack([pad_to(w) for w in ws], dim=0)
        L = torch.stack([pad_to(l) for l in ls], dim=0)
        yield W, L

@torch.no_grad()
def avg_reward(rm, seqs):
    # seqs: list of LongTensors (B=1,T) potentially on CPU; move them to rm's device
    device = next(rm.parameters()).device
    vals = []
    for s in seqs:
        if s.dim() == 1:  # ensure (1,T)
            s = s.unsqueeze(0)
        s = s.to(device)
        vals.append(rm(s).item())
    return float(sum(vals) / max(1, len(vals)))

@torch.no_grad()
def eval_avg_reward(policy, rm, stoi, n=16, gen_len=96, top_k=50):
    policy.eval()
    device = next(policy.parameters()).device
    bos_tok = stoi.get("\n", 0)
    start = torch.tensor([[bos_tok]], dtype=torch.long, device=device)

    seqs = []
    for _ in range(n):
        seq = sample_safe(policy, start, gen_len, top_k=top_k)  # stays on device
        seqs.append(seq.squeeze(0))  # store (T,) still on device
    return avg_reward(rm, seqs)


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_out_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--reward_ckpt", required=True)
    ap.add_argument("--out", default="out-dpo-quick")
    ap.add_argument("--device", default="cuda")
    # speed knobs
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--train_steps", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gen_len", type=int, default=96)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"

    itos, stoi, vocab_size = load_meta(args.data_dir)
    policy, block_size, vs_pol = load_policy(args.policy_out_dir, device)
    ref = GPT(policy.config).to(device)
    ref.load_state_dict(policy.state_dict(), strict=True)
    ref.eval()

    assert vs_pol == vocab_size, f"vocab mismatch: {vs_pol} vs {vocab_size}"
    rm = load_reward_model(args.reward_ckpt, vocab_size, device)

    # measure BEFORE avg reward quickly
    t0 = time.time()
    pre_R = eval_avg_reward(policy, rm, stoi, n=16, gen_len=args.gen_len, top_k=args.top_k)
    print(f"[pre] avg reward: {pre_R:.3f}")

    # build small preference set
    pairs = make_pairs(policy, rm, stoi, n_pairs=args.pairs, gen_len=args.gen_len, top_k=args.top_k)

    # freeze everything except lm_head
    for p in policy.parameters():
        p.requires_grad_(False)
    policy.lm_head.weight.requires_grad_(True)  # tied to wte

    opt = torch.optim.AdamW([policy.lm_head.weight], lr=args.lr)

    # tiny DPO training
    policy.train()
    step = 0
    for epoch in range(999999):
        for W, L in batchify_pairs(pairs, args.batch_size):
            W = W.to(device); L = L.to(device)
            loss = dpo_loss(policy, ref, W, L, beta=args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([policy.lm_head.weight], 1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{args.train_steps} | dpo_loss {loss.item():.4f}")
            if step >= args.train_steps:
                break
        if step >= args.train_steps:
            break

    # measure AFTER avg reward
    post_R = eval_avg_reward(policy, rm, stoi, n=16, gen_len=args.gen_len, top_k=args.top_k)
    dt = time.time() - t0
    print(f"[post] avg reward: {post_R:.3f}  |  Δ {post_R - pre_R:+.3f}  |  time {dt:.1f}s")

    # save
    torch.save({"model": policy.state_dict(),
                "model_args": policy.config.__dict__},
               os.path.join(args.out, "ckpt.pt"))

    # print a few top samples
    with torch.no_grad():
        bos_id = stoi.get("\n", 0)
        start = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        outs = []
        for _ in range(6):
            s = sample_safe(policy.eval(), start, args.gen_len, top_k=args.top_k).squeeze(0).cpu()
            r = rm(s.unsqueeze(0).to(device)).item()
            outs.append((r, s))
        outs.sort(key=lambda x: -x[0])
        print("\n=== Samples after DPO (sorted by reward) ===")
        for r, s in outs[:3]:
            snippet = decode(s, itos)
            snippet = snippet.replace("\n", " ")  # do backslash work outside the f-string
            snippet = snippet[:160]
            print(f"[{r:.3f}] {snippet}")  # or: print("[{:.3f}] {}".format(r, snippet))


if __name__ == "__main__":
    main()
