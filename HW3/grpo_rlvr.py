# HW3/grpo_rlvr.py  (GRPO with memory-light teacher-forcing logprobs)
import os, sys, time, csv, random, math, argparse, pickle
from typing import List, Tuple
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from model import GPT, GPTConfig

# ---------------- token helpers ----------------
def encode_str(s: str, stoi: dict) -> torch.Tensor:
    return torch.tensor([stoi.get(ch, 0) for ch in s], dtype=torch.long)

def decode_ids(ids: torch.Tensor, itos: List[str]) -> str:
    return "".join(itos[int(x)] for x in ids.tolist())

def pad_batch_tokenize(prompts: List[str], stoi: dict, pad_ch: str = "\n") -> Tuple[torch.Tensor, int]:
    toks = [encode_str(p, stoi) for p in prompts]
    pad_id = stoi.get(pad_ch, 0)
    T0 = max(t.numel() for t in toks)
    B = len(toks)
    out = torch.full((B, T0), pad_id, dtype=torch.long)
    for i, t in enumerate(toks):
        out[i, : t.numel()] = t
    return out, T0

# ---------------- models ----------------
def load_policy(out_dir, device):
    ck = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    cfg = ck["model_args"]
    gpt = GPT(GPTConfig(**cfg)).to(device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    gpt.load_state_dict(state, strict=True)
    gpt.train()
    print("number of parameters: %.2fM" % (gpt.get_num_params()/1e6,))
    return gpt

# ---------------- sampling (no grad) ----------------
@torch.no_grad()
def sample_ids(model: GPT, start_ids: torch.Tensor, max_new_tokens: int, top_k: int = 50):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = start_ids.to(device)

    for _ in range(max_new_tokens):
        idx_cond = ids if ids.size(1) <= model.config.block_size else ids[:, -model.config.block_size:]
        # targets=None => model returns only last-step logits (B,1,V)
        logits, _ = model(idx_cond, targets=None)
        last = logits[:, -1, :]  # (B,V)
        if top_k is not None:
            v, ix = torch.topk(last, min(top_k, last.size(-1)))
            mask = torch.full_like(last, -float("inf"))
            mask.scatter_(1, ix, v)
            last = mask
        probs = F.softmax(last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
        ids = torch.cat([ids, next_id], dim=1)

    if was_training: model.train()
    return ids

# ---------------- log-probs via teacher-forcing (with grad) ----------------
def continuation_logprobs(model: GPT, seq: torch.Tensor, L: int):
    """
    seq: (B, T0+L). Returns per-token logp of the last L tokens under the model.
    Uses one teacher-forced forward => low-memory but differentiable.
    """
    logits, _ = model(seq[:, :-1].contiguous(), targets=seq[:, 1:].contiguous())  # (B, T-1, V)
    logp_all = F.log_softmax(logits, dim=-1)                                      # (B, T-1, V)
    taken = seq[:, 1:].unsqueeze(-1)                                              # (B, T-1, 1)
    logp_taken = logp_all.gather(-1, taken).squeeze(-1)                           # (B, T-1)
    return logp_taken[:, -L:]                                                     # (B, L)

# ---------------- simple verifier: count 's'/'S' ----------------
def verifier_score_str(s: str, rmax: int = 40) -> float:
    cnt = sum(ch in ("s", "S") for ch in s)
    return min(cnt, rmax) / float(rmax)

def batch_prompts(n: int) -> List[str]:
    seeds = ["\n", "O ", "The ", "Sing ", "When ", "In ", "On ", "For ", "As ", "By "]
    out = []
    for i in range(n):
        base = seeds[i % len(seeds)]
        tail = random.choice(["", " the", " a", " of", " to", " my", " your"])
        out.append(base + tail)
    return out

@torch.no_grad()
def eval_mean_verifier(policy, itos, stoi, prompts, gen_len=64, top_k=50, device="cuda"):
    starts, _ = pad_batch_tokenize(prompts, stoi)
    starts = starts.to(device)
    ids = sample_ids(policy, starts, gen_len, top_k=top_k)
    cont = ids[:, -gen_len:]
    scores = []
    for i in range(cont.size(0)):
        s = decode_ids(cont[i], itos)
        scores.append(verifier_score_str(s))
    return sum(scores) / len(scores)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_out_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="out-grpo-rlvr")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_steps", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)    # prompts per step (B)
    ap.add_argument("--group_size", type=int, default=4)     # completions per prompt (K)
    ap.add_argument("--gen_len", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--beta", type=float, default=5.0)       # GRPO softmax temperature
    ap.add_argument("--kl_coef", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--prompts_eval", type=int, default=80)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"

    # tokenizer
    with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    itos, stoi, vocab_size = meta["itos"], meta["stoi"], meta["vocab_size"]

    # policy + frozen reference
    policy = load_policy(args.policy_out_dir, device)
    ref = load_policy(args.policy_out_dir, device)
    ref.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # eval prompts
    eval_prompts = batch_prompts(args.prompts_eval)
    pre_mean = eval_mean_verifier(policy, itos, stoi, eval_prompts, gen_len=args.gen_len,
                                  top_k=args.top_k, device=device)
    print(f"[pre] mean verifier: {pre_mean:.3f}")

    trace_path = os.path.join(args.out, "reward_trace.csv")
    with open(trace_path, "w", newline="") as fcsv:
        csv.writer(fcsv).writerow(["step", "mean_verifier"])

    B, K, L = args.batch_size, args.group_size, args.gen_len
    t0 = time.time()

    for step in range(1, args.train_steps + 1):
        prompts = batch_prompts(B)
        starts, _ = pad_batch_tokenize(prompts, stoi)
        starts = starts.to(device)

        # replicate each prompt K times: shape (B*K, T0)
        starts_rep = starts.repeat_interleave(K, dim=0)

        # ---- 1) SAMPLE WITHOUT GRAD (low memory)
        seq = sample_ids(policy, starts_rep, L, top_k=args.top_k)              # (B*K, T0+L)

        # ---- 2) COMPUTE LOGP WITH GRAD (teacher forcing)
        logp = continuation_logprobs(policy, seq, L)                           # (B*K, L)

        # ---- reference log-probs for KL (no grads)
        with torch.no_grad():
            logp_ref = continuation_logprobs(ref, seq, L)                      # (B*K, L)

        # ---- verifier scores on continuation strings (no grads)
        with torch.no_grad():
            cont = seq[:, -L:]
            scores = []
            for i in range(cont.size(0)):
                s = decode_ids(cont[i], itos)
                scores.append(verifier_score_str(s))
            scores = torch.tensor(scores, device=device).view(B, K)            # (B,K)
            mean_i = scores.mean(dim=1, keepdim=True)
            w_ij = F.softmax(args.beta * (scores - mean_i), dim=1)             # (B,K)

        # ---- GRPO objective
        sum_logp = logp.sum(dim=1).view(B, K)                                  # (B,K) grad flows
        reinforce_loss = -(w_ij * sum_logp).sum(dim=1).mean()

        kl_tok = (logp - logp_ref).mean()                                      # scalar
        loss = reinforce_loss + args.kl_coef * kl_tok

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step == 1:
            mean_now = eval_mean_verifier(policy, itos, stoi, eval_prompts, gen_len=L,
                                          top_k=args.top_k, device=device)
            with open(trace_path, "a", newline="") as fcsv:
                csv.writer(fcsv).writerow([step, f"{mean_now:.6f}"])
            elapsed = time.time() - t0
            print(f"step {step}/{args.train_steps} | loss {loss.item():.4f} | mean_v {mean_now:.3f} | {elapsed:.1f}s")

    post_mean = eval_mean_verifier(policy, itos, stoi, eval_prompts, gen_len=L,
                                   top_k=args.top_k, device=device)
    print(f"[post] mean verifier: {post_mean:.3f} | Δ {post_mean - pre_mean:+.3f} | time {time.time()-t0:.1f}s")

    # show a few samples
    fresh = batch_prompts(8)
    starts, _ = pad_batch_tokenize(fresh, stoi)
    starts = starts.to(device)
    ids = sample_ids(policy, starts, L, top_k=args.top_k)
    cont = ids[:, -L:]
    scored = []
    for i in range(cont.size(0)):
        s = decode_ids(cont[i], itos)
        scored.append((verifier_score_str(s), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    print("\n=== Samples after GRPO (sorted by verifier) ===")
    for r, s in scored[:3]:
        clean = s.replace("\n", " ")  # precompute to avoid backslashes inside f-string expression
        print(f"[{r:.3f}] {clean[:160]}")


    torch.save({"model": policy.state_dict(), "model_args": policy.config.__dict__},
               os.path.join(args.out, "ckpt.pt"))

if __name__ == "__main__":
    main()
