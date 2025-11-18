# eval_metrics.py
# Usage:
#   python eval_metrics.py --out_dir=4 --data_dir=data/shakespeare_char --device=cuda
import os, math, json, argparse, pickle, numpy as np, torch
from collections import Counter

def load_meta(p):
    with open(p, "rb") as f:
        m = pickle.load(f)
    return m["stoi"], m["itos"], m["vocab_size"]

def load_bin(p):
    arr = np.memmap(p, dtype=np.uint16, mode="r")
    return torch.from_numpy(np.array(arr, dtype=np.int64))

@torch.no_grad()
def eval_ppl(model, data, block_size, device):
    model.eval()
    T = block_size
    tot_loss, steps = 0.0, 0
    for i in range(0, len(data) - T - 1, T):
        x = data[i:i+T].unsqueeze(0).to(device)
        y = data[i+1:i+T+1].unsqueeze(0).to(device)
        _, loss = model(x, y)
        tot_loss += loss.item(); steps += 1
    avg = tot_loss / max(steps, 1)
    return avg, math.exp(avg)  # loss in nats => perplexity = e^loss

@torch.no_grad()
def generate(model, start_id, max_new_tokens, block_size, device, temperature=0.8, top_k=50):
    model.eval()
    idx = torch.tensor([[start_id]], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        x = idx[:, -block_size:]
        logits, _ = model(x, None)
        logits = logits[:, -1, :] / max(temperature, 1e-7)
        # --- safe top-k ---
        if top_k and top_k > 0:
            k = min(top_k, logits.size(-1))  # clamp to vocab
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

def txt_from_ids(ids, itos): return "".join(itos[int(i)] for i in ids)

def trigram_counts(text):
    c = Counter()
    for i in range(len(text)-2):
        c[(text[i], text[i+1], text[i+2])] += 1
    return c

def kl_train_to_gen(train_text, gen_text, alpha=0.5):
    # Specific metric: char 3-gram KL (lower is better)
    ct = trigram_counts(train_text)
    cg = trigram_counts(gen_text)
    keys = set(ct.keys()) | set(cg.keys())
    Nt = sum(ct.values()); Ng = sum(cg.values())
    kl = 0.0
    for k in keys:
        pt = (ct.get(k, 0) + alpha) / (Nt + alpha * len(keys))
        pg = (cg.get(k, 0) + alpha) / (Ng + alpha * len(keys))
        kl += pt * (math.log(pt) - math.log(pg))
    return kl

def distinct_n(text, n=3):
    if len(text) < n: return 0.0
    grams = set(text[i:i+n] for i in range(len(text)-n+1))
    return len(grams) / (len(text)-n+1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",  default="out-shakespeare-char")
    ap.add_argument("--data_dir", default="data/shakespeare_char")
    ap.add_argument("--device",   default="cuda")
    ap.add_argument("--gen_chars", type=int, default=50000)  # generate ~50k chars for stable stats
    args = ap.parse_args()

    # load vocab/meta + splits
    stoi, itos, _ = load_meta(os.path.join(args.data_dir, "meta.pkl"))
    train_ids = load_bin(os.path.join(args.data_dir, "train.bin"))
    val_ids   = load_bin(os.path.join(args.data_dir, "val.bin"))

    # load model
    ckpt = None
    if os.path.exists(os.path.join(args.out_dir, "ckpt.pt")):
        ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location="cpu")
    else:
        # fallback to latest model*.pt
        cands = [f for f in os.listdir(args.out_dir) if f.startswith("model") and f.endswith(".pt")]
        if not cands: raise FileNotFoundError("No checkpoint found in out_dir")
        ckpt = torch.load(os.path.join(args.out_dir, sorted(cands)[-1]), map_location="cpu")

    from model import GPTConfig, GPT
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    block_size = cfg.block_size

    # GENERAL metric 1: PPL on val
    val_loss, val_ppl = eval_ppl(model, val_ids, block_size, args.device)

    # Generate text to evaluate distributional metrics
    start_id = stoi.get("\n", 0)
    gen_ids  = generate(model, start_id, args.gen_chars, block_size, args.device)
    gen_txt  = txt_from_ids(gen_ids, itos)

    # SPECIFIC metric: 3-gram KL(train→gen); GENERAL metric 2: Distinct-3
    # limit train text for speed (first 200k chars is enough)
    train_txt = txt_from_ids(train_ids[:200000].tolist(), itos)
    kld = kl_train_to_gen(train_txt, gen_txt, alpha=0.5)
    d3  = distinct_n(gen_txt, n=3)

    out = {
        "out_dir": args.out_dir,
        "val_loss_nats_per_char": val_loss,
        "val_perplexity": val_ppl,
        "char_3gram_KL_train_to_gen": kld,
        "distinct_3": d3,
        "sample_first_two_lines": "\n".join(gen_txt.splitlines()[:2]),
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
