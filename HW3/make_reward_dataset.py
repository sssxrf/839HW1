# align/make_reward_dataset.py
import os, pickle, argparse, numpy as np, torch
from prefs import s_density_reward

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="e.g., data/whitman_shk")
parser.add_argument("--window", type=int, default=128)
parser.add_argument("--num_train", type=int, default=20000)
parser.add_argument("--num_val", type=int, default=2000)
parser.add_argument("--out", default="HW3/reward_data.pt")
args = parser.parse_args()

meta = pickle.load(open(os.path.join(args.data_dir,"meta.pkl"), "rb"))
itos, stoi, vocab_size = meta["itos"], meta["stoi"], meta["vocab_size"]

train_ids = np.memmap(os.path.join(args.data_dir,"train.bin"), dtype=np.uint16, mode="r")
val_ids   = np.memmap(os.path.join(args.data_dir,"val.bin"),   dtype=np.uint16, mode="r")

def sample_windows(memmap_ids, n):
    L = len(memmap_ids)
    W = args.window
    xs, ys = [], []
    for _ in range(n):
        i = np.random.randint(0, L - W - 1)
        seq = memmap_ids[i:i+W].astype(np.int64)
        # decode to text for heuristic
        text = "".join(itos[int(t)] for t in seq)
        r = s_density_reward(text)
        xs.append(seq)
        ys.append(r)
    x = torch.tensor(np.stack(xs), dtype=torch.long)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    return x, y

xtr, ytr = sample_windows(train_ids, args.num_train)
xva, yva = sample_windows(val_ids,   args.num_val)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({
    "vocab_size": vocab_size,
    "window": args.window,
    "train_x": xtr, "train_y": ytr,
    "val_x": xva,   "val_y": yva,
}, args.out)
print("Saved", args.out, xtr.shape, ytr.shape, xva.shape, yva.shape)
