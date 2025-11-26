# align/train_reward.py
import argparse, torch, torch.nn as nn, torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="HW3/reward_data.pt")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--out", default="HW3/reward_model.pt")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

blob = torch.load(args.data, map_location="cpu")
vocab_size = blob["vocab_size"]; W = blob["window"]
xtr, ytr = blob["train_x"], blob["train_y"]
xva, yva = blob["val_x"],   blob["val_y"]

class RewardGRU(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.emb(x)          # [B,T,E]
        _, hT = self.gru(h)      # hT: [1,B,H]
        out = self.head(hT[0]).squeeze(-1)  # [B]
        return out

device = torch.device(args.device if torch.cuda.is_available() and args.device=="cuda" else "cpu")
model = RewardGRU(vocab_size).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

def batches(x, y, bs):
    n = len(x)
    idx = torch.randperm(n)
    for i in range(0,n,bs):
        j = idx[i:i+bs]
        yield x[j], y[j]

for ep in range(args.epochs):
    model.train()
    tr_loss = 0.0; ntr = 0
    for xb, yb in batches(xtr, ytr, args.batch_size):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tr_loss += loss.item()*len(xb); ntr += len(xb)
    model.eval()
    with torch.no_grad():
        pred = model(xva.to(device))
        val = F.mse_loss(pred, yva.to(device)).item()
    print(f"epoch {ep+1}: train MSE {tr_loss/ntr:.5f}  val MSE {val:.5f}")

torch.save({"state_dict": model.state_dict(), "vocab_size": vocab_size, "window": W}, args.out)
print("Saved", args.out)
