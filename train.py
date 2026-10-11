import argparse, json, os, time
import torch
from torch import nn

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--hidden", type=int, default=256)
args = p.parse_args()

run_dir = os.environ.get("RUN_DIR", "./run")   # 잡 안에서는 플랫폼이 넣어 준다
os.makedirs(run_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)

def spiral(n, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * 4 * torch.pi
    y = torch.randint(0, 2, (n,), generator=g)
    r, a = t / (4 * torch.pi), t + y * torch.pi
    x = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1) + 0.02 * torch.randn(n, 2, generator=g)
    return x, y

xtr, ytr = spiral(200_000, 0)
xte, yte = spiral(20_000, 1)
xtr, ytr, xte, yte = [t.to(device) for t in (xtr, ytr, xte, yte)]

model = nn.Sequential(
    nn.Linear(2, args.hidden), nn.ReLU(),
    nn.Linear(args.hidden, args.hidden), nn.ReLU(),
    nn.Linear(args.hidden, 2),
).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

def evaluate():
    model.eval()
    with torch.no_grad():
        return (model(xte).argmax(1) == yte).float().mean().item()

# 선점되어 다시 시작하면 체크포인트에서 이어 간다
ckpt = os.path.join(run_dir, "checkpoint.pt")
start = 0
if os.path.exists(ckpt):
    s = torch.load(ckpt, map_location=device)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["optim"]); start = s["epoch"] + 1
    print(f"resume from epoch {start}", flush=True)

for epoch in range(start, args.epochs):
    t0 = time.time()
    model.train()
    perm = torch.randperm(len(xtr), device=device)
    for i in range(0, len(xtr), 1024):
        idx = perm[i:i + 1024]
        loss = loss_fn(model(xtr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    acc = evaluate()
    print(f"epoch {epoch + 1}/{args.epochs}  loss {loss.item():.4f}  test_acc {acc:.4f}  ({time.time() - t0:.1f}s)", flush=True)
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "epoch": epoch}, ckpt + ".tmp")
    os.replace(ckpt + ".tmp", ckpt)   # 저장 도중 선점돼도 깨진 파일이 남지 않게

with open(os.path.join(run_dir, "result.json"), "w") as f:
    json.dump({"test_acc": evaluate(), **vars(args)}, f, indent=2)
print("done:", run_dir, flush=True)
