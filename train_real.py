"""Train GNet on real CDnet videos using SuBSENSE pseudo-labels.
Videos listed in `--train-videos` are used for training; the rest are held out
for evaluation. Tests whether target-domain (pseudo-labelled) data is needed.
"""
import argparse, os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.gnet import GNet
from data.real import RealCDnetTrain
from utils.losses import BCEDiceLoss


def split_videos(root, holdout_cats):
    from data.cdnet import CDnetVideo
    cats = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    train, test = [], []
    for c in cats:
        for v in sorted(os.listdir(os.path.join(root, c))):
            if not os.path.isdir(os.path.join(root, c, v)):
                continue
            (test if c in holdout_cats else train).append((c, v))
    return train, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="datasets/cdnet/CDnet14")
    p.add_argument("--holdout", default="badWeather,nightVideos,turbulence,thermal")
    p.add_argument("--split-file", default="", help="train split 'cat/video' lines")
    p.add_argument("--label", default="subsense", choices=["subsense", "gt"])
    p.add_argument("--bg", default="causal_med", choices=["running", "causal_med"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=513)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--steps-per-epoch", type=int, default=600)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tsam-depth", type=int, default=12)
    p.add_argument("--fusion", default="cat")
    p.add_argument("--out", default="checkpoints/gnet_real")
    return p.parse_args()


def main_run(a):
    import random
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.split_file:
        rows = [x.strip() for x in open(a.split_file) if x.strip()]
        train_v = [tuple(x.split("/")) for x in rows]
        test_v = []
    else:
        train_v, test_v = split_videos(a.root, set(a.holdout.split(",")))
    print(f"train videos {len(train_v)} test videos {len(test_v)}", flush=True)
    ds = RealCDnetTrain(a.root, train_v, size=a.size, label=a.label, bg=a.bg)
    print(f"train samples {len(ds)}", flush=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=(a.workers > 0))
    m = GNet(num_classes=1, tsam_depth=a.tsam_depth, fusion=a.fusion).to(dev)
    crit = BCEDiceLoss()
    opt = torch.optim.SGD(m.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    total = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=a.lr * 0.01)
    log = open(os.path.join(a.out, "train.log"), "a")
    def L(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    L(json.dumps(vars(a)))
    it = 0; t0 = time.time()
    for ep in range(a.epochs):
        m.train(); run = 0.0; n = 0
        for step, (img, bg, mask) in enumerate(dl):
            if step >= a.steps_per_epoch: break
            img, bg, mask = img.to(dev), bg.to(dev), mask.to(dev)
            loss = crit(m(img, bg), mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item(); n += 1; it += 1
            if it % 20 == 0:
                L(f"ep{ep} it{it} loss {run/n:.4f} {(time.time()-t0)/it:.2f}s/it"); run = 0; n = 0
        torch.save({"model": m.state_dict(), "args": vars(a), "iter": it},
                   os.path.join(a.out, "last.pt"))
        L(f"[epoch {ep}] saved it={it}")
    L("DONE")


if __name__ == "__main__":
    main_run(main())
