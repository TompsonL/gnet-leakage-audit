"""Train GNet on synthetic Cityscapes+COCO data."""
import argparse
import os
import time
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.gnet import GNet
from data.synth import SynthDataset, COCOObjects, CityscapesBackgrounds
from utils.losses import BCEDiceLoss, BoundaryAwareLoss


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cityscapes", default="datasets/raw/cityscapes_train.parquet")
    p.add_argument("--cityscapes-val", default=None)
    p.add_argument("--coco-ann", default="datasets/raw/annotations/instances_val2017.json")
    p.add_argument("--coco-img", default="datasets/raw/val2017")
    p.add_argument("--size", type=int, default=513)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fusion", default="cat", choices=["sub", "cat"])
    p.add_argument("--tsam-depth", type=int, default=12)
    p.add_argument("--tsam-heads", type=int, default=8)
    p.add_argument("--tsam-patch", type=int, default=5)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--no-dda", action="store_true")
    p.add_argument("--no-pmb", action="store_true")
    p.add_argument("--no-tsam", action="store_true")
    p.add_argument("--no-bg", action="store_true",
                   help="current-frame-only ablation: feed img as its own background")
    p.add_argument("--rel-gate", action="store_true",
                   help="M1: gate the differential branch by a learnable reliability map")
    p.add_argument("--boundary", action="store_true",
                   help="M3: boundary-aware differential attention + boundary loss")
    p.add_argument("--boundary-weight", type=float, default=1.0)
    p.add_argument("--degrade", action="store_true",
                   help="T3: add causal-background degradation simulation")
    p.add_argument("--shadow", action="store_true",
                   help="T3: add cast shadow during compositing")
    p.add_argument("--synth-bg", default="pmb", choices=["pmb", "causal"],
                   help="L2b: synthetic background model (pmb or causal estimator)")
    p.add_argument("--motion-labels", action="store_true",
                   help="T4: train on label-free teacher masks from raw video")
    p.add_argument("--motion-root", default="datasets/cdnet/CDnet14")
    p.add_argument("--motion-categories", default="")
    p.add_argument("--motion-window", type=int, default=30)
    p.add_argument("--motion-thr", type=int, default=35)
    p.add_argument("--motion-split", default="")
    p.add_argument("--motion-label-stride", type=int, default=5)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/run1")
    p.add_argument("--max-iter", type=int, default=0, help="stop after N iters (smoke test)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_model(a):
    depth = 1 if a.no_tsam else a.tsam_depth
    return GNet(num_classes=1, embed_dim=a.embed_dim, tsam_depth=depth,
                tsam_heads=a.tsam_heads, tsam_patch=a.tsam_patch,
                fusion=a.fusion, pretrained=a.pretrained,
                rel_gate=a.rel_gate, boundary=a.boundary)


def main():
    a = get_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    def _seed_worker(worker_id):
        ws = (torch.initial_seed() + worker_id) % (2 ** 32)
        random.seed(ws); np.random.seed(ws)

    if a.motion_labels:
        from data.motion_dataset import MotionLabelDataset
        cats = a.motion_categories.split(",") if a.motion_categories else None
        ds = MotionLabelDataset(a.motion_root, categories=cats,
                                split_file=(a.motion_split or None), size=a.size,
                                K=a.motion_window, thr=a.motion_thr,
                                label_stride=a.motion_label_stride,
                                with_rel=a.rel_gate)
    else:
        cs = CityscapesBackgrounds(a.cityscapes)
        coco = COCOObjects(a.coco_ann, a.coco_img)
        ds = SynthDataset(cs, coco, size=a.size, length=10 ** 9,
                          dda=not a.no_dda, pmb=not a.no_pmb, with_rel=a.rel_gate,
                          degrade=a.degrade, shadow=a.shadow, bg_mode=a.synth_bg)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                    pin_memory=True, drop_last=True,
                    persistent_workers=(a.workers > 0),
                    worker_init_fn=_seed_worker)

    model = build_model(a).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params: {nparam/1e6:.1f}M", flush=True)
    crit = (BoundaryAwareLoss(boundary_weight=a.boundary_weight)
            if a.boundary else BCEDiceLoss())
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum,
                          weight_decay=a.weight_decay)
    total = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=a.lr * 0.01)

    log = open(os.path.join(a.out, "train.log"), "a")
    def L(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    L(json.dumps(vars(a)))
    it = 0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        run = 0.0; n = 0
        for step, batch in enumerate(dl):
            if step >= a.steps_per_epoch:
                break
            img, bg, mask = batch[0], batch[1], batch[2]
            rel = batch[3] if a.rel_gate else None
            img = img.to(device, non_blocking=True)
            if a.no_bg:
                bg = img
            bg = bg.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if rel is not None:
                rel = rel.to(device, non_blocking=True)
            out = model(img, bg, rel)
            loss = crit(out, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            run += loss.item(); n += 1; it += 1
            if it % 20 == 0:
                L(f"ep{ep} it{it} loss {run/n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time()-t0)/it:.2f}s/it")
                run = 0.0; n = 0
            if it % 200 == 0:
                torch.save({"model": model.state_dict(), "args": vars(a), "iter": it},
                           os.path.join(a.out, "last.pt"))
            if a.max_iter and it >= a.max_iter:
                break
        torch.save({"model": model.state_dict(), "args": vars(a), "iter": it},
                   os.path.join(a.out, "last.pt"))
        L(f"[epoch {ep}] saved. it={it}")
        if a.max_iter and it >= a.max_iter:
            L("max-iter reached, stop"); break
    L("DONE")


if __name__ == "__main__":
    main()
