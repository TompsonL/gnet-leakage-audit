"""Synthetic training data construction (paper Sections 3.3-3.4).

- Backgrounds: Cityscapes street scenes (unlabeled).
- Foreground objects: COCO 2017 instance masks.
- Compositing + differential data augmentation (Algorithm 2).
- Optional projective-motion-blur (PMB) background to mimic SuBSENSE output.
"""
import io
import json
import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------- COCO
class COCOObjects:
    def __init__(self, ann_file, img_dir, min_area=1024):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_ids = [i for i in self.coco.getImgIds()
                        if os.path.isfile(self._path(i))]
        self.ann_ids = []
        for i in self.img_ids:
            for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=i, iscrowd=0)):
                if a["area"] >= min_area and a["segmentation"]:
                    self.ann_ids.append((i, a["id"]))
        self.min_area = min_area
        self.cache = {}

    def _path(self, i):
        info = self.coco.loadImgs(i)[0]
        return os.path.join(self.img_dir, info["file_name"])

    def _load_img(self, i):
        if i not in self.cache:
            im = cv2.imread(self._path(i), cv2.IMREAD_COLOR)
            self.cache[i] = im
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def random_object(self):
        i, aid = random.choice(self.ann_ids)
        img = self._load_img(i)
        ann = self.coco.loadAnns(aid)[0]
        m = self.coco.annToMask(ann)
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return None
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if (x1 - x0) < 4 or (y1 - y0) < 4:
            return None
        obj = img[y0:y1, x0:x1].copy()
        om = (m[y0:y1, x0:x1] > 0).astype(np.uint8)
        return obj, om


# ----------------------------------------------------------- Cityscapes
class CityscapesBackgrounds:
    def __init__(self, parquet_file):
        import pandas as pd
        from PIL import Image
        self.df = pd.read_parquet(parquet_file)
        self.col = "image" if "image" in self.df.columns else self.df.columns[0]
        self.Image = Image

    def __len__(self):
        return len(self.df)

    def get(self, idx=None):
        if idx is None:
            idx = random.randrange(len(self.df))
        v = self.df.iloc[idx][self.col]
        if isinstance(v, dict):
            if v.get("bytes") is not None:
                img = self.Image.open(io.BytesIO(v["bytes"])).convert("RGB")
            else:
                img = self.Image.open(v["path"]).convert("RGB")
        elif isinstance(v, (bytes, bytearray)):
            img = self.Image.open(io.BytesIO(v)).convert("RGB")
        else:
            img = self.Image.fromarray(np.asarray(v))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------- helpers
def _affine(img, mask, angle, tx, ty, scale, border=cv2.BORDER_REFLECT101):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=border)
    mask2 = None
    if mask is not None:
        mask2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img2, mask2


def _photometric(img, b, c, noise=0.0, blur=0):
    x = img.astype(np.float32)
    x = x * b + c
    x = np.clip(x, 0, 255)
    if blur > 0:
        k = int(blur) * 2 + 1
        x = cv2.GaussianBlur(x, (k, k), 0)
    if noise > 0:
        n = np.random.randn(*x.shape).astype(np.float32) * noise * 255
        x = x + n
    return np.clip(x, 0, 255).astype(np.uint8)


def _salt_pepper(img, amount):
    if amount <= 0:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    n = int(amount * h * w)
    ys = np.random.randint(0, h, n)
    xs = np.random.randint(0, w, n)
    vals = np.random.randint(0, 2, n) * 255
    out[ys, xs] = vals[:, None]
    return out


# ------------------------------------------------------------ dataset
class SynthDataset(Dataset):
    def __init__(self, cityscapes, coco, size=513, length=4000,
                 max_objects=5, dda=True, pmb=True, pmb_frames=(5, 15),
                 pmb_lambda=0.12, max_obj_frac=0.5, with_rel=False,
                 degrade=False, shadow=False, rel_tau=100.0, bg_mode="pmb"):
        self.cs = cityscapes
        self.coco = coco
        self.size = size
        self.length = length
        self.max_objects = max_objects
        self.dda = dda
        self.pmb = pmb
        self.pmb_frames = pmb_frames
        self.pmb_lambda = pmb_lambda
        self.max_obj_frac = max_obj_frac
        self.with_rel = with_rel
        self.degrade = degrade
        self.shadow = shadow
        self.rel_tau = rel_tau
        self.bg_mode = bg_mode

    def __len__(self):
        return self.length

    def _compose(self):
        bg = self.cs.get()
        h, w = bg.shape[:2]
        n = random.randint(0, self.max_objects)
        scene = bg.copy()
        mask = np.zeros((h, w), np.uint8)
        placed = []
        for _ in range(n):
            obj = self.coco.random_object()
            if obj is None:
                continue
            o, om = obj
            s = random.uniform(0.5, 2.0)
            oh, ow = o.shape[:2]
            s = min(s, self.max_obj_frac * h / max(oh, 1), self.max_obj_frac * w / max(ow, 1))
            oh2, ow2 = max(4, int(oh * s)), max(4, int(ow * s))
            o = cv2.resize(o, (ow2, oh2), interpolation=cv2.INTER_LINEAR)
            om = cv2.resize(om, (ow2, oh2), interpolation=cv2.INTER_NEAREST)
            if random.random() < 0.5:
                o = o[:, ::-1]
                om = om[:, ::-1]
            px = random.randint(0, max(0, w - ow2))
            py = random.randint(0, max(0, h - oh2))
            if self.shadow:
                # cast shadow: darken a shifted, blurred copy before pasting
                soff_x = random.randint(2, max(3, ow2 // 5))
                soff_y = random.randint(2, max(3, oh2 // 5))
                sx, sy = px + soff_x, py + soff_y
                sx0, sy0 = max(0, sx), max(0, sy)
                sx1, sy1 = min(w, sx + ow2), min(h, sy + oh2)
                if sx1 > sx0 and sy1 > sy0:
                    omx0, omy0 = sx0 - sx, sy0 - sy
                    sm = om[omy0:omy0 + (sy1 - sy0), omx0:omx0 + (sx1 - sx0)]
                    sm = cv2.GaussianBlur(sm.astype(np.float32), (7, 7), 0)[..., None]
                    sroi = scene[sy0:sy1, sx0:sx1].astype(np.float32)
                    scene[sy0:sy1, sx0:sx1] = np.clip(
                        sroi * (1 - 0.55 * sm), 0, 255).astype(np.uint8)
            roi = scene[py:py + oh2, px:px + ow2]
            alpha = cv2.GaussianBlur(om.astype(np.float32), (5, 5), 0)[..., None]
            roi[:] = (alpha * o + (1 - alpha) * roi).astype(np.uint8)
            mroi = mask[py:py + oh2, px:px + ow2]
            mroi[:] = np.maximum(mroi, (alpha[..., 0] > 0.5).astype(np.uint8))
            placed.append((o, om, px, py))
        return bg, scene, mask, placed

    def _pmb_background(self, bg, placed):
        """Projective-motion-blur background + variance-based reliability.

        Reliability uses the SAME definition as the causal estimator at
        inference: R = 1/(1 + V/tau), where V is the per-pixel temporal
        variance (channel-averaged) of the synthetic frame stack. This makes
        the training and inference reliability semantics identical (M1).
        """
        h, w = bg.shape[:2]
        if not placed:
            return bg.copy(), np.ones((h, w), np.float32)
        n = random.randint(*self.pmb_frames)
        theta = random.uniform(0, 2 * np.pi)
        d = random.uniform(3, 5)
        mean = np.zeros((h, w, 3), np.float64)
        M2 = np.zeros((h, w, 3), np.float64)
        for i in range(n):
            frame = bg.astype(np.float32).copy()
            dx, dy = i * d * np.sin(theta), i * d * np.cos(theta)
            for o, om, px, py in placed:
                oh, ow = o.shape[:2]
                x = int(round(px + dx)); y = int(round(py + dy))
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(w, x + ow), min(h, y + oh)
                if x1 <= x0 or y1 <= y0:
                    continue
                ox0, oy0 = x0 - x, y0 - y
                sub = o[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]
                msub = om[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]
                region = frame[y0:y1, x0:x1]
                af = (msub.astype(np.float32) * self.pmb_lambda)[..., None]
                region[:] = af * sub + (1 - af) * region
            k = i + 1
            delta = frame.astype(np.float64) - mean
            mean += delta / k
            M2 += delta * (frame.astype(np.float64) - mean)
        var = (M2 / max(n, 1)).mean(axis=2)  # (h,w) channel-averaged variance
        rel = (1.0 / (1.0 + var / self.rel_tau)).astype(np.float32)
        bg_out = np.clip(mean, 0, 255).astype(np.uint8)
        return bg_out, rel

    def _causal_background(self, bg, placed, K=10):
        """L2b/L2a: render a short clip and run the SAME causal estimator as at
        inference -- causal temporal MEDIAN (K=10), which L2a found best
        (residual-F1 0.619 vs 0.538 for EMA). Reliability = 1/(1 + V/tau) with
        V the window variance, matching `running_background_median`.
        Returns (bg_uint8, rel_float32).
        """
        h, w = bg.shape[:2]
        if not placed:
            return bg.copy(), np.ones((h, w), np.float32)
        theta = random.uniform(0, 2 * np.pi)
        speed = random.uniform(2.0, 8.0)
        dx, dy = speed * np.cos(theta), speed * np.sin(theta)
        stack = []
        for t in range(K):
            frame = bg.astype(np.float32).copy()
            for o, om, px, py in placed:
                oh, ow = o.shape[:2]
                x = int(round(px + t * dx)); y = int(round(py + t * dy))
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(w, x + ow), min(h, y + oh)
                if x1 <= x0 or y1 <= y0:
                    continue
                ox0, oy0 = x0 - x, y0 - y
                sub = o[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)].astype(np.float32)
                msub = om[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)].astype(np.float32)
                af = msub[..., None]
                region = frame[y0:y1, x0:x1]
                region[:] = af * sub + (1 - af) * region
            stack.append(frame)
        S = np.stack(stack, 0)
        med = np.median(S, axis=0)
        var = S.var(axis=0).mean(axis=2)
        rel = (1.0 / (1.0 + var / self.rel_tau)).astype(np.float32)
        return np.clip(med, 0, 255).astype(np.uint8), rel

    def _degrade_background(self, bg, placed):
        """T3: simulate the error statistics of a *causal* background estimate.

        Adds (i) ghost/static-foreground residue, (ii) illumination drift,
        (iii) mild blur + noise. Returns degraded background and a reliability
        map (1 - normalized degradation strength). Unlike PMB (motion blur),
        this targets the low-frequency biases a causal running estimate exhibits.
        """
        h, w = bg.shape[:2]
        out = bg.astype(np.float32).copy()
        ghost = np.zeros((h, w), np.float32)
        # (i) ghost / static foreground residue
        for o, om, px, py in placed:
            if random.random() > 0.7:
                continue
            oh, ow = o.shape[:2]
            dx = random.randint(-max(3, ow // 4), max(3, ow // 4))
            dy = random.randint(-max(3, oh // 4), max(3, oh // 4))
            x, y = px + dx, py + dy
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + ow), min(h, y + oh)
            if x1 <= x0 or y1 <= y0:
                continue
            ox0, oy0 = x0 - x, y0 - y
            sub = o[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)].astype(np.float32)
            msub = om[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)].astype(np.float32)
            a = msub[..., None] * random.uniform(0.05, 0.20)
            region = out[y0:y1, x0:x1]
            region[:] = a * sub + (1 - a) * region
            ghost[y0:y1, x0:x1] += a[..., 0]
        # (ii) illumination drift
        gain = random.uniform(0.92, 1.08)
        bias = random.uniform(-12, 12)
        out = out * gain + bias
        # (iii) mild blur + noise
        if random.random() < 0.4:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        out = out + np.random.randn(h, w, 1).astype(np.float32) * random.uniform(0, 4)
        out = np.clip(out, 0, 255).astype(np.uint8)
        # same reliability formula as inference: R = 1/(1 + V/tau), with the
        # degradation deviation energy as the per-pixel variance proxy.
        var_deg = ((out.astype(np.float32) - bg.astype(np.float32)) ** 2).mean(axis=2)
        rel = (1.0 / (1.0 + var_deg / self.rel_tau)).astype(np.float32)
        return out, rel

    def __getitem__(self, k):
        bg, scene, mask, placed = self._compose()
        h, w = scene.shape[:2]
        if self.bg_mode == "causal":
            train_bg, rel = self._causal_background(bg, placed)
        elif self.pmb:
            train_bg, rel = self._pmb_background(bg, placed)
        else:
            train_bg, rel = bg.copy(), np.ones((h, w), np.float32)
        if self.degrade:
            train_bg, rel_deg = self._degrade_background(train_bg, placed)
            rel = np.minimum(rel, rel_deg)

        if self.dda:
            # foreground transform
            ang = random.uniform(-15, 15)
            scale = random.uniform(0.9, 1.1)
            tx = random.uniform(-0.05, 0.05) * w
            ty = random.uniform(-0.05, 0.05) * h
            scene, mask = _affine(scene, mask, ang, tx, ty, scale)
            if random.random() > 0.5:
                bang, btx, bty, bscale = ang, tx, ty, scale
                train_bg, _ = _affine(train_bg, None, bang, btx, bty, bscale)
            else:
                bang = random.uniform(-15, 15)
                btx = random.uniform(-0.05, 0.05) * w
                bty = random.uniform(-0.05, 0.05) * h
                bscale = random.uniform(0.9, 1.1)
                train_bg, _ = _affine(train_bg, None, bang, btx, bty, bscale)
            rel, _ = _affine(rel.astype(np.float32), None, bang, btx, bty, bscale)
            scene = _photometric(scene, random.uniform(0.8, 1.2),
                                 random.uniform(-30, 30),
                                 noise=random.uniform(0, 0.03),
                                 blur=random.choice([0, 0, 1]))
            train_bg = _photometric(train_bg, random.uniform(0.8, 1.2),
                                    random.uniform(-30, 30),
                                    noise=random.uniform(0, 0.03),
                                    blur=random.choice([0, 0, 1]))
            scene = _salt_pepper(scene, random.choice([0, 0, 0, 0.002, 0.005]))
            train_bg = _salt_pepper(train_bg, random.choice([0, 0, 0, 0.002, 0.005]))

        # resize
        scene = cv2.resize(scene, (self.size, self.size))
        train_bg = cv2.resize(train_bg, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        rel = cv2.resize(rel, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        train_bg = cv2.cvtColor(train_bg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rel = np.clip(rel, 0.0, 1.0)
        if self.with_rel:
            return (torch.from_numpy(scene.transpose(2, 0, 1)),
                    torch.from_numpy(train_bg.transpose(2, 0, 1)),
                    torch.from_numpy(mask[None].astype(np.float32)),
                    torch.from_numpy(rel[None].astype(np.float32)))
        return (torch.from_numpy(scene.transpose(2, 0, 1)),
                torch.from_numpy(train_bg.transpose(2, 0, 1)),
                torch.from_numpy(mask[None].astype(np.float32)))
