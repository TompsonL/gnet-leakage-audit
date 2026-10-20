"""CDnet 2014 / BMC 2012 sequence dataset for background-guided segmentation.

The released packaging (D2-CDNet) stores, per video:
    input/          current frames (in%06d.png/jpg)
    groundtruth/    GT label maps (gt%06d.png or same basename)
    backgrounds/    (optional) static/estimated background reference
    bgs_subsense/   (optional) SuBSENSE binary BGS masks
    ROI.bmp         spatial ROI (255 inside)
    temporalROI.txt two frame indices delimiting the evaluated range
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _sorted_frames(d):
    fs = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
    return fs


def read_temporal_roi(path):
    if not os.path.isfile(path):
        return 1, 10 ** 9
    with open(path) as f:
        nums = [int(x) for x in f.read().split()]
    return nums[0], nums[-1]


def read_roi(path):
    roi = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return roi


class CDnetVideo:
    def __init__(self, vdir):
        self.dir = vdir
        self.inputs = _sorted_frames(os.path.join(vdir, "input"))
        self.gts = _sorted_frames(os.path.join(vdir, "groundtruth"))
        self.f0, self.f1 = read_temporal_roi(os.path.join(vdir, "temporalROI.txt"))
        rp = os.path.join(vdir, "ROI.bmp")
        if not os.path.isfile(rp):
            rp = os.path.join(vdir, "ROI.jpg")
        self.roi = read_roi(rp) if os.path.isfile(rp) else None
        self.bgs = _sorted_frames(os.path.join(vdir, "bgs_subsense")) \
            if os.path.isdir(os.path.join(vdir, "bgs_subsense")) else []
        self.bg_frame = None
        bdir = os.path.join(vdir, "backgrounds")
        if os.path.isdir(bdir):
            bf = _sorted_frames(bdir)
            if bf:
                self.bg_frame = bf  # per-frame background if same count, else static

    def background(self, idx):
        if self.bg_frame:
            j = min(idx, len(self.bg_frame) - 1)
            return cv2.imread(self.bg_frame[j], cv2.IMREAD_COLOR)
        return None


class CDnetDataset(Dataset):
    """Yields (img, bg, gt, meta). Background source configurable."""

    def __init__(self, root, split_file=None, size=513, bg_source="provided",
                 frame_stride=1, max_frames=None, categories=None, limit_videos=None):
        self.root = root
        self.size = size
        self.bg_source = bg_source
        self.videos = []
        cats = categories if categories else sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        for cat in cats:
            cdir = os.path.join(root, cat)
            vids = sorted(d for d in os.listdir(cdir) if os.path.isdir(os.path.join(cdir, d)))
            if limit_videos:
                vids = vids[:limit_videos]
            for v in vids:
                self.videos.append((cat, v, os.path.join(cdir, v)))
        self.videos = self._apply_split(split_file)
        self._bg_cache = {}
        self._vid_cache = {}
        self.samples = []
        for cat, v, vdir in self.videos:
            vid = self._video(vdir)
            n = len(vid.inputs)
            i0, i1 = vid.f0 - 1, min(vid.f1, n) - 1
            idxs = list(range(max(i0, 0), i1 + 1, frame_stride))
            if max_frames:
                idxs = idxs[:max_frames]
            for i in idxs:
                self.samples.append((cat, v, vdir, i))

    def _apply_split(self, split_file):
        if not split_file or not os.path.isfile(split_file):
            return self.videos
        keep = set(open(split_file).read().split())
        return [t for t in self.videos if f"{t[0]}/{t[1]}" in keep]

    def _video(self, vdir):
        if vdir not in self._vid_cache:
            self._vid_cache[vdir] = CDnetVideo(vdir)
        return self._vid_cache[vdir]

    def _first_frame(self, vid):
        if not vid.inputs:
            return None
        img = cv2.imread(vid.inputs[0], cv2.IMREAD_COLOR)
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        cat, v, vdir, i = self.samples[k]
        vid = self._video(vdir)
        img = cv2.imread(vid.inputs[i], cv2.IMREAD_COLOR)
        gt = cv2.imread(vid.gts[i], cv2.IMREAD_GRAYSCALE) if i < len(vid.gts) else None
        rel = None
        if self.bg_source in ("causal_rel", "causal_relfb", "causal_med"):
            if vdir not in self._bg_cache:
                if self.bg_source == "causal_med":
                    from data.background import estimate_causal_background_median
                    r = estimate_causal_background_median(vdir)
                else:
                    from data.background import estimate_causal_background_rel
                    r = estimate_causal_background_rel(vdir)
                self._bg_cache[vdir] = r[1] if r else None
            cache = self._bg_cache[vdir]
            entry = cache.get(i) if cache else None
            if entry is None:
                bg = None
            else:
                bg, rel = entry
                if self.bg_source == "causal_relfb":
                    # Where the background is unreliable (low rel), fall back to
                    # the current frame so the differential branch carries less
                    # spurious signal; where reliable, keep the causal estimate.
                    r3 = rel[..., None]
                    bg = (r3 * bg.astype(np.float32) +
                          (1.0 - r3) * img.astype(np.float32)).astype(np.uint8)
        elif self.bg_source == "firstframe":
            bg = self._first_frame(vid)
        elif self.bg_source == "causal":
            if vdir not in self._bg_cache:
                from data.background import estimate_causal_background
                r = estimate_causal_background(vdir)
                self._bg_cache[vdir] = r[1] if r else None
            cache = self._bg_cache[vdir]
            bg = cache.get(i) if cache else None
        elif self.bg_source in ("reconstructed", "running", "median"):
            if vdir not in self._bg_cache:
                from data.background import estimate_video_background
                method = "median" if self.bg_source == "median" else "running"
                r = estimate_video_background(vdir, method=method)
                self._bg_cache[vdir] = r[1] if r else None
            cache = self._bg_cache[vdir]
            bg = cache.get(i) if cache else None
        elif self.bg_source == "provided":
            bg = vid.background(i)
        elif self.bg_source == "self":
            # Current-frame-only ablation: background == current frame, so the
            # differential branch carries no temporal information.
            bg = img.copy()
        else:
            bg = None
        if bg is None:
            bg = self._first_frame(vid)
        if bg is None:
            bg = img.copy()
        h0, w0 = img.shape[:2]
        img_r = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        bg_r = cv2.resize(bg, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bg_r = cv2.cvtColor(bg_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_r.transpose(2, 0, 1))
        bg_t = torch.from_numpy(bg_r.transpose(2, 0, 1))
        meta = {"cat": cat, "video": v, "idx": i, "h": h0, "w": w0,
                "frame": vid.inputs[i]}
        if rel is not None:
            rel_r = cv2.resize(rel, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            meta["rel"] = torch.from_numpy(rel_r[None].astype(np.float32))
        if gt is not None:
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            meta["gt"] = gt
        if vid.roi is not None:
            roi = vid.roi
            if gt is not None and (roi.shape[0] != gt.shape[0] or roi.shape[1] != gt.shape[1]):
                roi = cv2.resize(roi, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            meta["roi"] = roi
        return img_t, bg_t, meta


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    bgs = torch.stack([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return imgs, bgs, metas
