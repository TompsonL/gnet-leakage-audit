"""CDnet-standard foreground segmentation metrics: F1 and PWC.

Ground-truth labels (CDnet 2014):
    0   static
    50  hard shadow
    85  outside ROI
    170 unknown motion
    255 motion

Following the official changedetection.net evaluation:
    foreground = (GT == 255)
    background = (GT == 0) or (GT == 50)     # shadows counted as background
    ignored    = (GT == 85) or (GT == 170)
"""
import numpy as np


def confusion(pred, gt, ignore_shadow=False):
    """pred: binary {0,1}; gt: CDnet label map. Returns TP,TN,FP,FN."""
    pred = pred.astype(bool)
    fg = gt == 255
    if ignore_shadow:
        bg = gt == 0
    else:
        bg = (gt == 0) | (gt == 50)
    ign = (gt == 85) | (gt == 170)
    valid = ~ign
    p = pred & valid
    g = fg & valid
    b = bg & valid
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & b))
    fn = int(np.sum(~p & g & valid))
    tn = int(np.sum(~p & b))
    return tp, tn, fp, fn


def f1_pwc(tp, tn, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 1.0
    total = tp + tn + fp + fn
    pwc = 100.0 * (fn + fp) / total if total > 0 else 0.0
    return f1, pwc, prec, rec


def average(per_video):
    """per_video: dict video -> (f1,pwc). Returns mean F1, mean PWC."""
    if not per_video:
        return 0.0, 0.0
    f1s = [v[0] for v in per_video.values()]
    pwcs = [v[1] for v in per_video.values()]
    return float(np.mean(f1s)), float(np.mean(pwcs))
