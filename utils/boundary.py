"""Boundary quality metric (Boundary-F1 with tolerance based on distance transform).

For each frame: extract boundary pixels of prediction and GT, then a predicted
boundary pixel is a true positive if it lies within `tol` pixels of a GT
boundary pixel and vice versa.
"""
import numpy as np
import cv2


def _boundary(binary):
    b = (binary > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    er = cv2.erode(b, k)
    return (b - er).astype(bool)


def boundary_f1(pred, gt, tol=2, ignore_mask=None):
    pb = _boundary(pred)
    gb = _boundary((gt == 255).astype(np.uint8))
    if ignore_mask is not None:
        pb &= ~ignore_mask
        gb &= ~ignore_mask
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0, 1.0, 1.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0, 0.0, 0.0
    dt_g = cv2.distanceTransform((~gb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)
    prec = float(np.mean(dt_g[pb] <= tol))
    rec = float(np.mean(dt_p[gb] <= tol))
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec
