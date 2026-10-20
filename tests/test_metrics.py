import numpy as np

from utils.metrics import average, confusion, f1_pwc


def test_gt_label_mapping_shadow_background_and_ignore():
    gt = np.array([[255, 0, 50, 85, 170]], dtype=np.uint8)
    pred = np.ones_like(gt)
    tp, tn, fp, fn = confusion(pred, gt)
    assert (tp, tn, fp, fn) == (1, 0, 2, 0)


def test_perfect_prediction():
    gt = np.array([[255, 0, 50]], dtype=np.uint8)
    pred = np.array([[1, 0, 0]], dtype=np.uint8)
    tp, tn, fp, fn = confusion(pred, gt)
    assert (tp, tn, fp, fn) == (1, 2, 0, 0)
    f1, pwc, prec, rec = f1_pwc(tp, tn, fp, fn)
    assert f1 == 1.0 and pwc == 0.0 and prec == 1.0 and rec == 1.0


def test_f1_pwc_known_values():
    f1, pwc, prec, rec = f1_pwc(9, 0, 1, 1)
    assert abs(f1 - 0.9) < 1e-12
    assert abs(prec - 0.9) < 1e-12
    assert abs(rec - 0.9) < 1e-12
    assert abs(pwc - 100.0 * 2 / 11) < 1e-12


def test_category_average_simple_mean():
    per_video = {"a": (0.2, 10.0), "b": (0.4, 20.0), "c": (0.3, 5.0)}
    f1, pwc = average(per_video)
    assert abs(f1 - 0.3) < 1e-12
    assert abs(pwc - (35.0 / 3)) < 1e-12
