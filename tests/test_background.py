import numpy as np

from data.background import running_background_median, running_background_reliability


def _frames(n=8, h=32, w=24, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_running_background_is_causal():
    frames = _frames()
    bgs, rels = running_background_reliability(frames, alpha=0.1)
    bgs_prefix, rels_prefix = running_background_reliability(frames[:5], alpha=0.1)
    assert np.array_equal(bgs[4], bgs_prefix[4])
    assert np.array_equal(rels[4], rels_prefix[4])
    assert bgs[0].shape == frames[0].shape
    assert rels[0].shape == frames[0].shape[:2]
    assert float(rels[0].min()) >= 0.0 and float(rels[0].max()) <= 1.0


def test_causal_median_is_causal_and_bounded():
    frames = _frames()
    bgs, rels = running_background_median(frames, K=4)
    bgs_prefix, _ = running_background_median(frames[:5], K=4)
    assert np.array_equal(bgs[4], bgs_prefix[4])
    assert bgs[0].dtype == np.uint8
    assert float(rels[2].min()) >= 0.0 and float(rels[2].max()) <= 1.0
