"""Frozen CDnet 2014 evaluation protocol (official, fair, causal).

Rules used by every reportable number in this project:
  - Evaluate every frame inside temporalROI (frame_stride == 1).
  - GT labels: foreground = 255; background = 0 or 50 (shadow counts as
    background); ignored = 85 (outside ROI) or 170 (unknown).
  - ROI.bmp: pixels (<128) are forced to the ignore label.
  - F1 / PWC computed per video, then averaged over the 11 categories, then
    over categories ("AVG_category"). AVG_video is reported as a secondary.
  - Inference background must be CAUSAL: it may depend only on frames <= t and
    must not use any segmentation label or BGS mask decision.

Background-source taxonomy
  CAUSAL   : no label/BGS information, only past frames (causal, firstframe, self)
  LEAKY    : causal in time but updated from SuBSENSE decisions (reconstructed, running)
  ORACLE   : uses future frames or the whole sequence (provided, median, oracle)

Reportable results must use CAUSAL sources. LEAKY sources are allowed only for
the leakage-control experiment; ORACLE only for diagnostic upper bounds.
"""
import sys

PROTOCOL_VERSION = "cdnet14-official-v1"

CAUSAL_BG = {"causal", "causal_rel", "causal_relfb", "causal_med", "firstframe", "self"}
LEAKY_BG = {"reconstructed", "running"}
ORACLE_BG = {"provided", "median", "oracle"}

CATEGORIES = [
    "PTZ", "badWeather", "baseline", "cameraJitter", "dynamicBackground",
    "intermittentObjectMotion", "lowFramerate", "nightVideos", "shadow",
    "thermal", "turbulence",
]


def check_bg_source(bg_source, allow_leaky=False, allow_oracle=False):
    """Abort if a non-causal background source is used without an explicit opt-in."""
    if bg_source in ORACLE_BG and not allow_oracle:
        sys.exit(f"[protocol] '{bg_source}' is a future-frame ORACLE background. "
                 f"Pass --allow-oracle only for diagnostics; reportable results must be causal.")
    if bg_source in LEAKY_BG and not (allow_leaky or allow_oracle):
        sys.exit(f"[protocol] '{bg_source}' is a LEAKY background (uses SuBSENSE decisions). "
                 f"Pass --allow-leaky only for the leakage-control experiment.")
    if bg_source in CAUSAL_BG:
        return "causal"
    if bg_source in LEAKY_BG:
        return "leaky"
    if bg_source in ORACLE_BG:
        return "oracle"
    return "reference"  # e.g. an independently produced mask (SuBSENSE floor)


def protocol_record(bg_source, frame_stride, threshold, boundary=False):
    """Machine-readable record of the protocol actually applied."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "bg_source": bg_source,
        "bg_class": ("causal" if bg_source in CAUSAL_BG else
                     "leaky" if bg_source in LEAKY_BG else
                     "oracle" if bg_source in ORACLE_BG else "reference"),
        "frame_stride": frame_stride,
        "threshold": threshold,
        "shadow": "background",
        "roi": "ignore_outside",
        "aggregation": "per_video->category_mean",
        "boundary_f1": bool(boundary),
        "official_frames": frame_stride == 1,
    }
