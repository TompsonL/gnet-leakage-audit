import pytest

from utils.protocol import check_bg_source, protocol_record


def test_source_classes():
    assert check_bg_source("causal") == "causal"
    assert check_bg_source("firstframe") == "causal"
    assert check_bg_source("reconstructed", allow_leaky=True) == "leaky"
    assert check_bg_source("provided", allow_oracle=True) == "oracle"
    assert check_bg_source("subsense_mask") == "reference"


def test_leaky_and_oracle_are_guarded():
    with pytest.raises(SystemExit):
        check_bg_source("reconstructed")
    with pytest.raises(SystemExit):
        check_bg_source("provided")


def test_protocol_record_official_frames():
    rec = protocol_record("causal", frame_stride=1, threshold=0.5, boundary=True)
    assert rec["protocol_version"] == "cdnet14-official-v1"
    assert rec["bg_class"] == "causal"
    assert rec["official_frames"] is True
    assert rec["boundary_f1"] is True
    assert rec["shadow"] == "background"
    assert rec["roi"] == "ignore_outside"
    rec5 = protocol_record("causal", frame_stride=5, threshold=0.5)
    assert rec5["official_frames"] is False
