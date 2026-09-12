"""degraded() never raises and counts by tag."""

from __future__ import annotations

from common.obs import degraded, reset_obs, snapshot


def setup_function() -> None:
    reset_obs()


def test_degraded_counts_by_tag():
    degraded("redis", ConnectionError("down"))
    degraded("redis", ConnectionError("down again"))
    degraded("faiss_warmup", RuntimeError("missing index"))
    snap = snapshot()
    assert snap["total"] == 3
    assert snap["byTag"]["redis"] == 2
    assert snap["byTag"]["faiss_warmup"] == 1
    assert "ConnectionError" in snap["last"]["redis"]


def test_degraded_never_raises():
    class BrokenLogger:
        def warning(self, *args, **kwargs):
            raise RuntimeError("log failed")

    import common.obs as obs

    orig = obs.logger
    obs.logger = BrokenLogger()
    try:
        degraded("x", RuntimeError("down"))
        degraded("", None, "empty tag")
    finally:
        obs.logger = orig
    snap = snapshot()
    assert snap["total"] >= 1
    assert snap["byTag"]["x"] >= 1
