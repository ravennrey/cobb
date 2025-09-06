"""Utility functions for Cobb angle measurement."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Optional, Sequence, Tuple, Dict
import numpy as np

log = logging.getLogger(__name__)

@dataclass
class CobbConfig:
    """Configuration parameters for Cobb angle computation."""
    smooth_win: int = 3
    slope_win: int = 5
    angle_thr: float = 5.0
    max_curves: int = 2
    min_apex_gap: int = 6
    search_margin: int = 2
    edge_pad: float = 2.0
    top_n: int = 4
    bot_n: int = 4
    # slope clipping to avoid extreme values
    max_slope: float = 1000.0


def moving_avg(y: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average with odd window size."""
    win = max(1, int(win))
    if win % 2 == 0:
        win += 1
    if win == 1 or len(y) < win:
        return y.copy()
    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")


def local_slopes(z: np.ndarray, x: np.ndarray, win: int) -> np.ndarray:
    """Estimate local slopes dx/dz using small linear fits."""
    n = len(z)
    if win % 2 == 0:
        win += 1
    half = win // 2
    a = np.zeros(n, dtype=float)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        if e - s < 2:
            a[i] = 0.0
        else:
            a[i], _ = np.polyfit(z[s:e], x[s:e], 1)
    return a


def angle_between_line_slopes(m1: float, m2: float) -> float:
    """Return the acute angle between two line slopes (0..90 degrees)."""
    ang = np.degrees(np.arctan2(abs(m2 - m1), 1.0 + m1 * m2))
    if ang > 90:
        ang = 180 - ang
    if ang < 0:
        ang = -ang
    return float(ang)


def endplate_slope_from_tangent(a: float, max_slope: float) -> float:
    """Return slope of endplate line perpendicular to tangent slope *a*.

    Uses clipping to avoid extremely large slopes when *a* is near zero.
    """
    if abs(a) < 1e-6:
        slope = max_slope * np.sign(a if a != 0 else 1.0)
    else:
        slope = -1.0 / a
    return float(np.clip(slope, -max_slope, max_slope))


def fit_line(zs: Sequence[float], xs: Sequence[float], pad: float) -> Optional[Dict[str, np.ndarray]]:
    """Fit line x = a*z + b to given points and extend with padding."""
    zs = np.asarray(zs, float)
    xs = np.asarray(xs, float)
    if len(zs) < 2:
        return None
    a, b = np.polyfit(zs, xs, 1)
    z1 = float(np.min(zs)) - pad
    z2 = float(np.max(zs)) + pad
    x1 = a * z1 + b
    x2 = a * z2 + b
    v = np.array([a, 1.0], float)
    v /= (np.linalg.norm(v) + 1e-9)
    return {
        "a": a,
        "b": b,
        "p1": np.array([x1, z1]),
        "p2": np.array([x2, z2]),
        "vec": v,
    }


def find_apices(z: np.ndarray, x: np.ndarray, cfg: CobbConfig) -> Tuple[List[int], np.ndarray]:
    """Return indices of apex points based on derivative of local slopes."""
    a = local_slopes(z, x, cfg.slope_win)
    da = np.gradient(a, z)
    score = np.abs(da)
    order = np.argsort(-score)
    apices: List[int] = []
    for idx in order:
        if len(apices) >= cfg.max_curves:
            break
        if idx <= 1 or idx >= len(z) - 2:
            continue
        if all(abs(idx - p) >= cfg.min_apex_gap for p in apices):
            apices.append(int(idx))
    apices.sort()
    return apices, a


def pick_endvertebra_by_max_tilt(z: np.ndarray, a: np.ndarray, center_idx: int, side: str, cfg: CobbConfig) -> Optional[int]:
    """Select end-vertebra with maximum tilt around given apex."""
    n = len(z)
    if side == "up":
        s = 0
        e = max(0, center_idx - cfg.search_margin)
        if e <= s:
            return None
        idxs = np.arange(s, e)
    else:
        s = min(n - 1, center_idx + cfg.search_margin)
        e = n
        if e <= s:
            return None
        idxs = np.arange(s, e)
    i = idxs[np.argmax(np.abs(np.degrees(np.arctan(a[idxs]))))]
    return int(i)


def build_curve_region(x: np.ndarray, z: np.ndarray, center_idx: int, a: np.ndarray, side: str, span: int, cfg: CobbConfig):
    """Construct region around end-vertebra and fit a line."""
    i0 = pick_endvertebra_by_max_tilt(z, a, center_idx, side, cfg)
    if i0 is None:
        return None
    s = max(0, i0 - span // 2)
    e = min(len(z), i0 + span // 2 + 1)
    fit = fit_line(z[s:e], x[s:e], pad=cfg.edge_pad)
    if fit is None:
        return None
    return {
        "line": (fit["p1"], fit["p2"]),
        "vec": fit["vec"],
        "set_data": (z[s:e].copy(), x[s:e].copy(), (fit["a"], fit["b"])),
        "idx": i0,
    }


def compute_cobb_regions(spine_xy_px: np.ndarray, cfg: CobbConfig) -> List[Dict[str, object]]:
    """Compute Cobb angle regions given spine coordinates in pixels."""
    x = spine_xy_px[:, 0].astype(float)
    z = spine_xy_px[:, 1].astype(float)

    if cfg.smooth_win > 1:
        x = moving_avg(x, cfg.smooth_win)

    apices, a = find_apices(z, x, cfg)
    regions: List[Dict[str, object]] = []

    for apex in apices:
        top = build_curve_region(x, z, apex, a, side="up", span=cfg.top_n, cfg=cfg)
        bot = build_curve_region(x, z, apex, a, side="down", span=cfg.bot_n, cfg=cfg)
        if top is None or bot is None:
            continue

        m_up = endplate_slope_from_tangent(a[top["idx"]], cfg.max_slope)
        m_lo = endplate_slope_from_tangent(a[bot["idx"]], cfg.max_slope)

        cobb = angle_between_line_slopes(m_up, m_lo)
        if cobb < cfg.angle_thr:
            continue

        regions.append(
            {
                "cobb": float(cobb),
                "top_line": top["line"],
                "bot_line": bot["line"],
                "top_set": top["set_data"],
                "bot_set": bot["set_data"],
                "center_z": float(z[apex]),
            }
        )

    regions.sort(key=lambda r: r["cobb"], reverse=True)
    return regions[: cfg.max_curves]
