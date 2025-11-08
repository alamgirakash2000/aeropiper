import numpy as np
from typing import Iterable

# Per-DOF MuJoCo control ranges (min, max) in the physical/MuJoCo order:
# [thumb_abd, thumb_flex (th1), thumb_tendon (th2), index, middle, ring, pinky]
MJ_MIN = np.array([
    -0.1,      # thumb_abd
    0.026152,  # thumb_flex (thumb1)
    0.081568,  # thumb_tendon (thumb2)
    0.05852,   # index
    0.05852,   # middle
    0.05852,   # ring
    0.05852,   # pinky
], dtype=float)

MJ_MAX = np.array([
    1.75,      # thumb_abd
    0.038389,  # thumb_flex (thumb1)
    0.112138,  # thumb_tendon (thumb2)
    0.110387,  # index
    0.110387,  # middle
    0.110387,  # ring
    0.110387,  # pinky
], dtype=float)

def _normalize_physical(values: np.ndarray) -> np.ndarray:
    """Accept values in 0..100 or 0..1, return normalized 0..1."""
    scale = 100.0 if values.max(initial=0.0) > 1.0 or values.min(initial=0.0) < 0.0 else 1.0
    return np.clip(values / scale, 0.0, 1.0)

def _normalize_mujoco(values: np.ndarray) -> np.ndarray:
    """Normalize MuJoCo values to 0..1 relative to MJ_MIN..MJ_MAX per-DOF."""
    span = MJ_MAX - MJ_MIN
    # Avoid divide-by-zero if any span were zero (not expected)
    span = np.where(span == 0.0, 1.0, span)
    return np.clip((values - MJ_MIN) / span, 0.0, 1.0)

def physical_to_mujoco(physical_values: Iterable[float]) -> np.ndarray:
    """
    Convert physical values [thumb_abd, thumb_flex, thumb_tendon, index, middle, ring, pinky]
    to MuJoCo control values using per-DOF directions:
      - thumb_abd (DOF 0): non-inverted (0->min, 100->max)
      - others: inverted (0->max, 100->min)
    Accepts either 0..100 or 0..1 inputs.
    """
    values = np.asarray(list(physical_values), dtype=float)
    normalized = _normalize_physical(values)
    # Default inverted mapping for all DOFs
    out = MJ_MAX - (MJ_MAX - MJ_MIN) * normalized
    # DOF 0 non-inverted
    out[0] = MJ_MIN[0] + (MJ_MAX[0] - MJ_MIN[0]) * normalized[0]
    return out

def mujoco_to_physical(mujoco_values: Iterable[float]) -> np.ndarray:
    """
    Convert MuJoCo control values back to physical values in 0..100 scale with per-DOF directions:
      - thumb_abd (DOF 0): non-inverted (min->0, max->100)
      - others: inverted   (min->100, max->0)
    """
    values = np.asarray(list(mujoco_values), dtype=float)
    norm = _normalize_mujoco(values)

    physical = np.empty_like(norm)
    # DOF 0 non-inverted
    physical[0] = norm[0] * 100.0
    # Others inverted
    physical[1:] = (1.0 - norm[1:]) * 100.0
    return np.clip(physical, 0.0, 100.0)


