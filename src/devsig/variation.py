"""Benign run-to-run variation (the null model). Seeded, applied after run_ic()."""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

KT2FPS = 1.6878


@dataclass
class Variation:
    seed: int
    wind_kts: float
    wind_dir_deg: float
    fuel_frac: float
    payload_delta_lbs: float

    def as_dict(self):
        return asdict(self)


def draw(seed: int, wind_max_kts: float = 8.0, fuel_range=(0.55, 1.0), payload_sd_lbs: float = 15.0, rng=None) -> Variation:
    rng = np.random.default_rng(seed) if rng is None else rng
    return Variation(seed=seed,
                     wind_kts=float(rng.uniform(0.0, wind_max_kts)),
                     wind_dir_deg=float(rng.uniform(0.0, 360.0)),
                     fuel_frac=float(rng.uniform(*fuel_range)),
                     payload_delta_lbs=float(rng.normal(0.0, payload_sd_lbs)))


def apply(fdm, v: Variation) -> None:
    """Apply a Variation to a loaded, initialised fdm (call after run_ic)."""
    pm = fdm.get_property_manager()
    rad = math.radians(v.wind_dir_deg)
    fdm["atmosphere/wind-north-fps"] = v.wind_kts * KT2FPS * math.cos(rad)
    fdm["atmosphere/wind-east-fps"] = v.wind_kts * KT2FPS * math.sin(rad)
    i = 0
    while pm.hasNode(f"propulsion/tank[{i}]/contents-lbs"):
        fdm[f"propulsion/tank[{i}]/contents-lbs"] = fdm[f"propulsion/tank[{i}]/contents-lbs"] * v.fuel_frac
        i += 1
    if pm.hasNode("inertia/pointmass-weight-lbs"):
        w = fdm["inertia/pointmass-weight-lbs"]
        fdm["inertia/pointmass-weight-lbs"] = max(0.0, w + v.payload_delta_lbs)
