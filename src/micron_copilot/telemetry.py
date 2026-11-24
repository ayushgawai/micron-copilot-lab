"""Synthetic telemetry generation and z-score anomaly detection."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetryPoint:
    day: int
    site: str
    metric: str
    value: float


@dataclass(frozen=True)
class Anomaly:
    day: int
    site: str
    metric: str
    value: float
    z_score: float
    direction: str

    def to_dict(self) -> dict:
        return {"day": self.day, "site": self.site, "metric": self.metric,
                "value": round(self.value, 3), "z_score": round(self.z_score, 2),
                "direction": self.direction}


METRICS = {"yield_pct": (95.0, 1.2), "defect_density": (1.0, 0.25),
           "cycle_time_h": (9.0, 1.0), "downtime_h": (3.0, 0.8)}


def generate_telemetry(days: int = 28, seed: int = 7,
                       sites: tuple[str, ...] = ("Fab-A", "Fab-B"),
                       inject: bool = True) -> list[TelemetryPoint]:
    """Gaussian series per site/metric with (optionally) injected spikes."""
    rng = random.Random(seed)
    points: list[TelemetryPoint] = []
    for site in sites:
        for metric, (mean, sd) in METRICS.items():
            for day in range(1, days + 1):
                value = rng.gauss(mean, sd)
                if inject and day in (10, 21) and metric in ("defect_density", "downtime_h"):
                    value = mean + 4.5 * sd  # known ground-truth anomaly
                if inject and day == 15 and metric == "yield_pct":
                    value = mean - 4.5 * sd
                points.append(TelemetryPoint(day, site, metric, round(value, 3)))
    return points


def detect_anomalies(points: list[TelemetryPoint],
                     threshold: float = 2.5) -> list[Anomaly]:
    groups: dict[tuple[str, str], list[TelemetryPoint]] = {}
    for p in points:
        groups.setdefault((p.site, p.metric), []).append(p)
    anomalies: list[Anomaly] = []
    for (site, metric), series in sorted(groups.items()):
        vals = [p.value for p in series]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
        sd = math.sqrt(var) or 1.0
        for p in series:
            z = (p.value - mean) / sd
            if abs(z) >= threshold:
                anomalies.append(Anomaly(p.day, site, metric, p.value, z,
                                         "high" if z > 0 else "low"))
    anomalies.sort(key=lambda a: (-abs(a.z_score), a.day))
    return anomalies
