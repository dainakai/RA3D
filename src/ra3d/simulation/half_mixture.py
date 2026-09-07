"""25 um half-mixture allocation and collision-geometry helpers."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

FORMAT_VERSION = "collision-benchmark-half-mixture/v1"
DIAMETER_BIN_WIDTH_UM = 25.0
SPEED_TIER_COUNT = 5
IMPACT_TIER_COUNT = 3
DEFAULT_CANDIDATES_PER_PAIR = 6000
DEFAULT_PARENT_LEAD_FRAMES = 1500
CONTACT_RADIUS_FRACTION = 0.995
IMPACT_TIER_EDGE_MARGIN = 0.025
GEOMETRY_EPSILON_UM = 1e-06


@dataclass(frozen=True)
class CoverageAllocation:
    pair_low: int
    pair_high: int
    speed_tier: int
    impact_tier: int
    is_extra: int


@dataclass
class CandidatePool:
    d1_um: np.ndarray
    d2_um: np.ndarray
    velocity1_um_s: np.ndarray
    velocity2_um_s: np.ndarray
    relative_speed_um_s: np.ndarray
    sorted_relative_speed_um_s: np.ndarray
    quantile_edges_um_s: np.ndarray
    tier_indices: dict[int, np.ndarray]


def diameter_bin_edges(diameter_min_um: float, diameter_max_um: float) -> np.ndarray:
    span = diameter_max_um - diameter_min_um
    count = int(round(span / DIAMETER_BIN_WIDTH_UM))
    if count <= 0 or not math.isclose(
        diameter_min_um + count * DIAMETER_BIN_WIDTH_UM, diameter_max_um, rel_tol=0.0, abs_tol=1e-09
    ):
        raise ValueError("Half-mixture mode requires a diameter range divisible by 25 um")
    return np.linspace(diameter_min_um, diameter_max_um, count + 1)


def unordered_pairs(bin_count: int) -> list[tuple[int, int]]:
    return [(low, high) for low in range(bin_count) for high in range(low, bin_count)]


def diameter_bin_index(diameter_um: float, edges: np.ndarray) -> int:
    index = int(np.searchsorted(edges, diameter_um, side="right") - 1)
    return min(max(index, 0), len(edges) - 2)


def pair_label(pair: tuple[int, int], edges: np.ndarray) -> str:
    (low, high) = pair
    return f"{edges[low]:.0f}-{edges[low + 1]:.0f} x {edges[high]:.0f}-{edges[high + 1]:.0f}"


def build_diameter_bin_samples(samples: np.ndarray, edges: np.ndarray) -> dict[int, np.ndarray]:
    bins: dict[int, np.ndarray] = {}
    for index in range(len(edges) - 1):
        mask = (samples >= edges[index]) & (samples < edges[index + 1])
        selected = np.asarray(samples[mask], dtype=np.float64)
        if selected.size == 0:
            raise RuntimeError(f"No diameter source sample is available in 25 um bin {index}")
        bins[index] = selected
    return bins


def make_coverage_allocations(
    coverage_count: int, bin_count: int, rng: np.random.Generator
) -> list[CoverageAllocation]:
    """Allocate coverage events while preserving the full v1 design at n=1000."""
    if coverage_count < 0:
        raise ValueError("coverage_count must be non-negative")
    if coverage_count == 0:
        return []
    pairs = unordered_pairs(bin_count)
    joint_strata = [
        (speed_tier, impact_tier)
        for speed_tier in range(SPEED_TIER_COUNT)
        for impact_tier in range(IMPACT_TIER_COUNT)
    ]
    allocations: list[CoverageAllocation] = []
    (complete_pair_count, remainder) = divmod(coverage_count, len(joint_strata))
    pair_order = pairs[:]
    rng.shuffle(pair_order)
    for pair_index in range(complete_pair_count):
        pair = pair_order[pair_index % len(pair_order)]
        for speed_tier, impact_tier in joint_strata:
            allocations.append(
                CoverageAllocation(
                    pair_low=pair[0],
                    pair_high=pair[1],
                    speed_tier=speed_tier,
                    impact_tier=impact_tier,
                    is_extra=int(pair_index >= len(pairs)),
                )
            )
    if coverage_count == len(pairs) * len(joint_strata) + bin_count - 1:
        allocations = [
            CoverageAllocation(
                pair_low=pair[0],
                pair_high=pair[1],
                speed_tier=speed_tier,
                impact_tier=impact_tier,
                is_extra=0,
            )
            for pair in pairs
            for (speed_tier, impact_tier) in joint_strata
        ]
        for index in range(bin_count - 1):
            allocations.append(
                CoverageAllocation(
                    pair_low=index,
                    pair_high=index + 1,
                    speed_tier=index % SPEED_TIER_COUNT,
                    impact_tier=index % IMPACT_TIER_COUNT,
                    is_extra=1,
                )
            )
        remainder = 0
    elif remainder:
        pair = pair_order[complete_pair_count % len(pair_order)]
        strata_order = joint_strata[:]
        rng.shuffle(strata_order)
        for speed_tier, impact_tier in strata_order[:remainder]:
            allocations.append(
                CoverageAllocation(
                    pair_low=pair[0],
                    pair_high=pair[1],
                    speed_tier=speed_tier,
                    impact_tier=impact_tier,
                    is_extra=1,
                )
            )
    if len(allocations) != coverage_count:
        raise RuntimeError(f"Coverage allocation produced {len(allocations)} rows, expected {coverage_count}")
    rng.shuffle(allocations)
    return allocations


def random_perpendicular(unit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        candidate = rng.normal(0.0, 1.0, size=3)
        candidate -= float(np.dot(candidate, unit)) * unit
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-12:
            return candidate / norm
    raise RuntimeError("Could not construct a direction perpendicular to relative velocity")


def sample_impact_parameter(impact_tier: int, rng: np.random.Generator) -> float:
    low = impact_tier / IMPACT_TIER_COUNT + IMPACT_TIER_EDGE_MARGIN
    high = (impact_tier + 1) / IMPACT_TIER_COUNT - IMPACT_TIER_EDGE_MARGIN
    if impact_tier == IMPACT_TIER_COUNT - 1:
        high = min(high, 0.95)
    return float(rng.uniform(low, high))


def _mass(diameter_um: float) -> float:
    return diameter_um**3


def _merged_diameter(d1_um: float, d2_um: float) -> float:
    return float((d1_um**3 + d2_um**3) ** (1.0 / 3.0))


def _intersect(
    lower: np.ndarray, upper: np.ndarray, candidate_lower: np.ndarray, candidate_upper: np.ndarray
) -> bool:
    np.maximum(lower, candidate_lower, out=lower)
    np.minimum(upper, candidate_upper, out=upper)
    return bool(np.all(lower + GEOMETRY_EPSILON_UM < upper))


def _event_center_bounds(
    offset: np.ndarray, extent: np.ndarray, margin_um: float
) -> tuple[np.ndarray, np.ndarray]:
    return (margin_um - offset, extent - margin_um - offset)


def _activation_center_bounds(
    offset: np.ndarray,
    velocity_um_s: np.ndarray,
    lead_seconds: float,
    frame_seconds: float,
    extent: np.ndarray,
    face: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    activation_shift = offset - velocity_um_s * lead_seconds
    next_shift = activation_shift + velocity_um_s * frame_seconds
    lower = np.full(3, -np.inf, dtype=np.float64)
    upper = np.full(3, np.inf, dtype=np.float64)
    face_axis = 0 if face.startswith("x_") else 1
    face_is_min = face.endswith("_min")
    face_velocity = float(velocity_um_s[face_axis])
    if face_is_min and face_velocity <= 0.0:
        return None
    if not face_is_min and face_velocity >= 0.0:
        return None
    for axis in range(3):
        if axis == face_axis:
            if face_is_min:
                lower[axis] = -next_shift[axis] + GEOMETRY_EPSILON_UM
                upper[axis] = -activation_shift[axis] - GEOMETRY_EPSILON_UM
            else:
                lower[axis] = extent[axis] - activation_shift[axis] + GEOMETRY_EPSILON_UM
                upper[axis] = extent[axis] - next_shift[axis] - GEOMETRY_EPSILON_UM
        else:
            lower[axis] = max(-activation_shift[axis], -next_shift[axis]) + GEOMETRY_EPSILON_UM
            upper[axis] = (
                min(extent[axis] - activation_shift[axis], extent[axis] - next_shift[axis])
                - GEOMETRY_EPSILON_UM
            )
    return (lower, upper)


def _entry_faces(velocity_um_s: np.ndarray) -> list[str]:
    faces: list[str] = []
    if velocity_um_s[0] > 1e-12:
        faces.append("x_min")
    elif velocity_um_s[0] < -1e-12:
        faces.append("x_max")
    if velocity_um_s[1] > 1e-12:
        faces.append("y_min")
    elif velocity_um_s[1] < -1e-12:
        faces.append("y_max")
    return faces


class HalfMixtureDesigner:
    """Create conditional speed pools and entry-compatible coverage geometries."""

    def __init__(
        self,
        *,
        diameter_min_um: float,
        diameter_max_um: float,
        diameter_samples: np.ndarray,
        draw_horizontal_velocity_um_s: Callable[[], float],
        draw_vertical_velocity_um_s: Callable[[float], float],
        rng: np.random.Generator,
        coverage_count: int,
        candidates_per_pair: int = DEFAULT_CANDIDATES_PER_PAIR,
        parent_lead_frames: int = DEFAULT_PARENT_LEAD_FRAMES,
    ) -> None:
        self.edges = diameter_bin_edges(diameter_min_um, diameter_max_um)
        self.bin_count = len(self.edges) - 1
        self.pairs = unordered_pairs(self.bin_count)
        self.bin_samples = build_diameter_bin_samples(diameter_samples, self.edges)
        self.draw_horizontal_velocity_um_s = draw_horizontal_velocity_um_s
        self.draw_vertical_velocity_um_s = draw_vertical_velocity_um_s
        self.rng = rng
        self.coverage_count = int(coverage_count)
        self.candidates_per_pair = int(candidates_per_pair)
        self.parent_lead_frames = int(parent_lead_frames)
        self.allocations = make_coverage_allocations(self.coverage_count, self.bin_count, self.rng)
        self.pools: dict[tuple[int, int], CandidatePool] = {}

    def build_pool(self, pair: tuple[int, int]) -> CandidatePool:
        existing = self.pools.get(pair)
        if existing is not None:
            return existing
        d1_values = np.empty(self.candidates_per_pair, dtype=np.float64)
        d2_values = np.empty(self.candidates_per_pair, dtype=np.float64)
        velocity1 = np.empty((self.candidates_per_pair, 3), dtype=np.float64)
        velocity2 = np.empty((self.candidates_per_pair, 3), dtype=np.float64)
        relative_speeds = np.empty(self.candidates_per_pair, dtype=np.float64)
        accepted = 0
        while accepted < self.candidates_per_pair:
            d1 = float(self.rng.choice(self.bin_samples[pair[0]]))
            d2 = float(self.rng.choice(self.bin_samples[pair[1]]))
            v1 = np.asarray(
                [
                    self.draw_horizontal_velocity_um_s(),
                    self.draw_vertical_velocity_um_s(d1),
                    self.draw_horizontal_velocity_um_s(),
                ],
                dtype=np.float64,
            )
            v2 = np.asarray(
                [
                    self.draw_horizontal_velocity_um_s(),
                    self.draw_vertical_velocity_um_s(d2),
                    self.draw_horizontal_velocity_um_s(),
                ],
                dtype=np.float64,
            )
            relative_speed = float(np.linalg.norm(v2 - v1))
            if relative_speed < 1000.0:
                continue
            d1_values[accepted] = d1
            d2_values[accepted] = d2
            velocity1[accepted] = v1
            velocity2[accepted] = v2
            relative_speeds[accepted] = relative_speed
            accepted += 1
        quantile_edges = np.quantile(relative_speeds, np.linspace(0.0, 1.0, SPEED_TIER_COUNT + 1))
        tier_numbers = np.searchsorted(quantile_edges[1:-1], relative_speeds, side="right")
        pool = CandidatePool(
            d1_um=d1_values,
            d2_um=d2_values,
            velocity1_um_s=velocity1,
            velocity2_um_s=velocity2,
            relative_speed_um_s=relative_speeds,
            sorted_relative_speed_um_s=np.sort(relative_speeds),
            quantile_edges_um_s=quantile_edges,
            tier_indices={tier: np.flatnonzero(tier_numbers == tier) for tier in range(SPEED_TIER_COUNT)},
        )
        self.pools[pair] = pool
        return pool

    def build_all_pools(self) -> None:
        for pair in self.pairs:
            self.build_pool(pair)

    def speed_percentile(self, pair: tuple[int, int], relative_speed_um_s: float) -> float:
        sorted_speeds = self.build_pool(pair).sorted_relative_speed_um_s
        left = int(np.searchsorted(sorted_speeds, relative_speed_um_s, side="left"))
        right = int(np.searchsorted(sorted_speeds, relative_speed_um_s, side="right"))
        return float(0.5 * (left + right) / sorted_speeds.size)

    def speed_tier(self, pair: tuple[int, int], relative_speed_um_s: float) -> int:
        percentile = self.speed_percentile(pair, relative_speed_um_s)
        return min(int(percentile * SPEED_TIER_COUNT), SPEED_TIER_COUNT - 1)

    def quantile_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for pair in self.pairs:
            pool = self.build_pool(pair)
            row: dict[str, Any] = {
                "pair_low_25um": pair[0],
                "pair_high_25um": pair[1],
                "pair_label_25um": pair_label(pair, self.edges),
            }
            for quantile, value in zip(
                np.linspace(0.0, 1.0, SPEED_TIER_COUNT + 1), pool.quantile_edges_um_s, strict=True
            ):
                row[f"q{int(round(quantile * 100)):02d}_um_s"] = float(value)
            rows.append(row)
        return rows

    def try_geometry(
        self,
        *,
        allocation: CoverageAllocation,
        activation_frame: int,
        collision_frame: int,
        fps: float,
        width_um: float,
        height_um: float,
        depth_um: float,
        propagation_margin_um: float,
        accept: Callable[[dict[str, Any]], bool],
        max_candidate_trials: int = 400,
    ) -> dict[str, Any] | None:
        birth_frame = collision_frame - self.parent_lead_frames
        if birth_frame < 0 or activation_frame < birth_frame:
            return None
        activation_lead_frames = collision_frame - activation_frame
        if activation_lead_frames < 2:
            return None
        pair = (allocation.pair_low, allocation.pair_high)
        pool = self.build_pool(pair)
        indices = pool.tier_indices[allocation.speed_tier].copy()
        self.rng.shuffle(indices)
        extent = np.asarray([width_um, height_um, depth_um], dtype=np.float64)
        activation_lead_seconds = activation_lead_frames / fps
        total_lead_seconds = self.parent_lead_frames / fps
        frame_seconds = 1.0 / fps
        for candidate_index in indices[:max_candidate_trials]:
            index = int(candidate_index)
            d1 = float(pool.d1_um[index])
            d2 = float(pool.d2_um[index])
            velocity1 = pool.velocity1_um_s[index].copy()
            velocity2 = pool.velocity2_um_s[index].copy()
            relative_velocity = velocity2 - velocity1
            relative_speed = float(pool.relative_speed_um_s[index])
            relative_unit = relative_velocity / relative_speed
            radius_sum = 0.5 * (d1 + d2)
            mass1 = _mass(d1)
            mass2 = _mass(d2)
            faces1 = _entry_faces(velocity1)
            faces2 = _entry_faces(velocity2)
            if not faces1 or not faces2:
                continue
            for _ in range(8):
                impact_normalized = sample_impact_parameter(allocation.impact_tier, self.rng)
                transverse = impact_normalized * radius_sum
                longitudinal_squared = (CONTACT_RADIUS_FRACTION * radius_sum) ** 2 - transverse**2
                if longitudinal_squared <= 0.0:
                    continue
                perpendicular = random_perpendicular(relative_unit, self.rng)
                separation = -relative_unit * math.sqrt(longitudinal_squared) + perpendicular * transverse
                offset1 = -(mass2 / (mass1 + mass2)) * separation
                offset2 = mass1 / (mass1 + mass2) * separation
                face_pairs = [(face1, face2) for face1 in faces1 for face2 in faces2]
                self.rng.shuffle(face_pairs)
                for face1, face2 in face_pairs:
                    event_margin = max(0.5 * max(d1, d2) + 20.0, 60.0)
                    lower = np.full(3, -np.inf, dtype=np.float64)
                    upper = np.full(3, np.inf, dtype=np.float64)
                    for offset in (offset1, offset2):
                        (event_lower, event_upper) = _event_center_bounds(offset, extent, event_margin)
                        if not _intersect(lower, upper, event_lower, event_upper):
                            break
                    else:
                        bounds1 = _activation_center_bounds(
                            offset1, velocity1, activation_lead_seconds, frame_seconds, extent, face1
                        )
                        bounds2 = _activation_center_bounds(
                            offset2, velocity2, activation_lead_seconds, frame_seconds, extent, face2
                        )
                        if bounds1 is None or bounds2 is None:
                            continue
                        if not _intersect(lower, upper, *bounds1):
                            continue
                        if not _intersect(lower, upper, *bounds2):
                            continue
                        birth_z_lower = np.asarray([-np.inf, -np.inf, GEOMETRY_EPSILON_UM], dtype=np.float64)
                        birth_z_upper = np.asarray(
                            [np.inf, np.inf, depth_um - GEOMETRY_EPSILON_UM], dtype=np.float64
                        )
                        for offset, velocity in ((offset1, velocity1), (offset2, velocity2)):
                            shift = offset - velocity * total_lead_seconds
                            shifted_lower = birth_z_lower - shift
                            shifted_upper = birth_z_upper - shift
                            if not _intersect(lower, upper, shifted_lower, shifted_upper):
                                break
                        else:
                            child_velocity = (mass1 * velocity1 + mass2 * velocity2) / (mass1 + mass2)
                            child_margin = max(0.5 * _merged_diameter(d1, d2) + 20.0, 60.0)
                            if not _intersect(
                                lower,
                                upper,
                                np.asarray([child_margin, child_margin, child_margin], dtype=np.float64),
                                extent - child_margin,
                            ):
                                continue
                            fraction = self.rng.uniform(0.1, 0.9, size=3)
                            center = lower + fraction * (upper - lower)
                            event1 = center + offset1
                            event2 = center + offset2
                            activation1 = event1 - velocity1 * activation_lead_seconds
                            activation2 = event2 - velocity2 * activation_lead_seconds
                            birth1 = event1 - velocity1 * total_lead_seconds
                            birth2 = event2 - velocity2 * total_lead_seconds
                            margin_low = -propagation_margin_um
                            margin_high = np.asarray(
                                [width_um + propagation_margin_um, height_um + propagation_margin_um]
                            )
                            birth1_outside = bool(
                                birth1[0] < margin_low
                                or birth1[0] >= margin_high[0]
                                or birth1[1] < margin_low
                                or (birth1[1] >= margin_high[1])
                            )
                            birth2_outside = bool(
                                birth2[0] < margin_low
                                or birth2[0] >= margin_high[0]
                                or birth2[1] < margin_low
                                or (birth2[1] >= margin_high[1])
                            )
                            if not birth1_outside or not birth2_outside:
                                continue
                            actual_impact = float(
                                np.linalg.norm(
                                    separation - float(np.dot(separation, relative_unit)) * relative_unit
                                )
                                / radius_sum
                            )
                            geometry = {
                                "pair": pair,
                                "pair_label_25um": pair_label(pair, self.edges),
                                "d1_um": d1,
                                "d2_um": d2,
                                "child_diameter_um": _merged_diameter(d1, d2),
                                "velocity1_um_s": velocity1,
                                "velocity2_um_s": velocity2,
                                "child_velocity_um_s": child_velocity,
                                "relative_velocity_um_s": relative_velocity,
                                "relative_speed_um_s": relative_speed,
                                "conditional_speed_percentile": self.speed_percentile(pair, relative_speed),
                                "relative_speed_tier": allocation.speed_tier + 1,
                                "impact_parameter_tier": allocation.impact_tier + 1,
                                "normalized_impact_parameter": actual_impact,
                                "separation_um": separation,
                                "radius_sum_um": radius_sum,
                                "approach_dot_um2_s": float(np.dot(separation, relative_velocity)),
                                "event_center_um": center,
                                "event1_um": event1,
                                "event2_um": event2,
                                "activation1_um": activation1,
                                "activation2_um": activation2,
                                "birth1_um": birth1,
                                "birth2_um": birth2,
                                "birth_frame": birth_frame,
                                "activation_frame": activation_frame,
                                "collision_frame": collision_frame,
                                "entry_face1": face1,
                                "entry_face2": face2,
                                "coverage_extra_assignment": allocation.is_extra,
                            }
                            if accept(geometry):
                                return geometry
        return None
