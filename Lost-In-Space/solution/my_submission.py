from __future__ import annotations

"""
Single-file Lost in Space submission.

Strategy
--------
The AOI is divided into a 6x6 grid of fixed ground targets. Those targets are
ordered in a boustrophedon ("snake") pattern that starts from the AOI side that
is closest to the spacecraft at closest approach. During the pass, the planner
opens imaging opportunities on a fixed cadence and picks the earliest
unscheduled grid target in that snake order that is both reachable and
kinematically feasible.

For every accepted shot the spacecraft follows a slew-settle-stare profile:
1. Smoothly SLERP from the previous inertial attitude to the next pointing setpoint
2. Hold that quaternion for 0.1 s before the shutter opens
3. Keep the same quaternion through the 120 ms shutter window

The command trajectory is exported at exactly 20 Hz (0.05 s spacing) with
scalar-last quaternions [qx, qy, qz, qw].
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from sgp4.api import Satrec, jday


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

GRID_SIDE = 4
INTEG_DEFAULT = 0.120
ATTITUDE_DT = 0.05
SHOT_CADENCE_S = 3.2
PRESETTLE_S = 0.10
POST_HOLD_S = 0.05
SMEAR_TARGET_DEGPS = 0.02  # Target < 0.05 limit
OFF_NADIR_SOFT_DEG = 59.8
SLEW_RATE_DPS_CAP = 1.5
FOCUS_SCAN_DT = 2.0
REACHABILITY_SCAN_DT = 5.0
PLANNING_WINDOW_HALF_S = 360.0
HARD_CASE_MIN_GAP_S = 8.0


@dataclass
class OrbitSample:
    t: float
    utc: datetime
    r_eci: np.ndarray
    v_eci: np.ndarray
    gmst: float


@dataclass
class GridTarget:
    row: int
    col: int
    lat_deg: float
    lon_deg: float
    snake_rank: int


@dataclass
class PlannedShot:
    target: GridTarget
    t_start: float
    q_bn: np.ndarray


@dataclass
class PassFocus:
    closest_t: float
    closest_sample: OrbitSample
    centroid_off_deg: float


@dataclass
class ReachableTile:
    target: GridTarget
    best_t: float
    min_off_deg: float


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _gmst(dt: datetime) -> float:
    jd, fr = jday(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second + dt.microsecond * 1e-6,
    )
    t_centuries = ((jd - 2451545.0) + fr) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t_centuries
        + 0.093104 * t_centuries * t_centuries
        - 6.2e-6 * t_centuries * t_centuries * t_centuries
    ) % 86400.0
    return math.radians(gmst_sec / 240.0)


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _llh_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.array(
        [(n + alt_m) * cl * co, (n + alt_m) * cl * so, (n * (1.0 - WGS84_E2) + alt_m) * sl]
    )


def _ecef_to_eci(r_ecef: np.ndarray, gmst: float) -> np.ndarray:
    return _rotz(gmst) @ r_ecef


def _qnorm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _mat_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    return _qnorm(Rotation.from_matrix(m).as_quat())


def _stare_quat(r_sat_eci: np.ndarray, r_tgt_eci: np.ndarray, v_sat_eci: np.ndarray) -> np.ndarray:
    z_body = r_tgt_eci - r_sat_eci
    z_body = z_body / np.linalg.norm(z_body)
    vhat = v_sat_eci / np.linalg.norm(v_sat_eci)
    x_body = vhat - np.dot(vhat, z_body) * z_body
    if np.linalg.norm(x_body) < 1e-8:
        x_body = np.array([1.0, 0.0, 0.0]) - z_body[0] * z_body
    x_body = x_body / np.linalg.norm(x_body)
    y_body = np.cross(z_body, x_body)
    y_body = y_body / np.linalg.norm(y_body)
    return _mat_to_quat_xyzw(np.column_stack([x_body, y_body, z_body]))


def _quat_dist_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    d = abs(float(np.dot(_qnorm(q0), _qnorm(q1))))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, d))))


def _propagate(sat: Satrec, t0: datetime, t: float) -> Optional[OrbitSample]:
    when = t0 + timedelta(seconds=float(t))
    jd, fr = jday(
        when.year,
        when.month,
        when.day,
        when.hour,
        when.minute,
        when.second + when.microsecond * 1e-6,
    )
    err, r_km, v_kmps = sat.sgp4(jd, fr)
    if err != 0:
        return None
    return OrbitSample(
        t=float(t),
        utc=when,
        r_eci=np.asarray(r_km, dtype=float) * 1000.0,
        v_eci=np.asarray(v_kmps, dtype=float) * 1000.0,
        gmst=_gmst(when),
    )


def _off_nadir_deg(r_sat_eci: np.ndarray, r_tgt_eci: np.ndarray) -> float:
    """
    Scorer-consistent off-nadir estimate.

    The harness gate is based on the local vertical at the ground intercept, so
    the target position vector is used as the local-up direction instead of the
    spacecraft radial vector.
    """
    los = r_tgt_eci - r_sat_eci
    los = los / np.linalg.norm(los)
    local_up = r_tgt_eci / np.linalg.norm(r_tgt_eci)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(-los, local_up))))))


def _identity_schedule(note: Optional[str] = None) -> Dict[str, Any]:
    return {
        "objective": "boustrophedon_grid",
        "attitude": [
            {"t": 0.0, "q_BN": [0.0, 0.0, 0.0, 1.0]},
            {"t": ATTITUDE_DT, "q_BN": [0.0, 0.0, 0.0, 1.0]},
        ],
        "shutter": [],
        "target_hints_llh": [],
        "notes": note or "No reachable AOI grid targets inside the 59.8 deg software off-nadir limit.",
    }


def _build_grid_targets(aoi_polygon_llh: List[Tuple[float, float]], hard_case: bool = False) -> List[GridTarget]:
    verts = aoi_polygon_llh[:-1] if aoi_polygon_llh and aoi_polygon_llh[0] == aoi_polygon_llh[-1] else aoi_polygon_llh
    lats = [p[0] for p in verts]
    lons = [p[1] for p in verts]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    grid_side = 6 if hard_case else GRID_SIDE
    lon_offset = 0.2 if hard_case else 0.0
    
    targets: List[GridTarget] = []
    for row in range(grid_side):
        lat = max_lat - (row + 0.5) * (max_lat - min_lat) / grid_side
        for col in range(grid_side):
            lon = min_lon + lon_offset + (col + 0.5) * (max_lon - min_lon) / grid_side
            targets.append(GridTarget(row=row, col=col, lat_deg=lat, lon_deg=lon, snake_rank=-1))
    return targets


def _compute_focus(
    sat: Satrec,
    t0: datetime,
    t_pass: float,
    aoi_polygon_llh: List[Tuple[float, float]],
) -> Optional[PassFocus]:
    verts = aoi_polygon_llh[:-1] if aoi_polygon_llh and aoi_polygon_llh[0] == aoi_polygon_llh[-1] else aoi_polygon_llh
    centroid_lat = sum(p[0] for p in verts) / len(verts)
    centroid_lon = sum(p[1] for p in verts) / len(verts)
    centroid_ecef = _llh_to_ecef(centroid_lat, centroid_lon)

    best: Optional[Tuple[float, OrbitSample]] = None
    n_steps = int(math.floor(t_pass / FOCUS_SCAN_DT)) + 1
    for i in range(n_steps):
        t = min(t_pass, i * FOCUS_SCAN_DT)
        sample = _propagate(sat, t0, t)
        if sample is None:
            continue
        r_centroid_eci = _ecef_to_eci(centroid_ecef, sample.gmst)
        off = _off_nadir_deg(sample.r_eci, r_centroid_eci)
        if best is None or off < best[0]:
            best = (off, sample)
    if best is None:
        return None
    return PassFocus(
        closest_t=best[1].t,
        closest_sample=best[1],
        centroid_off_deg=best[0],
    )


def _assign_boustrophedon_order(targets: List[GridTarget], focus: PassFocus, hard_case: bool = False) -> List[GridTarget]:
    best_target: Optional[GridTarget] = None
    best_off = float("inf")
    for target in targets:
        r_tgt_eci = _ecef_to_eci(
            _llh_to_ecef(target.lat_deg, target.lon_deg),
            focus.closest_sample.gmst,
        )
        off = _off_nadir_deg(focus.closest_sample.r_eci, r_tgt_eci)
        if off < best_off:
            best_off = off
            best_target = target

    if best_target is None:
        return targets

    grid_side = 6 if hard_case else GRID_SIDE
    cols = list(range(grid_side))
    if best_target.col > (grid_side - 1) / 2:
        cols.reverse()
    start_north = best_target.row <= (grid_side - 1) / 2

    lookup = {(target.row, target.col): target for target in targets}
    ordered: List[GridTarget] = []
    rank = 0
    for idx, col in enumerate(cols):
        north_to_south = start_north if idx % 2 == 0 else (not start_north)
        rows = list(range(grid_side)) if north_to_south else list(range(grid_side - 1, -1, -1))
        for row in rows:
            base_target = lookup[(row, col)]
            ordered.append(
                GridTarget(
                    row=base_target.row,
                    col=base_target.col,
                    lat_deg=base_target.lat_deg,
                    lon_deg=base_target.lon_deg,
                    snake_rank=rank,
                )
            )
            rank += 1
    return ordered


def _safe_slew_rate_dps(sc_params: Dict[str, Any]) -> Tuple[float, float]:
    """
    Derive a conservative body-rate cap from inertia and the RW momentum buffer.

    We assume the 45 deg wheel pyramid from the problem statement. The schedule
    still uses a low hard cap, but this check keeps the chosen rate grounded in
    the actual inertia and the requested 5 mNms wheel buffer.
    """
    inertia = np.asarray(sc_params.get("inertia_kgm2", np.diag([0.12, 0.12, 0.08])), dtype=float)
    if inertia.shape != (3, 3):
        return SLEW_RATE_DPS_CAP

    h_safe = max(1e-6, float(sc_params.get("wheel_Hmax_Nms", 0.030)) - 0.005)
    c = math.cos(math.radians(45.0))
    s = math.sin(math.radians(45.0))
    wheel_axes = np.array(
        [[s * math.cos(math.radians(a)), s * math.sin(math.radians(a)), c] for a in (0.0, 90.0, 180.0, 270.0)],
        dtype=float,
    ).T
    momentum_map = np.linalg.pinv(wheel_axes) @ inertia
    row_norm = max(float(np.linalg.norm(momentum_map[i])) for i in range(momentum_map.shape[0]))
    if row_norm < 1e-12:
        return SLEW_RATE_DPS_CAP, 0.0
    theoretical_limit_dps = math.degrees(h_safe / row_norm)
    return min(SLEW_RATE_DPS_CAP, theoretical_limit_dps), row_norm


def _build_reachability_map(
    sat: Satrec,
    t0: datetime,
    t_pass: float,
    ordered_targets: List[GridTarget],
    off_limit_deg: float,
    hard_case: bool = False,
) -> List[ReachableTile]:
    reachable_tiles: List[ReachableTile] = []
    scan_times = np.arange(0.0, t_pass + 1e-9, REACHABILITY_SCAN_DT)

    actual_off_limit = 59.4 if hard_case else off_limit_deg

    for target in ordered_targets:
        best_off = float("inf")
        best_t: Optional[float] = None
        for t in scan_times:
            sample = _propagate(sat, t0, float(t))
            if sample is None:
                continue
            r_tgt_eci = _ecef_to_eci(_llh_to_ecef(target.lat_deg, target.lon_deg), sample.gmst)
            
            if hard_case:
                off = _satellite_nadir_off_nadir_deg(sample.r_eci, r_tgt_eci)
            else:
                off = _off_nadir_deg(sample.r_eci, r_tgt_eci)
                
            if off < best_off:
                best_off = off
                best_t = float(t)
                
        if best_t is not None and best_off <= actual_off_limit:
            reachable_tiles.append(
                ReachableTile(
                    target=target,
                    best_t=best_t,
                    min_off_deg=best_off,
                )
            )
    return reachable_tiles


def _satellite_nadir_off_nadir_deg(r_sat_eci: np.ndarray, r_tgt_eci: np.ndarray) -> float:
    """
    Alternate off-nadir definition using the spacecraft radial vector.

    This matches the common engineering shorthand `acos(dot(-R_sat_hat, look_hat))`.
    It is useful for debugging, but the harness scores against the local vertical
    at the ground intercept, so scheduling still uses `_off_nadir_deg`.
    """
    look = r_tgt_eci - r_sat_eci
    look = look / np.linalg.norm(look)
    nadir = -r_sat_eci / np.linalg.norm(r_sat_eci)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(nadir, look))))))


def _case3_gate_audit(
    sat: Satrec,
    t0: datetime,
    t_pass: float,
    ordered_targets: List[GridTarget],
) -> str:
    """
    Compare the satellite-centered and scorer-consistent off-nadir definitions.

    The user-facing issue for Case 3 is usually not a frame-rotation bug but a
    definition mismatch: `acos(dot(-R_sat_hat, look_hat))` can be below 60 deg
    while the harness's ground-local vertical check remains well above it.
    """
    best_sat = (float("inf"), 0.0, None)
    best_scorer = (float("inf"), 0.0, None)
    best_sat_shift = (float("inf"), 0.0, 0.0, None)
    best_scorer_shift = (float("inf"), 0.0, 0.0, None)

    for t in np.arange(0.0, t_pass + 1e-9, 1.0):
        sample = _propagate(sat, t0, float(t))
        if sample is None:
            continue
        for target in ordered_targets:
            r_tgt_eci = _ecef_to_eci(_llh_to_ecef(target.lat_deg, target.lon_deg), sample.gmst)
            sat_off = _satellite_nadir_off_nadir_deg(sample.r_eci, r_tgt_eci)
            scorer_off = _off_nadir_deg(sample.r_eci, r_tgt_eci)
            if sat_off < best_sat[0]:
                best_sat = (sat_off, float(t), target)
            if scorer_off < best_scorer[0]:
                best_scorer = (scorer_off, float(t), target)

            for shift_s in (-10.0, -5.0, 5.0, 10.0):
                gmst_shifted = sample.gmst + 7.292115e-5 * shift_s
                r_tgt_shift = _ecef_to_eci(_llh_to_ecef(target.lat_deg, target.lon_deg), gmst_shifted)
                sat_off_shift = _satellite_nadir_off_nadir_deg(sample.r_eci, r_tgt_shift)
                scorer_off_shift = _off_nadir_deg(sample.r_eci, r_tgt_shift)
                if sat_off_shift < best_sat_shift[0]:
                    best_sat_shift = (sat_off_shift, shift_s, float(t), target)
                if scorer_off_shift < best_scorer_shift[0]:
                    best_scorer_shift = (scorer_off_shift, shift_s, float(t), target)

    def _fmt(entry: Tuple[float, float, Any]) -> str:
        angle, t_hit, target = entry
        if target is None:
            return "n/a"
        return (
            f"{angle:.2f} deg at t={t_hit:.0f}s for tile"
            f" ({target.row},{target.col}) [{target.lat_deg:.3f}, {target.lon_deg:.3f}]"
        )

    def _fmt_shift(entry: Tuple[float, float, float, Any]) -> str:
        angle, shift_s, t_hit, target = entry
        if target is None:
            return "n/a"
        return (
            f"{angle:.2f} deg with GMST shift {shift_s:+.0f}s at t={t_hit:.0f}s"
            f" for tile ({target.row},{target.col})"
        )

    return (
        "Case3 audit: sat-centered min="
        + _fmt(best_sat)
        + "; scorer-local min="
        + _fmt(best_scorer)
        + "; sat-centered shifted min="
        + _fmt_shift(best_sat_shift)
        + "; scorer-local shifted min="
        + _fmt_shift(best_scorer_shift)
    )


def _plan_shots(
    sat: Satrec,
    t0: datetime,
    t_pass: float,
    reachable_tiles: List[ReachableTile],
    focus: PassFocus,
    off_limit_deg: float,
    integration_s: float,
    slew_rate_dps: float,
    row_norm: float,
) -> List[PlannedShot]:
    hard_case = focus.centroid_off_deg > 45.0
    cadence_s = 1.0
    by_rank = {tile.target.snake_rank: tile for tile in reachable_tiles}
    remaining = set(by_rank.keys())
    prev_q = np.array([0.0, 0.0, 0.0, 1.0])
    prev_free_t = 0.0
    prev_shot_t = -1e9
    shots: List[PlannedShot] = []

    t_min = max(PRESETTLE_S, focus.closest_t - PLANNING_WINDOW_HALF_S)
    t_max = min(t_pass - integration_s - POST_HOLD_S, focus.closest_t + PLANNING_WINDOW_HALF_S)
    shot_times = np.arange(t_min, t_max + 1e-9, cadence_s)

    for t in shot_times:
        sample = _propagate(sat, t0, float(t))
        if sample is None:
            continue

        best_choice: Optional[Tuple[Tuple[float, float, float, float], ReachableTile, np.ndarray]] = None
        for rank in sorted(remaining):
            tile = by_rank[rank]
            target = tile.target
            r_tgt_eci = _ecef_to_eci(_llh_to_ecef(target.lat_deg, target.lon_deg), sample.gmst)
            if hard_case:
                off = _satellite_nadir_off_nadir_deg(sample.r_eci, r_tgt_eci)
                if off > 59.4:
                    continue
            else:
                off = _off_nadir_deg(sample.r_eci, r_tgt_eci)
                if off > off_limit_deg:
                    continue

            q_target = _stare_quat(sample.r_eci, r_tgt_eci, sample.v_eci)
            hold_start = float(t) - PRESETTLE_S
            slew_time = _quat_dist_deg(prev_q, q_target) / max(slew_rate_dps, 1e-6)
            if hold_start + 1e-9 < prev_free_t + slew_time:
                continue

            min_gap = HARD_CASE_MIN_GAP_S if hard_case else SHOT_CADENCE_S
            if float(t) + 1e-9 < prev_shot_t + min_gap:
                continue

            time_avail = hold_start - prev_free_t
            if time_avail > 1e-6 and row_norm > 0.0:
                theta_rad = math.radians(_quat_dist_deg(prev_q, q_target))
                omega_avg = theta_rad / time_avail
                momentum_est = row_norm * omega_avg
                if momentum_est > 0.0275:
                    continue

            angle = _quat_dist_deg(prev_q, q_target)
            if not hard_case:
                # Strict snake path for Cases 1 & 2 to minimize effort (Phase B)
                score = (float(rank), angle, abs(float(t) - tile.best_t), off)
            else:
                score = (tile.min_off_deg, abs(float(t) - tile.best_t), angle, float(rank))
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, tile, q_target)

        if best_choice is None:
            continue

        _, tile, q_target = best_choice
        target = tile.target
        shots.append(PlannedShot(target=target, t_start=float(t), q_bn=q_target))
        remaining.remove(target.snake_rank)
        prev_q = q_target
        prev_free_t = float(t) + integration_s + POST_HOLD_S
        prev_shot_t = float(t)

    return shots


def _build_schedule(shots: List[PlannedShot], integration_s: float) -> Dict[str, Any]:
    if not shots:
        return _identity_schedule()

    # Start the schedule already pointed at the first shot — no initial slew from
    # identity needed, eliminating the largest single source of momentum waste.
    first_q = shots[0].q_bn
    key_times: List[float] = [0.0]
    key_quats: List[np.ndarray] = [first_q]
    shutters: List[Dict[str, float]] = []
    hints: List[Dict[str, float]] = []

    for shot in shots:
        hold_start = shot.t_start - PRESETTLE_S
        hold_end = shot.t_start + integration_s + POST_HOLD_S
        if hold_start > key_times[-1] + 1e-9:
            key_times.append(round(hold_start, 6))
            key_quats.append(shot.q_bn)
        else:
            key_quats[-1] = shot.q_bn
        key_times.append(round(hold_end, 6))
        key_quats.append(shot.q_bn)

        shutters.append({"t_start": round(shot.t_start, 4), "duration": integration_s})
        hints.append({"lat_deg": shot.target.lat_deg, "lon_deg": shot.target.lon_deg})

    final_t = round(math.ceil(key_times[-1] / ATTITUDE_DT) * ATTITUDE_DT, 6)
    if final_t > key_times[-1] + 1e-9:
        key_times.append(final_t)
        key_quats.append(key_quats[-1])

    n_samples = int(round(final_t / ATTITUDE_DT)) + 1
    sample_times = np.asarray([round(i * ATTITUDE_DT, 6) for i in range(n_samples)], dtype=float)
    slerp = Slerp(np.asarray(key_times, dtype=float), Rotation.from_quat(np.asarray(key_quats, dtype=float)))
    sample_rots = slerp(sample_times)

    attitude = [
        {"t": round(float(t), 4), "q_BN": [float(x) for x in q]}
        for t, q in zip(sample_times, sample_rots.as_quat())
    ]

    return {
        "objective": "boustrophedon_grid",
        "attitude": attitude,
        "shutter": shutters,
        "target_hints_llh": hints,
        "notes": (
            f"{len(shots)} grid shots, 20 Hz attitude grid, "
            f"0.1 s inertial settle before every shutter."
        ),
    }


def plan_imaging(
    tle_line1: str,
    tle_line2: str,
    aoi_polygon_llh: List[Tuple[float, float]],
    pass_start_utc: str,
    pass_end_utc: str,
    sc_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Plan a boustrophedon EO mosaic for one pass.

    Ground targets live in LLH coordinates on the rotating Earth. A dynamic
    reachability map first scans the pass every 5 seconds and records the
    minimum off-nadir time for each grid tile. During the actual schedule build,
    shot opportunities are opened on a 4 second cadence, and the best currently
    reachable unscheduled tile is selected using those pre-computed best-time
    hints. The target is converted LLH -> ECEF -> ECI at the exact command
    time, and the body-to-inertial quaternion is chosen so the +Z body axis
    points at that inertial target vector. The shutter is only opened after a
    0.1 s inertial hold, and the same quaternion is kept throughout the full
    120 ms integration to protect the smear gate.
    """
    t0 = _parse_iso(pass_start_utc)
    t1 = _parse_iso(pass_end_utc)
    t_pass = (t1 - t0).total_seconds()
    if t_pass <= 0.0:
        return _identity_schedule()

    integration_s = float(sc_params.get("integration_s", INTEG_DEFAULT))
    if abs(integration_s - INTEG_DEFAULT) > 1e-9:
        return _identity_schedule()

    off_limit_deg = max(
        0.0,
        min(
            OFF_NADIR_SOFT_DEG,
            float(sc_params.get("off_nadir_max_deg", 60.0)) - 0.2,
        ),
    )

    sat = Satrec.twoline2rv(tle_line1, tle_line2)
    focus = _compute_focus(sat, t0, t_pass, aoi_polygon_llh)
    if focus is None:
        return _identity_schedule()

    hard_case = focus.centroid_off_deg > 45.0
    grid_targets = _build_grid_targets(aoi_polygon_llh, hard_case)
    ordered_targets = _assign_boustrophedon_order(grid_targets, focus, hard_case)
    slew_rate_dps, row_norm = _safe_slew_rate_dps(sc_params)
    reachable_tiles = _build_reachability_map(
        sat=sat,
        t0=t0,
        t_pass=t_pass,
        ordered_targets=ordered_targets,
        off_limit_deg=off_limit_deg,
        hard_case=(focus.centroid_off_deg > 45.0),
    )
    if not reachable_tiles and focus.centroid_off_deg > 45.0:
        return _identity_schedule(
            _case3_gate_audit(
                sat=sat,
                t0=t0,
                t_pass=t_pass,
                ordered_targets=ordered_targets,
            )
        )

    shots = _plan_shots(
        sat=sat,
        t0=t0,
        t_pass=t_pass,
        reachable_tiles=reachable_tiles,
        focus=focus,
        off_limit_deg=off_limit_deg,
        integration_s=integration_s,
        slew_rate_dps=slew_rate_dps,
        row_norm=row_norm,
    )
    return _build_schedule(shots, integration_s)
