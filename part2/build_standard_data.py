from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT_DIR / "part2" / "data" / "raw.csv"
DEFAULT_PLANET = ROOT_DIR / "part2" / "data" / "planet.csv"
DEFAULT_BRIDGE = ROOT_DIR / "part2" / "data" / "data.csv"
DEFAULT_OUTPUT = ROOT_DIR / "part2" / "data" / "planet_full.csv"
DEFAULT_REPORT = ROOT_DIR / "part2" / "data" / "planet_full_match_report.csv"


EXACT_STAGES = (
    ("pl_orbper", "pl_rade", "pl_bmasse", "pl_orbsmax"),
    ("pl_orbper", "pl_rade", "pl_bmasse"),
    ("pl_orbper", "pl_rade"),
    ("pl_rade", "sy_dist"),
    ("pl_trandep",),
    ("sy_dist", "pl_bmasse"),
    ("sy_dist",),
)

FUZZY_COLUMNS = (
    "pl_orbper",
    "pl_rade",
    "pl_bmasse",
    "pl_orbsmax",
    "pl_trandep",
    "pl_trandur",
    "pl_imppar",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "sy_dist",
)


def read_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False)


def normalized_value(value: object) -> object:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isfinite(number):
        return round(number, 10)
    return None


def build_key(row: pd.Series, columns: tuple[str, ...]) -> tuple[object, ...] | None:
    values = tuple(normalized_value(row[col]) for col in columns)
    if any(value is None for value in values):
        return None
    return values


def exact_match_stage(
    planet: pd.DataFrame,
    bridge: pd.DataFrame,
    columns: tuple[str, ...],
    matched: dict[int, dict],
    used_bridge_indices: set[int],
) -> int:
    key_to_indices: dict[tuple[object, ...], list[int]] = {}
    for bridge_index, row in bridge.iterrows():
        key = build_key(row, columns)
        if key is None:
            continue
        key_to_indices.setdefault(key, []).append(int(bridge_index))

    added = 0
    for planet_index, row in planet.iterrows():
        if int(planet_index) in matched:
            continue
        key = build_key(row, columns)
        if key is None:
            continue
        candidates = key_to_indices.get(key, [])
        available = [idx for idx in candidates if idx not in used_bridge_indices]
        if len(available) == 1:
            bridge_index = available[0]
            matched[int(planet_index)] = {
                "bridge_index": bridge_index,
                "strategy": "exact:" + "+".join(columns),
                "score": 0.0,
                "candidate_count": len(candidates),
            }
            used_bridge_indices.add(bridge_index)
            added += 1
    return added


def relative_diff(left: object, right: object) -> float | None:
    if pd.isna(left) or pd.isna(right):
        return None
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return 0.0 if str(left).strip() == str(right).strip() else 1.0
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    denominator = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denominator


def fuzzy_score(planet_row: pd.Series, bridge_row: pd.Series, columns: list[str]) -> tuple[float, int]:
    total = 0.0
    used = 0
    for col in columns:
        diff = relative_diff(planet_row[col], bridge_row[col])
        if diff is None:
            continue
        weight = 3.0 if col in {"pl_orbper", "pl_rade", "sy_dist", "pl_trandep"} else 1.0
        total += diff * weight
        used += 1
    if used == 0:
        return float("inf"), 0
    return total / used, used


def fuzzy_match_remaining(
    planet: pd.DataFrame,
    bridge: pd.DataFrame,
    matched: dict[int, dict],
    used_bridge_indices: set[int],
) -> int:
    common = [col for col in FUZZY_COLUMNS if col in planet.columns and col in bridge.columns]
    added = 0

    for planet_index, planet_row in planet.iterrows():
        planet_index = int(planet_index)
        if planet_index in matched:
            continue

        candidates = bridge.loc[[idx for idx in bridge.index if int(idx) not in used_bridge_indices]]
        sy_dist = normalized_value(planet_row["sy_dist"]) if "sy_dist" in planet.columns else None
        if sy_dist is not None and "sy_dist" in bridge.columns:
            same_system = candidates[candidates["sy_dist"].map(normalized_value) == sy_dist]
            if not same_system.empty:
                candidates = same_system

        best_index: int | None = None
        best_score = float("inf")
        best_used = 0
        for bridge_index, bridge_row in candidates.iterrows():
            score, used = fuzzy_score(planet_row, bridge_row, common)
            if used > best_used or (used == best_used and score < best_score):
                best_index = int(bridge_index)
                best_score = score
                best_used = used

        if best_index is not None and math.isfinite(best_score):
            matched[planet_index] = {
                "bridge_index": best_index,
                "strategy": "fuzzy:nearest_common_features",
                "score": best_score,
                "candidate_count": len(candidates),
            }
            used_bridge_indices.add(best_index)
            added += 1

    return added


def build_standard_data(raw_path: Path, planet_path: Path, bridge_path: Path, output_path: Path, report_path: Path) -> None:
    raw = read_raw(raw_path)
    planet = pd.read_csv(planet_path)
    bridge = pd.read_csv(bridge_path)

    if len(raw) != len(bridge):
        raise ValueError(
            f"raw and bridge must have the same number of rows; got raw={len(raw)}, bridge={len(bridge)}"
        )

    matched: dict[int, dict] = {}
    used_bridge_indices: set[int] = set()
    stage_stats = []

    for columns in EXACT_STAGES:
        if not all(col in planet.columns and col in bridge.columns for col in columns):
            continue
        added = exact_match_stage(planet, bridge, columns, matched, used_bridge_indices)
        stage_stats.append({"stage": "exact:" + "+".join(columns), "added": added})

    fuzzy_added = fuzzy_match_remaining(planet, bridge, matched, used_bridge_indices)
    stage_stats.append({"stage": "fuzzy:nearest_common_features", "added": fuzzy_added})

    missing = [idx for idx in range(len(planet)) if idx not in matched]
    if missing:
        raise RuntimeError(f"Could not match {len(missing)} planet rows: {missing[:20]}")

    ordered_bridge_indices = [matched[idx]["bridge_index"] for idx in range(len(planet))]
    standard = raw.iloc[ordered_bridge_indices].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    standard.to_csv(output_path, index=False)

    report_rows = []
    for planet_index in range(len(planet)):
        info = matched[planet_index]
        bridge_index = info["bridge_index"]
        raw_row = raw.iloc[bridge_index]
        report_rows.append(
            {
                "planet_index": planet_index,
                "bridge_index": bridge_index,
                "raw_rowid": raw_row.get("rowid", ""),
                "pl_name": raw_row.get("pl_name", ""),
                "hostname": raw_row.get("hostname", ""),
                "strategy": info["strategy"],
                "score": info["score"],
                "candidate_count": info["candidate_count"],
            }
        )
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    print("Standard data generated")
    print(f"  raw:      {raw.shape}")
    print(f"  planet:   {planet.shape}")
    print(f"  output:   {standard.shape} -> {output_path}")
    print(f"  report:   {report_path}")
    print("  match stages:")
    for stat in stage_stats:
        print(f"    {stat['stage']}: {stat['added']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a full-column raw-data subset whose rows match part2/data/planet.csv."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--planet", type=Path, default=DEFAULT_PLANET)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    build_standard_data(args.raw, args.planet, args.bridge, args.output, args.report)


if __name__ == "__main__":
    main()
