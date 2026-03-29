#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cpp_propagation.propbin_reader import load_propbin
from cpp_propagation.propbin_writer import write_propbin_v2


SOURCE_RE = re.compile(r"source_(\d+)\.propbin(?:\.gz)?$")
MAP_RE = re.compile(r"map_(\d+)$")


def wrap_angle_rad(angle: float) -> float:
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return wrapped


def sinc_power_gain(
    departure_az: float,
    departure_el: float,
    boresight_az: float,
    boresight_el: float,
    az_mainlobe_width: float,
    el_mainlobe_width: float,
    min_gain_floor: float,
) -> float:
    delta_az = wrap_angle_rad(departure_az - boresight_az)
    delta_el = departure_el - boresight_el
    az_amp = float(math.sin(math.pi * (delta_az / az_mainlobe_width)) / (math.pi * (delta_az / az_mainlobe_width))) if delta_az != 0.0 else 1.0
    el_amp = float(math.sin(math.pi * (delta_el / el_mainlobe_width)) / (math.pi * (delta_el / el_mainlobe_width))) if delta_el != 0.0 else 1.0
    gain = az_amp * el_amp
    return max(gain * gain, min_gain_floor)


def infer_tx_power_dbm(propbin_data: Dict) -> float:
    candidates: List[float] = []
    for receiver in propbin_data.get("receivers", {}).values():
        total_intensity = receiver.get("total_intensity_dBm")
        total_loss = receiver.get("total_loss_dB")
        if total_intensity is None or total_loss is None:
            continue
        if not math.isfinite(total_intensity) or not math.isfinite(total_loss):
            continue
        candidates.append(float(total_intensity) + float(total_loss))

    if not candidates:
        raise ValueError("could not infer tx power from propbin receiver totals")

    candidates.sort()
    return candidates[len(candidates) // 2]


def apply_tx_beam_in_place(
    propbin_data: Dict,
    tx_power_dbm: float,
    boresight_az: float,
    boresight_el: float,
    az_mainlobe_width: float,
    el_mainlobe_width: float,
    min_gain_floor: float,
) -> Dict:
    result = copy.deepcopy(propbin_data)

    for receiver in result.get("receivers", {}).values():
        linear_sum = 0.0
        for path in receiver.get("path_info", []):
            departure = path.get("departure_angle")
            if departure is None or len(departure) != 2:
                raise ValueError("path is missing departure_angle")

            gain = sinc_power_gain(
                departure_az=float(departure[0]),
                departure_el=float(departure[1]),
                boresight_az=boresight_az,
                boresight_el=boresight_el,
                az_mainlobe_width=az_mainlobe_width,
                el_mainlobe_width=el_mainlobe_width,
                min_gain_floor=min_gain_floor,
            )
            new_loss = float(path["loss"]) - 10.0 * math.log10(gain)
            path["loss"] = new_loss

            linear_sum += math.pow(10.0, (tx_power_dbm - new_loss) / 10.0)

        if linear_sum > 0.0:
            total_intensity = 10.0 * math.log10(linear_sum)
            total_loss = tx_power_dbm - total_intensity
        else:
            total_intensity = -999.0
            total_loss = float("inf")

        receiver["total_intensity_dBm"] = total_intensity
        receiver["total_loss_dB"] = total_loss

    return result


def extract_map_id(path: Path) -> Optional[int]:
    for part in path.parts:
        match = MAP_RE.match(part)
        if match:
            return int(match.group(1))
    return None


def extract_source_idx(path: Path) -> Optional[int]:
    match = SOURCE_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def resolve_map_ids(args: argparse.Namespace) -> Optional[set[int]]:
    map_ids: set[int] = set()
    if args.map_id is not None:
        map_ids.add(args.map_id)
    if args.map_ids:
        map_ids.update(args.map_ids)
    if args.map_id_start is not None or args.map_id_end is not None:
        if args.map_id_start is None or args.map_id_end is None:
            raise SystemExit("--map-id-start and --map-id-end must be used together")
        if args.map_id_start > args.map_id_end:
            raise SystemExit("--map-id-start must be <= --map-id-end")
        map_ids.update(range(args.map_id_start, args.map_id_end + 1))
    return map_ids or None


def find_input_files(input_root: Path, map_ids: Optional[set[int]], source_idx: Optional[int]) -> List[Path]:
    files: List[Path] = []
    for path in sorted(input_root.rglob("source_*.propbin*")):
        file_source_idx = extract_source_idx(path)
        if file_source_idx is None:
            continue
        if source_idx is not None and file_source_idx != source_idx:
            continue

        file_map_id = extract_map_id(path)
        if map_ids is not None and file_map_id not in map_ids:
            continue

        files.append(path)
    return files


def validate_args(args: argparse.Namespace) -> None:
    if (args.input_file is None) == (args.input_root is None):
        raise SystemExit("exactly one of --input-file or --input-root is required")
    if args.input_file is not None and args.output_file is None:
        raise SystemExit("--output-file is required when using --input-file")
    if args.input_root is not None and args.output_root is None:
        raise SystemExit("--output-root is required when using --input-root")
    if args.az_mainlobe_width <= 0.0 or args.el_mainlobe_width <= 0.0:
        raise SystemExit("mainlobe widths must be > 0")
    if args.min_gain_floor <= 0.0 or args.min_gain_floor > 1.0:
        raise SystemExit("min_gain_floor must be in (0, 1]")


def process_one_file(
    input_path: Path,
    output_path: Path,
    boresight_az: float,
    boresight_el: float,
    az_mainlobe_width: float,
    el_mainlobe_width: float,
    min_gain_floor: float,
) -> None:
    propbin_data = load_propbin(input_path)
    version = int(propbin_data.get("version", 0))
    if version != 2:
        raise ValueError(f"only PROP v2 inputs are supported for write-back, got version={version}")

    tx_power_dbm = infer_tx_power_dbm(propbin_data)
    weighted = apply_tx_beam_in_place(
        propbin_data=propbin_data,
        tx_power_dbm=tx_power_dbm,
        boresight_az=boresight_az,
        boresight_el=boresight_el,
        az_mainlobe_width=az_mainlobe_width,
        el_mainlobe_width=el_mainlobe_width,
        min_gain_floor=min_gain_floor,
    )
    write_propbin_v2(output_path, weighted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a TX-side 2D sinc beam pattern to PROP v2 propbin files."
    )
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--map-id", type=int)
    parser.add_argument("--map-ids", type=int, nargs="*")
    parser.add_argument("--map-id-start", type=int)
    parser.add_argument("--map-id-end", type=int)
    parser.add_argument("--source-idx", type=int)
    parser.add_argument("--tx-boresight-az", type=float, required=True)
    parser.add_argument("--tx-boresight-el", type=float, required=True)
    parser.add_argument("--az-mainlobe-width", type=float, required=True)
    parser.add_argument("--el-mainlobe-width", type=float, required=True)
    parser.add_argument("--min-gain-floor", type=float, default=1e-6)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.input_file is not None:
        if not args.input_file.exists():
            raise SystemExit(f"input file not found: {args.input_file}")
        if args.output_file.exists() and not args.force:
            print(f"SKIP existing output: {args.output_file}")
            return 0

        print(f"[1/1] {args.input_file}")
        process_one_file(
            input_path=args.input_file,
            output_path=args.output_file,
            boresight_az=args.tx_boresight_az,
            boresight_el=args.tx_boresight_el,
            az_mainlobe_width=args.az_mainlobe_width,
            el_mainlobe_width=args.el_mainlobe_width,
            min_gain_floor=args.min_gain_floor,
        )
        print(f"OK -> {args.output_file}")
        return 0

    if not args.input_root.exists():
        raise SystemExit(f"input root not found: {args.input_root}")

    map_ids = resolve_map_ids(args)
    files = find_input_files(args.input_root, map_ids, args.source_idx)
    if not files:
        raise SystemExit("no matching propbin files found")

    failed: List[Path] = []
    skipped = 0
    total = len(files)
    for idx, input_path in enumerate(files, start=1):
        rel_path = input_path.relative_to(args.input_root)
        output_path = args.output_root / rel_path
        print(f"[{idx}/{total}] {input_path}")
        if output_path.exists() and not args.force:
            print(f"  SKIP existing output: {output_path}")
            skipped += 1
            continue

        try:
            process_one_file(
                input_path=input_path,
                output_path=output_path,
                boresight_az=args.tx_boresight_az,
                boresight_el=args.tx_boresight_el,
                az_mainlobe_width=args.az_mainlobe_width,
                el_mainlobe_width=args.el_mainlobe_width,
                min_gain_floor=args.min_gain_floor,
            )
            print(f"  OK -> {output_path}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append(input_path)

    print("\n===== Summary =====")
    print(f"total   : {total}")
    print(f"failed  : {len(failed)}")
    print(f"skipped : {skipped}")
    if failed:
        for path in failed:
            print(f"  - {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
