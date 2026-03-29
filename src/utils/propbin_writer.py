"""
Python writer for PROP v2 `.propbin` / `.propbin.gz` files.

Input format matches the structure returned by `cpp_propagation.propbin_reader.load_propbin`
after the local extensions in this repo:
  - result["source_location"] = (x, y, z)
  - result["receivers"][key]["rx_id"]
  - result["receivers"][key]["location"] = (x, y, z)
  - result["receivers"][key]["total_intensity_dBm"]
  - result["receivers"][key]["total_loss_dB"]
  - result["receivers"][key]["path_info"] = list of paths
  - path["path_type"] / path["level"]
  - path["chain_vertex_ids"] for normal BFS paths
  - path["rooftop_points"] for rooftop diffraction paths
  - path["departure_angle"], path["arrival_angle"]
  - path["distance"], path["loss"], path["delay"]
"""

from __future__ import annotations

import gzip
import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np


MAGIC = 0x50524F50  # "PROP"
VERSION = 2


def float_to_half(x: float) -> int:
    return int(np.asarray(np.float16(x), dtype=np.float16).view(np.uint16))


def _pack_receiver_path_count(path_info: Sequence[Dict[str, Any]]) -> int:
    return len(path_info)


def _get_path_type(path: Dict[str, Any]) -> int:
    return int(path.get("path_type", path.get("level", 0)))


def _get_angles(path: Dict[str, Any]) -> Tuple[float, float, float, float]:
    dep = path.get("departure_angle")
    arr = path.get("arrival_angle")
    if dep is None or arr is None or len(dep) != 2 or len(arr) != 2:
        raise ValueError("path is missing departure_angle / arrival_angle")
    return float(dep[0]), float(dep[1]), float(arr[0]), float(arr[1])


def _get_rooftop_points(path: Dict[str, Any]) -> List[Tuple[float, float]]:
    points = path.get("rooftop_points")
    if points is None:
        points = path.get("path_chain", [])
    out: List[Tuple[float, float]] = []
    for point in points:
        if len(point) < 2:
            raise ValueError("rooftop point must have at least x/y")
        out.append((float(point[0]), float(point[1])))
    return out


def _get_chain_vertex_ids(path: Dict[str, Any]) -> List[int]:
    ids = path.get("chain_vertex_ids")
    if ids is None:
        raise ValueError("normal BFS path requires chain_vertex_ids for V2 writing")
    return [int(v) for v in ids]


def _receiver_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
    receiver = item[1]
    rx_id = receiver.get("rx_id")
    if rx_id is not None:
        return int(rx_id), item[0]
    loc = receiver.get("location", (0.0, 0.0, 0.0))
    x = int(round(float(loc[0])))
    y = int(round(float(loc[1])))
    return y * 257 + x, item[0]


def _infer_rx_id(receiver: Dict[str, Any]) -> int:
    if "rx_id" in receiver:
        return int(receiver["rx_id"])
    loc = receiver.get("location", (0.0, 0.0, 0.0))
    x = int(round(float(loc[0])))
    y = int(round(float(loc[1])))
    return y * 257 + x


def write_propbin_v2(output_path: Union[str, Path], result: Dict[str, Any]) -> None:
    output_path = Path(output_path)

    source_xyz = result.get("source_location")
    if source_xyz is None or len(source_xyz) != 3:
        raise ValueError("result must contain source_location=(x, y, z)")

    receiver_items = sorted(result.get("receivers", {}).items(), key=_receiver_sort_key)
    n_receivers = len(receiver_items)

    n_paths_total = 0
    n_chain_values_total = 0
    path_entries: List[Dict[str, Any]] = []

    for _, receiver in receiver_items:
        path_info = receiver.get("path_info", [])
        n_paths_total += _pack_receiver_path_count(path_info)
        for path in path_info:
            path_type = _get_path_type(path)
            if path_type == -1:
                rooftop_points = _get_rooftop_points(path)
                chain_len = len(rooftop_points) * 2
                chain_values = []
                for x, y in rooftop_points:
                    chain_values.append(float_to_half(x))
                    chain_values.append(float_to_half(y))
            else:
                chain_vertex_ids = _get_chain_vertex_ids(path)
                chain_len = len(chain_vertex_ids)
                chain_values = [int(v) & 0xFFFF for v in chain_vertex_ids]

            dep_az, dep_el, arr_az, arr_el = _get_angles(path)
            path_entries.append(
                {
                    "loss": float(path["loss"]),
                    "distance": float(path["distance"]),
                    "delay": float(path["delay"]),
                    "dep_az": dep_az,
                    "dep_el": dep_el,
                    "arr_az": arr_az,
                    "arr_el": arr_el,
                    "chain_len": chain_len,
                    "path_type": path_type,
                    "chain_values": chain_values,
                }
            )
            n_chain_values_total += chain_len

    buffer = bytearray()
    buffer.extend(struct.pack("<II", MAGIC, VERSION))
    buffer.extend(struct.pack("<3f", float(source_xyz[0]), float(source_xyz[1]), float(source_xyz[2])))
    buffer.extend(struct.pack("<III", n_receivers, n_paths_total, n_chain_values_total))

    current_path_offset = 0
    for _, receiver in receiver_items:
        location = receiver.get("location")
        if location is None or len(location) != 3:
            raise ValueError("receiver must contain location=(x, y, z)")
        path_info = receiver.get("path_info", [])
        buffer.extend(
            struct.pack(
                "<i3f2fHHI",
                _infer_rx_id(receiver),
                float(location[0]),
                float(location[1]),
                float(location[2]),
                float(receiver["total_intensity_dBm"]),
                float(receiver["total_loss_dB"]),
                len(path_info),
                0,
                current_path_offset,
            )
        )
        current_path_offset += len(path_info)

    current_chain_offset = 0
    for entry in path_entries:
        buffer.extend(
            struct.pack(
                "<HHHHHHHHbBHI",
                float_to_half(entry["loss"]),
                float_to_half(entry["distance"]),
                float_to_half(entry["delay"]),
                float_to_half(entry["dep_az"]),
                float_to_half(entry["dep_el"]),
                float_to_half(entry["arr_az"]),
                float_to_half(entry["arr_el"]),
                entry["chain_len"],
                entry["path_type"],
                0,
                0,
                current_chain_offset,
            )
        )
        current_chain_offset += entry["chain_len"]

    for entry in path_entries:
        for value in entry["chain_values"]:
            buffer.extend(struct.pack("<H", int(value) & 0xFFFF))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".gz" or str(output_path).endswith(".propbin.gz"):
        with gzip.open(output_path, "wb") as f:
            f.write(buffer)
    else:
        with output_path.open("wb") as f:
            f.write(buffer)
