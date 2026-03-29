#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灰度图像生成器 - 支持.propbin和.pkl.gz格式
Gray Image Generator - Supports both .propbin (C++) and .pkl.gz (Python) formats

用法:
    python generate_gray_image.py <数据文件> [--map-id ID]
    
示例:
    # 从C++生成的.propbin文件
    python generate_gray_image.py prop_out/map_0/scenario_A/source_0.propbin --map-id 0
    
    # 从Python生成的.pkl.gz文件
    python generate_gray_image.py propagation_results/map_0/scenario_A/source_0_propagation.pkl.gz
"""

import os
import sys
import json
import pickle
import gzip
import struct
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import argparse

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not available, building mask without GPU acceleration")


import sys
from pathlib import Path

# 获取数据根目录 (脚本在 3D_scripts 文件夹中)
DATA_ROOT = Path(__file__).resolve().parent

# 确保可以导入 propbin_reader
sys.path.append(str(DATA_ROOT / "cpp_propagation"))
try:
    from propbin_reader import load_propbin
except ImportError:
    print(f"Error: Could not import propbin_reader from {DATA_ROOT / 'cpp_propagation'}")
    sys.exit(1)

def load_propbin_data(file_path: str) -> Dict:
    """
    使用 propbin_reader 加载 .propbin 数据
    """
    return load_propbin(file_path)


def load_pkl_data(file_path: str) -> Dict:
    """加载.pkl.gz文件 (Python输出格式)"""
    if file_path.endswith('.pkl.gz'):
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    return data


def detect_and_load_data(file_path: str) -> Dict:
    """自动检测格式并加载数据"""
    print(f"Loading: {file_path}")
    
    if '.propbin' in file_path:
        print("  Format: propbin (C++ output)")
        return load_propbin_data(file_path)
    elif file_path.endswith('.pkl.gz') or file_path.endswith('.pkl'):
        print("  Format: pickle (Python output)")
        return load_pkl_data(file_path)
    else:
        raise ValueError(f"Unsupported format: {file_path}")


class Point:
    def __init__(self, x: float, y: float, z: float = 1.5):
        self.x = x
        self.y = y
        self.z = z


def point_in_polygon_cpu(x: float, y: float, polygon: List[Point]) -> bool:
    """CPU版本的点在多边形内检测"""
    n = len(polygon)
    if n < 3:
        return False
    
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


BUILDING_HEIGHT_SCALE = 5.0
COORD_SCALE = 1.0 # 保持 1:1 映射

def load_buildings_with_height(json_path: str, coord_scale: float = 1.0):
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    buildings = []
    for item in data:
        # Supported formats:
        # 1) [polygon_points, [height]] or [polygon_points, height]
        # 2) polygon_points only
        if (
            isinstance(item, list)
            and len(item) >= 2
            and isinstance(item[0], list)
            and item[0]
            and isinstance(item[0][0], (list, tuple))
        ):
            poly_raw, h_data = item[0], item[1]
            if isinstance(h_data, list):
                h_val = float(h_data[0])
            else:
                h_val = float(h_data)
        else:
            poly_raw = item
            h_val = 20.0
        
        scaled_h = h_val * BUILDING_HEIGHT_SCALE
        
        # [MODIFIED] Apply coordinate scaling to buildings (x5 for full scale)
        poly_pts = [Point(p[0] * coord_scale, p[1] * coord_scale, 0.0) for p in poly_raw]
        
        buildings.append({
            "poly": poly_pts,
            "height": scaled_h
        })
    return buildings


def generate_gray_image(data: Dict, buildings_list: List[Dict], output_path: str) -> None:
    """生成灰度强度图像"""
    intensity_image = build_gray_image(data, buildings_list)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    Image.fromarray(intensity_image, mode='L').save(output_path)
    print(f"Saved: {output_path}")
    return

    source_location = data['source_location']
    source_x, source_y, source_z = source_location
    print(f"Source: ({source_x:.1f}, {source_y:.1f}, {source_z:.1f})")
    
    # 提取接收点强度
    intensity_values = []
    receiver_positions = []
    rx_z_sum = 0.0
    rx_count = 0
    
    DEFAULT_TX_DBM = 23.0
    
    is_large_map = False # 强制 1:1
    
    if is_large_map:
        print(f"  Detected large map coordinates. Mode: Full Scale (1:1) - Dynamic Size.")

    max_px, max_py = 0, 0

    skipped_source_self = 0

    for receiver_key, receiver_info in data['receivers'].items():
        if 'total_intensity_dBm' in receiver_info:
            intensity_dbm = receiver_info['total_intensity_dBm']
        elif 'total_loss_dB' in receiver_info:
            intensity_dbm = DEFAULT_TX_DBM - receiver_info['total_loss_dB']
        else:
            continue
        
        location = receiver_info['location']
        rx_z_sum += location[2]
        rx_count += 1
        
        if isinstance(intensity_dbm, (int, float)) and not math.isnan(intensity_dbm) and not math.isinf(intensity_dbm):
            intensity_values.append(float(intensity_dbm))
            
            # [MODIFIED] No downscaling, 1:1 mapping
            px = int(round(location[0]))
            py = int(round(location[1]))
            
            receiver_positions.append((px, py))
            max_px = max(max_px, px)
            max_py = max(max_py, py)
    
    if not intensity_values:
        print("Error: no valid intensity values found")
        return
    
    avg_rx_z = rx_z_sum / rx_count if rx_count > 0 else 1.5
    print(f"Receivers: {len(intensity_values)}, Avg Height: {avg_rx_z:.2f}m")
    
    actual_min_int = min(intensity_values)
    actual_max_int = max(intensity_values)
    print(f"Intensity range: {actual_min_int:.2f} to {actual_max_int:.2f} dBm")
    
    # [MODIFIED] Dynamic image size
    W, H_img = max_px + 1, max_py + 1
    print(f"Dynamic Image Size: {W} x {H_img}")

    intensity_image = np.zeros((H_img, W), dtype=np.uint8)
    
    for i, (x, y) in enumerate(receiver_positions):
        if 0 <= x < W and 0 <= y < H_img:
            intensity_dbm = intensity_values[i]
            
            if actual_max_int > actual_min_int:
                normalized = (intensity_dbm - actual_min_int) / (actual_max_int - actual_min_int)
            else:
                normalized = 0.0
            
            pixel_value = int(255 * normalized)
            pixel_value = max(0, min(255, pixel_value))
            
            image_y = (H_img - 1) - y
            intensity_image[image_y, x] = pixel_value
    
    # 将建筑物内部设为0 (仅当 Rx 高度 < 建筑高度时)
    print("Masking building interiors...")
    masked_count = 0
    
    # Pre-select relevant buildings (those taller than Rx)
    blocking_buildings = [b for b in buildings_list if avg_rx_z < b["height"]]
    print(f"  Blocking buildings (H > {avg_rx_z:.1f}m): {len(blocking_buildings)} / {len(buildings_list)}")
    
    if blocking_buildings:
        for x in range(W):
            for y in range(H_img):
                # Check against blocking buildings
                for b in blocking_buildings:
                    if point_in_polygon_cpu(float(x), float(y), b["poly"]):
                        image_y = (H_img - 1) - y
                        intensity_image[image_y, x] = 0
                        masked_count += 1
                        break
                        
    print(f"  Masked pixels: {masked_count}")
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    Image.fromarray(intensity_image, mode='L').save(output_path)
    print(f"Saved: {output_path}")


def build_rss_map(data: Dict, buildings_list: List[Dict]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Render an RSS map in dBm and return it with the TX pixel in image coordinates."""
    source_location = data['source_location']
    source_x, source_y, source_z = source_location
    print(f"Source: ({source_x:.1f}, {source_y:.1f}, {source_z:.1f})")

    intensity_values = []
    receiver_positions = []
    rx_z_sum = 0.0
    rx_count = 0
    skipped_source_self = 0
    default_tx_dbm = 23.0
    max_px, max_py = 0, 0

    for receiver_info in data['receivers'].values():
        if 'total_intensity_dBm' in receiver_info:
            intensity_dbm = receiver_info['total_intensity_dBm']
        elif 'total_loss_dB' in receiver_info:
            intensity_dbm = default_tx_dbm - receiver_info['total_loss_dB']
        else:
            continue

        location = receiver_info['location']
        if (
            abs(float(location[0]) - float(source_x)) < 1e-6
            and abs(float(location[1]) - float(source_y)) < 1e-6
        ):
            skipped_source_self += 1
            continue
        rx_z_sum += location[2]
        rx_count += 1

        if isinstance(intensity_dbm, (int, float)) and not math.isnan(intensity_dbm) and not math.isinf(intensity_dbm):
            intensity_values.append(float(intensity_dbm))
            px = int(round(location[0]))
            py = int(round(location[1]))
            receiver_positions.append((px, py))
            max_px = max(max_px, px)
            max_py = max(max_py, py)

    if not intensity_values:
        print("Error: no valid intensity values found")
        tx_x = int(round(source_x))
        tx_y = int(round(source_y))
        return np.full((1, 1), np.nan, dtype=np.float32), (tx_x, tx_y)

    avg_rx_z = rx_z_sum / rx_count if rx_count > 0 else 1.5
    print(f"Receivers: {len(intensity_values)}, Avg Height: {avg_rx_z:.2f}m")
    if skipped_source_self:
        print(f"Skipped source-self cells: {skipped_source_self}")

    actual_min_int = min(intensity_values)
    actual_max_int = max(intensity_values)
    print(f"Intensity range: {actual_min_int:.2f} to {actual_max_int:.2f} dBm")

    width, height = max_px + 1, max_py + 1
    print(f"Dynamic Image Size: {width} x {height}")

    rss_map = np.full((height, width), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(receiver_positions):
        if 0 <= x < width and 0 <= y < height:
            intensity_dbm = intensity_values[i]
            image_y = (height - 1) - y
            rss_map[image_y, x] = float(intensity_dbm)

    print("Masking building interiors...")
    masked_count = 0
    blocking_buildings = [b for b in buildings_list if avg_rx_z < b["height"]]
    print(f"  Blocking buildings (H > {avg_rx_z:.1f}m): {len(blocking_buildings)} / {len(buildings_list)}")

    if blocking_buildings:
        for x in range(width):
            for y in range(height):
                for building in blocking_buildings:
                    if point_in_polygon_cpu(float(x), float(y), building["poly"]):
                        image_y = (height - 1) - y
                        rss_map[image_y, x] = np.nan
                        masked_count += 1
                        break

    print(f"  Masked pixels: {masked_count}")
    tx_px = int(round(source_x))
    tx_py = (height - 1) - int(round(source_y))
    return rss_map, (tx_px, tx_py)


def normalize_rss_map(
    rss_map: np.ndarray,
    tx_pixel: Optional[Tuple[int, int]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Normalize an RSS map to uint8 grayscale and force the TX pixel to white."""

    image = np.zeros(rss_map.shape, dtype=np.uint8)
    valid = np.isfinite(rss_map)
    if np.any(valid):
        valid_vals = rss_map[valid].astype(np.float32)
        if vmin is None:
            vmin = float(np.min(valid_vals))
        if vmax is None:
            vmax = float(np.max(valid_vals))
        if vmax <= vmin:
            scaled = np.zeros(valid_vals.shape, dtype=np.float32)
        else:
            clipped = np.clip(valid_vals, vmin, vmax)
            scaled = (clipped - vmin) / (vmax - vmin)
        image[valid] = np.clip(np.round(255.0 * scaled), 0, 255).astype(np.uint8)

    if tx_pixel is not None:
        tx_x, tx_y = tx_pixel
        if 0 <= tx_y < image.shape[0] and 0 <= tx_x < image.shape[1]:
            image[tx_y, tx_x] = 255
    return image


def build_gray_image(
    data: Dict,
    buildings_list: List[Dict],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Render a grayscale RSS image and return it as a uint8 array."""

    rss_map, tx_pixel = build_rss_map(data, buildings_list)
    return normalize_rss_map(rss_map, tx_pixel=tx_pixel, vmin=vmin, vmax=vmax)


def resolve_buildings_json(map_id: str) -> Path:
    """Resolve the buildings json for the given map id in this workspace."""

    candidates = [
        DATA_ROOT / "buildings_complete" / f"{map_id}.json",
        DATA_ROOT / "Result" / "buildings_complete" / f"{map_id}.json",
        DATA_ROOT / "Building_Infomation" / "buildings" / f"{map_id}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="生成灰度强度图像")
    parser.add_argument("data_file", help="数据文件路径 (.propbin, .propbin.gz, .pkl.gz)")
    parser.add_argument("--map-id", default="0", help="地图ID (默认: 0)")
    parser.add_argument("--output", help="输出文件路径 (默认: 自动生成)")
    
    args = parser.parse_args()
    
    # 加载建筑物数据
    # 改为从 DATA_ROOT 读取
    buildings_json_path = resolve_buildings_json(args.map_id)
    print(f"Loading buildings from: {buildings_json_path}")
    buildings_list = load_buildings_with_height(str(buildings_json_path), coord_scale=1.0)
    print(f"Loaded {len(buildings_list)} buildings (Scaled H x{BUILDING_HEIGHT_SCALE})")
    
    # 加载传播数据
    data = detect_and_load_data(args.data_file)
    
    # 确定输出路径
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        data_path = Path(args.data_file)
        stem = data_path.stem.replace('.propbin', '').replace('.pkl', '')
        
        # 尝试保留原有结构 (scenario_X/rx_zy)
        # 查找路径中的关键部分
        parts = data_path.parts
        sub_dir = ""
        
        for i, p in enumerate(parts):
            if p.startswith("scenario_"):
                # 获取 scenario_X 及其之后的部分直到文件名
                # parts[i] = scenario_A or scenario_B
                # parts[i+1] might be rx_z50 or file
                remaining = parts[i:-1] # excluding filename
                sub_dir = os.path.join(*remaining)
                break
        
        if not sub_dir:
            # Fallback path if standard structure not found
            if "rx_z" in str(data_path):
                # Try to extract rx_z part manually if possible or just append unique hash
                pass
                
        output_dir = DATA_ROOT / f"gray_images/map_{args.map_id}" / sub_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{stem}_gray.png")
    
    # 生成图像
    generate_gray_image(data, buildings_list, output_path)


if __name__ == "__main__":
    main()
