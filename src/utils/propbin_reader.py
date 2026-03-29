"""
Python读取器：读取C++生成的 .propbin / .propbin.gz 文件
支持 Version 1 (Full) 和 Version 2 (Compact) 格式。

使用方法:
    from propbin_reader import load_propbin
    
    # 加载单个文件 (自动寻找 feature_points.npy)
    result = load_propbin("source_0.propbin")
    
    # 指定 feature_points
    result = load_propbin("source_0.propbin", feature_points_path="vertex_out/map_0/feature_points_0.npy")
"""

import numpy as np
import struct
import gzip
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import glob
import os

SPEED_OF_LIGHT = 3e8

def load_propbin(path: Union[str, Path], feature_points_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载 .propbin 或 .propbin.gz 文件
    对于 Version 2 (Compact) 格式，需要 feature_points_path 来重建坐标。
    如果未提供，尝试根据文件路径自动推断。
    """
    path = Path(path)
    
    # 打开文件（支持gzip，自动回退）
    if path.suffix == '.gz' or str(path).endswith('.propbin.gz'):
        try:
            with gzip.open(path, 'rb') as f:
                data = f.read()
        except (gzip.BadGzipFile, OSError):
            # 文件名带 .gz 但实际未压缩，回退到普通读取
            with open(path, 'rb') as f:
                data = f.read()
    else:
        with open(path, 'rb') as f:
            data = f.read()
    
    return _parse_propbin(data, path, feature_points_path)


def _parse_propbin(data: bytes, file_path: Path, feature_points_path: Optional[str] = None) -> Dict[str, Any]:
    """解析二进制数据"""
    offset = 0
    
    # Header (32 bytes)
    magic, version = struct.unpack_from('<II', data, offset)
    offset += 8
    
    if magic != 0x50524F50:  # "PROP"
        raise ValueError(f"Invalid magic number: {hex(magic)}")
    
    if version not in [1, 2]:
        raise ValueError(f"Unsupported version: {version}")
        
    source_xyz = struct.unpack_from('<3f', data, offset)
    offset += 12
    
    n_receivers, n_paths_total, n_chain_points = struct.unpack_from('<III', data, offset)
    offset += 12
    
    # ---------------------------------------------------------
    # Version 2 Specific: Load Feature Points if needed
    # ---------------------------------------------------------
    feature_pts = None
    if version == 2:
        if n_chain_points > 0: # Only need features if there are chains
            try:
                feature_pts = _load_feature_points_auto(file_path, feature_points_path)
            except RuntimeError:
                feature_pts = None  # May only have rooftop paths (no vertex IDs needed)
    
    # ---------------------------------------------------------
    # Receiver Table (Same for V1 and V2)
    # ---------------------------------------------------------
    receivers_raw = []
    for _ in range(n_receivers):
        rx_id = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        location = struct.unpack_from('<3f', data, offset)
        offset += 12
        total_intensity_dbm, total_loss_db = struct.unpack_from('<2f', data, offset)
        offset += 8
        n_paths, _pad, paths_offset = struct.unpack_from('<HHI', data, offset)
        offset += 8
        
        receivers_raw.append({
            'rx_id': rx_id,
            'location': location,
            'total_intensity_dBm': total_intensity_dbm,
            'total_loss_dB': total_loss_db,
            'n_paths': n_paths,
            'paths_offset': paths_offset
        })
    
    # ---------------------------------------------------------
    # Path Table & Chain Data
    # ---------------------------------------------------------
    
    if version == 1:
        # V1: Full precision, explicit coord chains
        paths_raw = []
        for _ in range(n_paths_total):
            loss, dist = struct.unpack_from('<2f', data, offset)
            offset += 8
            dep_az, dep_el, arr_az, arr_el = struct.unpack_from('<4f', data, offset)
            offset += 16
            delay_ns = struct.unpack_from('<f', data, offset)[0]
            offset += 4
            chain_len, _pad, chain_offset = struct.unpack_from('<HHI', data, offset)
            offset += 8
            
            paths_raw.append({
                'loss': loss, 'dist': dist,
                'departure_angle': (dep_az, dep_el),
                'arrival_angle': (arr_az, arr_el),
                'delay': delay_ns,
                'chain_len': chain_len, 'chain_offset': chain_offset
            })
            
        # V1 Chain Points (float32 x 3)
        chain_data = np.frombuffer(data, dtype=np.float32, count=n_chain_points * 3, offset=offset)
        chain_data = chain_data.reshape(-1, 3)
        
    elif version == 2:
        # V2: 包含完整信息 (24 bytes per path, #pragma pack(1) 无额外padding)
        PATH_V2_SIZE = 24  # 8*H + b + B + H + I
        paths_raw = []
        for _ in range(n_paths_total):
            # loss(2), dist(2), delay(2), dep_az(2), dep_el(2), arr_az(2), arr_el(2),
            # chain_len(2), path_type(1), pad1(1), pad2(2), chain_offset(4) = 24 bytes
            loss_fp16, dist_fp16, delay_fp16, dep_az_fp16, dep_el_fp16, arr_az_fp16, arr_el_fp16, \
            chain_len, path_type, _pad1, _pad2, chain_offset = struct.unpack_from('<HHHHHHHHbBHI', data, offset)
            offset += PATH_V2_SIZE
            
            # Dequantize all FP16 values
            loss = _fp16_to_float(loss_fp16)
            dist = _fp16_to_float(dist_fp16)
            delay = _fp16_to_float(delay_fp16)
            dep_az = _fp16_to_float(dep_az_fp16)
            dep_el = _fp16_to_float(dep_el_fp16)
            arr_az = _fp16_to_float(arr_az_fp16)
            arr_el = _fp16_to_float(arr_el_fp16)
            
            paths_raw.append({
                'loss': loss,
                'dist': dist,
                'delay': delay,
                'departure_az': dep_az,
                'departure_el': dep_el,
                'arrival_az': arr_az,
                'arrival_el': arr_el,
                'chain_len': chain_len,
                'path_type': path_type,  # -1=刀缘衍射, 0=普通多跳
                'chain_offset': chain_offset
            })
            
        # V2 Chain IDs (uint16)
        chain_ids = np.frombuffer(data, dtype=np.uint16, count=n_chain_points, offset=offset)
        
    # ---------------------------------------------------------
    # Reconstruct Result
    # ---------------------------------------------------------
    result = {
        'version': version,
        'source_location': source_xyz,
        'receivers': {}
    }
    
    source_pt = np.array(source_xyz, dtype=np.float32)
    
    for rx in receivers_raw:
        rx_loc = rx['location']
        rx_pt = np.array(rx_loc, dtype=np.float32)
        key = f"receiver_{int(rx_loc[0])}_{int(rx_loc[1])}"
        
        path_info = []
        for i in range(rx['n_paths']):
            p = paths_raw[rx['paths_offset'] + i]
            
            if version == 1:
                # V1: Direct copy
                chain_start = p['chain_offset']
                chain_end = chain_start + p['chain_len']
                chain = chain_data[chain_start:chain_end].tolist()
                
                if len(chain) >= 2:
                    parent = (chain[-2][0], chain[-2][1])
                    level = len(chain) - 2
                else:
                    parent = (rx_loc[0], rx_loc[1])
                    level = 0
                
                path_info.append({
                    'parent': parent, 'level': level,
                    'path_type': level,
                    'departure_angle': list(p['departure_angle']),
                    'arrival_angle': list(p['arrival_angle']),
                    'distance': p['dist'],
                    'loss': p['loss'],
                    'delay': p['delay'],
                    'path_chain': chain,
                    'chain_vertex_ids': [],
                    'rooftop_points': [],
                })
                
            elif version == 2:
                # V2: 直接使用存储的值，不需要重建
                chain_start = p['chain_offset']
                chain_end = chain_start + p['chain_len']
                c_ids = chain_ids[chain_start:chain_end]
                
                if p.get('path_type', 0) == -1:
                    # Rooftop diffraction: chain stores float16 coords (x1,y1,x2,y2)
                    level = -1
                    parent = (source_xyz[0], source_xyz[1])
                    rooftop_coords = []
                    for ci in range(0, len(c_ids), 2):
                        fx = _fp16_to_float(int(c_ids[ci]))
                        fy = _fp16_to_float(int(c_ids[ci + 1]))
                        rooftop_coords.append([fx, fy, source_xyz[2]])
                    mid_points_list = rooftop_coords
                    chain_vertex_ids = []
                else:
                    # Normal BFS path: chain stores vertex IDs
                    if feature_pts is not None and len(c_ids) > 0:
                        mid_points = feature_pts[c_ids]
                    else:
                        mid_points = np.zeros((0, 3), dtype=np.float32)
                    
                    if len(mid_points) > 0:
                        parent = (float(mid_points[-1][0]), float(mid_points[-1][1]))
                        level = len(mid_points)
                    else:
                        parent = (source_xyz[0], source_xyz[1])
                        level = 0
                    mid_points_list = mid_points.tolist()
                    chain_vertex_ids = [int(v) for v in c_ids.tolist()]
                    
                path_info.append({
                    'parent': parent, 'level': level,
                    'path_type': int(p.get('path_type', level)),
                    'departure_angle': [float(p['departure_az']), float(p['departure_el'])],
                    'arrival_angle': [float(p['arrival_az']), float(p['arrival_el'])],
                    'distance': float(p['dist']),
                    'loss': float(p['loss']),
                    'delay': float(p['delay']),
                    'path_chain': mid_points_list,
                    'chain_vertex_ids': chain_vertex_ids,
                    'rooftop_points': mid_points_list if int(p.get('path_type', 0)) == -1 else [],
                })

        result['receivers'][key] = {
            'rx_id': rx['rx_id'],
            'location': rx_loc,
            'total_intensity_dBm': rx['total_intensity_dBm'],
            'total_loss_dB': rx['total_loss_dB'],
            'path_info': path_info
        }
        
    return result

def _compute_angle(p1, p2):
    dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
    xy = np.sqrt(dx*dx + dy*dy)
    az = np.arctan2(dy, dx)
    el = np.arctan2(dz, xy)
    return az, el

def _load_feature_points_auto(propbin_path: Path, provided_path: Optional[str]) -> np.ndarray:
    if provided_path:
        return np.load(provided_path)
    
    # Auto discovery
    # Attempt to find map_ID from path: .../map_0/...
    parts = propbin_path.parts
    map_id = None
    for p in parts:
        if p.startswith("map_"):
            map_id = p
            break
            
    if map_id:
        map_num = map_id.split('_')[1]
        
        # 优先查找 essential_npy 文件夹（推荐位置）
        curr = propbin_path.parent
        for _ in range(5):
            candidate = curr / "../essential_npy" / f"feature_points_map_{map_num}.npy"
            try:
                candidate = candidate.resolve()
                if candidate.exists():
                    return np.load(candidate)
            except:
                pass
            curr = curr.parent
        
        # 回退到旧位置：vertex_out
        curr = propbin_path.parent
        for _ in range(5):
            candidate = curr / "../vertex_out" / map_id / f"feature_points_{map_num}.npy"
            try:
                candidate = candidate.resolve()
                if candidate.exists():
                    return np.load(candidate)
            except:
                pass
            
            # Try absolute from D:\3D logic?
            # Hardcoded fallback for this user environment
            hardcoded = Path(f"d:/3D/vertex_out/{map_id}/feature_points_{map_id.split('_')[1]}.npy")
            if hardcoded.exists():
                return np.load(hardcoded)
                
            curr = curr.parent
            
    raise RuntimeError(f"Could not automatically find feature_points.npy for V2 format. Please provide feature_points_path argument.")

def _fp16_to_float(h):
    # Python struct 'e' is float16 (IEEE 754 raw) since 3.6
    return struct.unpack('e', struct.pack('H', h))[0]

# 批量加载函数 (保持不变)
def load_propbin_batch(directory: Union[str, Path], pattern: str = "source_*.propbin") -> Dict[int, Dict[str, Any]]:
    # ... implementation mirrors previous ...
    # Simplified here
    directory = Path(directory)
    files = sorted(glob.glob(str(directory / pattern)))
    results = {}
    for fpath in files:
        fname = Path(fpath).stem
        if fname.endswith('.propbin'): fname = fname[:-8]
        try:
            tx_idx = int(fname.split('_')[1])
            results[tx_idx] = load_propbin(fpath)
        except: continue
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python propbin_reader.py <file.propbin> [feature_points.npy]")
        sys.exit(1)
    
    fpts = sys.argv[2] if len(sys.argv) > 2 else None
    result = load_propbin(sys.argv[1], fpts)
    print(f"Source: {result['source_location']}")
    print(f"Receivers: {len(result['receivers'])}")
