"""
proj_geometry.py
几何修正工具：将模型预测的投影点修正到物理合法位置。

坐标系统一使用 MAP 坐标（与 buildings JSON 一致，x 向右，y 向上）。
核心思路：从顶点沿 source→vertex 延伸方向投射射线，找到第一个建筑墙面/边界的交点。
"""

import json
import math
from typing import Optional, Tuple, List, Set, Dict

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None


def load_buildings(json_path: str) -> List:
    """从 JSON 文件加载建筑物多边形列表。"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[proj_geometry] 加载建筑物 JSON: {json_path}，共 {len(data)} 个建筑")
            return data
    except FileNotFoundError:
        print(f"[proj_geometry] 警告：JSON 未找到: {json_path}，edge_set 仅含边界边！")
        return []
    except Exception as e:
        print(f"[proj_geometry] 警告：加载 JSON 出错 {e}，edge_set 仅含边界边！")
        return []


def build_edge_set(buildings: List, map_size: int = 256) -> Set:
    """
    构建边集（frozenset），包含：
    - 所有建筑物多边形的相邻边
    - 地图四边界

    边用 frozenset({(x1,y1), (x2,y2)}) 表示（方向无关）。
    坐标为 MAP 坐标（与 JSON 一致）。
    """
    edge_set = set()
    for poly in buildings:
        n = len(poly)
        if n < 2:
            continue
        # 处理首尾闭合的多边形
        is_closed = list(poly[0]) == list(poly[-1])
        loop_n = n - 1 if is_closed else n
        for i in range(loop_n):
            p1 = (float(poly[i][0]),       float(poly[i][1]))
            p2 = (float(poly[(i+1) % n][0]), float(poly[(i+1) % n][1]))
            if p1 != p2:
                edge_set.add(frozenset([p1, p2]))
    # 地图四边界
    ms = float(map_size)
    for e in [((0., 0.), (ms, 0.)), ((ms, 0.), (ms, ms)),
              ((ms, ms), (0., ms)), ((0., ms), (0., 0.))]:
        edge_set.add(frozenset(e))
    return edge_set


def _ray_segment_intersect(
    ray_origin: Tuple[float, float],
    ray_angle: float,
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float],
    max_dist: float = 1000.0,
    tol: float = 1e-6,
) -> Optional[Tuple[float, float]]:
    """
    计算射线与线段的交点。
    射线从 ray_origin 出发，方向 ray_angle（弧度）。
    返回交点坐标，或 None（不相交/平行）。
    """
    cos_a, sin_a = math.cos(ray_angle), math.sin(ray_angle)
    x1, y1 = ray_origin
    x2, y2 = x1 + cos_a * max_dist, y1 + sin_a * max_dist
    x3, y3 = float(seg_start[0]), float(seg_start[1])
    x4, y4 = float(seg_end[0]),   float(seg_end[1])

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < tol:
        return None  # 平行或共线
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if t > tol and 0.0 <= u <= 1.0:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _get_edges_in_circle(
    center: Tuple[float, float],
    radius: float,
    edge_set: Set,
) -> List:
    """返回圆内或圆相交的边，用于圆圈式扫描。"""
    cx, cy = center
    result = []
    for edge in edge_set:
        pts = list(edge)
        p1, p2 = pts[0], pts[1]
        d1 = math.hypot(p1[0] - cx, p1[1] - cy)
        d2 = math.hypot(p2[0] - cx, p2[1] - cy)
        if d1 <= radius + 1e-6 or d2 <= radius + 1e-6:
            result.append((p1, p2))
        else:
            # 检查线段本身是否与圆相交（点到线段最近距离）
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            len_sq = dx * dx + dy * dy
            if len_sq < 1e-10:
                continue
            t = max(0.0, min(1.0, ((cx - p1[0]) * dx + (cy - p1[1]) * dy) / len_sq))
            near_x, near_y = p1[0] + t * dx, p1[1] + t * dy
            if math.hypot(near_x - cx, near_y - cy) <= radius + 1e-6:
                result.append((p1, p2))
    return result


def _point_in_polygon(x: float, y: float, poly: List) -> bool:
    """射线法判断点是否在多边形内部（MAP 坐标）。"""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_any_building(x: float, y: float, buildings: List) -> bool:
    """判断点是否在任一建筑物内部。"""
    for poly in buildings:
        if _point_in_polygon(x, y, poly):
            return True
    return False


def _is_path_clear(
    start: Tuple[float, float],
    end: Tuple[float, float],
    buildings: List,
    num_samples: int = 30,
) -> bool:
    """
    检查从 start 到 end 的路径是否穿过建筑物内部。
    沿路径均匀采样（跳过首尾端点），任一采样点落入建筑物即判定穿墙。
    """
    for i in range(1, num_samples - 1):
        t = i / (num_samples - 1)
        px = start[0] + t * (end[0] - start[0])
        py = start[1] + t * (end[1] - start[1])
        if _point_in_any_building(px, py, buildings):
            return False
    return True


def _point_to_segment_distance(
    point: Tuple[float, float],
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float],
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(seg_start[0]), float(seg_start[1])
    bx, by = float(seg_end[0]), float(seg_end[1])
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def _point_near_polygon_boundary(
    point: Tuple[float, float],
    poly: List,
    tol: float,
) -> bool:
    if len(poly) < 2:
        return False
    pts = [(float(p[0]), float(p[1])) for p in poly]
    for i in range(len(pts)):
        if _point_to_segment_distance(point, pts[i], pts[(i + 1) % len(pts)]) <= tol:
            return True
    return False


def _is_path_clear_allow_boundary_hugging(
    start: Tuple[float, float],
    end: Tuple[float, float],
    buildings: List,
    num_samples: int = 30,
    boundary_tol: float = 2,
) -> bool:
    """
    比 _is_path_clear 略宽松：
    若采样点只是在建筑边界附近轻微落入 polygon，仍视作贴边合法；
    只有明显进入建筑内部时才判定为不清晰。
    """
    for i in range(1, num_samples - 1):
        t = i / (num_samples - 1)
        px = start[0] + t * (end[0] - start[0])
        py = start[1] + t * (end[1] - start[1])
        inside_polys = [poly for poly in buildings if _point_in_polygon(px, py, poly)]
        if not inside_polys:
            continue
        if not any(_point_near_polygon_boundary((px, py), poly, boundary_tol) for poly in inside_polys):
            return False
    return True


def _point_on_map_boundary(
    point_xy: Tuple[float, float],
    map_size: int = 256,
    tol: float = 1e-6,
) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    ms = float(map_size)
    return (
        abs(x) < tol or abs(x - ms) < tol or
        abs(y) < tol or abs(y - ms) < tol
    )


def _clamp_point_to_map(
    point_xy: Tuple[float, float],
    map_size: int = 256,
) -> Tuple[float, float]:
    ms = float(map_size)
    x = float(point_xy[0])
    y = float(point_xy[1])
    return (min(max(x, 0.0), ms), min(max(y, 0.0), ms))


def _cast_ray_from_vertex(
    vertex_xy: Tuple[float, float],
    ray_angle: float,
    edge_set: Set,
    buildings: Optional[List] = None,
    max_dist: float = 1000.0,
    min_dist: float = 1.0,
    excluded_edge_keys: Optional[Set] = None,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """从顶点沿射线重新传播，返回最近的合法交点及其所在边。"""
    vx, vy = float(vertex_xy[0]), float(vertex_xy[1])
    best_dist = float("inf")
    best_pt = None
    best_seg = None

    for edge in edge_set:
        pts = list(edge)
        seg = (pts[0], pts[1])
        seg_key = frozenset([seg[0], seg[1]])
        if excluded_edge_keys and seg_key in excluded_edge_keys:
            continue
        hit = _ray_segment_intersect((vx, vy), ray_angle, seg[0], seg[1], max_dist)
        if hit is None:
            continue
        dist = math.hypot(hit[0] - vx, hit[1] - vy)
        if dist <= min_dist or dist >= best_dist:
            continue
        if buildings and not _is_path_clear_allow_boundary_hugging((vx, vy), hit, buildings):
            continue
        best_dist = dist
        best_pt = hit
        best_seg = seg

    return best_pt, best_seg


def _build_reprop_excluded_edge_keys(
    current_seg: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    buildings: Optional[List],
    map_size: int,
) -> Set:
    excluded: Set = set()
    if current_seg is None:
        return excluded

    seg = (
        (float(current_seg[0][0]), float(current_seg[0][1])),
        (float(current_seg[1][0]), float(current_seg[1][1])),
    )
    seg_key = frozenset([seg[0], seg[1]])
    excluded.add(seg_key)

    if buildings is None:
        return excluded

    loops, edge_to_loop = _build_loop_catalog(buildings, map_size)
    loop_idx = edge_to_loop.get(seg_key)
    if loop_idx is None:
        return excluded

    loop_pts = loops[loop_idx].get('pts', [])
    loop_n = len(loop_pts)
    if loop_n < 2:
        return excluded

    edge_idx = next(
        (
            i for i in range(loop_n)
            if frozenset([loop_pts[i], loop_pts[(i + 1) % loop_n]]) == seg_key
        ),
        None,
    )
    if edge_idx is None:
        return excluded

    prev_edge = frozenset([
        loop_pts[(edge_idx - 1) % loop_n],
        loop_pts[edge_idx],
    ])
    next_edge = frozenset([
        loop_pts[(edge_idx + 1) % loop_n],
        loop_pts[(edge_idx + 2) % loop_n],
    ])
    excluded.add(prev_edge)
    excluded.add(next_edge)
    return excluded


def _maybe_repropagate_near_vertex(
    source_xy: Tuple[float, float],
    vertex_xy: Tuple[float, float],
    current_hit: Tuple[float, float],
    current_seg: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    edge_set: Set,
    buildings: Optional[List],
    max_dist: float,
    min_dist: float,
    map_size: int,
    near_tol: float = 5.0,
    probe_dist: float = 5.0,
    probe_radius: float = 5.0,
    probe_samples: int = 12,
) -> Tuple[Tuple[float, float], Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    vx, vy = float(vertex_xy[0]), float(vertex_xy[1])
    hx, hy = float(current_hit[0]), float(current_hit[1])

    if math.hypot(hx - vx, hy - vy) > near_tol:
        return current_hit, current_seg

    if not _probe_ahead_has_outside(
        source_xy,
        vertex_xy,
        buildings,
        map_size,
        probe_dist=probe_dist,
        probe_radius=probe_radius,
        probe_samples=probe_samples,
    ):
        return current_hit, current_seg

    sx, sy = float(source_xy[0]), float(source_xy[1])
    ray_angle = math.atan2(vy - sy, vx - sx)
    excluded_edge_keys = _build_reprop_excluded_edge_keys(current_seg, buildings, map_size)
    hit, seg = _cast_ray_from_vertex(
        (vx, vy),
        ray_angle,
        edge_set,
        buildings=buildings,
        max_dist=max_dist,
        min_dist=min_dist,
        excluded_edge_keys=excluded_edge_keys,
    )
    if hit is None:
        return current_hit, current_seg
    if buildings is not None and not _is_path_clear((sx, sy), hit, buildings):
        return current_hit, current_seg
    return _clamp_point_to_map(hit, map_size), seg


def audit_near_vertex_projection_with_edge(
    source_xy: Tuple[float, float],
    vertex_xy: Tuple[float, float],
    proj_xy: Tuple[float, float],
    edge: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    edge_set: Set,
    buildings: Optional[List] = None,
    map_size: int = 256,
    min_dist: float = 1.0,
    near_tol: float = 5.0,
    probe_dist: float = 5.0,
    probe_radius: float = 5.0,
    probe_samples: int = 12,
) -> Tuple[Tuple[float, float], Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    sx, sy = float(source_xy[0]), float(source_xy[1])
    vx, vy = float(vertex_xy[0]), float(vertex_xy[1])
    max_dist = math.hypot(vx - sx, vy - sy) * 4.0 + map_size * 1.5
    return _maybe_repropagate_near_vertex(
        source_xy,
        vertex_xy,
        _clamp_point_to_map((float(proj_xy[0]), float(proj_xy[1])), map_size),
        edge,
        edge_set,
        buildings,
        max_dist,
        min_dist,
        map_size,
        near_tol=near_tol,
        probe_dist=probe_dist,
        probe_radius=probe_radius,
        probe_samples=probe_samples,
    )


def _probe_ahead_has_outside(
    source_xy: Tuple[float, float],
    vertex_xy: Tuple[float, float],
    buildings: Optional[List],
    map_size: int,
    probe_dist: float = 5.0,
    probe_radius: float = 5.0,
    probe_samples: int = 12,
) -> bool:
    if buildings is None:
        return True

    sx, sy = float(source_xy[0]), float(source_xy[1])
    vx, vy = float(vertex_xy[0]), float(vertex_xy[1])
    ray_angle = math.atan2(vy - sy, vx - sx)
    probe_center = _clamp_point_to_map(
        (
            vx + probe_dist * math.cos(ray_angle),
            vy + probe_dist * math.sin(ray_angle),
        ),
        map_size,
    )
    if math.hypot(probe_center[0] - vx, probe_center[1] - vy) < 1e-6:
        return False

    probe_points = [probe_center]
    for i in range(probe_samples):
        ang = (2.0 * math.pi * i) / probe_samples
        probe_points.append(
            _clamp_point_to_map(
                (
                    probe_center[0] + probe_radius * math.cos(ang),
                    probe_center[1] + probe_radius * math.sin(ang),
                ),
                map_size,
            )
        )
    return any(not _point_in_any_building(pt[0], pt[1], buildings) for pt in probe_points)


def _estimate_reprop_seed_seg(
    vertex_xy: Tuple[float, float],
    ray_angle: float,
    pred_proj_xy: Optional[Tuple[float, float]],
    edge_set: Set,
    max_dist: float,
    min_dist: float,
    search_radius: float = 10.0,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if pred_proj_xy is None:
        return None

    vx, vy = float(vertex_xy[0]), float(vertex_xy[1])
    px, py = float(pred_proj_xy[0]), float(pred_proj_xy[1])
    candidate_edges = list(_get_edges_in_circle((px, py), search_radius, edge_set))
    if not candidate_edges:
        candidate_edges = [tuple(edge) for edge in edge_set]

    best_seg = None
    best_score = float("inf")
    for edge in candidate_edges:
        seg = (
            (float(edge[0][0]), float(edge[0][1])),
            (float(edge[1][0]), float(edge[1][1])),
        )
        hit = _ray_segment_intersect((vx, vy), ray_angle, seg[0], seg[1], max_dist)
        if hit is None:
            continue
        ray_dist = math.hypot(hit[0] - vx, hit[1] - vy)
        if ray_dist <= min_dist:
            continue
        score = math.hypot(hit[0] - px, hit[1] - py)
        score += 0.25 * _point_to_segment_distance((px, py), seg[0], seg[1])
        if score < best_score:
            best_score = score
            best_seg = seg
    return best_seg


def correct_proj_point(
    source_xy: Tuple[float, float],
    vertex_xy: Tuple[float, float],
    edge_set: Set,
    pred_proj_xy: Optional[Tuple[float, float]] = None,
    buildings: Optional[List] = None,
    map_size: int = 256,
    min_dist: float = 1.0,
) -> Tuple[float, float]:
    """
    几何修正：给定 source 和可见 vertex，计算物理合法的投影点。

    逻辑与 test_single_unet_local.py _geometric_correction 一致：
    1. 以模型预测点为圆心，逐步扩大半径寻找附近的建筑棱（圆圈式搜索）
    2. 从 vertex 出发沿 source→vertex 方向射线，找到与附近边的第一个合法交点
    3. 若找不到则退化为全图搜索；仍找不到则回退到顶点坐标自身

    Args:
        source_xy:    发射源坐标 (x, y)，MAP 坐标。
        vertex_xy:    可见顶点坐标 (x, y)，MAP 坐标。
        edge_set:     由 build_edge_set() 构建的边集。
        pred_proj_xy: 模型预测的投影点坐标 (x, y)，MAP 坐标（可选，None 时退化为全图搜索）。
        map_size:     地图尺寸（默认 256）。
        min_dist:     最小有效距离（避免与顶点自身重合）。

    Returns:
        (corrected_x, corrected_y)，MAP 坐标。
    """
    sx, sy = float(source_xy[0]), float(source_xy[1])
    vx, vy = float(vertex_xy[0]), float(vertex_xy[1])

    if _point_on_map_boundary((vx, vy), map_size) and (
        buildings is None or _is_path_clear((sx, sy), (vx, vy), buildings)
    ):
        return _clamp_point_to_map((vx, vy), map_size)

    ray_angle = math.atan2(vy - sy, vx - sx)
    max_dist = math.hypot(vx - sx, vy - sy) * 4.0 + map_size * 1.5
    pred_blocked = (
        pred_proj_xy is not None
        and buildings is not None
        and not _is_path_clear((vx, vy), (float(pred_proj_xy[0]), float(pred_proj_xy[1])), buildings)
    )
    force_vertex_reprop = (
        pred_proj_xy is not None
        and math.hypot(float(pred_proj_xy[0]) - vx, float(pred_proj_xy[1]) - vy) <= 5.0
        and _probe_ahead_has_outside(source_xy, vertex_xy, buildings, map_size)
    )
    reprop_seed_seg = (
        _estimate_reprop_seed_seg(
            vertex_xy,
            ray_angle,
            pred_proj_xy,
            edge_set,
            max_dist,
            min_dist,
        )
        if force_vertex_reprop else None
    )

    def _first_hit(edges):
        """在给定 edges 中，从顶点沿射线找第一个合法交点（距离最近且 > min_dist，且不穿墙）。"""
        best_dist = float("inf")
        best_pt = None
        for seg in edges:
            hit = _ray_segment_intersect((vx, vy), ray_angle, seg[0], seg[1], max_dist)
            if hit is None:
                continue
            dist = math.hypot(hit[0] - vx, hit[1] - vy)
            if dist > min_dist and dist < best_dist:
                if buildings and not _is_path_clear_allow_boundary_hugging((vx, vy), hit, buildings):
                    continue
                best_dist = dist
                best_pt = hit
        return best_pt

    # ------------------------------------------------------------------
    # 有模型预测点时：圆圈式扫描找附近边，再做射线吸附（同 test_single_unet_local.py）
    # ------------------------------------------------------------------
    if pred_proj_xy is not None and not pred_blocked and not force_vertex_reprop:
        px, py = float(pred_proj_xy[0]), float(pred_proj_xy[1])

        # 第一阶段：逐步扩圆找到至少一条边
        init_r, step_r = 10.0, 5.0
        radius = init_r
        found_keys: Set = set()   # frozenset key，方向无关去重
        found_edges: List = []

        while radius <= map_size and len(found_edges) == 0:
            for e in _get_edges_in_circle((px, py), radius, edge_set):
                k = frozenset([e[0], e[1]])   # 方向无关 key
                if k not in found_keys:
                    found_keys.add(k)
                    found_edges.append(e)
            if found_edges:
                break
            radius += step_r

        # 第二阶段：继续扩圆，连续两次无新边则停止
        if found_edges:
            no_new = 0
            while radius <= map_size and no_new < 2:
                radius += step_r
                added = 0
                for e in _get_edges_in_circle((px, py), radius, edge_set):
                    k = frozenset([e[0], e[1]])   # 方向无关 key
                    if k not in found_keys:
                        found_keys.add(k)
                        found_edges.append(e)
                        added += 1
                no_new = 0 if added > 0 else no_new + 1

        hit = _first_hit(found_edges) if found_edges else None

        if hit is not None and math.hypot(hit[0] - vx, hit[1] - vy) > min_dist:
            # 检查 source→proj 是否穿墙
            if buildings is None or _is_path_clear((sx, sy), hit, buildings):
                reproj_hit, _ = _maybe_repropagate_near_vertex(
                    source_xy,
                    vertex_xy,
                    _clamp_point_to_map(hit, map_size),
                    None,
                    edge_set,
                    buildings,
                    max_dist,
                    min_dist,
                    map_size,
                )
                return reproj_hit

    # ------------------------------------------------------------------
    # 无预测点 / 圆圈搜索无结果：退化为全图搜索
    # ------------------------------------------------------------------
    excluded_edge_keys = _build_reprop_excluded_edge_keys(reprop_seed_seg, buildings, map_size)
    hit, _ = _cast_ray_from_vertex(
        (vx, vy), ray_angle, edge_set,
        buildings=buildings,
        max_dist=max_dist,
        min_dist=min_dist,
        excluded_edge_keys=excluded_edge_keys if force_vertex_reprop else None,
    )
    if hit is not None:
        # 检查 source→proj 是否穿墙
        if buildings is None or _is_path_clear((sx, sy), hit, buildings):
            reproj_hit, _ = _maybe_repropagate_near_vertex(
                source_xy,
                vertex_xy,
                _clamp_point_to_map(hit, map_size),
                None,
                edge_set,
                buildings,
                max_dist,
                min_dist,
                map_size,
            )
            return reproj_hit
    reproj_hit, _ = _maybe_repropagate_near_vertex(
        source_xy,
        vertex_xy,
        _clamp_point_to_map((vx, vy), map_size),
        None,
        edge_set,
        buildings,
        max_dist,
        min_dist,
        map_size,
    )
    return reproj_hit


def correct_proj_point_with_edge(
    source_xy: Tuple[float, float],
    vertex_xy: Tuple[float, float],
    edge_set: Set,
    pred_proj_xy: Optional[Tuple[float, float]] = None,
    buildings: Optional[List] = None,
    map_size: int = 256,
    min_dist: float = 1.0,
) -> Tuple[Tuple[float, float], Optional[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    """
    与 correct_proj_point 逻辑相同，但额外返回投影点所在的边。

    Returns:
        ((cx, cy), edge_or_None)
        edge 为 ((ex1,ey1),(ex2,ey2))，fallback 到顶点自身时为 None。
    """
    sx, sy = float(source_xy[0]), float(source_xy[1])
    vx, vy = float(vertex_xy[0]), float(vertex_xy[1])

    if _point_on_map_boundary((vx, vy), map_size) and (
        buildings is None or _is_path_clear((sx, sy), (vx, vy), buildings)
    ):
        return _clamp_point_to_map((vx, vy), map_size), None

    ray_angle = math.atan2(vy - sy, vx - sx)
    max_dist = math.hypot(vx - sx, vy - sy) * 4.0 + map_size * 1.5
    pred_blocked = (
        pred_proj_xy is not None
        and buildings is not None
        and not _is_path_clear((vx, vy), (float(pred_proj_xy[0]), float(pred_proj_xy[1])), buildings)
    )
    force_vertex_reprop = (
        pred_proj_xy is not None
        and math.hypot(float(pred_proj_xy[0]) - vx, float(pred_proj_xy[1]) - vy) <= 5.0
        and _probe_ahead_has_outside(source_xy, vertex_xy, buildings, map_size)
    )
    reprop_seed_seg = (
        _estimate_reprop_seed_seg(
            vertex_xy,
            ray_angle,
            pred_proj_xy,
            edge_set,
            max_dist,
            min_dist,
        )
        if force_vertex_reprop else None
    )

    def _first_hit_with_edge(edges):
        best_dist = float("inf")
        best_pt = None
        best_seg = None
        for seg in edges:
            hit = _ray_segment_intersect((vx, vy), ray_angle, seg[0], seg[1], max_dist)
            if hit is None:
                continue
            dist = math.hypot(hit[0] - vx, hit[1] - vy)
            if dist > min_dist and dist < best_dist:
                if buildings and not _is_path_clear_allow_boundary_hugging((vx, vy), hit, buildings):
                    continue
                best_dist = dist
                best_pt = hit
                best_seg = seg
        return best_pt, best_seg

    # 圆圈式扫描
    if pred_proj_xy is not None and not pred_blocked and not force_vertex_reprop:
        px, py = float(pred_proj_xy[0]), float(pred_proj_xy[1])
        init_r, step_r = 10.0, 5.0
        radius = init_r
        found_keys: Set = set()
        found_edges: List = []

        while radius <= map_size and len(found_edges) == 0:
            for e in _get_edges_in_circle((px, py), radius, edge_set):
                k = frozenset([e[0], e[1]])
                if k not in found_keys:
                    found_keys.add(k)
                    found_edges.append(e)
            if found_edges:
                break
            radius += step_r

        if found_edges:
            no_new = 0
            while radius <= map_size and no_new < 2:
                radius += step_r
                added = 0
                for e in _get_edges_in_circle((px, py), radius, edge_set):
                    k = frozenset([e[0], e[1]])
                    if k not in found_keys:
                        found_keys.add(k)
                        found_edges.append(e)
                        added += 1
                no_new = 0 if added > 0 else no_new + 1

        if found_edges:
            hit, seg = _first_hit_with_edge(found_edges)
            if hit is not None and math.hypot(hit[0] - vx, hit[1] - vy) > min_dist:
                # 检查 source→proj 是否穿墙
                if buildings is None or _is_path_clear((sx, sy), hit, buildings):
                    return _maybe_repropagate_near_vertex(
                        source_xy,
                        vertex_xy,
                        _clamp_point_to_map(hit, map_size),
                        seg,
                        edge_set,
                        buildings,
                        max_dist,
                        min_dist,
                        map_size,
                    )

    # 全图搜索
    excluded_edge_keys = _build_reprop_excluded_edge_keys(reprop_seed_seg, buildings, map_size)
    hit, seg = _cast_ray_from_vertex(
        (vx, vy), ray_angle, edge_set,
        buildings=buildings,
        max_dist=max_dist,
        min_dist=min_dist,
        excluded_edge_keys=excluded_edge_keys if force_vertex_reprop else None,
    )
    if hit is not None:
        # 检查 source→proj 是否穿墙
        if buildings is None or _is_path_clear((sx, sy), hit, buildings):
            return _maybe_repropagate_near_vertex(
                source_xy,
                vertex_xy,
                _clamp_point_to_map(hit, map_size),
                seg,
                edge_set,
                buildings,
                max_dist,
                min_dist,
                map_size,
            )
    return _maybe_repropagate_near_vertex(
        source_xy,
        vertex_xy,
        _clamp_point_to_map((vx, vy), map_size),
        None,
        edge_set,
        buildings,
        max_dist,
        min_dist,
        map_size,
    )


# =========================================================================
# 可视多边形构建
# =========================================================================

'''
def build_edge_adjacency(buildings: List, map_size: int = 256) -> Dict:
    """
    构建顶点邻接图，用于可视多边形的边路径追踪。

    返回 dict: (x,y) → set of (nx,ny)
    包含：
    - 每个建筑物多边形中相邻顶点的连接
    - 4个地图边界角点的环形连接
    """
    adj: Dict = defaultdict(set)

    for poly in buildings:
        n = len(poly)
        if n < 2:
            continue
        is_closed = list(poly[0]) == list(poly[-1])
        loop_n = n - 1 if is_closed else n
        pts = [(float(poly[i][0]), float(poly[i][1])) for i in range(loop_n)]
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            if p1 != p2:
                adj[p1].add(p2)
                adj[p2].add(p1)

    # 地图四边界角点环形连接
    ms = float(map_size)
    corners = [(0., 0.), (ms, 0.), (ms, ms), (0., ms)]
    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i + 1) % 4]
        adj[c1].add(c2)
        adj[c2].add(c1)

    return dict(adj)
'''


def _angle_from(source: Tuple[float, float], pt: Tuple[float, float]) -> float:
    """计算 pt 相对 source 的角度（弧度，-π ~ π）。"""
    return math.atan2(pt[1] - source[1], pt[0] - source[0])


def _seg_seg_cross(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
    tol: float = 1e-6,
) -> bool:
    """
    判断线段 AB 和线段 CD 是否真正相交（开区间，不含共享端点）。
    用于射线遮挡检测。
    """
    denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(denom) < tol:
        return False
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
    u = -((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / denom
    return tol < t < 1.0 - tol and tol < u < 1.0 - tol


def _corner_visible(
    source_xy: Tuple[float, float],
    corner: Tuple[float, float],
    edge_set: Set,
    endpoint_tol: float = 0.5,
) -> bool:
    """
    判断从 source 到地图角点 corner 的连线是否被任意建筑边真正遮挡。
    使用线段真正相交（开区间）检测，天然排除共享端点的边界边。

    endpoint_tol: 跳过端点接近 source 或 corner 的边（防止 vertex_points
                  与 buildings JSON 坐标精度不一致导致误判遮挡）。
    """
    sx, sy = float(source_xy[0]), float(source_xy[1])
    cx, cy = float(corner[0]), float(corner[1])
    for edge in edge_set:
        pts = list(edge)
        p1x, p1y = float(pts[0][0]), float(pts[0][1])
        p2x, p2y = float(pts[1][0]), float(pts[1][1])
        # 跳过端点接近 source 或 target 的边——
        # 这些边与 source/target 共享端点，不应构成遮挡
        if (math.hypot(p1x - sx, p1y - sy) < endpoint_tol or
            math.hypot(p2x - sx, p2y - sy) < endpoint_tol or
            math.hypot(p1x - cx, p1y - cy) < endpoint_tol or
            math.hypot(p2x - cx, p2y - cy) < endpoint_tol):
            continue
        if _seg_seg_cross(sx, sy, cx, cy, p1x, p1y, p2x, p2y):
            return False
    return True


def is_visible(
    source_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    edge_set: Set,
) -> bool:
    """
    判断从 source 到 target 的连线是否未被任何边遮挡。
    使用线段真正相交（开区间）检测，天然排除共享端点的边。
    """
    return _corner_visible(source_xy, target_xy, edge_set)


'''
def _are_adjacent_on_building(
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    buildings: List,
) -> bool:
    """判断两个顶点是否是同一建筑物上的相邻顶点。"""
    for poly in buildings:
        n = len(poly)
        if n < 2:
            continue
        is_closed = list(poly[0]) == list(poly[-1])
        loop_n = n - 1 if is_closed else n
        pts = [(float(poly[i][0]), float(poly[i][1])) for i in range(loop_n)]

        idx1, idx2 = None, None
        for i, p in enumerate(pts):
            if abs(p[0] - v1[0]) < 1e-6 and abs(p[1] - v1[1]) < 1e-6:
                idx1 = i
            if abs(p[0] - v2[0]) < 1e-6 and abs(p[1] - v2[1]) < 1e-6:
                idx2 = i

        if idx1 is not None and idx2 is not None:
            m = len(pts)
            if (idx1 + 1) % m == idx2 or (idx2 + 1) % m == idx1:
                return True
    return False


def _on_same_building(
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    buildings: List,
) -> bool:
    """判断两个顶点是否属于同一建筑物（不要求相邻）。"""
    for poly in buildings:
        n = len(poly)
        if n < 2:
            continue
        is_closed = list(poly[0]) == list(poly[-1])
        loop_n = n - 1 if is_closed else n
        pts = [(float(poly[i][0]), float(poly[i][1])) for i in range(loop_n)]
        found1 = any(abs(p[0] - v1[0]) < 1e-6 and abs(p[1] - v1[1]) < 1e-6 for p in pts)
        found2 = any(abs(p[0] - v2[0]) < 1e-6 and abs(p[1] - v2[1]) < 1e-6 for p in pts)
        if found1 and found2:
            return True
    return False


def _edge_on_same_building(
    vertex: Tuple[float, float],
    edge: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    buildings: List,
) -> bool:
    """判断命中边是否属于包含给定顶点的同一栋建筑。"""
    if edge is None:
        return False
    e1 = (float(edge[0][0]), float(edge[0][1]))
    e2 = (float(edge[1][0]), float(edge[1][1]))
    for poly in buildings:
        n = len(poly)
        if n < 2:
            continue
        is_closed = list(poly[0]) == list(poly[-1])
        loop_n = n - 1 if is_closed else n
        pts = [(float(poly[i][0]), float(poly[i][1])) for i in range(loop_n)]
        has_vertex = any(abs(p[0] - vertex[0]) < 1e-6 and abs(p[1] - vertex[1]) < 1e-6 for p in pts)
        if not has_vertex:
            continue
        has_e1 = any(abs(p[0] - e1[0]) < 1e-6 and abs(p[1] - e1[1]) < 1e-6 for p in pts)
        has_e2 = any(abs(p[0] - e2[0]) < 1e-6 and abs(p[1] - e2[1]) < 1e-6 for p in pts)
        if has_e1 and has_e2:
            return True
    return False


def _is_boundary_edge(edge, map_size: int = 256) -> bool:
    """
    判断边是否是地图四条边界边之一。
    条件：两个端点在同一条边界线上（x=0、x=ms、y=0 或 y=ms）。
    """
    if edge is None:
        return False
    ms = float(map_size)
    (x1, y1), (x2, y2) = (float(edge[0][0]), float(edge[0][1])), (float(edge[1][0]), float(edge[1][1]))
    tol = 1e-6
    return (
        (abs(x1) < tol       and abs(x2) < tol)       or  # 左边界
        (abs(x1 - ms) < tol  and abs(x2 - ms) < tol)  or  # 右边界
        (abs(y1) < tol       and abs(y2) < tol)       or  # 下边界
        (abs(y1 - ms) < tol  and abs(y2 - ms) < tol)       # 上边界
    )


def _boundary_edge_side(
    edge: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
    map_size: int = 256,
) -> Optional[str]:
    """返回边界边所属的具体边界：left/right/bottom/top。"""
    if edge is None:
        return None
    ms = float(map_size)
    (x1, y1), (x2, y2) = (float(edge[0][0]), float(edge[0][1])), (float(edge[1][0]), float(edge[1][1]))
    tol = 1e-6
    if abs(x1) < tol and abs(x2) < tol:
        return 'left'
    if abs(x1 - ms) < tol and abs(x2 - ms) < tol:
        return 'right'
    if abs(y1) < tol and abs(y2) < tol:
        return 'bottom'
    if abs(y1 - ms) < tol and abs(y2 - ms) < tol:
        return 'top'
    return None
'''


def _same_point(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    tol: float = 1e-6,
) -> bool:
    return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol


def _dedupe_consecutive_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    deduped: List[Tuple[float, float]] = []
    for pt in points:
        pt_xy = (float(pt[0]), float(pt[1]))
        if not deduped or not _same_point(deduped[-1], pt_xy):
            deduped.append(pt_xy)
    return deduped


def _path_length(points: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    return total


def _normalize_angle(angle: float) -> float:
    two_pi = 2.0 * math.pi
    angle = math.fmod(angle, two_pi)
    if angle < 0.0:
        angle += two_pi
    return angle


def _angle_in_interval(angle: float, start: float, end: float, tol: float = 1e-6) -> bool:
    a = _normalize_angle(angle)
    s = _normalize_angle(start)
    e = _normalize_angle(end)
    if e < s:
        e += 2.0 * math.pi
    if a < s:
        a += 2.0 * math.pi
    return s - tol <= a <= e + tol


def _point_on_segment(
    pt: Tuple[float, float],
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float],
    tol: float = 1e-6,
) -> bool:
    ax, ay = float(seg_start[0]), float(seg_start[1])
    bx, by = float(seg_end[0]), float(seg_end[1])
    px, py = float(pt[0]), float(pt[1])
    dx, dy = bx - ax, by - ay
    seg_len = math.hypot(dx, dy)
    if seg_len < tol:
        return math.hypot(px - ax, py - ay) < tol
    cross = abs((px - ax) * dy - (py - ay) * dx)
    if cross > tol * max(1.0, seg_len):
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= tol


def _build_loop_catalog(buildings: List, map_size: int = 256) -> Tuple[List[Dict], Dict]:
    loops: List[Dict] = []
    edge_to_loop: Dict = {}

    def _register_loop(kind: str, loop_id, pts: List[Tuple[float, float]]):
        if len(pts) < 2:
            return
        edge_lengths: List[float] = []
        cum = [0.0]
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            edge_lengths.append(length)
            cum.append(cum[-1] + length)
        loop_idx = len(loops)
        loops.append({
            'kind': kind,
            'id': loop_id,
            'pts': pts,
            'edge_lengths': edge_lengths,
            'cum': cum,
            'total': cum[-1],
        })
        for i in range(len(pts)):
            edge_to_loop[frozenset([pts[i], pts[(i + 1) % len(pts)]])] = loop_idx

    for building_idx, poly in enumerate(buildings):
        n = len(poly)
        if n < 2:
            continue
        is_closed = list(poly[0]) == list(poly[-1])
        loop_n = n - 1 if is_closed else n
        pts = [(float(poly[i][0]), float(poly[i][1])) for i in range(loop_n)]
        _register_loop('building', building_idx, pts)

    ms = float(map_size)
    boundary_pts = [(0.0, 0.0), (ms, 0.0), (ms, ms), (0.0, ms)]
    _register_loop('boundary', 'boundary', boundary_pts)
    return loops, edge_to_loop


def get_building_neighbor_vertices(
    source_xy: Tuple[float, float],
    buildings: List,
) -> List[Tuple[float, float]]:
    neighbors: List[Tuple[float, float]] = []

    for poly in buildings:
        n = len(poly)
        if n < 2:
            continue
        is_closed = list(poly[0]) == list(poly[-1])
        loop_n = n - 1 if is_closed else n
        pts = [(float(poly[i][0]), float(poly[i][1])) for i in range(loop_n)]
        m = len(pts)
        if m < 2:
            continue

        for i, pt in enumerate(pts):
            if not _same_point(pt, source_xy):
                continue
            neighbors.extend([pts[(i - 1) % m], pts[(i + 1) % m]])
            break

    deduped: List[Tuple[float, float]] = []
    for pt in neighbors:
        if not any(_same_point(pt, existing) for existing in deduped):
            deduped.append(pt)
    return deduped


def _locate_point_on_loop(loop: Dict, point: Tuple[float, float]) -> List[float]:
    pts = loop['pts']
    cum = loop['cum']
    edge_lengths = loop['edge_lengths']
    total = loop['total']
    matches: List[float] = []

    for i in range(len(pts)):
        start = pts[i]
        end = pts[(i + 1) % len(pts)]
        seg_len = edge_lengths[i]
        if seg_len < 1e-10:
            continue
        if not _point_on_segment(point, start, end):
            continue
        if _same_point(point, start):
            pos = cum[i]
        elif _same_point(point, end):
            pos = cum[i] + seg_len
        else:
            ratio = math.hypot(point[0] - start[0], point[1] - start[1]) / seg_len
            pos = cum[i] + ratio * seg_len
        if total > 0.0 and pos >= total - 1e-6:
            pos = 0.0
        if all(abs(pos - existing) > 1e-6 for existing in matches):
            matches.append(pos)

    matches.sort()
    return matches


def _build_forward_loop_path(
    loop: Dict,
    start_s: float,
    end_s: float,
    start_pt: Tuple[float, float],
    end_pt: Tuple[float, float],
) -> List[Tuple[float, float]]:
    pts = loop['pts']
    cum = loop['cum']
    total = loop['total']
    if total < 1e-10:
        return [start_pt, end_pt]

    s0 = start_s
    s1 = end_s
    if s1 < s0:
        s1 += total

    path = [start_pt]
    for i, vertex in enumerate(pts):
        pos = cum[i]
        if pos <= s0 + 1e-6:
            pos += total
        if s0 + 1e-6 < pos < s1 - 1e-6 and not _same_point(path[-1], vertex):
            path.append(vertex)
    if not _same_point(path[-1], end_pt):
        path.append(end_pt)
    return _dedupe_consecutive_points(path)


def _build_loop_path_between(
    source_xy: Tuple[float, float],
    loop: Optional[Dict],
    start_pt: Tuple[float, float],
    end_pt: Tuple[float, float],
    start_angle: float,
    end_angle: float,
) -> List[Tuple[float, float]]:
    if loop is None or _same_point(start_pt, end_pt):
        return _dedupe_consecutive_points([start_pt, end_pt])

    start_positions = _locate_point_on_loop(loop, start_pt)
    end_positions = _locate_point_on_loop(loop, end_pt)
    if not start_positions or not end_positions:
        return _dedupe_consecutive_points([start_pt, end_pt])

    best_path = None
    best_score = None
    for start_s in start_positions:
        for end_s in end_positions:
            candidates = [
                _build_forward_loop_path(loop, start_s, end_s, start_pt, end_pt),
                list(reversed(_build_forward_loop_path(loop, end_s, start_s, end_pt, start_pt))),
            ]
            for candidate in candidates:
                candidate = _dedupe_consecutive_points(candidate)
                mid_pts = candidate[1:-1]
                violations = sum(
                    0 if _angle_in_interval(_angle_from(source_xy, pt), start_angle, end_angle) else 1
                    for pt in mid_pts
                )
                score = (violations, _path_length(candidate))
                if best_score is None or score < best_score:
                    best_score = score
                    best_path = candidate

    return best_path if best_path is not None else _dedupe_consecutive_points([start_pt, end_pt])


def _point_on_building_loop(
    point: Tuple[float, float],
    loop_idx: Optional[int],
    loops: List[Dict],
) -> bool:
    if loop_idx is None or loop_idx < 0 or loop_idx >= len(loops):
        return False
    loop = loops[loop_idx]
    if loop.get('kind') != 'building':
        return False
    return len(_locate_point_on_loop(loop, point)) > 0


def _find_source_neighbor_building_loop(
    source_xy: Tuple[float, float],
    curr_v: Tuple[float, float],
    next_v: Tuple[float, float],
    loops: List[Dict],
) -> Optional[Dict]:
    for loop in loops:
        if loop.get('kind') != 'building':
            continue
        pts = loop.get('pts', [])
        src_idx = None
        for i, pt in enumerate(pts):
            if _same_point(pt, source_xy):
                src_idx = i
                break
        if src_idx is None:
            continue
        m = len(pts)
        if m < 3:
            continue
        prev_v = pts[(src_idx - 1) % m]
        next_n = pts[(src_idx + 1) % m]
        if (
            (_same_point(curr_v, prev_v) and _same_point(next_v, next_n)) or
            (_same_point(curr_v, next_n) and _same_point(next_v, prev_v))
        ):
            return loop
    return None


def _interval_mid_angle(start_angle: float, end_angle: float) -> float:
    s = _normalize_angle(start_angle)
    e = _normalize_angle(end_angle)
    if e < s:
        e += 2.0 * math.pi
    return _normalize_angle(s + 0.5 * (e - s))


def _sector_between_neighbors_points_into_building(
    source_xy: Tuple[float, float],
    curr_angle: float,
    next_angle: float,
    building_loop: Dict,
) -> bool:
    pts = building_loop.get('pts', [])
    src_idx = None
    for i, pt in enumerate(pts):
        if _same_point(pt, source_xy):
            src_idx = i
            break
    if src_idx is None or len(pts) < 3:
        return False

    prev_v = pts[(src_idx - 1) % len(pts)]
    next_v = pts[(src_idx + 1) % len(pts)]
    eps = 2.0
    mid_angle = _interval_mid_angle(curr_angle, next_angle)
    test_pt = (
        float(source_xy[0]) + eps * math.cos(mid_angle),
        float(source_xy[1]) + eps * math.sin(mid_angle),
    )
    return _point_in_polygon(test_pt[0], test_pt[1], pts)


def _sector_cell_hits_building(
    cell_pts: List[Tuple[float, float]],
    buildings: List,
) -> bool:
    if len(cell_pts) < 3 or not buildings:
        return False

    source_xy = cell_pts[0]
    for i in range(1, len(cell_pts) - 1):
        p1 = cell_pts[i]
        p2 = cell_pts[i + 1]
        samples = [
            (
                (source_xy[0] + p1[0] + p2[0]) / 3.0,
                (source_xy[1] + p1[1] + p2[1]) / 3.0,
            ),
            (
                (2.0 * source_xy[0] + p1[0] + p2[0]) / 4.0,
                (2.0 * source_xy[1] + p1[1] + p2[1]) / 4.0,
            ),
        ]
        for sx, sy in samples:
            if _point_in_any_building(sx, sy, buildings):
                return True
    return False


def _pair_vertex_on_own_building(
    pair: Dict,
    loops: List[Dict],
) -> bool:
    return _point_on_building_loop(pair['v'], pair.get('loop_idx'), loops)


def _vertices_share_building_edge(
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    loops: List[Dict],
) -> bool:
    for loop in loops:
        if loop.get('kind') != 'building':
            continue
        pts = loop.get('pts', [])
        idx1 = idx2 = None
        for i, pt in enumerate(pts):
            if _same_point(pt, v1):
                idx1 = i
            if _same_point(pt, v2):
                idx2 = i
        if idx1 is None or idx2 is None:
            continue
        m = len(pts)
        if m > 1 and ((idx1 + 1) % m == idx2 or (idx2 + 1) % m == idx1):
            return True
    return False


def _build_cross_pair_path(
    curr_pair: Dict,
    next_pair: Dict,
    loops: List[Dict],
) -> Optional[List[Tuple[float, float]]]:
    curr_p = curr_pair['p']
    curr_v = curr_pair['v']
    next_p = next_pair['p']
    next_v = next_pair['v']

    if _vertices_share_building_edge(curr_v, next_v, loops):
        return _dedupe_consecutive_points([curr_p, curr_v, next_v, next_p])

    if _point_on_building_loop(curr_v, next_pair.get('loop_idx'), loops):
        return _dedupe_consecutive_points([curr_p, curr_v, next_p])

    if _point_on_building_loop(next_v, curr_pair.get('loop_idx'), loops):
        return _dedupe_consecutive_points([curr_p, next_v, next_p])

    return None


def _collect_visibility_events(
    source_xy: Tuple[float, float],
    proj_data: List[Tuple],
    buildings: List,
    edge_set: Optional[Set] = None,
    map_size: int = 256,
) -> Tuple[List[Dict], List[Dict]]:
    if len(proj_data) < 2:
        return [], []

    seen_verts = set()
    unique_data = []
    for entry in proj_data:
        vk = (round(float(entry[0][0]), 6), round(float(entry[0][1]), 6))
        if vk not in seen_verts:
            seen_verts.add(vk)
            unique_data.append(entry)

    loops, edge_to_loop = _build_loop_catalog(buildings, map_size)
    boundary_loop_idx = next((i for i, loop in enumerate(loops) if loop['kind'] == 'boundary'), None)
    pair_entries = []

    for idx, (v, p, e) in enumerate(unique_data):
        vx, vy = float(v[0]), float(v[1])
        px, py = float(p[0]), float(p[1])
        if edge_set is not None and not is_visible(source_xy, (vx, vy), edge_set):
            continue

        edge = None
        loop_idx = None
        if e is not None:
            edge = (
                (float(e[0][0]), float(e[0][1])),
                (float(e[1][0]), float(e[1][1])),
            )
            loop_idx = edge_to_loop.get(frozenset(edge))

        pair_entries.append({
            'kind': 'pair',
            'idx': idx,
            'v': (vx, vy),
            'p': (px, py),
            'e': edge,
            'loop_idx': loop_idx,
            'angle': _angle_from(source_xy, (vx, vy)),
        })

    if len(pair_entries) < 2:
        return [], loops

    pair_entries.sort(key=lambda item: (item['angle'], item['idx']))
    events: List[Dict] = []
    for pair in pair_entries:
        events.append({
            'kind': 'pair',
            'angle': pair['angle'],
            'coord': pair['p'],
            'loop_idx': pair['loop_idx'],
            'pair': pair,
        })

    if edge_set is not None and boundary_loop_idx is not None:
        ms = float(map_size)
        for corner in [(0.0, 0.0), (ms, 0.0), (ms, ms), (0.0, ms)]:
            if _corner_visible(source_xy, corner, edge_set):
                events.append({
                    'kind': 'corner',
                    'angle': _angle_from(source_xy, corner),
                    'coord': corner,
                    'loop_idx': boundary_loop_idx,
                })

    kind_priority = {'pair': 0, 'corner': 1}
    events.sort(key=lambda item: (item['angle'], kind_priority.get(item['kind'], 99)))
    return events, loops


def _build_sector_cell(
    source_xy: Tuple[float, float],
    curr_event: Dict,
    next_event: Dict,
    loops: List[Dict],
    buildings: List,
) -> List[Tuple[float, float]]:
    cell_pts: List[Tuple[float, float]] = [source_xy]
    curr_pair = curr_event.get('pair') if curr_event['kind'] == 'pair' else None
    next_pair = next_event.get('pair') if next_event['kind'] == 'pair' else None

    if curr_pair is not None and next_pair is not None:
        building_loop = _find_source_neighbor_building_loop(
            source_xy,
            curr_pair['v'],
            next_pair['v'],
            loops,
        )
        if (
            building_loop is not None and
            _sector_between_neighbors_points_into_building(
                source_xy,
                curr_event['angle'],
                next_event['angle'],
                building_loop,
            )
        ):
            return []

    if curr_event['kind'] == 'pair':
        cell_pts.extend([curr_pair['v'], curr_pair['p']])
    else:
        cell_pts.append(curr_event['coord'])

    boundary_path = None
    if curr_pair is not None and next_pair is not None:
        if _vertices_share_building_edge(curr_pair['v'], next_pair['v'], loops):
            boundary_path = _dedupe_consecutive_points(
                [curr_pair['p'], curr_pair['v'], next_pair['v'], next_pair['p']]
            )
        curr_pair_same_building = _pair_vertex_on_own_building(curr_pair, loops)
        next_pair_same_building = _pair_vertex_on_own_building(next_pair, loops)

        # 当前 pair 的 V/P 不在同一栋楼上，而下一个 pair 在同一栋楼上时，
        # 保持普通连接：V0 -> P0 -> P1 -> V1。
        if boundary_path is None and (not curr_pair_same_building) and next_pair_same_building:
            next_loop_idx = next_pair.get('loop_idx')
            if (
                next_loop_idx is not None
                and 0 <= next_loop_idx < len(loops)
                and _point_on_building_loop(curr_pair['v'], next_loop_idx, loops)
            ):
                loop_path = _build_loop_path_between(
                    source_xy,
                    loops[next_loop_idx],
                    curr_pair['v'],
                    next_pair['p'],
                    curr_event['angle'],
                    next_event['angle'],
                )
                boundary_path = _dedupe_consecutive_points([curr_pair['p']] + loop_path[1:])
            else:
                boundary_path = _dedupe_consecutive_points([curr_pair['p'], next_pair['p']])
        elif boundary_path is None:
            boundary_path = _build_cross_pair_path(curr_pair, next_pair, loops)

    if boundary_path is None:
        shared_loop = None
        if (
            curr_event.get('loop_idx') is not None
            and curr_event.get('loop_idx') == next_event.get('loop_idx')
            and curr_event.get('loop_idx') < len(loops)
        ):
            shared_loop = loops[curr_event['loop_idx']]

        boundary_path = _build_loop_path_between(
            source_xy,
            shared_loop,
            curr_event['coord'],
            next_event['coord'],
            curr_event['angle'],
            next_event['angle'],
        )

    same_building_proj_loop = (
        curr_pair is not None
        and next_pair is not None
        and curr_pair.get('loop_idx') is not None
        and curr_pair.get('loop_idx') == next_pair.get('loop_idx')
        and 0 <= curr_pair['loop_idx'] < len(loops)
        and loops[curr_pair['loop_idx']].get('kind') == 'building'
    )

    if (
        curr_pair is not None
        and next_pair is not None
        and boundary_path is not None
        and len(boundary_path) == 2
        and _same_point(boundary_path[0], curr_pair['p'])
        and _same_point(boundary_path[1], next_pair['p'])
        and not same_building_proj_loop
        and not _is_path_clear_allow_boundary_hugging(curr_pair['p'], next_pair['p'], buildings)
    ):
        fallback_candidates = [
            [curr_pair['p'], next_pair['v'], next_pair['p']],
            [curr_pair['p'], curr_pair['v'], next_pair['p']],
            [curr_pair['p'], curr_pair['v'], next_pair['v'], next_pair['p']],
        ]
        chosen_candidate = None
        for raw_candidate in fallback_candidates:
            candidate = _dedupe_consecutive_points(raw_candidate)
            valid = True
            for i in range(len(candidate) - 1):
                if not _is_path_clear_allow_boundary_hugging(candidate[i], candidate[i + 1], buildings):
                    valid = False
                    break
            if valid:
                chosen_candidate = candidate
                break
        if chosen_candidate is None:
            return []
        boundary_path = chosen_candidate

    if boundary_path:
        cell_pts.extend(boundary_path[1:])

    if next_pair is not None:
        next_v = next_pair['v']
        boundary_has_next_v = boundary_path is not None and any(_same_point(pt, next_v) for pt in boundary_path)
        if not boundary_has_next_v and not _same_point(cell_pts[-1], next_v):
            cell_pts.append(next_v)

    cell_pts = _dedupe_consecutive_points(cell_pts)
    if _sector_cell_hits_building(cell_pts, buildings):
        return []
    return cell_pts


def build_visibility_mask(
    source_xy: Tuple[float, float],
    proj_data: List[Tuple],
    adjacency: Dict,
    buildings: List,
    edge_set: Optional[Set] = None,
    map_size: int = 256,
) -> np.ndarray:
    """按相邻事件生成局部 sector cell，并对所有 cell 做并集。"""
    del adjacency

    mask = np.zeros((map_size + 1, map_size + 1), dtype=np.uint8)
    pil_image = None
    pil_draw = None
    if cv2 is None:
        pil_image = Image.new('L', (map_size + 1, map_size + 1), 0)
        pil_draw = ImageDraw.Draw(pil_image)

    events, loops = _collect_visibility_events(
        source_xy,
        proj_data,
        buildings,
        edge_set=edge_set,
        map_size=map_size,
    )
    if len(events) < 2:
        return mask

    for idx, curr_event in enumerate(events):
        next_event = events[(idx + 1) % len(events)]
        cell_pts = _build_sector_cell(source_xy, curr_event, next_event, loops, buildings)
        if len(cell_pts) < 3:
            continue
        raster_pts = []
        for x, y in cell_pts:
            raster_pts.append((
                int(round(min(max(x, 0.0), float(map_size)))),
                int(round(min(max(y, 0.0), float(map_size)))),
            ))
        raster_pts = _dedupe_consecutive_points(raster_pts)
        if len(raster_pts) < 3:
            continue
        if cv2 is not None:
            pts_arr = np.array(raster_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts_arr], 1)
        else:
            pil_draw.polygon(raster_pts, outline=1, fill=1)

    if pil_image is not None:
        mask = (np.array(pil_image, dtype=np.uint8) > 0).astype(np.uint8)
    return mask


'''
def _build_visibility_polygon_legacy(
    source_xy: Tuple[float, float],
    proj_data: List[Tuple],
    adjacency: Dict,
    buildings: List,
    edge_set: Optional[Set] = None,
    map_size: int = 256,
) -> List[Tuple[float, float]]:
    """
    从投影点 + 建筑物边 + 地图边界构建可视多边形。

    算法：
    1. 收集可视 V-P 对与可见地图角点
    2. 先按顶点方位角排序 V-P 对，再与角点按方位角合并
    3. 按 pair 级规则展开点链，处理边界连续段与 corner -> P -> V
    4. 顺序连接形成闭合多边形

    Args:
        source_xy: 光源坐标 (x, y)，MAP 坐标
        proj_data: [(vertex_xy, proj_xy, edge), ...] 每个可见顶点的信息
        adjacency: 保留参数（向后兼容）
        buildings: 建筑物多边形列表
        edge_set: 由 build_edge_set() 构建的边集（用于角点可视判断）
        map_size: 地图尺寸

    Returns:
        有序多边形顶点列表（MAP 坐标），可直接用于 fillPoly
    """
    if len(proj_data) < 2:
        return []

    # vertex 去重：同一个顶点只保留一条
    seen_verts = set()
    unique_data = []
    for entry in proj_data:
        vk = (round(float(entry[0][0]), 6), round(float(entry[0][1]), 6))
        if vk not in seen_verts:
            seen_verts.add(vk)
            unique_data.append(entry)
    proj_data = unique_data
    if len(proj_data) < 2:
        return []

    ms = float(map_size)

    pair_entries = []
    for idx, (v, p, e) in enumerate(proj_data):
        vx, vy = float(v[0]), float(v[1])
        px, py = float(p[0]), float(p[1])
        if edge_set is not None and not is_visible(source_xy, (vx, vy), edge_set):
            continue
        pair_entries.append({
            'idx': idx,
            'v': (vx, vy),
            'p': (px, py),
            'e': e,
            'angle_v': _angle_from(source_xy, (vx, vy)),
        })

    if len(pair_entries) < 2:
        return []

    pair_entries.sort(key=lambda item: (item['angle_v'], item['idx']))

    events = []
    for pair in pair_entries:
        events.append({'kind': 'pair', 'angle': pair['angle_v'], 'pair': pair})

    if edge_set is not None:
        for corner in [(0., 0.), (ms, 0.), (ms, ms), (0., ms)]:
            if _corner_visible(source_xy, corner, edge_set):
                events.append({
                    'kind': 'corner',
                    'angle': _angle_from(source_xy, corner),
                    'coord': corner,
                })

    if not events:
        return []

    kind_priority = {'pair': 0, 'corner': 1}
    events.sort(key=lambda item: (item['angle'], kind_priority.get(item['kind'], 99)))

    def _is_boundary_pair_event(event) -> bool:
        return event['kind'] == 'pair' and _is_boundary_edge(event['pair']['e'], map_size)

    def _boundary_pair_side(event) -> Optional[str]:
        if event['kind'] != 'pair':
            return None
        return _boundary_edge_side(event['pair']['e'], map_size)

    rotate_idx = 0
    for i, event in enumerate(events):
        prev_event = events[i - 1]
        if not _is_boundary_pair_event(event) or not _is_boundary_pair_event(prev_event):
            rotate_idx = i
            break
        if _boundary_pair_side(event) != _boundary_pair_side(prev_event):
            rotate_idx = i
            break
    if rotate_idx:
        events = events[rotate_idx:] + events[:rotate_idx]

    boundary_roles = {}
    i = 0
    while i < len(events):
        if not _is_boundary_pair_event(events[i]):
            i += 1
            continue
        run_side = _boundary_pair_side(events[i])
        j = i
        while j < len(events) and _is_boundary_pair_event(events[j]) and _boundary_pair_side(events[j]) == run_side:
            j += 1
        run = [events[k]['pair']['idx'] for k in range(i, j)]
        if len(run) == 2:
            boundary_roles[run[0]] = 'boundary_first'
            boundary_roles[run[1]] = 'boundary_last'
        elif len(run) > 2:
            boundary_roles[run[0]] = 'boundary_first'
            boundary_roles[run[-1]] = 'boundary_last'
            for mid_idx in run[1:-1]:
                boundary_roles[mid_idx] = 'boundary_middle'
        i = j

    def _pair_should_vertex_first(prev_pair, curr_pair) -> bool:
        curr_v = curr_pair['v']
        curr_edge = curr_pair['e']
        prev_v = prev_pair['v']
        prev_pts = [prev_v]
        prev_edge = prev_pair['e']
        if prev_edge is not None:
            prev_pts.append((float(prev_edge[0][0]), float(prev_edge[0][1])))
            prev_pts.append((float(prev_edge[1][0]), float(prev_edge[1][1])))
        same = any(_are_adjacent_on_building(pt, curr_v, buildings) for pt in prev_pts)
        if not same:
            return False
        if (
            curr_edge is not None
            and _edge_on_same_building(prev_v, curr_edge, buildings)
            and not _on_same_building(prev_v, curr_v, buildings)
        ):
            return False
        return True

    prev_pair = next((event['pair'] for event in reversed(events) if event['kind'] == 'pair'), None)
    prev_event_kind = events[-1]['kind']
    polygon_pts = []

    num_events = len(events)
    for event_idx, event in enumerate(events):
        if event['kind'] == 'corner':
            polygon_pts.append(event['coord'])
            prev_event_kind = 'corner'
            continue

        curr_pair = event['pair']
        role = boundary_roles.get(curr_pair['idx'])
        next_event = events[(event_idx + 1) % num_events] if num_events > 1 else None

        if role == 'boundary_middle':
            local_pts = [curr_pair['p'], curr_pair['v'], curr_pair['p']]
        elif role == 'boundary_last':
            local_pts = [curr_pair['p'], curr_pair['v']]
        elif role == 'boundary_first':
            if prev_event_kind == 'corner':
                local_pts = [curr_pair['p'], curr_pair['v']]
            else:
                local_pts = [curr_pair['v'], curr_pair['p']]
        else:
            if prev_event_kind == 'corner':
                local_pts = [curr_pair['p'], curr_pair['v']]
            elif prev_pair is not None and _pair_should_vertex_first(prev_pair, curr_pair):
                local_pts = [curr_pair['v'], curr_pair['p']]
            else:
                local_pts = [curr_pair['p'], curr_pair['v']]

        if (
            next_event is not None
            and next_event['kind'] == 'corner'
            and _is_boundary_edge(curr_pair['e'], map_size)
            and local_pts[-1] != curr_pair['p']
        ):
            local_pts.append(curr_pair['p'])

        polygon_pts.extend(local_pts)
        prev_pair = curr_pair
        prev_event_kind = 'pair'

    if len(polygon_pts) < 2:
        return polygon_pts
    deduped = [polygon_pts[0]]
    for pt in polygon_pts[1:]:
        if math.hypot(pt[0] - deduped[-1][0], pt[1] - deduped[-1][1]) > 1e-6:
            deduped.append(pt)
    if len(deduped) > 1 and math.hypot(deduped[0][0] - deduped[-1][0], deduped[0][1] - deduped[-1][1]) < 1e-6:
        deduped.pop()

    return deduped
'''


def build_visibility_polygon(
    source_xy: Tuple[float, float],
    proj_data: List[Tuple],
    adjacency: Dict,
    buildings: List,
    edge_set: Optional[Set] = None,
    map_size: int = 256,
) -> List[Tuple[float, float]]:
    """兼容接口：内部先构建 sector-cell mask，再提取外轮廓。"""
    mask = build_visibility_mask(
        source_xy,
        proj_data,
        adjacency,
        buildings,
        edge_set=edge_set,
        map_size=map_size,
    )
    if mask is None or int(mask.sum()) == 0:
        return []

    if cv2 is None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []
        boundary_pts = []
        for x, y in zip(xs, ys):
            x0 = max(0, x - 1)
            x1 = min(mask.shape[1], x + 2)
            y0 = max(0, y - 1)
            y1 = min(mask.shape[0], y + 2)
            if np.any(mask[y0:y1, x0:x1] == 0):
                boundary_pts.append((float(x), float(y)))
        if not boundary_pts:
            boundary_pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
        cx = sum(pt[0] for pt in boundary_pts) / len(boundary_pts)
        cy = sum(pt[1] for pt in boundary_pts) / len(boundary_pts)
        boundary_pts.sort(key=lambda pt: math.atan2(pt[1] - cy, pt[0] - cx))
        return _dedupe_consecutive_points(boundary_pts)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    polygon_pts = [(float(pt[0][0]), float(pt[0][1])) for pt in contour]
    return _dedupe_consecutive_points(polygon_pts)

