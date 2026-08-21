"""
Topological Angular Gap Healing and Road Network Reconstruction.
Bridges tree canopy and shadow occlusion gaps using directional ray casting and geometric alignment.
"""

from typing import List, Tuple
import cv2
import networkx as nx
import numpy as np


def get_endpoint_tangent(G: nx.Graph, node: int, lookback_pts: int = 5) -> Tuple[float, float, float]:
    """
    Computes outgoing tangent direction (unit_y, unit_x, angle_rad) for a degree-1 endpoint node.
    Direction points away from the road body into the gap.
    """
    pos = np.array(G.nodes[node]["pos"], dtype=float)
    edges = list(G.edges(node, data=True))
    if not edges:
        return 0.0, 0.0, 0.0

    u, v, data = edges[0]
    pts = data.get("pts", [])
    if len(pts) >= 2:
        if tuple(pts[0]) == tuple(G.nodes[node]["pos"]):
            # Node is at start of path, vector points outward from pts[k] -> pts[0]
            k = min(lookback_pts, len(pts) - 1)
            inward_pt = np.array(pts[k], dtype=float)
        else:
            # Node is at end of path, vector points outward from pts[-k] -> pts[-1]
            k = min(lookback_pts, len(pts) - 1)
            inward_pt = np.array(pts[-k - 1], dtype=float)
        vec = pos - inward_pt
    else:
        other_node = v if u == node else u
        other_pos = np.array(G.nodes[other_node]["pos"], dtype=float)
        vec = pos - other_pos

    norm = np.linalg.norm(vec)
    if norm < 1e-5:
        return 0.0, 0.0, 0.0

    unit_vec = vec / norm
    angle = np.arctan2(unit_vec[0], unit_vec[1])
    return float(unit_vec[0]), float(unit_vec[1]), float(angle)


def heal_road_network(
    G: nx.Graph,
    max_gap_distance: float = 60.0,
    max_angle_diff_deg: float = 35.0,
) -> Tuple[nx.Graph, int]:
    """
    Directional Angular Gap Healing algorithm.
    Identifies degree-1 dead-ends caused by canopies/shadows and bridges
    collinear gap endpoints within the search cone.

    Args:
        G: Input NetworkX road graph
        max_gap_distance: Maximum occlusion gap width in pixels to bridge
        max_angle_diff_deg: Maximum allowed angular deviation between ray and candidate

    Returns:
        (G_healed, num_healed_gaps)
    """
    G_healed = G.copy()
    max_angle_rad = np.radians(max_angle_diff_deg)

    # Collect degree-1 dead-end endpoints
    endpoints = [n for n in G_healed.nodes() if G_healed.degree(n) == 1]
    healed_count = 0
    connected_pairs = set()

    for i, u in enumerate(endpoints):
        if u in connected_pairs:
            continue
        uy, ux, u_angle = get_endpoint_tangent(G_healed, u)
        u_pos = np.array(G_healed.nodes[u]["pos"], dtype=float)
        u_dir = np.array([uy, ux])

        best_v = None
        best_dist = float("inf")

        for j, v in enumerate(endpoints):
            if i == j or v in connected_pairs:
                continue
            if G_healed.has_edge(u, v):
                continue

            v_pos = np.array(G_healed.nodes[v]["pos"], dtype=float)
            gap_vec = v_pos - u_pos
            dist = np.linalg.norm(gap_vec)

            if dist > max_gap_distance or dist < 2.0:
                continue

            gap_dir = gap_vec / dist

            # Check if candidate is inside outgoing search cone of u
            cos_u = np.dot(u_dir, gap_dir)
            if cos_u < np.cos(max_angle_rad):
                continue

            # Check if candidate v is facing toward u (opposite directional heading)
            vy, vx, _ = get_endpoint_tangent(G_healed, v)
            v_dir = np.array([vy, vx])
            cos_v = np.dot(v_dir, -gap_dir)
            if cos_v < np.cos(max_angle_rad):
                continue

            # Candidate passed geometric alignment tests
            if dist < best_dist:
                best_dist = dist
                best_v = v

        if best_v is not None:
            # Create healed bridge edge
            u_pt = G_healed.nodes[u]["pos"]
            v_pt = G_healed.nodes[best_v]["pos"]
            
            # Generate interpolated straight line points for the healed gap
            n_steps = max(2, int(best_dist))
            ys = np.linspace(u_pt[0], v_pt[0], n_steps).astype(int)
            xs = np.linspace(u_pt[1], v_pt[1], n_steps).astype(int)
            line_pts = list(zip(ys.tolist(), xs.tolist()))

            G_healed.add_edge(u, best_v, weight=float(best_dist), pts=line_pts, healed=True)
            connected_pairs.add(u)
            connected_pairs.add(best_v)
            healed_count += 1

    return G_healed, healed_count


def rasterize_graph(
    G: nx.Graph,
    height: int = 512,
    width: int = 512,
    road_width: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterizes graph back into 2D binary road mask and healed-only overlay.

    Returns:
        (full_mask, healed_mask) uint8 arrays in {0, 1}
    """
    mask_full = np.zeros((height, width), dtype=np.uint8)
    mask_healed = np.zeros((height, width), dtype=np.uint8)

    for u, v, data in G.edges(data=True):
        pts = data.get("pts", [])
        is_healed = data.get("healed", False)

        if len(pts) >= 2:
            pts_cv = np.array([[x, y] for y, x in pts], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask_full, [pts_cv], isClosed=False, color=1, thickness=road_width)
            if is_healed:
                cv2.polylines(mask_healed, [pts_cv], isClosed=False, color=1, thickness=road_width + 2)
        else:
            p1 = G.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]
            cv2.line(mask_full, (p1[1], p1[0]), (p2[1], p2[0]), color=1, thickness=road_width)
            if is_healed:
                cv2.line(mask_healed, (p1[1], p1[0]), (p2[1], p2[0]), color=1, thickness=road_width + 2)

    return mask_full, mask_healed
