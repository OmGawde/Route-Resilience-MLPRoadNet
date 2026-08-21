"""
Road Skeleton Vectorization and Graph Construction.
Converts 2D road segmentation masks into NetworkX planar graphs.
"""

from typing import Dict, List, Tuple
import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Computes single-pixel wide skeleton from binary road mask."""
    bin_mask = (mask > 0).astype(bool)
    skel = skeletonize(bin_mask).astype(np.uint8)
    return skel


def get_neighbors_8(y: int, x: int, h: int, w: int) -> List[Tuple[int, int]]:
    """Returns 8-connected neighbor coordinates within image bounds."""
    nbrs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx_ < w:
                nbrs.append((ny, nx_))
    return nbrs


def mask_to_road_graph(
    mask: np.ndarray,
    min_branch_len: int = 8,
) -> Tuple[nx.Graph, np.ndarray]:
    """
    Extracts topological NetworkX road graph G=(V, E) from binary road mask.

    Args:
        mask: (H, W) binary mask in {0, 1}
        min_branch_len: Prune spurious spur branches shorter than this pixel threshold

    Returns:
        (G, skeleton) where G has:
          - nodes: integer IDs, attribute 'pos' = (y, x), 'deg' = degree
          - edges: attribute 'weight' = length in pixels, 'pts' = list of (y, x)
    """
    skel = skeletonize_mask(mask)
    h, w = skel.shape
    G = nx.Graph()

    # Find all skeleton pixels
    skel_pts = np.argwhere(skel > 0)
    if len(skel_pts) == 0:
        return G, skel

    # Map (y, x) -> neighbor count in skeleton
    skel_set = set((int(y), int(x)) for y, x in skel_pts)
    deg_map: Dict[Tuple[int, int], int] = {}

    for y, x in skel_set:
        nbrs = [p for p in get_neighbors_8(y, x, h, w) if p in skel_set]
        deg_map[(y, x)] = len(nbrs)

    # Key nodes are endpoints (deg == 1) and junctions (deg >= 3)
    key_points = set(p for p, d in deg_map.items() if d != 2)
    
    # If the skeleton is a single loop with no deg != 2, pick arbitrary point
    if not key_points and skel_set:
        key_points.add(next(iter(skel_set)))

    # Assign node IDs to key points
    pt_to_node = {}
    for node_id, pt in enumerate(key_points):
        pt_to_node[pt] = node_id
        G.add_node(node_id, pos=pt, y=pt[0], x=pt[1])

    # Trace edges along deg == 2 paths between key points
    visited_edges = set()

    for start_pt in key_points:
        start_id = pt_to_node[start_pt]
        nbrs = [p for p in get_neighbors_8(start_pt[0], start_pt[1], h, w) if p in skel_set]

        for nbr in nbrs:
            edge_key = tuple(sorted([start_pt, nbr]))
            if edge_key in visited_edges:
                continue

            path = [start_pt, nbr]
            curr = nbr
            prev = start_pt
            visited_edges.add(edge_key)

            while curr not in key_points:
                next_nbrs = [p for p in get_neighbors_8(curr[0], curr[1], h, w) if p in skel_set and p != prev]
                if not next_nbrs:
                    break
                next_p = next_nbrs[0]
                visited_edges.add(tuple(sorted([curr, next_p])))
                path.append(next_p)
                prev, curr = curr, next_p

            if curr in key_points:
                end_id = pt_to_node[curr]
                if start_id != end_id or len(path) > 3:
                    # Calculate path length
                    pts_arr = np.array(path)
                    diffs = np.diff(pts_arr, axis=0)
                    length = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
                    G.add_edge(start_id, end_id, weight=length, pts=path, healed=False)

    # Prune short dead-end spur branches
    if min_branch_len > 0:
        spurs = [
            n for n in G.nodes()
            if G.degree(n) == 1 and G.edges(n) and list(G.edges(n, data=True))[0][2].get("weight", 0) < min_branch_len
        ]
        G.remove_nodes_from(spurs)

    return G, skel
