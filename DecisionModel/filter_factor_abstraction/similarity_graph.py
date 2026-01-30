"""
基于余弦相似度构建相似度图，并通过阈值连接或连通分量得到语义簇。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    embeddings: (N, D)，假定已 L2 归一化。
    返回 (N, N) 余弦相似度矩阵。
    """
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.dot(embeddings, embeddings.T).astype(np.float32)


def build_similarity_graph(
    embeddings: np.ndarray,
    threshold: float,
    texts: List[str],
) -> List[Tuple[int, int]]:
    """
    根据阈值将相似度矩阵转为边列表。相似度 >= threshold 且 i < j 的 (i, j) 作为边。
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    sim = cosine_similarity_matrix(embeddings)
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                edges.append((i, j))
    return edges


def connected_components_from_graph(
    n: int,
    edges: List[Tuple[int, int]],
) -> List[List[int]]:
    """
    根据边列表计算连通分量，返回每个分量中的节点索引列表。
    使用并查集。
    """
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in edges:
        union(i, j)

    comp_id: dict = {}
    for i in range(n):
        r = find(i)
        if r not in comp_id:
            comp_id[r] = []
        comp_id[r].append(i)

    return list(comp_id.values())


def clusters_from_embeddings(
    embeddings: np.ndarray,
    texts: List[str],
    threshold: float,
) -> List[List[int]]:
    """
    一站式：embeddings + threshold → 相似度图 → 连通分量 → 簇（每个簇为索引列表）。
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    edges = build_similarity_graph(embeddings, threshold, texts)
    return connected_components_from_graph(n, edges)
