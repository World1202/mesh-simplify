from __future__ import annotations

import time
from enum import Enum
from typing import Dict, List, Set, Tuple

import numpy as np

from .core import TriangleMesh

EdgeKey = Tuple[int, int]


class SubdivisionScheme(str, Enum):
    LOOP = "loop"
    LINEAR = "linear"
    MIDPOINT = "midpoint"
    BUTTERFLY = "butterfly"


SUBDIVISION_SCHEME_LABELS: dict[SubdivisionScheme, str] = {
    SubdivisionScheme.LOOP: "Loop 平滑细分",
    SubdivisionScheme.LINEAR: "Linear 一对四",
    SubdivisionScheme.MIDPOINT: "Centroid 面心细分",
    SubdivisionScheme.BUTTERFLY: "Modified Butterfly",
}


def normalize_subdivision_scheme(scheme: str | SubdivisionScheme) -> SubdivisionScheme:
    if isinstance(scheme, SubdivisionScheme):
        return scheme
    return SubdivisionScheme(str(scheme).strip().lower())


def subdivision_scheme_label(scheme: str | SubdivisionScheme) -> str:
    normalized = normalize_subdivision_scheme(scheme)
    return SUBDIVISION_SCHEME_LABELS[normalized]


class MeshTopology:
    def __init__(
        self,
        edge_to_opposites: Dict[EdgeKey, List[int]],
        edge_to_faces: Dict[EdgeKey, List[int]],
        vertex_neighbors: List[Set[int]],
    ) -> None:
        self.edge_to_opposites = edge_to_opposites
        self.edge_to_faces = edge_to_faces
        self.vertex_neighbors = vertex_neighbors


def _sorted_edge(a: int, b: int) -> EdgeKey:
    return (a, b) if a < b else (b, a)


def _build_topology(mesh: TriangleMesh) -> MeshTopology:
    faces = mesh.faces
    num_vertices = mesh.vertex_count
    edge_to_opposites: Dict[EdgeKey, List[int]] = {}
    edge_to_faces: Dict[EdgeKey, List[int]] = {}
    vertex_neighbors: List[Set[int]] = [set() for _ in range(num_vertices)]

    for face_index, (a, b, c) in enumerate(faces):
        a, b, c = int(a), int(b), int(c)
        vertex_neighbors[a].update([b, c])
        vertex_neighbors[b].update([a, c])
        vertex_neighbors[c].update([a, b])

        for u, v, opp in ((a, b, c), (b, c, a), (c, a, b)):
            edge = _sorted_edge(u, v)
            edge_to_opposites.setdefault(edge, []).append(opp)
            edge_to_faces.setdefault(edge, []).append(face_index)

    return MeshTopology(
        edge_to_opposites=edge_to_opposites,
        edge_to_faces=edge_to_faces,
        vertex_neighbors=vertex_neighbors,
    )


def _rebuild_faces_from_edge_vertices(
    faces: np.ndarray,
    edge_to_new_index: Dict[EdgeKey, int],
) -> np.ndarray:
    new_faces = []
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        ab = edge_to_new_index[_sorted_edge(a, b)]
        bc = edge_to_new_index[_sorted_edge(b, c)]
        ca = edge_to_new_index[_sorted_edge(c, a)]
        new_faces.extend(
            [
                (a, ab, ca),
                (ab, b, bc),
                (ca, bc, c),
                (ab, bc, ca),
            ]
        )
    return np.asarray(new_faces, dtype=np.int64)


def _subdivide_linear(mesh: TriangleMesh) -> TriangleMesh:
    vertices = mesh.vertices
    topology = _build_topology(mesh)
    edge_to_new_index: Dict[EdgeKey, int] = {}
    new_vertices = [v.copy() for v in vertices]

    for edge in topology.edge_to_opposites:
        u, v = edge
        edge_to_new_index[edge] = len(new_vertices)
        new_vertices.append(0.5 * (vertices[u] + vertices[v]))

    new_faces = _rebuild_faces_from_edge_vertices(mesh.faces, edge_to_new_index)
    return TriangleMesh(vertices=np.asarray(new_vertices, dtype=np.float64), faces=new_faces)


def _subdivide_midpoint(mesh: TriangleMesh) -> TriangleMesh:
    vertices = mesh.vertices
    faces = mesh.faces
    new_vertices = [v.copy() for v in vertices]
    new_faces = []

    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        centroid = (vertices[a] + vertices[b] + vertices[c]) / 3.0
        centroid_index = len(new_vertices)
        new_vertices.append(centroid)
        new_faces.extend(
            [
                (a, b, centroid_index),
                (b, c, centroid_index),
                (c, a, centroid_index),
            ]
        )

    return TriangleMesh(
        vertices=np.asarray(new_vertices, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
    )


def _find_wing_vertex(
    edge_to_faces: Dict[EdgeKey, List[int]],
    faces: np.ndarray,
    a: int,
    b: int,
    exclude_vertex: int,
) -> int | None:
    face_indices = edge_to_faces.get(_sorted_edge(a, b), [])
    for face_index in face_indices:
        tri = [int(x) for x in faces[face_index]]
        for vertex in tri:
            if vertex not in {a, b, exclude_vertex}:
                return vertex
    return None


def _subdivide_butterfly(mesh: TriangleMesh) -> TriangleMesh:
    vertices = mesh.vertices
    topology = _build_topology(mesh)
    edge_to_new_index: Dict[EdgeKey, int] = {}
    new_vertices = [v.copy() for v in vertices]

    for edge, opposites in topology.edge_to_opposites.items():
        u, v = edge
        if len(opposites) == 2:
            p, q = opposites
            wing_up = _find_wing_vertex(topology.edge_to_faces, mesh.faces, u, p, v)
            wing_pv = _find_wing_vertex(topology.edge_to_faces, mesh.faces, p, v, u)
            wing_uq = _find_wing_vertex(topology.edge_to_faces, mesh.faces, u, q, v)
            wing_qv = _find_wing_vertex(topology.edge_to_faces, mesh.faces, q, v, u)

            if None not in {wing_up, wing_pv, wing_uq, wing_qv}:
                odd = (
                    0.5 * (vertices[u] + vertices[v])
                    + 0.125 * (vertices[p] + vertices[q])
                    - 0.0625
                    * (
                        vertices[int(wing_up)]
                        + vertices[int(wing_pv)]
                        + vertices[int(wing_uq)]
                        + vertices[int(wing_qv)]
                    )
                )
            else:
                odd = 0.5 * (vertices[u] + vertices[v]) + 0.125 * (vertices[p] + vertices[q])
        else:
            odd = 0.5 * (vertices[u] + vertices[v])

        edge_to_new_index[edge] = len(new_vertices)
        new_vertices.append(odd)

    new_faces = _rebuild_faces_from_edge_vertices(mesh.faces, edge_to_new_index)
    return TriangleMesh(vertices=np.asarray(new_vertices, dtype=np.float64), faces=new_faces)


def _subdivide_loop_slow(mesh: TriangleMesh) -> TriangleMesh:
    vertices = mesh.vertices
    faces = mesh.faces
    num_vertices = len(vertices)
    topology = _build_topology(mesh)
    edge_to_opposites = topology.edge_to_opposites
    vertex_neighbors = topology.vertex_neighbors

    is_boundary_vertex = [False] * num_vertices
    boundary_edges_of_vertex: List[List[int]] = [[] for _ in range(num_vertices)]
    for (u, v), opps in edge_to_opposites.items():
        if len(opps) == 1:
            is_boundary_vertex[u] = True
            is_boundary_vertex[v] = True
            boundary_edges_of_vertex[u].append(v)
            boundary_edges_of_vertex[v].append(u)

    new_even_vertices = np.zeros_like(vertices)
    for i in range(num_vertices):
        v_pos = vertices[i]
        if is_boundary_vertex[i]:
            neighbors = boundary_edges_of_vertex[i]
            if len(neighbors) >= 2:
                n1, n2 = neighbors[0], neighbors[1]
                new_even_vertices[i] = 0.75 * v_pos + 0.125 * vertices[n1] + 0.125 * vertices[n2]
            else:
                new_even_vertices[i] = v_pos
        else:
            n = len(vertex_neighbors[i])
            if n > 0:
                beta = (1.0 / n) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / n)) ** 2)
                neighbor_sum = np.sum(vertices[list(vertex_neighbors[i])], axis=0)
                new_even_vertices[i] = (1.0 - n * beta) * v_pos + beta * neighbor_sum
            else:
                new_even_vertices[i] = v_pos

    new_odd_vertices_list = []
    edge_to_new_index = {}

    for (u, v), opps in edge_to_opposites.items():
        if len(opps) == 2:
            o1, o2 = opps
            odd_pos = 0.375 * (vertices[u] + vertices[v]) + 0.125 * (vertices[o1] + vertices[o2])
        else:
            odd_pos = 0.5 * (vertices[u] + vertices[v])

        edge_to_new_index[(u, v)] = num_vertices + len(new_odd_vertices_list)
        new_odd_vertices_list.append(odd_pos)

    if new_odd_vertices_list:
        new_odd_array = np.array(new_odd_vertices_list, dtype=np.float64)
        new_vertices = np.vstack((new_even_vertices, new_odd_array))
    else:
        new_vertices = new_even_vertices

    new_faces = _rebuild_faces_from_edge_vertices(faces, edge_to_new_index)
    return TriangleMesh(vertices=new_vertices, faces=new_faces)


def _subdivide_loop_fast(mesh: TriangleMesh) -> TriangleMesh:
    vertices = mesh.vertices
    faces = mesh.faces
    num_vertices = len(vertices)
    num_faces = len(faces)

    he0 = faces[:, [0, 1]]
    opp0 = faces[:, 2]
    he1 = faces[:, [1, 2]]
    opp1 = faces[:, 0]
    he2 = faces[:, [2, 0]]
    opp2 = faces[:, 1]

    half_edges = np.vstack([he0, he1, he2])
    opposites = np.concatenate([opp0, opp1, opp2])
    half_edges_sorted = np.sort(half_edges, axis=1)

    unique_edges, inverse_indices, edge_counts = np.unique(
        half_edges_sorted, axis=0, return_inverse=True, return_counts=True
    )
    num_edges = len(unique_edges)

    u = unique_edges[:, 0]
    v = unique_edges[:, 1]

    opp_sum = np.zeros((num_edges, 3), dtype=np.float64)
    np.add.at(opp_sum, inverse_indices, vertices[opposites])

    is_interior_edge = (edge_counts == 2)[:, None]
    odd_vertices = np.where(
        is_interior_edge,
        0.375 * (vertices[u] + vertices[v]) + 0.125 * opp_sum,
        0.5 * (vertices[u] + vertices[v]),
    )

    neighbor_sum = np.zeros_like(vertices)
    degree = np.zeros(num_vertices, dtype=np.int64)
    np.add.at(neighbor_sum, u, vertices[v])
    np.add.at(neighbor_sum, v, vertices[u])
    np.add.at(degree, u, 1)
    np.add.at(degree, v, 1)

    boundary_mask = edge_counts == 1
    bu = u[boundary_mask]
    bv = v[boundary_mask]
    is_boundary_vertex = np.zeros(num_vertices, dtype=bool)
    is_boundary_vertex[bu] = True
    is_boundary_vertex[bv] = True

    boundary_neighbor_sum = np.zeros_like(vertices)
    np.add.at(boundary_neighbor_sum, bu, vertices[bv])
    np.add.at(boundary_neighbor_sum, bv, vertices[bu])

    beta = np.zeros(num_vertices, dtype=np.float64)
    valid = degree > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        n = degree[valid]
        beta[valid] = (1.0 / n) * (5.0 / 8.0 - (3.0 / 8.0 + 0.25 * np.cos(2.0 * np.pi / n)) ** 2)

    even_interior = (1.0 - degree[:, None] * beta[:, None]) * vertices + beta[:, None] * neighbor_sum
    even_boundary = 0.75 * vertices + 0.125 * boundary_neighbor_sum
    even_vertices = np.where(is_boundary_vertex[:, None], even_boundary, even_interior)

    new_vertices = np.vstack([even_vertices, odd_vertices])

    ab = num_vertices + inverse_indices[0:num_faces]
    bc = num_vertices + inverse_indices[num_faces:2 * num_faces]
    ca = num_vertices + inverse_indices[2 * num_faces:3 * num_faces]

    new_faces = np.vstack(
        [
            np.column_stack([faces[:, 0], ab, ca]),
            np.column_stack([ab, faces[:, 1], bc]),
            np.column_stack([ca, bc, faces[:, 2]]),
            np.column_stack([ab, bc, ca]),
        ]
    )
    return TriangleMesh(vertices=new_vertices, faces=new_faces)


def subdivide_triangle_mesh(
    mesh: TriangleMesh,
    scheme: str | SubdivisionScheme = SubdivisionScheme.LOOP,
    use_fast: bool = True,
) -> TriangleMesh:
    normalized_scheme = normalize_subdivision_scheme(scheme)
    if normalized_scheme == SubdivisionScheme.LOOP:
        return _subdivide_loop_fast(mesh) if use_fast else _subdivide_loop_slow(mesh)
    if normalized_scheme == SubdivisionScheme.LINEAR:
        return _subdivide_linear(mesh)
    if normalized_scheme == SubdivisionScheme.MIDPOINT:
        return _subdivide_midpoint(mesh)
    if normalized_scheme == SubdivisionScheme.BUTTERFLY:
        return _subdivide_butterfly(mesh)
    raise ValueError(f"Unsupported subdivision scheme: {scheme}")


def generate_mesh_levels(
    base_mesh: TriangleMesh,
    max_level: int,
    use_fast: bool = True,
    scheme: str | SubdivisionScheme = SubdivisionScheme.LOOP,
) -> tuple[list[TriangleMesh], float]:
    if max_level < 0:
        raise ValueError("max_level must be non-negative")

    start_time = time.perf_counter()
    levels = [base_mesh.copy()]
    current = base_mesh.copy()
    for _ in range(max_level):
        current = subdivide_triangle_mesh(current, scheme=scheme, use_fast=use_fast)
        levels.append(current)

    compute_time_ms = (time.perf_counter() - start_time) * 1000.0
    return levels, compute_time_ms
