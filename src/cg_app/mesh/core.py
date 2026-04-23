from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


def _as_vertices(vertices: np.ndarray) -> np.ndarray:
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    return array


def _as_faces(faces: np.ndarray) -> np.ndarray:
    array = np.asarray(faces, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("faces must have shape (m, 3)")
    return array


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return array / safe_norms


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts = _as_vertices(vertices)
    tri = _as_faces(faces)

    v0 = verts[tri[:, 0]]
    v1 = verts[tri[:, 1]]
    v2 = verts[tri[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    return normalize_vectors(normals)


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    verts = _as_vertices(vertices)
    tri = _as_faces(faces)

    v0 = verts[tri[:, 0]]
    v1 = verts[tri[:, 1]]
    v2 = verts[tri[:, 2]]
    face_vectors = np.cross(v1 - v0, v2 - v0)

    vertex_normals = np.zeros_like(verts)
    for face_index, face in enumerate(tri):
        normal = face_vectors[face_index]
        vertex_normals[face] += normal
    return normalize_vectors(vertex_normals)


@dataclass
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray | None = field(default=None)
    face_normals: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        self.vertices = _as_vertices(self.vertices)
        self.faces = _as_faces(self.faces)
        if self.vertex_normals is not None:
            self.vertex_normals = _as_vertices(self.vertex_normals)
            if self.vertex_normals.shape[0] != self.vertices.shape[0]:
                raise ValueError("vertex_normals must match vertex count")
        if self.face_normals is not None:
            self.face_normals = _as_vertices(self.face_normals)
            if self.face_normals.shape[0] != self.faces.shape[0]:
                raise ValueError("face_normals must match face count")

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def copy(self) -> "TriangleMesh":
        return TriangleMesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            vertex_normals=None if self.vertex_normals is None else self.vertex_normals.copy(),
            face_normals=None if self.face_normals is None else self.face_normals.copy(),
        )

    def with_computed_normals(self) -> "TriangleMesh":
        return TriangleMesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            vertex_normals=compute_vertex_normals(self.vertices, self.faces),
            face_normals=compute_face_normals(self.vertices, self.faces),
        )

    def compute_face_normals(self) -> np.ndarray:
        return compute_face_normals(self.vertices, self.faces)

    def compute_vertex_normals(self) -> np.ndarray:
        return compute_vertex_normals(self.vertices, self.faces)

    def levels(self, max_level: int) -> List["TriangleMesh"]:
        from .subdivision import generate_mesh_levels

        return generate_mesh_levels(self, max_level)
