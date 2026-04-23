from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    return (vectors / lengths).astype(np.float32, copy=False)


def _triangulate_face(face_indices: list[int]) -> list[tuple[int, int, int]]:
    if len(face_indices) < 3:
        return []
    first = face_indices[0]
    return [
        (first, face_indices[i], face_indices[i + 1])
        for i in range(1, len(face_indices) - 1)
    ]


def compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    triangles = positions[indices]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals[indices[:, 0]] += face_normals
    normals[indices[:, 1]] += face_normals
    normals[indices[:, 2]] += face_normals
    return _normalize_rows(normals)


@dataclass
class Mesh:
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    name: str = "mesh"

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float32)
        self.normals = np.asarray(self.normals, dtype=np.float32)
        self.indices = np.asarray(self.indices, dtype=np.int32)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.normals.shape != self.positions.shape:
            raise ValueError("normals must have the same shape as positions")
        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError("indices must have shape (M, 3)")

    @classmethod
    def from_triangles(
        cls,
        positions: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None = None,
        name: str = "mesh",
    ) -> "Mesh":
        positions = np.asarray(positions, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int32)
        if normals is None:
            normals = compute_vertex_normals(positions, indices)
        else:
            normals = np.asarray(normals, dtype=np.float32)
        return cls(positions=positions, normals=normals, indices=indices, name=name)

    @classmethod
    def from_triangle_mesh(cls, mesh, name: str | None = None) -> "Mesh":
        positions = getattr(mesh, "positions", None)
        if positions is None:
            positions = getattr(mesh, "vertices")
        indices = getattr(mesh, "indices", None)
        if indices is None:
            indices = getattr(mesh, "faces")
        normals = getattr(mesh, "normals", None)
        if normals is None:
            normals = getattr(mesh, "vertex_normals", None)
        mesh_name = name or getattr(mesh, "name", "mesh")
        render_mesh = cls.from_triangles(positions, indices, normals=normals, name=mesh_name)
        return render_mesh

    @classmethod
    def default_demo(cls) -> "Mesh":
        positions = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
                [5, 2, 1],
                [5, 3, 2],
                [5, 4, 3],
                [5, 1, 4],
            ],
            dtype=np.int32,
        )
        return cls.from_triangles(positions, indices, name="octahedron")

    @classmethod
    def load_obj(cls, path: str | Path, name: str | None = None) -> "Mesh":
        path = Path(path)
        vertices: list[list[float]] = []
        faces: list[tuple[int, int, int]] = []

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                tag, values = parts[0], parts[1:]
                if tag == "v":
                    vertices.append([float(values[0]), float(values[1]), float(values[2])])
                elif tag == "f":
                    face_indices: list[int] = []
                    for token in values:
                        vertex_index = token.split("/")[0]
                        index = int(vertex_index)
                        if index < 0:
                            index = len(vertices) + index + 1
                        face_indices.append(index - 1)
                    faces.extend(_triangulate_face(face_indices))

        if not vertices or not faces:
            raise ValueError(f"OBJ file {path} does not contain a valid triangulated mesh")

        positions = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(faces, dtype=np.int32)
        return cls.from_triangles(positions, indices, name=name or path.stem).centered()

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    def centered(self) -> "Mesh":
        mins = self.positions.min(axis=0)
        maxs = self.positions.max(axis=0)
        center = (mins + maxs) * 0.5
        extent = float(np.max(maxs - mins))
        scale = 1.0 if extent == 0.0 else 2.0 / extent
        positions = (self.positions - center) * scale
        return Mesh.from_triangles(positions, self.indices, self.normals, name=self.name)
