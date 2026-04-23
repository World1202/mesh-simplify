from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .core import TriangleMesh


def _resolve_obj_index(index: int, count: int) -> int:
    if index > 0:
        resolved = index - 1
    else:
        resolved = count + index
    if resolved < 0 or resolved >= count:
        raise ValueError("OBJ index out of range")
    return resolved


def _parse_face_vertex(token: str) -> int:
    head = token.split("/", 1)[0].strip()
    if not head:
        raise ValueError("empty OBJ face index")
    return int(head)


def _triangulate_face(indices: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    anchor = indices[0]
    return ((anchor, indices[i], indices[i + 1]) for i in range(1, len(indices) - 1))


def load_obj_text(text: str) -> TriangleMesh:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        prefix = parts[0]
        if prefix == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            continue

        if prefix == "f" and len(parts) >= 4:
            raw_indices = [_parse_face_vertex(token) for token in parts[1:]]
            resolved = [_resolve_obj_index(index, len(vertices)) for index in raw_indices]
            faces.extend(_triangulate_face(resolved))
            continue

    if not vertices:
        raise ValueError("OBJ file does not contain any vertices")
    if not faces:
        raise ValueError("OBJ file does not contain any triangular faces")

    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )


def load_obj(path: str | Path, encoding: str = "utf-8") -> TriangleMesh:
    obj_path = Path(path)
    return load_obj_text(obj_path.read_text(encoding=encoding, errors="ignore"))
