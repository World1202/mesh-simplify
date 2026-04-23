from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32, copy=True)
    return (vector / norm).astype(np.float32, copy=False)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    fov_rad = np.deg2rad(fov_y_degrees)
    focal = 1.0 / np.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = focal / aspect
    proj[1, 1] = focal
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


@dataclass
class OrbitCamera:
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    distance: float = 3.5
    azimuth: float = 0.0
    elevation: float = 0.35
    fov_y_degrees: float = 45.0
    near_plane: float = 0.1
    far_plane: float = 100.0
    orbit_sensitivity: float = 0.01
    pan_sensitivity: float = 0.002
    zoom_sensitivity: float = 0.12
    up_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )

    def reset(
        self,
        target: np.ndarray | None = None,
        distance: float = 3.5,
        azimuth: float = 0.0,
        elevation: float = 0.35,
    ) -> None:
        if target is not None:
            self.target = np.asarray(target, dtype=np.float32)
        self.distance = float(distance)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)

    def eye(self) -> np.ndarray:
        cos_elevation = np.cos(self.elevation)
        eye = np.array(
            [
                self.distance * cos_elevation * np.sin(self.azimuth),
                self.distance * np.sin(self.elevation),
                self.distance * cos_elevation * np.cos(self.azimuth),
            ],
            dtype=np.float32,
        )
        return self.target + eye

    def rotate(self, delta_x: float, delta_y: float) -> None:
        self.azimuth += float(delta_x) * self.orbit_sensitivity
        self.elevation += float(delta_y) * self.orbit_sensitivity
        self.elevation = float(np.clip(self.elevation, -1.55, 1.55))

    def zoom(self, scroll_delta: float) -> None:
        scale = np.exp(-float(scroll_delta) * self.zoom_sensitivity)
        self.distance = float(np.clip(self.distance * scale, 0.25, 200.0))

    def pan(self, delta_x: float, delta_y: float) -> None:
        eye = self.eye()
        forward = _normalize(self.target - eye)
        right = _normalize(np.cross(forward, self.up_axis))
        up = _normalize(np.cross(right, forward))
        offset = (
            -delta_x * self.pan_sensitivity * self.distance * right
            + delta_y * self.pan_sensitivity * self.distance * up
        )
        self.target = (self.target + offset).astype(np.float32)

    def view_matrix(self) -> np.ndarray:
        return look_at(self.eye(), self.target, self.up_axis)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        safe_aspect = max(float(aspect), 1e-6)
        return perspective(
            self.fov_y_degrees,
            safe_aspect,
            self.near_plane,
            self.far_plane,
        )

