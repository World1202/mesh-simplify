from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class DrawMode(str, Enum):
    SOLID = "solid"
    WIREFRAME = "wireframe"


class ShadingMode(str, Enum):
    GOURAUD = "gouraud"
    PHONG = "phong"


@dataclass
class LightingParams:
    ambient_strength: float = 0.35
    diffuse_strength: float = 0.85
    specular_strength: float = 0.35
    shininess: float = 32.0
    light_position: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 2.0], dtype=np.float32)
    )
    light_color: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    fill_light_position: np.ndarray = field(
        default_factory=lambda: np.array([-2.0, -1.0, -2.0], dtype=np.float32)
    )
    fill_light_color: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32)
    )
    fill_light_strength: float = 0.4
    object_color: np.ndarray = field(
        default_factory=lambda: np.array([0.78, 0.82, 0.92], dtype=np.float32)
    )


@dataclass
class RenderState:
    draw_mode: DrawMode = DrawMode.SOLID
    shading: ShadingMode = ShadingMode.PHONG
    subdivision_scheme: str = "loop"
    show_normals: bool = False
    background_color: np.ndarray = field(
        default_factory=lambda: np.array([0.06, 0.07, 0.09, 1.0], dtype=np.float32)
    )
    lighting: LightingParams = field(default_factory=LightingParams)
    use_vectorization: bool = True
    last_compute_time_ms: float = 0.0
    animate_light: bool = False
    auto_play_enabled: bool = False
    auto_play_interval_seconds: float = 0.8
    recompute_in_progress: bool = False
    subdivision_status_text: str = ""
