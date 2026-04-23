from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import math
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import glfw
import numpy as np

from cg_app.ai import DemoAIAssistant
from cg_app.config import AppConfig, resolve_mesh_path
from cg_app.input.controller import InputController
from cg_app.mesh import (
    SubdivisionScheme,
    TriangleMesh,
    generate_mesh_levels,
    load_obj,
    normalize_subdivision_scheme,
    subdivision_scheme_label,
)
from cg_app.rendering.camera import OrbitCamera
from cg_app.rendering.renderer import OpenGLRenderer
from cg_app.rendering.state import DrawMode, LightingParams, RenderState, ShadingMode
from cg_app.ui.panel import create_control_panel


@dataclass
class MeshLevels:
    name: str
    levels: list[TriangleMesh]

    @property
    def max_level(self) -> int:
        return max(len(self.levels) - 1, 0)

    def at(self, level: int) -> TriangleMesh:
        return self.levels[int(np.clip(level, 0, self.max_level))]


@dataclass
class SubdivisionRecomputeRequest:
    scheme: SubdivisionScheme
    use_fast: bool
    max_level: int


@dataclass
class SubdivisionRecomputeResult:
    request: SubdivisionRecomputeRequest
    levels: list[TriangleMesh]
    compute_time_ms: float


def _normalize_triangle_mesh(mesh: TriangleMesh) -> TriangleMesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    extent = float(np.max(maxs - mins))
    scale = 1.0 if extent == 0.0 else 2.0 / extent
    normalized = (vertices - center) * scale
    return TriangleMesh(normalized, mesh.faces.copy()).with_computed_normals()


def _create_demo_mesh() -> TriangleMesh:
    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
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
        dtype=np.int64,
    )
    return TriangleMesh(vertices, faces).with_computed_normals()


def _load_mesh_levels(
    mesh_path: Path | None,
    max_level: int,
    scheme: str | SubdivisionScheme,
    use_fast: bool = True,
) -> tuple[MeshLevels, float]:
    if mesh_path is None:
        base_mesh = _create_demo_mesh()
        name = "demo-octahedron"
    else:
        base_mesh = load_obj(mesh_path)
        name = mesh_path.stem
    normalized = _normalize_triangle_mesh(base_mesh)
    levels, compute_time = generate_mesh_levels(
        normalized,
        max_level,
        use_fast=use_fast,
        scheme=scheme,
    )
    return MeshLevels(name=name, levels=levels), compute_time


def _compute_levels_in_background(
    base_mesh: TriangleMesh,
    request: SubdivisionRecomputeRequest,
) -> SubdivisionRecomputeResult:
    levels, compute_time_ms = generate_mesh_levels(
        base_mesh,
        request.max_level,
        use_fast=request.use_fast,
        scheme=request.scheme,
    )
    return SubdivisionRecomputeResult(
        request=request,
        levels=levels,
        compute_time_ms=compute_time_ms,
    )


class GraphicsApp:
    def __init__(
        self,
        mesh_levels: MeshLevels,
        width: int = 1280,
        height: int = 720,
        title: str = "CG Renderer",
    ) -> None:
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.config = AppConfig()
        self.mesh_levels = mesh_levels
        self.base_mesh = mesh_levels.at(0).copy()
        self.current_level = 0
        self.max_subdivision_level = mesh_levels.max_level

        state = RenderState(
            draw_mode=DrawMode(self.config.default_draw_mode),
            shading=ShadingMode(self.config.default_shading),
            subdivision_scheme=self.config.default_subdivision_scheme,
            background_color=np.array(self.config.background_color, dtype=np.float32),
            lighting=LightingParams(
                ambient_strength=self.config.ambient_strength,
                diffuse_strength=self.config.diffuse_strength,
                specular_strength=self.config.specular_strength,
                shininess=self.config.shininess,
                light_position=np.array(self.config.light_position, dtype=np.float32),
                light_color=np.array(self.config.light_color, dtype=np.float32),
                fill_light_position=np.array(self.config.fill_light_position, dtype=np.float32),
                fill_light_color=np.array(self.config.fill_light_color, dtype=np.float32),
                fill_light_strength=self.config.fill_light_strength,
                object_color=np.array(self.config.object_color, dtype=np.float32),
            ),
        )
        state.show_normals = False
        state.use_vectorization = self.config.default_use_vectorization
        state.last_compute_time_ms = 0.0
        state.animate_light = False
        state.auto_play_enabled = self.config.default_auto_play_enabled
        state.auto_play_interval_seconds = self.config.default_auto_play_interval_seconds
        state.recompute_in_progress = False
        state.subdivision_status_text = "当前结果可用。"

        self.camera = OrbitCamera(
            distance=self.config.camera_distance,
            azimuth=self.config.camera_azimuth,
            elevation=self.config.camera_elevation,
            fov_y_degrees=self.config.camera_fov,
            orbit_sensitivity=self.config.orbit_sensitivity,
            pan_sensitivity=self.config.pan_sensitivity,
            zoom_sensitivity=self.config.zoom_sensitivity,
        )

        self.renderer = OpenGLRenderer(state=state)
        self.renderer.initialize()
        self.renderer.resize(width, height)
        self.renderer.set_mesh(self.mesh_levels.at(self.current_level))

        self.input = InputController(
            self.window,
            camera=self.camera,
            state=self.renderer.state,
        )
        self.panel = create_control_panel(
            self.window,
            panel_width=self.config.panel_width,
            ai_panel_height=self.config.ai_panel_height,
            font_candidates=self.config.chinese_font_candidates,
            font_size_px=self.config.chinese_font_size_px,
        )
        self.ai_assistant = DemoAIAssistant(self.config)
        self._subdivision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="subdivision")
        self._recompute_future: Future[SubdivisionRecomputeResult] | None = None
        self._queued_recompute_request: SubdivisionRecomputeRequest | None = None
        self._last_framebuffer_size = (width, height)
        self._last_auto_play_step_time = float(glfw.get_time())
        self._running = True

    def _render_region(self, framebuffer_width: int, framebuffer_height: int) -> tuple[int, int, int, int]:
        panel_width = min(max(int(self.config.panel_width), 0), max(int(framebuffer_width) - 1, 0))
        ai_panel_height = 0
        if self.config.ai_panel_enabled:
            ai_panel_height = min(max(int(self.config.ai_panel_height), 0), max(int(framebuffer_height) - 1, 0))

        render_width = max(int(framebuffer_width) - panel_width, 1)
        render_height = max(int(framebuffer_height) - ai_panel_height, 1)
        render_x = 0
        render_y = ai_panel_height
        return render_x, render_y, render_width, render_height

    def _build_ai_runtime_context(self, active_mesh: TriangleMesh) -> str:
        shading_label = "Phong" if self.renderer.state.shading == ShadingMode.PHONG else "Gouraud"
        draw_mode_label = "实体" if self.renderer.state.draw_mode == DrawMode.SOLID else "线框"
        scheme_label = subdivision_scheme_label(self.renderer.state.subdivision_scheme)
        return (
            "当前程序状态："
            f"模型名称={self.mesh_levels.name}；"
            f"当前细分算法={scheme_label}；"
            f"当前细分层级={self.current_level}/{self.mesh_levels.max_level}；"
            f"当前网格顶点数={active_mesh.vertex_count}；"
            f"当前网格面数={active_mesh.face_count}；"
            f"当前着色模式={shading_label}；"
            f"当前绘制模式={draw_mode_label}；"
            f"顶点法线显示={'开启' if self.renderer.state.show_normals else '关闭'}；"
            f"主光源动画={'开启' if self.renderer.state.animate_light else '关闭'}。"
        )

    def _make_recompute_request(self) -> SubdivisionRecomputeRequest:
        return SubdivisionRecomputeRequest(
            scheme=normalize_subdivision_scheme(self.renderer.state.subdivision_scheme),
            use_fast=bool(self.renderer.state.use_vectorization),
            max_level=self.max_subdivision_level,
        )

    def _submit_recompute(self, request: SubdivisionRecomputeRequest) -> None:
        self.renderer.state.auto_play_enabled = False
        if self._recompute_future is not None:
            self._queued_recompute_request = request
            self.renderer.state.subdivision_status_text = "已有细分重算进行中，已记录新的请求。"
            return

        self.renderer.state.recompute_in_progress = True
        self.renderer.state.subdivision_status_text = (
            f"正在重算：{subdivision_scheme_label(request.scheme)}，最高层级 {request.max_level}。"
        )
        self._recompute_future = self._subdivision_executor.submit(
            _compute_levels_in_background,
            self.base_mesh.copy(),
            request,
        )

    def _poll_recompute_result(self) -> None:
        if self._recompute_future is None or not self._recompute_future.done():
            return

        try:
            result = self._recompute_future.result()
            self.mesh_levels.levels = result.levels
            self.current_level = min(self.current_level, self.mesh_levels.max_level)
            self.renderer.state.last_compute_time_ms = result.compute_time_ms
            self.renderer.state.subdivision_status_text = (
                f"{subdivision_scheme_label(result.request.scheme)} 重算完成，"
                f"耗时 {result.compute_time_ms:.2f} ms。"
            )
            self.renderer.set_mesh(self.mesh_levels.at(self.current_level))
        except Exception as exc:
            self.renderer.state.subdivision_status_text = f"细分重算失败：{exc}"
        finally:
            self.renderer.state.recompute_in_progress = False
            self._recompute_future = None

        if self._queued_recompute_request is not None:
            queued = self._queued_recompute_request
            self._queued_recompute_request = None
            self._submit_recompute(queued)

    def _advance_auto_play(self) -> None:
        if self.renderer.state.recompute_in_progress or not self.renderer.state.auto_play_enabled:
            return
        if self.current_level >= self.mesh_levels.max_level:
            self.renderer.state.auto_play_enabled = False
            return

        now = float(glfw.get_time())
        if now - self._last_auto_play_step_time < float(self.renderer.state.auto_play_interval_seconds):
            return

        self._last_auto_play_step_time = now
        self.current_level = min(self.current_level + 1, self.mesh_levels.max_level)
        self.renderer.set_mesh(self.mesh_levels.at(self.current_level))
        if self.current_level >= self.mesh_levels.max_level:
            self.renderer.state.auto_play_enabled = False

    def _toggle_auto_play(self) -> None:
        if self.renderer.state.auto_play_enabled:
            self.renderer.state.auto_play_enabled = False
            return
        if self.renderer.state.recompute_in_progress:
            return
        if self.current_level >= self.mesh_levels.max_level:
            self.current_level = 0
            self.renderer.set_mesh(self.mesh_levels.at(self.current_level))
        self.renderer.state.auto_play_enabled = True
        self._last_auto_play_step_time = float(glfw.get_time())

    def reset_scene(self) -> None:
        defaults = self.config
        old_use_vec = self.renderer.state.use_vectorization
        old_scheme = self.renderer.state.subdivision_scheme

        self.renderer.state.draw_mode = DrawMode(defaults.default_draw_mode)
        self.renderer.state.shading = ShadingMode(defaults.default_shading)
        self.renderer.state.subdivision_scheme = defaults.default_subdivision_scheme
        self.renderer.state.use_vectorization = defaults.default_use_vectorization
        self.renderer.state.show_normals = False
        self.renderer.state.animate_light = False
        self.renderer.state.auto_play_enabled = defaults.default_auto_play_enabled
        self.renderer.state.auto_play_interval_seconds = defaults.default_auto_play_interval_seconds
        self.renderer.state.background_color = np.array(defaults.background_color, dtype=np.float32)
        self.renderer.state.lighting.ambient_strength = defaults.ambient_strength
        self.renderer.state.lighting.diffuse_strength = defaults.diffuse_strength
        self.renderer.state.lighting.specular_strength = defaults.specular_strength
        self.renderer.state.lighting.shininess = defaults.shininess
        self.renderer.state.lighting.light_position = np.array(defaults.light_position, dtype=np.float32)
        self.renderer.state.lighting.light_color = np.array(defaults.light_color, dtype=np.float32)
        self.renderer.state.lighting.fill_light_position = np.array(defaults.fill_light_position, dtype=np.float32)
        self.renderer.state.lighting.fill_light_color = np.array(defaults.fill_light_color, dtype=np.float32)
        self.renderer.state.lighting.fill_light_strength = defaults.fill_light_strength
        self.renderer.state.lighting.object_color = np.array(defaults.object_color, dtype=np.float32)

        self.camera.reset(
            target=np.zeros(3, dtype=np.float32),
            distance=defaults.camera_distance,
            azimuth=defaults.camera_azimuth,
            elevation=defaults.camera_elevation,
        )
        self.current_level = 0
        self.renderer.set_mesh(self.mesh_levels.at(self.current_level))
        self._last_auto_play_step_time = float(glfw.get_time())

        if (
            old_use_vec != self.renderer.state.use_vectorization
            or old_scheme != self.renderer.state.subdivision_scheme
        ):
            self._submit_recompute(self._make_recompute_request())
        else:
            self.renderer.state.subdivision_status_text = "场景已重置为默认状态。"

    def run(self) -> None:
        while self._running and not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.panel.process_inputs()
            self.ai_assistant.poll()
            self._poll_recompute_result()
            self._advance_auto_play()

            width, height = glfw.get_framebuffer_size(self.window)
            if (width, height) != self._last_framebuffer_size:
                self.renderer.resize(width, height)
                self._last_framebuffer_size = (width, height)

            if self.renderer.state.animate_light:
                time_val = glfw.get_time()
                radius = 3.0
                self.renderer.state.lighting.light_position[0] = radius * math.cos(time_val)
                self.renderer.state.lighting.light_position[2] = radius * math.sin(time_val)

            self.input.update(
                capture_mouse=self.panel.wants_mouse_capture(),
                capture_keyboard=self.panel.wants_keyboard_capture(),
                scroll_delta=self.panel.scroll_delta(),
            )

            if not self.panel.is_open():
                glfw.set_window_should_close(self.window, True)

            self.panel.begin_frame()
            active_mesh = self.mesh_levels.at(self.current_level)

            old_use_vec = self.renderer.state.use_vectorization
            old_scheme = self.renderer.state.subdivision_scheme

            ai_view_state = self.ai_assistant.view_state
            selected_level, submitted_question, auto_play_toggle_requested, reset_scene_requested = self.panel.draw(
                self.renderer.state,
                self.camera,
                current_level=self.current_level,
                max_level=self.mesh_levels.max_level,
                mesh_name=self.mesh_levels.name,
                vertex_count=active_mesh.vertex_count,
                face_count=active_mesh.face_count,
                window_width=width,
                window_height=height,
                ai_enabled=ai_view_state.enabled,
                ai_available=ai_view_state.available,
                ai_busy=ai_view_state.busy,
                ai_status=ai_view_state.status_text,
                ai_last_question=ai_view_state.last_question,
                ai_last_answer=ai_view_state.last_answer,
            )

            if submitted_question:
                self.ai_assistant.submit(submitted_question, self._build_ai_runtime_context(active_mesh))

            if auto_play_toggle_requested:
                self._toggle_auto_play()

            if reset_scene_requested:
                self.reset_scene()

            if (
                old_use_vec != self.renderer.state.use_vectorization
                or old_scheme != self.renderer.state.subdivision_scheme
            ):
                self._submit_recompute(self._make_recompute_request())

            if selected_level != self.current_level and not reset_scene_requested:
                self.renderer.state.auto_play_enabled = False
                self._last_auto_play_step_time = float(glfw.get_time())
                self.current_level = selected_level
                self.renderer.set_mesh(self.mesh_levels.at(self.current_level))

            render_x, render_y, render_width, render_height = self._render_region(width, height)
            self.renderer.draw(
                self.camera,
                width,
                height,
                viewport_x=render_x,
                viewport_y=render_y,
                viewport_width=render_width,
                viewport_height=render_height,
            )
            self.panel.render()
            glfw.swap_buffers(self.window)

        self.shutdown()

    def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self.panel.close()
        except Exception:
            pass
        self.ai_assistant.shutdown()
        if self._recompute_future is not None:
            self._recompute_future.cancel()
            self._recompute_future = None
        self._subdivision_executor.shutdown(wait=False, cancel_futures=True)
        self.renderer.cleanup()
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()


def main(argv: list[str] | None = None) -> int:
    defaults = AppConfig()

    parser = ArgumentParser(description="Run the triangle mesh subdivision viewer.")
    parser.add_argument("--mesh", type=Path, default=None, help="Optional OBJ file to load")
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--title", type=str, default=defaults.window_title)
    parser.add_argument(
        "--max-level",
        type=int,
        default=defaults.max_subdivision_level,
        help="Maximum precomputed subdivision level",
    )
    args = parser.parse_args(argv)

    mesh_path = resolve_mesh_path(args.mesh, defaults.mesh_path)
    scheme = normalize_subdivision_scheme(defaults.default_subdivision_scheme)
    mesh_levels, init_time = _load_mesh_levels(
        mesh_path,
        args.max_level,
        scheme=scheme,
        use_fast=defaults.default_use_vectorization,
    )

    app = GraphicsApp(mesh_levels=mesh_levels, width=args.width, height=args.height, title=args.title)
    app.renderer.state.last_compute_time_ms = init_time
    app.renderer.state.subdivision_scheme = scheme.value
    app.renderer.state.subdivision_status_text = (
        f"{subdivision_scheme_label(scheme)} 初始生成完成，耗时 {init_time:.2f} ms。"
    )

    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
