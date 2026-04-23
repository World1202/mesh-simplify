from __future__ import annotations

from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - optional runtime dependency
    import imgui
    from imgui.integrations.glfw import GlfwRenderer
except Exception:  # pragma: no cover - fallback when pyimgui is unavailable
    imgui = None
    GlfwRenderer = None

from ..mesh import SUBDIVISION_SCHEME_LABELS, SubdivisionScheme, subdivision_scheme_label
from ..rendering.camera import OrbitCamera
from ..rendering.state import DrawMode, RenderState, ShadingMode

_PANEL_W = 400
_AI_PANEL_H = 210

# ── 配色常量 ──────────────────────────────────────────────────────────────
# 分组标题：蓝色系
_C_HDR_BG     = (0.18, 0.52, 0.80, 0.18)
_C_HDR_HOVER  = (0.18, 0.52, 0.80, 0.32)
_C_HDR_ACTIVE = (0.18, 0.52, 0.80, 0.50)

# 主操作按钮：蓝色
_C_BTN_BLUE   = (0.18, 0.52, 0.80, 1.00)
_C_BTN_BLUE_H = (0.26, 0.60, 0.87, 1.00)
_C_BTN_BLUE_A = (0.12, 0.42, 0.68, 1.00)

# 播放按钮：绿色
_C_BTN_GREEN   = (0.15, 0.62, 0.30, 1.00)
_C_BTN_GREEN_H = (0.18, 0.72, 0.36, 1.00)
_C_BTN_GREEN_A = (0.10, 0.52, 0.24, 1.00)

# 暂停按钮：橙色
_C_BTN_PAUSE   = (0.80, 0.45, 0.10, 1.00)
_C_BTN_PAUSE_H = (0.90, 0.53, 0.15, 1.00)
_C_BTN_PAUSE_A = (0.68, 0.37, 0.08, 1.00)

# 重置按钮：灰红色
_C_BTN_RESET   = (0.65, 0.22, 0.18, 1.00)
_C_BTN_RESET_H = (0.75, 0.28, 0.22, 1.00)
_C_BTN_RESET_A = (0.55, 0.18, 0.14, 1.00)


def _load_chinese_font(font_candidates: Sequence[str], size_px: float) -> None:
    if imgui is None:
        return
    io = imgui.get_io()
    for path in font_candidates:
        if Path(path).exists():
            io.fonts.add_font_from_file_ttf(
                path,
                size_px,
                glyph_ranges=io.fonts.get_glyph_ranges_chinese_full(),
            )
            return


def _apply_light_style() -> None:
    if imgui is None:
        return
    imgui.style_colors_light()
    style = imgui.get_style()
    style.window_rounding = 0.0
    style.frame_rounding = 5.0
    style.scrollbar_rounding = 5.0
    style.grab_rounding = 4.0
    style.item_spacing = (8.0, 8.0)
    style.item_inner_spacing = (6.0, 6.0)
    style.frame_padding = (10.0, 6.0)
    style.window_padding = (14.0, 12.0)


def _push_header_colors() -> None:
    """Push blue accent colors for collapsing headers."""
    if imgui is None:
        return
    imgui.push_style_color(imgui.COLOR_HEADER, *_C_HDR_BG)
    imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *_C_HDR_HOVER)
    imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, *_C_HDR_ACTIVE)


def _pop_header_colors() -> None:
    if imgui is None:
        return
    imgui.pop_style_color(3)


def _push_button_colors(color, hover, active) -> None:
    if imgui is None:
        return
    imgui.push_style_color(imgui.COLOR_BUTTON, *color)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *hover)
    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *active)


def _pop_button_colors() -> None:
    if imgui is None:
        return
    imgui.pop_style_color(3)


def _section_header(title: str, expanded: bool, width: float) -> bool:
    if imgui is None:
        return expanded
    _push_header_colors()
    prefix = "[-]" if expanded else "[+]"
    clicked = imgui.button(f"{prefix}  {title}", width=width)
    _pop_header_colors()
    return (not expanded) if clicked else expanded


class NullControlPanel:
    def process_inputs(self) -> None:
        return

    def begin_frame(self) -> None:
        return

    def draw(
        self,
        state: RenderState,
        camera: OrbitCamera | None,
        current_level: int = 0,
        max_level: int = 0,
        mesh_name: str = "",
        vertex_count: int = 0,
        face_count: int = 0,
        window_width: int = 1280,
        window_height: int = 720,
        ai_enabled: bool = False,
        ai_available: bool = False,
        ai_busy: bool = False,
        ai_status: str = "",
        ai_last_question: str = "",
        ai_last_answer: str = "",
    ) -> tuple[int, str | None, bool, bool]:
        return current_level, None, False, False

    def render(self) -> None:
        return

    def wants_mouse_capture(self) -> bool:
        return False

    def wants_keyboard_capture(self) -> bool:
        return False

    def scroll_delta(self) -> float:
        return 0.0

    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        return


class ImGuiControlPanel:
    def __init__(
        self,
        window: int,
        panel_width: int = _PANEL_W,
        ai_panel_height: int = _AI_PANEL_H,
        font_candidates: Sequence[str] = (),
        font_size_px: float = 22.0,
    ) -> None:
        if imgui is None or GlfwRenderer is None:
            raise RuntimeError("pyimgui is not available")
        imgui.create_context()
        _load_chinese_font(font_candidates, font_size_px)
        _apply_light_style()
        self._impl = GlfwRenderer(window)
        self._open = True
        self._panel_width = panel_width
        self._ai_panel_height = ai_panel_height
        self._ai_input_text = ""
        self._sections = {
            "mesh_info": True,
            "subdivision": True,
            "performance": True,
            "playback": True,
            "scene_ops": False,
            "shading": True,
            "lighting": False,
            "camera": False,
        }

    def process_inputs(self) -> None:
        self._impl.process_inputs()

    def begin_frame(self) -> None:
        imgui.new_frame()

    def draw(
        self,
        state: RenderState,
        camera: OrbitCamera | None,
        current_level: int = 0,
        max_level: int = 0,
        mesh_name: str = "",
        vertex_count: int = 0,
        face_count: int = 0,
        window_width: int = 1280,
        window_height: int = 720,
        ai_enabled: bool = False,
        ai_available: bool = False,
        ai_busy: bool = False,
        ai_status: str = "",
        ai_last_question: str = "",
        ai_last_answer: str = "",
    ) -> tuple[int, str | None, bool, bool]:
        selected_level = int(current_level)
        submitted_question: str | None = None
        auto_play_toggle_requested = False
        reset_scene_requested = False

        shading_modes = [ShadingMode.PHONG, ShadingMode.GOURAUD]
        shading_labels = ["Phong（逐片元）", "Gouraud（逐顶点）"]
        draw_modes = [DrawMode.SOLID, DrawMode.WIREFRAME]
        draw_labels = ["实体", "线框"]
        subdivision_schemes = list(SubdivisionScheme)
        subdivision_labels = [SUBDIVISION_SCHEME_LABELS[scheme] for scheme in subdivision_schemes]

        inner_w = self._panel_width - 28  # 控件有效宽度

        # ── 右侧控制面板 ──────────────────────────────────────────────────
        imgui.set_next_window_position(window_width - self._panel_width, 0, condition=imgui.ALWAYS)
        imgui.set_next_window_size(self._panel_width, window_height, condition=imgui.ALWAYS)

        win_flags = (
            imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_TITLE_BAR
        )
        imgui.begin("##panel", flags=win_flags)

        imgui.spacing()
        imgui.push_style_color(imgui.COLOR_TEXT, 0.15, 0.40, 0.75, 1.0)
        imgui.text("  三角形网格渐进细分")
        imgui.pop_style_color()
        imgui.separator()
        imgui.spacing()

        # ── 网格信息 ──────────────────────────────────────────────────────
        self._sections["mesh_info"] = _section_header("网格信息", self._sections["mesh_info"], inner_w)
        if self._sections["mesh_info"]:
            imgui.spacing()
            if mesh_name:
                imgui.text(f"名称：{mesh_name}")
            imgui.text(f"顶点：{vertex_count:,}")
            imgui.text(f"面数：{face_count:,}")
            imgui.spacing()

        # ── 细分控制 ──────────────────────────────────────────────────────
        self._sections["subdivision"] = _section_header("细分控制", self._sections["subdivision"], inner_w)
        if self._sections["subdivision"]:
            imgui.spacing()
            imgui.push_item_width(inner_w)
            scheme_index = subdivision_schemes.index(SubdivisionScheme(state.subdivision_scheme))
            changed, scheme_index = imgui.combo("##scheme", scheme_index, subdivision_labels)
            if changed:
                state.subdivision_scheme = subdivision_schemes[scheme_index].value

            imgui.spacing()
            changed, new_level = imgui.slider_int(
                "##subdiv",
                selected_level,
                0,
                max_level,
                f"层级 %d/{max_level}",
            )
            imgui.pop_item_width()
            if changed:
                selected_level = int(new_level)
            imgui.spacing()

        # ── 算法对比与性能 ────────────────────────────────────────────────
        self._sections["performance"] = _section_header("算法对比与性能", self._sections["performance"], inner_w)
        if self._sections["performance"]:
            imgui.spacing()
            changed, use_vec = imgui.checkbox("开启 NumPy 向量化加速", state.use_vectorization)
            if changed:
                state.use_vectorization = use_vec

            imgui.spacing()
            if state.subdivision_scheme != SubdivisionScheme.LOOP.value:
                imgui.text_disabled("当前算法统一使用通用 CPU 版本")
            elif state.use_vectorization:
                imgui.text_disabled("当前 Loop 使用 NumPy 向量化实现")
            else:
                imgui.text_disabled("当前 Loop 使用 Python 原生实现")

            imgui.spacing()
            if state.recompute_in_progress:
                imgui.text_colored("细分重算中...", 0.85, 0.45, 0.10, 1.0)
            else:
                imgui.text_colored(
                    f"最近计算耗时: {state.last_compute_time_ms:.2f} ms",
                    0.15, 0.62, 0.30, 1.0,
                )

            if state.subdivision_status_text:
                imgui.spacing()
                imgui.text_wrapped(state.subdivision_status_text)
            imgui.spacing()

        # ── 播放控制 ──────────────────────────────────────────────────────
        self._sections["playback"] = _section_header("播放控制", self._sections["playback"], inner_w)
        if self._sections["playback"]:
            imgui.spacing()
            imgui.push_item_width(inner_w)
            changed, interval_seconds = imgui.slider_float(
                "每层间隔（秒）",
                float(state.auto_play_interval_seconds),
                0.2,
                3.0,
                "%.1f",
            )
            imgui.pop_item_width()
            if changed:
                state.auto_play_interval_seconds = float(interval_seconds)

            imgui.spacing()
            if state.recompute_in_progress:
                imgui.text_disabled("细分重算中，自动播放已暂停。")
            elif state.auto_play_enabled:
                imgui.text_colored("自动播放进行中", 0.15, 0.62, 0.30, 1.0)
            else:
                imgui.text_disabled("自动播放已暂停")

            imgui.spacing()
            if state.auto_play_enabled:
                _push_button_colors(_C_BTN_PAUSE, _C_BTN_PAUSE_H, _C_BTN_PAUSE_A)
            else:
                _push_button_colors(_C_BTN_GREEN, _C_BTN_GREEN_H, _C_BTN_GREEN_A)
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
            play_label = "  暂停自动播放  " if state.auto_play_enabled else "  开始自动播放  "
            if imgui.button(play_label, width=inner_w):
                auto_play_toggle_requested = True
            imgui.pop_style_color()
            _pop_button_colors()
            imgui.spacing()

        # ── 场景操作 ──────────────────────────────────────────────────────
        self._sections["scene_ops"] = _section_header("场景操作", self._sections["scene_ops"], inner_w)
        if self._sections["scene_ops"]:
            imgui.spacing()
            _push_button_colors(_C_BTN_RESET, _C_BTN_RESET_H, _C_BTN_RESET_A)
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
            if imgui.button("  一键重置场景  ", width=inner_w):
                reset_scene_requested = True
            imgui.pop_style_color()
            _pop_button_colors()
            imgui.spacing()

        # ── 着色与绘制 ────────────────────────────────────────────────────
        self._sections["shading"] = _section_header("着色与绘制", self._sections["shading"], inner_w)
        if self._sections["shading"]:
            imgui.spacing()
            imgui.push_item_width(inner_w)

            imgui.text_disabled("着色模式")
            shading_index = shading_modes.index(state.shading)
            changed, shading_index = imgui.combo("##shading", shading_index, shading_labels)
            if changed:
                state.shading = shading_modes[shading_index]

            imgui.spacing()
            imgui.text_disabled("绘制模式")
            draw_index = draw_modes.index(state.draw_mode)
            changed, draw_index = imgui.combo("##drawmode", draw_index, draw_labels)
            if changed:
                state.draw_mode = draw_modes[draw_index]

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text_disabled("调试选项")
            changed, new_show_normals = imgui.checkbox(
                "显示顶点法线", bool(getattr(state, "show_normals", False))
            )
            if changed:
                state.show_normals = new_show_normals

            imgui.pop_item_width()
            imgui.spacing()

        # ── 光照参数 ──────────────────────────────────────────────────────
        self._sections["lighting"] = _section_header("光照参数", self._sections["lighting"], inner_w)
        if self._sections["lighting"]:
            imgui.spacing()
            imgui.push_item_width(inner_w)

            imgui.text_disabled("环境光")
            changed, v = imgui.slider_float("##ambient", state.lighting.ambient_strength, 0.0, 1.0, "%.2f")
            if changed:
                state.lighting.ambient_strength = float(v)

            imgui.text_disabled("漫反射")
            changed, v = imgui.slider_float("##diffuse", state.lighting.diffuse_strength, 0.0, 1.0, "%.2f")
            if changed:
                state.lighting.diffuse_strength = float(v)

            imgui.text_disabled("镜面反射")
            changed, v = imgui.slider_float("##specular", state.lighting.specular_strength, 0.0, 1.0, "%.2f")
            if changed:
                state.lighting.specular_strength = float(v)

            imgui.text_disabled("高光指数")
            changed, v = imgui.slider_float("##shininess", state.lighting.shininess, 1.0, 128.0, "%.0f")
            if changed:
                state.lighting.shininess = float(v)

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            imgui.text_disabled("动画选项")
            changed, new_animate = imgui.checkbox(
                "开启主光源环绕动画", bool(getattr(state, "animate_light", False))
            )
            if changed:
                state.animate_light = new_animate

            imgui.pop_item_width()
            imgui.spacing()

        # ── 相机 ──────────────────────────────────────────────────────────
        if camera is not None:
            self._sections["camera"] = _section_header("相机", self._sections["camera"], inner_w)
            if self._sections["camera"]:
                imgui.spacing()
                imgui.text(f"距离：{camera.distance:.2f}")
                imgui.text(f"方位角：{camera.azimuth:.2f}")
                imgui.text(f"仰角：{camera.elevation:.2f}")
                imgui.spacing()
                _push_button_colors(_C_BTN_BLUE, _C_BTN_BLUE_H, _C_BTN_BLUE_A)
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
                if imgui.button("  重置相机  ##btn", width=inner_w):
                    camera.reset()
                imgui.pop_style_color()
                _pop_button_colors()
                imgui.spacing()

        # ── 底部快捷键提示 ────────────────────────────────────────────────
        imgui.set_cursor_pos_y(window_height - 52)
        imgui.separator()
        imgui.spacing()
        imgui.text_disabled("1 线框  2 实体  G/P 切换着色  R 重置")

        imgui.end()

        # ── 底部 AI 面板 ──────────────────────────────────────────────────
        if ai_enabled:
            ai_w = window_width - self._panel_width
            ai_h = self._ai_panel_height
            imgui.set_next_window_position(0, window_height - ai_h, condition=imgui.ALWAYS)
            imgui.set_next_window_size(ai_w, ai_h, condition=imgui.ALWAYS)

            imgui.begin("##ai_panel", flags=win_flags)

            # 面板标题行
            imgui.push_style_color(imgui.COLOR_TEXT, 0.15, 0.40, 0.75, 1.0)
            imgui.text("  AI 讲解助手")
            imgui.pop_style_color()
            imgui.same_line(spacing=16)
            if ai_status:
                if ai_available:
                    imgui.text_colored(ai_status, 0.15, 0.62, 0.30, 1.0)
                else:
                    imgui.text_colored(ai_status, 0.80, 0.40, 0.10, 1.0)
            imgui.separator()
            imgui.spacing()

            # 左右双栏布局：左侧输入，右侧输出
            left_w = ai_w * 0.40 - 28
            right_w = ai_w * 0.60 - 28
            output_h = ai_h - 90  # 输出区高度，留标题行和 padding

            imgui.begin_group()

            # 左列：输入区
            imgui.text_disabled("输入问题")
            imgui.push_item_width(left_w)
            changed, self._ai_input_text = imgui.input_text(
                "##ai_input", self._ai_input_text, 512
            )
            imgui.pop_item_width()
            if changed:
                self._ai_input_text = self._ai_input_text.strip("\n\r")

            imgui.spacing()
            can_submit = ai_available and not ai_busy and bool(self._ai_input_text.strip())
            if ai_busy:
                imgui.text_disabled("请求中，请稍候…")
            else:
                _push_button_colors(_C_BTN_BLUE, _C_BTN_BLUE_H, _C_BTN_BLUE_A)
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 1.0)
                if imgui.button("  发送提问  ##ai", width=left_w):
                    if can_submit:
                        submitted_question = self._ai_input_text.strip()
                        self._ai_input_text = ""
                imgui.pop_style_color()
                _pop_button_colors()

            if ai_last_question:
                imgui.spacing()
                imgui.text_disabled(f"上次提问：{ai_last_question[:40]}{'…' if len(ai_last_question) > 40 else ''}")

            imgui.end_group()

            imgui.same_line(spacing=20)

            # 右列：输出区
            imgui.begin_group()
            imgui.text_disabled("讲解输出")
            imgui.begin_child("##ai_output", right_w, output_h, border=True)
            if ai_last_answer:
                imgui.text_wrapped(ai_last_answer)
            else:
                imgui.text_disabled("AI 的讲解结果会显示在这里。")
            imgui.end_child()
            imgui.end_group()

            imgui.end()

        return selected_level, submitted_question, auto_play_toggle_requested, reset_scene_requested

    def render(self) -> None:
        imgui.render()
        self._impl.render(imgui.get_draw_data())

    def wants_mouse_capture(self) -> bool:
        return bool(imgui.get_io().want_capture_mouse)

    def wants_keyboard_capture(self) -> bool:
        return bool(imgui.get_io().want_capture_keyboard)

    def scroll_delta(self) -> float:
        return float(getattr(imgui.get_io(), "mouse_wheel", 0.0))

    def is_open(self) -> bool:
        return self._open

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        try:
            self._impl.shutdown()
        except Exception:
            pass


def create_control_panel(
    window: int,
    panel_width: int = _PANEL_W,
    ai_panel_height: int = _AI_PANEL_H,
    font_candidates: Sequence[str] = (),
    font_size_px: float = 22.0,
):
    if imgui is None or GlfwRenderer is None:
        return NullControlPanel()
    try:
        return ImGuiControlPanel(
            window,
            panel_width=panel_width,
            ai_panel_height=ai_panel_height,
            font_candidates=font_candidates,
            font_size_px=font_size_px,
        )
    except Exception:
        return NullControlPanel()
