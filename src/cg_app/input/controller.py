from __future__ import annotations

from dataclasses import dataclass

import glfw

from ..rendering.camera import OrbitCamera
from ..rendering.state import DrawMode, RenderState, ShadingMode


@dataclass
class InputBindings:
    orbit_button: int = glfw.MOUSE_BUTTON_LEFT
    pan_button: int = glfw.MOUSE_BUTTON_RIGHT


class InputController:
    def __init__(
        self,
        window: int,
        camera: OrbitCamera,
        state: RenderState,
        bindings: InputBindings | None = None,
    ) -> None:
        self.window = window
        self.camera = camera
        self.state = state
        self.bindings = bindings or InputBindings()
        self._last_cursor: tuple[float, float] | None = None
        self._orbiting = False
        self._panning = False

        self._key_states: dict[int, bool] = {
            glfw.KEY_ESCAPE: False,
            glfw.KEY_1: False,
            glfw.KEY_2: False,
            glfw.KEY_G: False,
            glfw.KEY_P: False,
            glfw.KEY_R: False,
        }

    def update(
        self,
        capture_mouse: bool = False,
        capture_keyboard: bool = False,
        scroll_delta: float = 0.0,
    ) -> None:
        self._update_mouse(capture_mouse)
        self._update_keyboard(capture_keyboard)
        if not capture_mouse and scroll_delta != 0.0:
            self.camera.zoom(scroll_delta)

    def _update_mouse(self, capture_mouse: bool) -> None:
        orbit_down = glfw.get_mouse_button(self.window, self.bindings.orbit_button) == glfw.PRESS
        pan_down = glfw.get_mouse_button(self.window, self.bindings.pan_button) == glfw.PRESS
        xpos, ypos = glfw.get_cursor_pos(self.window)

        if capture_mouse:
            self._orbiting = orbit_down
            self._panning = pan_down
            self._last_cursor = (xpos, ypos)
            return

        if orbit_down and not self._orbiting:
            self._orbiting = True
            self._last_cursor = (xpos, ypos)
        elif not orbit_down and self._orbiting:
            self._orbiting = False
            self._last_cursor = None

        if pan_down and not self._panning:
            self._panning = True
            self._last_cursor = (xpos, ypos)
        elif not pan_down and self._panning:
            self._panning = False
            self._last_cursor = None

        if self._last_cursor is None:
            self._last_cursor = (xpos, ypos)
            return

        dx = xpos - self._last_cursor[0]
        dy = ypos - self._last_cursor[1]
        self._last_cursor = (xpos, ypos)

        if self._orbiting:
            self.camera.rotate(dx, dy)
        elif self._panning:
            self.camera.pan(dx, -dy)

    def _update_keyboard(self, capture_keyboard: bool) -> None:
        current = {
            key: glfw.get_key(self.window, key) == glfw.PRESS
            for key in self._key_states
        }

        if capture_keyboard:
            self._key_states.update(current)
            return

        for key, pressed in current.items():
            if pressed and not self._key_states[key]:
                if key == glfw.KEY_ESCAPE:
                    glfw.set_window_should_close(self.window, True)
                elif key == glfw.KEY_1:
                    self.state.draw_mode = DrawMode.WIREFRAME
                elif key == glfw.KEY_2:
                    self.state.draw_mode = DrawMode.SOLID
                elif key == glfw.KEY_G:
                    self.state.shading = ShadingMode.GOURAUD
                elif key == glfw.KEY_P:
                    self.state.shading = ShadingMode.PHONG
                elif key == glfw.KEY_R:
                    self.camera.reset()

        self._key_states.update(current)
