from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OpenGL import GL as gl


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _compile_shader(source: str, shader_type: int, label: str) -> int:
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    success = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if not success:
        log = gl.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        gl.glDeleteShader(shader)
        raise RuntimeError(f"Failed to compile {label} shader:\n{log}")
    return shader


@dataclass
class ShaderProgram:
    program: int
    _locations: dict[str, int] = field(default_factory=dict)

    def use(self) -> None:
        gl.glUseProgram(self.program)

    def location(self, name: str) -> int:
        if name not in self._locations:
            self._locations[name] = gl.glGetUniformLocation(self.program, name)
        return self._locations[name]

    def set_int(self, name: str, value: int) -> None:
        gl.glUniform1i(self.location(name), int(value))

    def set_float(self, name: str, value: float) -> None:
        gl.glUniform1f(self.location(name), float(value))

    def set_vec3(self, name: str, value: np.ndarray | tuple[float, float, float]) -> None:
        vec = np.asarray(value, dtype=np.float32).reshape(3)
        gl.glUniform3f(self.location(name), float(vec[0]), float(vec[1]), float(vec[2]))

    def set_mat4(self, name: str, value: np.ndarray) -> None:
        mat = np.asarray(value, dtype=np.float32)
        gl.glUniformMatrix4fv(self.location(name), 1, gl.GL_TRUE, mat)

    def set_mat3(self, name: str, value: np.ndarray) -> None:
        mat = np.asarray(value, dtype=np.float32)
        gl.glUniformMatrix3fv(self.location(name), 1, gl.GL_TRUE, mat)

    def delete(self) -> None:
        if self.program:
            gl.glDeleteProgram(self.program)
            self.program = 0


def load_shader_program(vertex_path: str | Path, fragment_path: str | Path) -> ShaderProgram:
    vertex_shader = _compile_shader(_read_text(vertex_path), gl.GL_VERTEX_SHADER, "vertex")
    fragment_shader = _compile_shader(_read_text(fragment_path), gl.GL_FRAGMENT_SHADER, "fragment")
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    linked = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    gl.glDeleteShader(vertex_shader)
    gl.glDeleteShader(fragment_shader)
    if not linked:
        log = gl.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        gl.glDeleteProgram(program)
        raise RuntimeError(f"Failed to link shader program:\n{log}")
    return ShaderProgram(program=program)
