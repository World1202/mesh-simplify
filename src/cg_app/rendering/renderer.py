from __future__ import annotations

from pathlib import Path
from typing import Any

import ctypes
import numpy as np
from OpenGL import GL as gl
from OpenGL.GL import shaders  # 新增导入，用于编译法线着色器

from .camera import OrbitCamera
from .mesh import Mesh
from .shaders import ShaderProgram, load_shader_program
from .state import DrawMode, RenderState, ShadingMode

# 新增：用于画法向量线的简易内置着色器
_LINE_VERT = """#version 330 core
layout (location = 0) in vec3 a_pos;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_pos, 1.0);
}
"""
_LINE_FRAG = """#version 330 core
out vec4 FragColor;
uniform vec3 u_color;
void main() {
    FragColor = vec4(u_color, 1.0);
}
"""


class OpenGLRenderer:
    def __init__(self, state: RenderState | None = None) -> None:
        self.state = state or RenderState()
        self.mesh: Mesh | None = None
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._index_count = 0

        # 新增法线可视化的缓冲标识
        self._normal_vao = 0
        self._normal_vbo = 0
        self._normal_count = 0
        self._line_program = None

        self._programs: dict[ShadingMode, ShaderProgram] = {}
        self._asset_root = Path(__file__).resolve().parents[3] / "assets" / "shaders"
        self._model_matrix = np.eye(4, dtype=np.float32)
        self._initialized = False

    def initialize(self, shader_root: str | Path | None = None) -> None:
        if shader_root is not None:
            self._asset_root = Path(shader_root)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(*self.state.background_color.tolist())
        gl.glEnable(gl.GL_MULTISAMPLE) 

        # 优化：开启背面剔除，只画朝向相机的面，大幅提升高细分级别下的性能
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glFrontFace(gl.GL_CCW)  # OBJ 默认是逆时针排列为正面

        self._programs[ShadingMode.GOURAUD] = load_shader_program(
            self._asset_root / "gouraud.vert",
            self._asset_root / "gouraud.frag",
        )
        self._programs[ShadingMode.PHONG] = load_shader_program(
            self._asset_root / "phong.vert",
            self._asset_root / "phong.frag",
        )

        #编译法线专用着色器
        self._line_program = shaders.compileProgram(
            shaders.compileShader(_LINE_VERT, gl.GL_VERTEX_SHADER),
            shaders.compileShader(_LINE_FRAG, gl.GL_FRAGMENT_SHADER)
        )

        self._initialized = True
        if self.mesh is not None:
            self._upload_mesh(self.mesh)

    def set_model_matrix(self, model_matrix: np.ndarray) -> None:
        self._model_matrix = np.asarray(model_matrix, dtype=np.float32).reshape(4, 4)

    def set_mesh(self, mesh: Mesh | Any) -> None:
        self.mesh = self._coerce_mesh(mesh)
        if self._initialized:
            self._upload_mesh(self.mesh)

    def resize(self, width: int, height: int) -> None:
        gl.glViewport(0, 0, max(int(width), 1), max(int(height), 1))

    def draw(
        self,
        camera: OrbitCamera,
        width: int,
        height: int,
        viewport_x: int = 0,
        viewport_y: int = 0,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            return
        draw_width = max(int(viewport_width if viewport_width is not None else width), 1)
        draw_height = max(int(viewport_height if viewport_height is not None else height), 1)
        draw_x = max(int(viewport_x), 0)
        draw_y = max(int(viewport_y), 0)

        gl.glViewport(draw_x, draw_y, draw_width, draw_height)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(draw_x, draw_y, draw_width, draw_height)
        gl.glClearColor(*self.state.background_color.tolist())
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        if self.mesh is None or self._vao == 0:
            return

        program = self._programs[self.state.shading]
        program.use()

        view = camera.view_matrix()
        projection = camera.projection_matrix(draw_width / float(draw_height))
        normal_matrix = np.linalg.inv(self._model_matrix[:3, :3]).T.astype(np.float32)

        program.set_mat4("u_model", self._model_matrix)
        program.set_mat4("u_view", view)
        program.set_mat4("u_projection", projection)
        program.set_mat3("u_normal_matrix", normal_matrix)
        program.set_vec3("u_camera_position", camera.eye())
        program.set_vec3("u_light_position", self.state.lighting.light_position)
        program.set_vec3("u_light_color", self.state.lighting.light_color)
        program.set_vec3("u_fill_light_position", self.state.lighting.fill_light_position)
        program.set_vec3("u_fill_light_color", self.state.lighting.fill_light_color)
        program.set_float("u_fill_light_strength", self.state.lighting.fill_light_strength)
        program.set_vec3("u_object_color", self.state.lighting.object_color)
        program.set_float("u_ambient_strength", self.state.lighting.ambient_strength)
        program.set_float("u_diffuse_strength", self.state.lighting.diffuse_strength)
        program.set_float("u_specular_strength", self.state.lighting.specular_strength)
        program.set_float("u_shininess", self.state.lighting.shininess)

        gl.glBindVertexArray(self._vao)
        # 如果是线框模式，最好关闭剔除，否则看不到背面的线框
        if self.state.draw_mode == DrawMode.WIREFRAME:
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        gl.glDrawElements(gl.GL_TRIANGLES, self._index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        #绘制法线可视化线段
        if getattr(self.state, "show_normals", False) and self._normal_vao > 0:
            gl.glUseProgram(self._line_program)
            loc_model = gl.glGetUniformLocation(self._line_program, "u_model")
            loc_view = gl.glGetUniformLocation(self._line_program, "u_view")
            loc_proj = gl.glGetUniformLocation(self._line_program, "u_projection")
            loc_color = gl.glGetUniformLocation(self._line_program, "u_color")

            gl.glUniformMatrix4fv(loc_model, 1, gl.GL_FALSE, self._model_matrix.T.astype(np.float32))
            gl.glUniformMatrix4fv(loc_view, 1, gl.GL_FALSE, view.T.astype(np.float32))
            gl.glUniformMatrix4fv(loc_proj, 1, gl.GL_FALSE, projection.T.astype(np.float32))
            gl.glUniform3f(loc_color, 0.0, 0.8, 1.0)  # 青蓝色法线段

            gl.glBindVertexArray(self._normal_vao)
            gl.glDrawArrays(gl.GL_LINES, 0, self._normal_count)
            gl.glBindVertexArray(0)

    def cleanup(self) -> None:
        if self._vao:
            gl.glDeleteVertexArrays(1, [self._vao])
            self._vao = 0
        if self._vbo:
            gl.glDeleteBuffers(1, [self._vbo])
            self._vbo = 0
        if self._ebo:
            gl.glDeleteBuffers(1, [self._ebo])
            self._ebo = 0
        # 清理法线缓冲
        if self._normal_vao:
            gl.glDeleteVertexArrays(1, [self._normal_vao])
            self._normal_vao = 0
        if self._normal_vbo:
            gl.glDeleteBuffers(1, [self._normal_vbo])
            self._normal_vbo = 0

        for program in self._programs.values():
            program.delete()
        if self._line_program:
            gl.glDeleteProgram(self._line_program)
            self._line_program = None

        self._programs.clear()
        self._initialized = False

    def _upload_mesh(self, mesh: Mesh) -> None:
        if self._vao:
            gl.glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            gl.glDeleteBuffers(1, [self._vbo])
        if self._ebo:
            gl.glDeleteBuffers(1, [self._ebo])

        self._vao = gl.glGenVertexArrays(1)
        self._vbo = gl.glGenBuffers(1)
        self._ebo = gl.glGenBuffers(1)
        self._index_count = int(mesh.indices.size)

        vertex_data = np.hstack(
            [
                np.asarray(mesh.positions, dtype=np.float32),
                np.asarray(mesh.normals, dtype=np.float32),
            ]
        ).astype(np.float32, copy=False)
        index_data = np.asarray(mesh.indices, dtype=np.uint32).reshape(-1)

        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, gl.GL_STATIC_DRAW)

        stride = 6 * vertex_data.dtype.itemsize
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(
            1,
            3,
            gl.GL_FLOAT,
            gl.GL_FALSE,
            stride,
            ctypes.c_void_p(3 * vertex_data.dtype.itemsize),
        )
        gl.glBindVertexArray(0)

        # 构建法线可视化数据（为每个顶点生成一个线段）
        pts = np.asarray(mesh.positions, dtype=np.float32)
        nrms = np.asarray(mesh.normals, dtype=np.float32)
        if len(pts) > 0:
            # 动态计算法线长度：取模型包围盒尺寸的 3% 作为法线可视化长度
            bbox_size = np.max(pts.max(axis=0) - pts.min(axis=0))
            n_scale = bbox_size * 0.03 if bbox_size > 0 else 0.1

            lines = np.empty((len(pts) * 2, 3), dtype=np.float32)
            lines[0::2] = pts  # 线段起点：顶点位置
            lines[1::2] = pts + nrms * n_scale  # 线段终点：沿着法向量延伸

            if self._normal_vao:
                gl.glDeleteVertexArrays(1, [self._normal_vao])
            if self._normal_vbo:
                gl.glDeleteBuffers(1, [self._normal_vbo])

            self._normal_vao = gl.glGenVertexArrays(1)
            self._normal_vbo = gl.glGenBuffers(1)
            self._normal_count = len(lines)

            gl.glBindVertexArray(self._normal_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._normal_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, lines.nbytes, lines, gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glBindVertexArray(0)

    def _coerce_mesh(self, mesh: Mesh | Any) -> Mesh:
        if isinstance(mesh, Mesh):
            return mesh
        return Mesh.from_triangle_mesh(mesh)
