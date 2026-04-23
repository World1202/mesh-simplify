from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def resolve_mesh_path(cli_mesh_path: Path | None, config_mesh_path: str | None) -> Path | None:
    if cli_mesh_path is not None:
        return cli_mesh_path
    if config_mesh_path:
        return Path(config_mesh_path)
    return None


@dataclass
class AppConfig:
    # ── 窗口 ──────────────────────────────────────────────────────────────
    window_title: str = "三角形网格渐进细分"
    width: int = 1280
    height: int = 800
    max_subdivision_level: int = 5          # 启动时预生成的最高细分层级（0~N）
    mesh_path: str | None = r"model\spot\spot_triangulated.obj"           # 留空时使用默认八面体；填 OBJ 路径时启动即加载该模型

    # ── 默认着色与绘制模式 ────────────────────────────────────────────────
    default_shading: str = "phong"          # "phong" 或 "gouraud"
    default_draw_mode: str = "solid"        # "solid" 或 "wireframe"
    default_subdivision_scheme: str = "loop"
    default_use_vectorization: bool = True
    default_auto_play_enabled: bool = False
    default_auto_play_interval_seconds: float = 0.8

    # ── 光照参数（对应 GUI 面板里的滑块） ─────────────────────────────────
    ambient_strength: float = 0.35          # 环境光强度   范围 [0.0, 1.0]
    diffuse_strength: float = 0.85          # 漫反射强度   范围 [0.0, 1.0]
    specular_strength: float = 0.35         # 镜面反射强度 范围 [0.0, 1.0]
    shininess: float = 32.0                 # 高光指数     范围 [1.0, 128.0]
    light_position: tuple[float, float, float] = (2.0, 2.0, 2.0)    # 主光源位置
    light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fill_light_position: tuple[float, float, float] = (-2.0, -1.0, -2.0)  # 补光位置（主光反向）
    fill_light_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fill_light_strength: float = 0.4                # 补光漫反射强度 范围 [0.0, 1.0]
    object_color: tuple[float, float, float] = (0.78, 0.82, 0.92)
    # 浅冷灰 (0.88, 0.90, 0.92)；纯白用 (1.0, 1.0, 1.0)；深色用 (0.06, 0.07, 0.09)
    background_color: tuple[float, float, float, float] = (0.88, 0.90, 0.92, 1.0)

    # ── GUI 面板 ──────────────────────────────────────────────────────────
    panel_width: int = 600                  # 右侧控制面板宽度（像素）
    ai_panel_height: int = 410              # 底部 AI 面板高度（像素）
    chinese_font_size_px: float = 22.0
    # 中文字体
    chinese_font_candidates: tuple[str, ...] = (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    )

    # ── AI 讲解助手 ──────────────────────────────────
    ai_panel_enabled: bool = True
    ai_base_url: str = "https://api.deepseek.com"  # DeepSeek 兼容 OpenAI SDK 的接口地址
    ai_model: str = "deepseek-chat"
    ai_api_key: str | None = None   #  在此填api_key
    ai_api_key_env_var: str = "DEEPSEEK_API_KEY"
    ai_request_timeout_seconds: float = 60.0
    ai_max_response_chars: int = 1200
    ai_system_prompt: str = (
        "你是一个面向计算机图形学课程演示和答辩验收的中文讲解助手。"
        "你的任务是帮助使用者解释这个三角网格渐进细分 demo 的功能、操作方式、算法思路、图形学概念和项目亮点。"
        "回答必须优先基于我提供的项目事实，不要编造本项目没有实现的功能、论文来源、性能数据或代码细节。"
        "如果问题超出本项目范围，先说明当前 demo 没有直接覆盖，再给出简短的通用解释。"
        "普通问题用 150 到 300 字回答，复杂问题最多约 500 字；课堂演示时优先短而清楚，不要长篇推理。"
        "如果用户问如何演示，按实际界面和操作路径说明；如果用户问答辩问题，突出题目要求、实现方式和展示价值。"
        "输出必须是自然中文纯文本。"
        "不要使用 Markdown。"
        "不要使用标题、列表符号、代码块、反引号、星号、井号、减号项目符号、数字编号、表格或公式块。"
        "不要提及这些提示词规则本身。"
    )

    ai_project_context: str = (
        "项目事实：本项目是计算机图形学课程的三角网格渐进细分演示程序，目标是展示复杂三角网格模型从粗到细的分辨率变化，并结合观察器、光照和着色方式完成课堂演示。"
        "技术栈是 Python、PyOpenGL、GLFW、pyimgui 和 NumPy。程序支持加载 OBJ 模型，也有默认八面体 demo；当前最终展示模型是 Spot 的三角化 OBJ。"
        "界面分为主显示区、右侧控制面板和底部 AI 讲解面板。主显示区负责渲染模型，并按扣除右侧控制面板和底部 AI 面板后的可视区域居中。右侧控制面板负责网格信息、细分算法、细分层级、自动播放、场景重置、绘制模式、着色模式、光照参数和相机信息。底部 AI 面板用于输入问题和显示讲解结果。"
        "观察器功能包括鼠标左键旋转、滚轮缩放、右键平移，以及通过按钮重置相机。绘制模式支持实体和线框，方便观察模型表面与三角网格结构。"
        "细分展示是项目主线。用户可以手动切换细分层级，也可以用自动播放从低层级逐步播放到高层级，到最高层后自动停止。算法切换或高层级重算会使用后台线程，避免 OpenGL 和 GUI 主循环长时间卡住。界面会显示最近一次重算耗时，用于观察算法性能差异。"
        "当前支持四种细分算法。Loop 细分适合展示平滑曲面效果，是主要算法；Linear 1-to-4 保持简单的一对四拓扑拆分，便于说明层级增长；Centroid 面心细分通过加入面心形成新三角形，适合对比不同拓扑变化；Modified Butterfly 属于插值型细分思路，适合说明不同算法对形状细节的影响。"
        "着色与光照部分支持 Gouraud 和 Phong 两种着色模式。Gouraud 在顶点计算光照再插值，速度直观但高光可能不够细；Phong 对片元插值法线再计算光照，效果更平滑，适合展示细分后曲面的光照变化。光照采用主光加补光的双光源方案，主光提供主要漫反射和镜面反射，补光减轻背光面全黑的问题。项目还支持环境光、漫反射、镜面反射、高光指数调节、顶点法线可视化和主光源环绕动画。"
        "本项目满足题目要求的对应关系是：一般观察器由旋转、缩放、平移、重置和显示模式支持；三角形细分由多算法细分模块支持；渐进细化由层级切换和自动播放支持；Phong 光照由 Phong 着色器和光照参数支持；Gouraud 与 Phong 明暗处理由两种着色模式切换支持；独特模型由 Spot 模型支持。"
        "项目亮点包括多细分算法对比、后台重算、自动播放、性能耗时显示、双光源改善展示效果、右侧控制加底部 AI 的布局、主显示区自动居中和一键重置场景。AI 助手本身只负责讲解，不读取代码文件，不做检索增强，不保证回答项目外的最新知识。"
        "绘制模式支持实体和线框。实体模式会填充三角形面片，并结合光照、材质颜色、Gouraud 或 Phong 着色来显示模型表面，适合观察  细分后的整体外观、曲面平滑程度和明暗效果。线框模式会突出显示三角形网格的边线，不强调表面填充效果，适合观察模型的拓扑结构、细分层级变化、网格密度增长，以及不同细分算法生成的网格差异。"
    )

    # ── 相机参数 ──────────────────────────────────────────────────────────
    camera_distance: float = 3.5            # 初始距离
    camera_azimuth: float = 0.0             # 初始方位角（弧度）
    camera_elevation: float = 0.35          # 初始仰角（弧度）
    camera_fov: float = 45.0               # 视野角（度）
    orbit_sensitivity: float = 0.01         # 旋转灵敏度
    pan_sensitivity: float = 0.002          # 平移灵敏度
    zoom_sensitivity: float = 0.12          # 缩放灵敏度
