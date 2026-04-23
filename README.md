# 三角形网格渐进细分演示程序

这是一个用于计算机图形学课程展示的三角网格渐进细分程序。项目使用 Python、PyOpenGL、GLFW、pyimgui 和 NumPy 实现，支持加载 OBJ 模型，并通过 GUI 展示网格从粗到细的细分过程。

## 功能简介

- 加载并显示三角网格模型，默认模型为 `model/spot/spot_triangulated.obj`
- 支持鼠标旋转、滚轮缩放、右键平移和相机重置
- 支持实体模式和线框模式切换
- 支持 Gouraud 着色和 Phong 着色切换
- 支持环境光、漫反射、镜面反射等光照参数调节
- 支持多种细分算法：Loop、Linear 1-to-4、Centroid、Modified Butterfly
- 支持手动切换细分层级和自动播放渐进细分过程
- 支持底部 AI 讲解助手，用于解释项目功能和相关图形学概念

## 环境配置

推荐使用 Conda 创建环境：

```powershell
conda env create -f environment.yml
conda activate mesh-simplify
```

`environment.yml` 会安装主要依赖，包括：

- Python 3.11
- NumPy
- PyOpenGL
- GLFW
- pyimgui
- OpenAI Python SDK

## 配置 API Key

AI 讲解助手需要 API Key。普通模型显示、细分、着色和交互功能不需要 API Key；如果不配置，程序仍可运行，但 AI 面板无法正常请求回答。

推荐使用环境变量配置：

```powershell
$env:DEEPSEEK_API_KEY="你的APIKey"   #deepseek api
python main.py
```

也可以在 `src/cg_app/config.py` 中填写 `ai_api_key`。

## 运行程序

使用配置文件中的默认模型运行：

```powershell
conda activate mesh-simplify
python main.py
```

临时指定其他 OBJ 模型：

```powershell
python main.py --mesh path\to\model.obj --max-level 4
```

如果想修改默认模型路径，可以编辑 `src/cg_app/config.py`：

```python
mesh_path: str | None = r"model\spot\spot_triangulated.obj"
```

如果将 `mesh_path` 设置为 `None`，程序会使用代码内置的默认八面体模型。

## 基本操作

- 鼠标左键拖动：旋转模型
- 鼠标滚轮：缩放模型
- 鼠标右键拖动：平移模型
- `1`：切换线框模式
- `2`：切换实体模式
- `G`：切换到 Gouraud 着色
- `P`：切换到 Phong 着色
- `R`：重置相机
- `ESC`：退出程序

## 主要配置

常用参数集中在 `src/cg_app/config.py` 中，包括：

- 默认模型路径
- 最大细分层级
- 默认细分算法
- 默认着色和绘制模式
- 光照参数
- GUI 面板尺寸
- 中文字体候选路径
- AI 接口地址、模型名、API Key 环境变量名
