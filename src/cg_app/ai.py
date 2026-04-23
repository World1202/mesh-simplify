from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import os

from .config import AppConfig

try:  # pragma: no cover - optional runtime dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback when dependency is missing
    OpenAI = None


@dataclass
class AIViewState:
    enabled: bool = False
    available: bool = False
    busy: bool = False
    status_text: str = "AI 助手未启用。"
    last_question: str = ""
    last_answer: str = ""


class DemoAIAssistant:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-ai")
        self._future: Future[str] | None = None
        self._client = None
        self._view_state = AIViewState(enabled=config.ai_panel_enabled)
        self._initialize_client()

    @property
    def view_state(self) -> AIViewState:
        return self._view_state

    def _initialize_client(self) -> None:
        if not self._config.ai_panel_enabled:
            self._view_state.status_text = "AI 助手已在 config.py 中关闭。"
            return

        if OpenAI is None:
            self._view_state.status_text = "未检测到 openai 依赖，请先安装 openai。"
            return

        api_key = self._config.ai_api_key or os.getenv(self._config.ai_api_key_env_var, "").strip()
        if not api_key:
            self._view_state.status_text = (
                f"未找到 API Key，请在 config.py 的 ai_api_key 中填写，"
                f"或设置环境变量 {self._config.ai_api_key_env_var}。"
            )
            return

        self._client = OpenAI(
            base_url=self._config.ai_base_url,
            api_key=api_key,
            timeout=self._config.ai_request_timeout_seconds,
        )
        self._view_state.available = True
        self._view_state.status_text = "AI 助手已就绪，可直接提问。"

    def submit(self, question: str, runtime_context: str) -> None:
        question = question.strip()
        if not question or not self._view_state.available or self._future is not None:
            return
        self._view_state.busy = True
        self._view_state.status_text = "AI 正在生成讲解，请稍候..."
        self._view_state.last_question = question
        self._future = self._executor.submit(self._request_completion, question, runtime_context)

    def poll(self) -> None:
        if self._future is None or not self._future.done():
            return
        try:
            self._view_state.last_answer = self._future.result()
            self._view_state.status_text = "讲解完成。"
        except Exception as exc:
            self._view_state.last_answer = ""
            self._view_state.status_text = f"请求失败：{exc}"
        finally:
            self._view_state.busy = False
            self._future = None

    def _request_completion(self, question: str, runtime_context: str) -> str:
        response = self._client.chat.completions.create(
            model=self._config.ai_model,
            messages=[
                {"role": "system", "content": self._config.ai_system_prompt},
                {"role": "system", "content": self._config.ai_project_context},
                {"role": "system", "content": runtime_context},
                {"role": "user", "content": question},
            ],
            stream=False,
        )

        message = response.choices[0].message.content or ""
        answer = message.strip() if isinstance(message, str) else str(message).strip()
        if not answer:
            answer = "AI 没有返回可显示的文本内容。"

        max_chars = self._config.ai_max_response_chars
        if max_chars > 0 and len(answer) > max_chars:
            answer = answer[:max_chars].rstrip() + "\n\n[回答过长，已按 config.py 中的 ai_max_response_chars 截断]"
        return answer

    def shutdown(self) -> None:
        if self._future is not None:
            self._future.cancel()
            self._future = None
        self._executor.shutdown(wait=False, cancel_futures=True)
