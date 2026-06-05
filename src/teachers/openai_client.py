"""OpenAI (GPT) teacher client."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from .base import TeacherClient, TeacherResponse


class OpenAITeacherClient(TeacherClient):
    provider = "openai"

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout, max_retries=max_retries)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install `openai` to use OpenAITeacherClient.") from e
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        kwargs = {"api_key": key, "timeout": self.timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._use_max_completion_tokens = False

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> TeacherResponse:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=float(temperature),
                )
                if self._use_max_completion_tokens:
                    kwargs["max_completion_tokens"] = int(max_tokens)
                else:
                    kwargs["max_tokens"] = int(max_tokens)
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as e:
                    msg = str(e)
                    if "max_tokens" in msg and "max_completion_tokens" in msg:
                        kwargs.pop("max_tokens", None)
                        kwargs["max_completion_tokens"] = int(max_tokens)
                        self._use_max_completion_tokens = True
                        resp = self._client.chat.completions.create(**kwargs)
                    else:
                        raise
                text = (resp.choices[0].message.content or "").strip()
                return TeacherResponse(
                    text=text, provider=self.provider, model=self.model,
                    raw={"id": getattr(resp, "id", "")},
                )
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                continue
        raise RuntimeError(f"OpenAI chat failed after retries: {last_err}")
