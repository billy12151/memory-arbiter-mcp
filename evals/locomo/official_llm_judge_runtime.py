from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from contextvars import ContextVar

from openai import AsyncOpenAI
from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt, wait_fixed, wait_random


def extract_json_object(text: str) -> str:
    stripped = _strip_code_fence(text).strip()
    in_string = False
    escape = False
    depth = 0
    start: int | None = None
    for index, ch in enumerate(stripped):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return stripped[start : index + 1]
    raise ValueError("failed to extract JSON object from judge response")


def _strip_code_fence(content: str) -> str:
    stripped = (content or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


class AsyncBinaryJudge:
    def __init__(
        self,
        *,
        prompt_template: str,
        default_model: str,
        default_base_url: str | None,
        default_api_key: str | None,
    ) -> None:
        self._prompt_template = prompt_template
        self._model = default_model
        self._base_url = default_base_url
        self._api_key = default_api_key
        self._enable_thinking_unsupported_models: set[str] = set()
        self._client: AsyncOpenAI | None = None
        self._retry_details: ContextVar[str | None] = ContextVar("judge_retry_details", default=None)

    def configure(self, *, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        self.close()
        if model:
            self._model = model
        if base_url is not None:
            self._base_url = base_url or None
        if api_key is not None:
            self._api_key = api_key or None

    async def evaluate(self, question: str, gold_answer: str | Sequence[str], generated_answer: str) -> int:
        return int(await self._evaluate_async(question, gold_answer, generated_answer))

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            closer = getattr(client, "aclose", None)
            if callable(closer):
                await closer()

    def close(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
        else:
            asyncio.create_task(self.aclose())

    async def _evaluate_async(self, question: str, gold_answer: str | Sequence[str], generated_answer: str) -> int:
        if isinstance(gold_answer, Sequence) and not isinstance(gold_answer, str):
            tasks = [
                self._evaluate_single_gold(question, str(candidate), generated_answer) for candidate in gold_answer
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            best = 0
            errors: list[Exception] = []
            for item in results:
                if isinstance(item, Exception):
                    errors.append(item)
                    continue
                if int(item) == 1:
                    return 1
                best = max(best, int(item))
            if errors:
                raise RuntimeError(f"llm judge failed for {len(errors)} answer candidate(s)") from errors[0]
            return best
        return await self._evaluate_single_gold(question, str(gold_answer), generated_answer)

    async def _evaluate_single_gold(self, question: str, gold_answer: str, generated_answer: str) -> int:
        token = self._retry_details.set(
            f"question='{question[:60]}', gold='{gold_answer[:40]}', generated='{generated_answer[:40]}'"
        )
        try:
            self._validate_configuration()
            async for attempt in AsyncRetrying(
                reraise=True,
                stop=stop_after_attempt(10),
                wait=self._custom_wait,
                before_sleep=self._log_retry,
            ):
                with attempt:
                    request_kwargs = self._build_request_kwargs(
                        question=question,
                        gold_answer=gold_answer,
                        generated_answer=generated_answer,
                    )
                    response = await self._create_completion(request_kwargs)
                    content = str(response.choices[0].message.content or "")
                    payload = json.loads(extract_json_object(content))
                    return 1 if str(payload.get("label", "")).strip().upper() == "CORRECT" else 0
        finally:
            self._retry_details.reset(token)

    def _get_client(self) -> AsyncOpenAI:
        self._validate_configuration()
        if self._client is None:
            client_kwargs: dict[str, str] = {}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            self._client = AsyncOpenAI(**client_kwargs)
        return self._client

    def _validate_configuration(self) -> None:
        if not self._model:
            raise RuntimeError("LLM judge model is empty. Set EVALUATOR_MODEL or --evaluator-model.")

    def _log_retry(self, retry_state: RetryCallState) -> None:
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        wait_action = getattr(retry_state.next_action, "sleep", None)
        max_attempts = getattr(getattr(retry_state.retry_object, "stop", None), "max_attempt_number", None)
        attempt = retry_state.attempt_number
        message = f"LLM judge retry {attempt}"
        if max_attempts:
            message += f"/{max_attempts}"
        if exception:
            message += f" due to: {exception}"
        if wait_action is not None:
            message += f"; waiting {wait_action:.2f}s"
        context = self._retry_details.get()
        if context:
            message += f" | {context}"
        print(message, file=sys.stderr, flush=True)

    @staticmethod
    def _custom_wait(retry_state: RetryCallState):
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        if exception and any(token in str(exception).lower() for token in ("tpm", "limit", "rate limit", "429")):
            return wait_random(min=20, max=60)(retry_state)
        return wait_fixed(1)(retry_state)

    def _build_request_kwargs(
        self,
        *,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> dict[str, object]:
        request_kwargs: dict[str, object] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt_template.format(
                        question=question,
                        gold_answer=gold_answer,
                        generated_answer=generated_answer,
                    ),
                }
            ],
            "temperature": 0.0,
        }
        if self._should_send_enable_thinking():
            request_kwargs["extra_body"] = {"enable_thinking": False}
        return request_kwargs

    async def _create_completion(self, request_kwargs: dict[str, object]):
        try:
            response = await self._get_client().chat.completions.create(**request_kwargs)
        except Exception as exc:
            if not self._should_retry_without_enable_thinking(exc, request_kwargs):
                raise
            self._enable_thinking_unsupported_models.add(self._model)
            fallback_kwargs = dict(request_kwargs)
            fallback_kwargs.pop("extra_body", None)
            response = await self._get_client().chat.completions.create(**fallback_kwargs)
        return response

    def _should_send_enable_thinking(self) -> bool:
        return self._model not in self._enable_thinking_unsupported_models

    @staticmethod
    def _should_retry_without_enable_thinking(exc: Exception, request_kwargs: dict[str, object]) -> bool:
        if "extra_body" not in request_kwargs:
            return False
        message = str(exc).lower()
        if "enable_thinking" not in message:
            return False
        unsupported_markers = (
            "unknown",
            "unsupported",
            "unexpected",
            "unrecognized",
            "not allowed",
            "not permit",
            "extra inputs",
            "additional properties",
            "invalid parameter",
        )
        return any(marker in message for marker in unsupported_markers)
