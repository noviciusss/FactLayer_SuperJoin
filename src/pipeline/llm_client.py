"""LLM client abstraction supporting Groq and OpenAI."""
import json
import base64
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple, Type, TypeVar
from pydantic import BaseModel
from src.config import settings

T = TypeVar("T", bound=BaseModel)

# Thread-local storage for tracking per-call status across threads
_thread_local = threading.local()


class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self._groq_client = None
        self._openai_client = None

    def get_last_call_status(self) -> Tuple[str, Optional[str]]:
        """
        Returns (status, error_message) for the most recent LLM call on the calling thread.
        Statuses: 'success', 'rate_limited', 'extraction_failed'
        """
        status = getattr(_thread_local, "last_status", "success")
        err = getattr(_thread_local, "last_error", None)
        return status, err

    def _get_client(self):
        timeout = getattr(settings, "LLM_REQUEST_TIMEOUT", 30.0)
        if self.provider == "groq":
            if not self._groq_client:
                from groq import Groq
                api_key = settings.GROQ_API_KEY
                self._groq_client = Groq(api_key=api_key, timeout=timeout)
            return self._groq_client
        else:
            if not self._openai_client:
                from openai import OpenAI
                api_key = settings.OPENAI_API_KEY
                self._openai_client = OpenAI(api_key=api_key, timeout=timeout)
            return self._openai_client

    def extract_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> T:
        """Extract structured output matching response_model using JSON mode with bounded rate-limit retry and model fallback."""
        _thread_local.last_status = "success"
        _thread_local.last_error = None

        client = self._get_client()
        candidate_models = [model or settings.LLM_MODEL, "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
        unique_models = []
        for m in candidate_models:
            if m and m not in unique_models:
                unique_models.append(m)

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        full_system_prompt = (
            f"{system_prompt}\n\n"
            f"CRITICAL: You MUST respond ONLY with a valid JSON object strictly matching this JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any conversational preamble, commentary, or markdown formatting outside the JSON."
        )

        def parse_json_from_text(raw_text: str) -> dict:
            clean = raw_text.strip()
            if clean.startswith("```json"):
                clean = clean.removeprefix("```json").removesuffix("```").strip()
            elif clean.startswith("```"):
                clean = clean.removeprefix("```").removesuffix("```").strip()
            if "{" in clean:
                start = clean.find("{")
                try:
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(clean[start:])
                    return obj
                except Exception:
                    end = clean.rfind("}") + 1
                    return json.loads(clean[start:end])
            return json.loads(clean)

        max_retries = getattr(settings, "LLM_MAX_RETRIES", 3)
        max_backoff = min(getattr(settings, "LLM_MAX_BACKOFF", 20.0), 30.0)  # Hard ceiling <= 30s
        req_timeout = getattr(settings, "LLM_REQUEST_TIMEOUT", 30.0)
        total_retries = 0
        last_error = None

        for current_model in unique_models:
            if total_retries >= max_retries:
                break
            for attempt in range(2):
                try:
                    chat_completion = client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": full_system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=600,
                        timeout=req_timeout,
                    )
                    raw_content = chat_completion.choices[0].message.content.strip()
                    parsed_json = parse_json_from_text(raw_content)
                    _thread_local.last_status = "success"
                    _thread_local.last_error = None
                    return response_model.model_validate(parsed_json)
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    is_rate_limit = "rate limit" in err_str or "429" in err_str or "otpm" in err_str or "tpm" in err_str
                    is_timeout = "timeout" in err_str or "timed out" in err_str

                    total_retries += 1
                    if is_rate_limit:
                        _thread_local.last_status = "rate_limited"
                        _thread_local.last_error = str(e)
                        if total_retries > max_retries:
                            print(f"[LLMClient RateLimit] Retry ceiling ({max_retries}) exceeded for {current_model}.")
                            break
                        match = re.search(r"try again in ([\d\.]+)s", err_str)
                        if match:
                            wait_sec = min(float(match.group(1)) + 0.5, max_backoff)
                        else:
                            wait_sec = min(2.0 * (2 ** (total_retries - 1)), max_backoff)
                        print(f"[LLMClient RateLimit] Pacing request for {wait_sec:.1f}s (retry {total_retries}/{max_retries})...")
                        time.sleep(wait_sec)
                        continue
                    elif is_timeout:
                        _thread_local.last_status = "extraction_failed"
                        _thread_local.last_error = f"Request timed out: {e}"
                        if total_retries > max_retries:
                            print(f"[LLMClient Timeout] Timeout retry ceiling ({max_retries}) reached.")
                            break
                        wait_sec = min(1.5 * (2 ** (total_retries - 1)), max_backoff)
                        print(f"[LLMClient Timeout] Retrying in {wait_sec:.1f}s (retry {total_retries}/{max_retries})...")
                        time.sleep(wait_sec)
                        continue
                    else:
                        _thread_local.last_status = "extraction_failed"
                        _thread_local.last_error = str(e)
                        break

        if _thread_local.last_status == "success":
            _thread_local.last_status = "extraction_failed"
            _thread_local.last_error = str(last_error)

        print(f"[LLMClient Warning] Extraction failed across models (status: {_thread_local.last_status}): {last_error}")
        return response_model()

    def extract_vision_structured(
        self,
        image_bytes: bytes,
        prompt: str,
        system_prompt: str,
        response_model: Type[T]
    ) -> T:
        """Multimodal extraction for image/chart-dense slides with timeout and bounded retries."""
        _thread_local.last_status = "success"
        _thread_local.last_error = None

        client = self._get_client()
        target_model = settings.VISION_MODEL

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        full_system_prompt = (
            f"{system_prompt}\n\n"
            f"CRITICAL: You MUST respond ONLY with a valid JSON object strictly matching this JSON Schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any preamble, commentary, or markdown code block fences outside the JSON."
        )

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_image}"

        max_retries = getattr(settings, "LLM_MAX_RETRIES", 3)
        max_backoff = min(getattr(settings, "LLM_MAX_BACKOFF", 20.0), 30.0)
        req_timeout = getattr(settings, "LLM_REQUEST_TIMEOUT", 30.0)

        for attempt in range(max_retries + 1):
            try:
                chat_completion = client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": full_system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    timeout=req_timeout,
                )
                raw_content = chat_completion.choices[0].message.content
                parsed_json = json.loads(raw_content)
                _thread_local.last_status = "success"
                _thread_local.last_error = None
                return response_model.model_validate(parsed_json)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "rate limit" in err_str or "429" in err_str or "otpm" in err_str
                is_timeout = "timeout" in err_str or "timed out" in err_str

                if attempt >= max_retries:
                    _thread_local.last_status = "rate_limited" if is_rate_limit else "extraction_failed"
                    _thread_local.last_error = str(e)
                    print(f"[LLMClient Vision Warning] Retries exhausted for {target_model}: {e}")
                    break

                if is_rate_limit:
                    _thread_local.last_status = "rate_limited"
                    match = re.search(r"try again in ([\d\.]+)s", err_str)
                    wait_sec = min(float(match.group(1)) + 0.5, max_backoff) if match else min(2.0 * (2 ** attempt), max_backoff)
                    print(f"[LLMClient Vision RateLimit] Pacing for {wait_sec:.1f}s (retry {attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                elif is_timeout:
                    _thread_local.last_status = "extraction_failed"
                    wait_sec = min(1.5 * (2 ** attempt), max_backoff)
                    print(f"[LLMClient Vision Timeout] Retrying in {wait_sec:.1f}s (retry {attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                else:
                    _thread_local.last_status = "extraction_failed"
                    _thread_local.last_error = str(e)
                    break

        return response_model()

    def transcribe_image(
        self,
        image_bytes: bytes,
        prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Sends page image to vision model (e.g. qwen/qwen3.6-27b) with timeout and bounded retries.
        """
        _thread_local.last_status = "success"
        _thread_local.last_error = None

        client = self._get_client()
        target_model = model or settings.VISION_MODEL
        default_prompt = (
            "Transcribe and describe all visible numbers, labels, financial metrics, table cells, "
            "chart data points, and their visual associations (which label belongs to which number) "
            "from this document page in plain text."
        )
        user_prompt = prompt or default_prompt

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_image}"

        max_retries = getattr(settings, "LLM_MAX_RETRIES", 3)
        max_backoff = min(getattr(settings, "LLM_MAX_BACKOFF", 20.0), 30.0)
        req_timeout = getattr(settings, "LLM_REQUEST_TIMEOUT", 30.0)

        for attempt in range(max_retries + 1):
            try:
                chat_completion = client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert document vision transcription assistant. "
                                "Accurately describe all text, numbers, metrics, and labels visible in the image, "
                                "explicitly indicating which label or metric name is associated with each number."
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    temperature=0.0,
                    max_tokens=1000,
                    timeout=req_timeout,
                )
                _thread_local.last_status = "success"
                _thread_local.last_error = None
                return chat_completion.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "rate limit" in err_str or "429" in err_str or "otpm" in err_str
                is_timeout = "timeout" in err_str or "timed out" in err_str

                if attempt >= max_retries:
                    _thread_local.last_status = "rate_limited" if is_rate_limit else "extraction_failed"
                    _thread_local.last_error = str(e)
                    print(f"[LLMClient Transcription] Retries exhausted for {target_model}: {e}")
                    break

                if is_rate_limit:
                    _thread_local.last_status = "rate_limited"
                    match = re.search(r"try again in ([\d\.]+)s", err_str)
                    wait_sec = min(float(match.group(1)) + 0.5, max_backoff) if match else min(2.0 * (2 ** attempt), max_backoff)
                    print(f"[LLMClient Vision RateLimit] Pacing transcription for {wait_sec:.1f}s (retry {attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                elif is_timeout:
                    _thread_local.last_status = "extraction_failed"
                    wait_sec = min(1.5 * (2 ** attempt), max_backoff)
                    print(f"[LLMClient Vision Timeout] Retrying transcription in {wait_sec:.1f}s (retry {attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                else:
                    _thread_local.last_status = "extraction_failed"
                    _thread_local.last_error = str(e)
                    break

        return ""


# Singleton instance
llm_client = LLMClient()
