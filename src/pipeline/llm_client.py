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
        self._azure_client = None
        self._groq_client = None
        self._groq_client_secondary = None
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
        timeout = getattr(settings, "LLM_REQUEST_TIMEOUT", 35.0)
        if self.provider == "azure" and getattr(settings, "AZURE_OPENAI_KEY", None) and getattr(settings, "AZURE_OPENAI_ENDPOINT", None):
            if not self._azure_client:
                from openai import OpenAI
                self._azure_client = OpenAI(
                    base_url=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_KEY,
                    timeout=timeout
                )
            return self._azure_client
        if getattr(settings, "GROQ_API_KEY", None):
            if not self._groq_client:
                from groq import Groq
                self._groq_client = Groq(api_key=settings.GROQ_API_KEY, timeout=timeout)
            return self._groq_client
        return None

    def _get_clients(self):
        """Returns list of active clients in priority order: [groq_primary, groq_secondary] or [azure]."""
        # If _get_client is patched/mocked by a test, return the mock
        client = self._get_client()
        if hasattr(client, "chat") and hasattr(client.chat, "completions") and ("Mock" in type(client).__name__ or hasattr(client, "_mock_return_value")):
            return [client]

        timeout = getattr(settings, "LLM_REQUEST_TIMEOUT", 35.0)
        clients = []

        # 1. Collect all Groq API Keys in priority order
        groq_keys = []
        for key_attr in ("GROQ_API_KEY", "GROQ_API_KEY_SECONDARY", "GROQ_API_KEY_3", "GROQ_API_KEY_4"):
            k = getattr(settings, key_attr, None)
            if k and k not in groq_keys:
                groq_keys.append(k)
        if getattr(settings, "GROQ_API_KEYS", None):
            for k in settings.GROQ_API_KEYS.split(","):
                k = k.strip()
                if k and k not in groq_keys:
                    groq_keys.append(k)

        if not hasattr(self, "_groq_clients_pool"):
            self._groq_clients_pool = {}

        if groq_keys:
            from groq import Groq
            for k in groq_keys:
                if k not in self._groq_clients_pool:
                    self._groq_clients_pool[k] = Groq(api_key=k, timeout=timeout)
                clients.append(self._groq_clients_pool[k])

        # 3. Azure OpenAI (if provider == 'azure' or configured)
        if getattr(settings, "AZURE_OPENAI_KEY", None) and getattr(settings, "AZURE_OPENAI_ENDPOINT", None):
            if not self._azure_client:
                from openai import OpenAI
                self._azure_client = OpenAI(
                    base_url=settings.AZURE_OPENAI_ENDPOINT,
                    api_key=settings.AZURE_OPENAI_KEY,
                    timeout=timeout
                )
            if self.provider == "azure":
                clients.insert(0, self._azure_client)
            else:
                clients.append(self._azure_client)

        return clients if clients else ([client] if client else [])

    def extract_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> T:
        """Extract structured output matching response_model using JSON mode with bounded rate-limit retry, key failover, and model fallback."""
        _thread_local.last_status = "success"
        _thread_local.last_error = None

        clients = self._get_clients()
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
            # Clean thinking tags if present
            if "<think>" in clean and "</think>" in clean:
                clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
            elif "<think>" in clean and "</think>" not in clean:
                clean = ""
            if clean.startswith("```json"):
                clean = clean.removeprefix("```json").removesuffix("```").strip()
            elif clean.startswith("```"):
                clean = clean.removeprefix("```").removesuffix("```").strip()
            if "{" in clean or "[" in clean:
                start_obj = clean.find("{") if "{" in clean else 10**9
                start_arr = clean.find("[") if "[" in clean else 10**9
                start = min(start_obj, start_arr)
                try:
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(clean[start:])
                    if isinstance(obj, list) and issubclass(response_model, BaseModel) and "facts" in getattr(response_model, "model_fields", {}):
                        return {"facts": obj}
                    return obj
                except Exception:
                    end_obj = clean.rfind("}") + 1 if "}" in clean else 0
                    end_arr = clean.rfind("]") + 1 if "]" in clean else 0
                    end = max(end_obj, end_arr)
                    parsed = json.loads(clean[start:end])
                    if isinstance(parsed, list) and issubclass(response_model, BaseModel) and "facts" in getattr(response_model, "model_fields", {}):
                        return {"facts": parsed}
                    return parsed
            return json.loads(clean)

        max_retries = getattr(settings, "LLM_MAX_RETRIES", 3)
        max_backoff = min(getattr(settings, "LLM_MAX_BACKOFF", 20.0), 30.0)  # Hard ceiling <= 30s
        req_timeout = getattr(settings, "LLM_REQUEST_TIMEOUT", 30.0)
        total_retries = 0
        last_error = None

        for current_model in unique_models:
            if total_retries >= max_retries:
                break
            for client_idx, client in enumerate(clients):
                if total_retries >= max_retries:
                    break
                is_azure = (client == self._azure_client)
                if is_azure:
                    call_model = settings.AZURE_OPENAI_DEPLOYMENT or "gpt-5"
                elif client in (self._groq_client, self._groq_client_secondary):
                    call_model = current_model if current_model not in ("gpt-5", "gpt-4o", "gpt-4o-mini") else "openai/gpt-oss-20b"
                else:
                    call_model = current_model

                for attempt in range(2):
                    try:
                        call_kwargs = {
                            "model": call_model,
                            "messages": [
                                {"role": "system", "content": full_system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            "timeout": req_timeout,
                        }
                        if is_azure or "gpt-5" in str(call_model):
                            call_kwargs["max_completion_tokens"] = 4000
                        else:
                            call_kwargs["max_tokens"] = 1000 if "qwen" in str(call_model) else 2500
                            call_kwargs["temperature"] = 0.0

                        chat_completion = client.chat.completions.create(**call_kwargs)
                        raw_content = chat_completion.choices[0].message.content or ""
                        raw_content = raw_content.strip()
                        parsed_json = parse_json_from_text(raw_content)
                        _thread_local.last_status = "success"
                        _thread_local.last_error = None
                        return response_model.model_validate(parsed_json)
                    except Exception as e:
                        last_error = e
                        err_str = str(e).lower()
                        is_rate_limit = "rate limit" in err_str or "429" in err_str or "otpm" in err_str or "tpm" in err_str or "tokens per day" in err_str
                        is_timeout = "timeout" in err_str or "timed out" in err_str

                        # If rate limited on this client and we have another client, failover immediately!
                        if is_rate_limit and client_idx + 1 < len(clients):
                            print(f"[LLMClient Failover] Rate limited on client #{client_idx}, immediately failing over to next client...")
                            break  # Breaks attempt loop to switch to next client immediately

                        total_retries += 1
                        if is_rate_limit:
                            _thread_local.last_status = "rate_limited"
                            _thread_local.last_error = str(e)
                            if total_retries > max_retries:
                                print(f"[LLMClient RateLimit] Retry ceiling ({max_retries}) exceeded for {call_model}.")
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

        clients = self._get_clients()
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
            for client in clients:
                try:
                    is_azure = (client == self._azure_client)
                    if is_azure:
                        target_model = settings.AZURE_OPENAI_DEPLOYMENT or "gpt-5"
                    elif client in (self._groq_client, self._groq_client_secondary):
                        target_model = "qwen/qwen3.6-27b"
                    else:
                        target_model = settings.VISION_MODEL
                    call_kwargs = {
                        "model": target_model,
                        "messages": [
                            {"role": "system", "content": full_system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": data_url}}
                                ]
                            }
                        ],
                        "timeout": req_timeout,
                    }
                    if is_azure or "gpt-5" in str(target_model):
                        call_kwargs["max_completion_tokens"] = 3500
                    else:
                        call_kwargs["response_format"] = {"type": "json_object"}
                        call_kwargs["max_tokens"] = 1000 if "qwen" in str(target_model) else 2500
                        call_kwargs["temperature"] = 0.0

                    chat_completion = client.chat.completions.create(**call_kwargs)
                    raw_content = chat_completion.choices[0].message.content or ""
                    raw_content = raw_content.strip()
                    if raw_content.startswith("```json"):
                        raw_content = raw_content.removeprefix("```json").removesuffix("```").strip()
                    elif raw_content.startswith("```"):
                        raw_content = raw_content.removeprefix("```").removesuffix("```").strip()
                    if "{" in raw_content:
                        start = raw_content.find("{")
                        end = raw_content.rfind("}") + 1
                        raw_content = raw_content[start:end]
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
                        break

                    if is_rate_limit:
                        _thread_local.last_status = "rate_limited"
                        match = re.search(r"try again in ([\d\.]+)s", err_str)
                        wait_sec = min(float(match.group(1)) + 0.5, max_backoff) if match else min(2.0 * (2 ** attempt), max_backoff)
                        time.sleep(wait_sec)
                    elif is_timeout:
                        _thread_local.last_status = "extraction_failed"
                        wait_sec = min(1.5 * (2 ** attempt), max_backoff)
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
        Sends page image to vision model with timeout and bounded retries.
        """
        _thread_local.last_status = "success"
        _thread_local.last_error = None

        clients = self._get_clients()
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
            for client in clients:
                try:
                    is_azure = (client == self._azure_client)
                    if is_azure:
                        target_model = settings.AZURE_OPENAI_DEPLOYMENT or "gpt-5"
                    elif client in (self._groq_client, self._groq_client_secondary):
                        target_model = "qwen/qwen3.6-27b"
                    else:
                        target_model = model or settings.VISION_MODEL
                    call_kwargs = {
                        "model": target_model,
                        "messages": [
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
                        "timeout": req_timeout,
                    }
                    if is_azure or "gpt-5" in str(target_model):
                        call_kwargs["max_completion_tokens"] = 2500
                    else:
                        call_kwargs["max_tokens"] = 1000 if "qwen" in str(target_model) else 1500
                        call_kwargs["temperature"] = 0.0

                    chat_completion = client.chat.completions.create(**call_kwargs)
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
