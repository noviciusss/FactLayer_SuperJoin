"""LLM client abstraction supporting Groq and OpenAI."""
import json
import base64
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel
from src.config import settings

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self._groq_client = None
        self._openai_client = None

    def _get_client(self):
        if self.provider == "groq":
            if not self._groq_client:
                from groq import Groq
                api_key = settings.GROQ_API_KEY
                self._groq_client = Groq(api_key=api_key)
            return self._groq_client
        else:
            if not self._openai_client:
                from openai import OpenAI
                api_key = settings.OPENAI_API_KEY
                self._openai_client = OpenAI(api_key=api_key)
            return self._openai_client

    def extract_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[T],
        model: Optional[str] = None
    ) -> T:
        """Extract structured output matching response_model using JSON mode with rate-limit retry and model fallback."""
        client = self._get_client()
        candidate_models = [model or settings.LLM_MODEL, "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
        # Deduplicate while preserving order
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

        last_error = None
        for current_model in unique_models:
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
                    )
                    raw_content = chat_completion.choices[0].message.content.strip()
                    parsed_json = parse_json_from_text(raw_content)
                    return response_model.model_validate(parsed_json)
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "rate limit" in err_str or "429" in err_str or "otpm" in err_str:
                        import time
                        import re
                        match = re.search(r"try again in ([\d\.]+)s", err_str)
                        if match:
                            wait_sec = min(float(match.group(1)) + 0.5, 35.0)
                        else:
                            wait_sec = 3 * (attempt + 1)
                        print(f"[LLMClient RateLimit] Pacing request for {wait_sec:.1f}s...")
                        time.sleep(wait_sec)
                        continue
                    break

        print(f"[LLMClient Warning] Extraction failed across models: {last_error}")
        return response_model()

    def extract_vision_structured(
        self,
        image_bytes: bytes,
        prompt: str,
        system_prompt: str,
        response_model: Type[T]
    ) -> T:
        """Multimodal extraction for image/chart-dense slides."""
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
                temperature=0.0
            )
            raw_content = chat_completion.choices[0].message.content
            parsed_json = json.loads(raw_content)
            return response_model.model_validate(parsed_json)
        except Exception as e:
            # If vision is not supported or fails, return empty result gracefully
            return response_model()


# Singleton instance
llm_client = LLMClient()
