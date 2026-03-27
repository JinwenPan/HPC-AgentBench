from typing import Dict, List, Optional, Sequence

from llm_config import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OFFLINE_DTYPE,
    DEFAULT_OFFLINE_ENABLE_CHUNKED_PREFILL,
    DEFAULT_OFFLINE_GPU_MEMORY_UTILIZATION,
    DEFAULT_OFFLINE_MAX_MODEL_LEN,
    DEFAULT_OFFLINE_QUANTIZATION,
    DEFAULT_OFFLINE_TENSOR_PARALLEL_SIZE,
    DEFAULT_SERVER_API_KEY,
    DEFAULT_SERVER_BASE_URL,
    DEFAULT_TEMPERATURE,
)


class LocalLLMClient:
    """Shared local inference client for vLLM server and offline modes."""

    def __init__(
        self,
        backend: str = DEFAULT_LLM_BACKEND,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        server_base_url: str = DEFAULT_SERVER_BASE_URL,
        server_api_key: str = DEFAULT_SERVER_API_KEY,
        offline_tensor_parallel_size: int = DEFAULT_OFFLINE_TENSOR_PARALLEL_SIZE,
        offline_gpu_memory_utilization: float = DEFAULT_OFFLINE_GPU_MEMORY_UTILIZATION,
        offline_max_model_len: int = DEFAULT_OFFLINE_MAX_MODEL_LEN,
        offline_dtype: str = DEFAULT_OFFLINE_DTYPE,
        offline_quantization: str = DEFAULT_OFFLINE_QUANTIZATION,
        offline_enable_chunked_prefill: bool = DEFAULT_OFFLINE_ENABLE_CHUNKED_PREFILL,
    ) -> None:
        if backend not in {"server", "offline"}:
            raise ValueError("backend must be 'server' or 'offline'")

        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = None
        self._llm = None
        self._SamplingParams = None
        self._tokenizer = None

        if backend == "server":
            from openai import OpenAI

            self._client = OpenAI(base_url=server_base_url, api_key=server_api_key)
        else:
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise ImportError(
                    "vLLM offline mode requires the 'vllm' package."
                ) from exc

            self._SamplingParams = SamplingParams

            llm_kwargs = {
                "model": model,
                "tensor_parallel_size": offline_tensor_parallel_size,
                "gpu_memory_utilization": offline_gpu_memory_utilization,
                "trust_remote_code": True,
                "enable_chunked_prefill": offline_enable_chunked_prefill,
            }
            if offline_max_model_len > 0:
                llm_kwargs["max_model_len"] = offline_max_model_len
            if offline_dtype:
                llm_kwargs["dtype"] = offline_dtype
            if offline_quantization:
                llm_kwargs["quantization"] = offline_quantization

            self._llm = LLM(**llm_kwargs)
            self._tokenizer = self._llm.get_tokenizer()

    @staticmethod
    def _build_fallback_prompt(messages: List[Dict[str, str]]) -> str:
        prompt_lines: List[str] = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            content = message.get("content", "")
            prompt_lines.append(f"{role}:\n{content}")
        prompt_lines.append("Assistant:\n")
        return "\n\n".join(prompt_lines)

    def _build_offline_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return self._build_fallback_prompt(messages)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.backend == "server":
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": resolved_temperature,
            }
            if resolved_max_tokens and resolved_max_tokens > 0:
                kwargs["max_tokens"] = resolved_max_tokens
            response = self._client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()

        prompt = self._build_offline_prompt(messages)
        sampling_kwargs = {"temperature": resolved_temperature}
        if resolved_max_tokens and resolved_max_tokens > 0:
            sampling_kwargs["max_tokens"] = resolved_max_tokens
        sampling_params = self._SamplingParams(**sampling_kwargs)
        outputs = self._llm.generate([prompt], sampling_params, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            return ""
        return (outputs[0].outputs[0].text or "").strip()

    def generate_batch(
        self,
        messages_batch: Sequence[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.backend == "server":
            return [
                self.generate(messages, temperature=resolved_temperature, max_tokens=resolved_max_tokens)
                for messages in messages_batch
            ]

        prompts = [self._build_offline_prompt(messages) for messages in messages_batch]
        sampling_kwargs = {"temperature": resolved_temperature}
        if resolved_max_tokens and resolved_max_tokens > 0:
            sampling_kwargs["max_tokens"] = resolved_max_tokens
        sampling_params = self._SamplingParams(**sampling_kwargs)
        outputs = self._llm.generate(prompts, sampling_params, use_tqdm=False)

        texts: List[str] = []
        for output in outputs:
            if not output.outputs:
                texts.append("")
                continue
            texts.append((output.outputs[0].text or "").strip())
        return texts

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
