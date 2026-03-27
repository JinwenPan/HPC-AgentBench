#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

from baseline_agent import NaiveLocalLLMAgent
from evaluator import evaluate_agent
from llm_config import (
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
from local_llm import LocalLLMClient

DEFAULT_DATASET_PATH = Path("backup/benchmark_results.cleaned.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample cleaned benchmark tickets and run the local baseline agent."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the cleaned benchmark dataset JSON.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of valid tickets to sample (0 = evaluate all valid tickets).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--backend",
        choices=["server", "offline"],
        default="offline",
        help="Local inference backend to use for both the agent and the judge.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help="Model name or path to use for both the agent and the judge.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for the baseline agent.",
    )
    parser.add_argument(
        "--agent-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum generated tokens per agent turn.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for each judge call.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum action steps per ticket.",
    )
    parser.add_argument(
        "--server-base-url",
        default=DEFAULT_SERVER_BASE_URL,
        help="Base URL for a local vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--server-api-key",
        default=DEFAULT_SERVER_API_KEY,
        help="API key value for a local vLLM OpenAI-compatible server.",
    )
    parser.add_argument(
        "--offline-tensor-parallel-size",
        type=int,
        default=DEFAULT_OFFLINE_TENSOR_PARALLEL_SIZE,
        help="Tensor parallel size for vLLM offline mode.",
    )
    parser.add_argument(
        "--offline-gpu-memory-utilization",
        type=float,
        default=DEFAULT_OFFLINE_GPU_MEMORY_UTILIZATION,
        help="GPU memory utilization fraction for vLLM offline mode.",
    )
    parser.add_argument(
        "--offline-max-model-len",
        type=int,
        default=DEFAULT_OFFLINE_MAX_MODEL_LEN,
        help="Maximum model context length for vLLM offline mode.",
    )
    parser.add_argument(
        "--offline-dtype",
        default=DEFAULT_OFFLINE_DTYPE,
        help="dtype for vLLM offline mode.",
    )
    parser.add_argument(
        "--offline-quantization",
        default=DEFAULT_OFFLINE_QUANTIZATION,
        help="Quantization mode for vLLM offline mode.",
    )
    parser.add_argument(
        "--offline-enable-chunked-prefill",
        action="store_true",
        default=DEFAULT_OFFLINE_ENABLE_CHUNKED_PREFILL,
        help="Enable chunked prefill in vLLM offline mode.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the aggregate metrics JSON.",
    )
    parser.add_argument(
        "--sample-output",
        default="",
        help="Optional path to save the sampled ticket subset JSON.",
    )
    parser.add_argument(
        "--verbose-agent",
        action="store_true",
        help="Print each observation, raw model output, and parsed action.",
    )
    parser.add_argument(
        "--include-reconstructed",
        action="store_true",
        help="Also evaluate reconstructed samples instead of grounded-only default behavior.",
    )
    return parser


def load_sample(dataset_path: Path, sample_size: int, seed: int):
    with dataset_path.open(encoding="utf-8") as fh:
        dataset = json.load(fh)

    valid_dataset = [ticket for ticket in dataset if ticket.get("is_valid", True) is not False]
    if sample_size == 0:
        return valid_dataset
    if sample_size > len(valid_dataset):
        raise ValueError(
            f"Requested sample size {sample_size}, but only {len(valid_dataset)} valid tickets are available."
        )
    return random.Random(seed).sample(valid_dataset, sample_size)


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    sample = load_sample(dataset_path, args.sample_size, args.seed)

    if args.sample_output:
        with Path(args.sample_output).open("w", encoding="utf-8") as fh:
            json.dump(sample, fh, ensure_ascii=False, indent=2)

    client = LocalLLMClient(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.agent_max_tokens,
        server_base_url=args.server_base_url,
        server_api_key=args.server_api_key,
        offline_tensor_parallel_size=args.offline_tensor_parallel_size,
        offline_gpu_memory_utilization=args.offline_gpu_memory_utilization,
        offline_max_model_len=args.offline_max_model_len,
        offline_dtype=args.offline_dtype,
        offline_quantization=args.offline_quantization,
        offline_enable_chunked_prefill=args.offline_enable_chunked_prefill,
    )

    try:
        agent = NaiveLocalLLMAgent(
            model=args.model,
            backend=args.backend,
            temperature=args.temperature,
            max_tokens=args.agent_max_tokens,
            verbose=args.verbose_agent,
            client=client,
        )
        results = evaluate_agent(
            agent,
            sample,
            max_steps=args.max_steps,
            judge_client=client,
            judge_max_tokens=args.judge_max_tokens,
            include_reconstructed=args.include_reconstructed,
        )
    finally:
        client.close()

    print("\nAggregate metrics:")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
