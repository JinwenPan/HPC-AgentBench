#!/usr/bin/env python3
"""
process_tickets.py — Async HPC Ticket → Agentic Benchmark Transformer

Reads tickets_combined.json    → calls local vLLM (OpenAI-compatible)
                                → writes benchmark_results.json

Features:
  • asyncio + openai.AsyncOpenAI for fully async I/O
  • Semaphore-based concurrency control (default 50)
  • Resumability: checkpoint JSONL + periodic JSON saves
  • Graceful shutdown on SIGINT/SIGTERM — always saves progress
  • Robust JSON parsing with markdown fence stripping + retries
  • Pre-filters empty / short tickets (<50 chars) without API calls
  • tqdm progress bar
"""

import argparse
import asyncio
import json
import logging
import re
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
INPUT_FILE = Path(__file__).parent / "tickets_combined.json"
OUTPUT_FILE = Path(__file__).parent / "benchmark_results.json"
CHECKPOINT_FILE = Path(__file__).parent / ".benchmark_checkpoint.jsonl"
API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
BACKEND_MODE = "server"        # one of: server, offline
MAX_CONCURRENT = 50          # asyncio.Semaphore limit
MAX_RETRIES = 5              # retries per ticket (covers both API + parse errors)
MIN_TICKET_CHARS = 0       # skip tickets shorter than this
TEMPERATURE = 0.0            # deterministic
REQUEST_TIMEOUT = 180        # seconds per API request
SAVE_EVERY = 500             # write JSON snapshot every N completed tickets
OFFLINE_MAX_TOKENS = 0
OFFLINE_TENSOR_PARALLEL_SIZE = 1
OFFLINE_GPU_MEMORY_UTILIZATION = 0.90
OFFLINE_MAX_MODEL_LEN = 65536
OFFLINE_DTYPE = "bfloat16"
OFFLINE_QUANTIZATION = "fp8"

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(Path(__file__).parent / "process_tickets.log"),
    ],
)
log = logging.getLogger(__name__)

# Suppress noisy per-request HTTP logs from the openai/httpx client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)

# ──────────────────────────────────────────────
# System prompt (embedded constant)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert HPC Sysadmin and Test Engineer. Convert the following \
support ticket into an interactive agent test case.

Assume the test agent can use exactly 3 tools:
    1) execute_bash(command)
         - Meaning: run a shell command to inspect/diagnose/fix cluster state.
         - Parameter: the exact bash command string.
    2) search_docs(query)
         - Meaning: search documentation, KB pages, or policy references.
         - Parameter: a concise search query.
    3) ask_user_for_info(question)
         - Meaning: ask the end user for missing context.
         - Parameter: the exact question asked to the user.

Extract the implied troubleshooting workflow from the full ticket and map it
into a flat list of traces. Each trace item must contain one action trigger
and one simulated output.

IMPORTANT:
    - Do NOT include submit_solution in traces.
    - A ticket may require multiple actions.
    - The same action type may appear multiple times.
    - trigger_command must always be exactly action(parameter) where action is
        one of execute_bash, search_docs, ask_user_for_info.

CRITICAL — Information Gating:
  If the admin checked a file or ran a diagnostic, the FIRST mock command \
    (e.g. execute_bash(squeue), execute_bash(scontrol show job)) MUST reveal a specific, hallucinated \
  absolute path (e.g. /work/01234/user/job.err).  The agent must then use \
    that exact absolute path in the subsequent trigger_command \
    (e.g. execute_bash(cat /work/01234/user/job.err)).  Do NOT allow relative paths.

If the ticket is empty, purely conversational, or lacks any technical \
troubleshooting steps, return ONLY:
  {"is_valid": false}

Otherwise return ONLY valid JSON matching this schema (no extra keys, \
no markdown fences, no commentary):
{
  "instance_id": "<ticket_id>",
  "is_valid": true,
  "instruction": "<The user's initial prompt/error. Do NOT include admin replies.>",
  "traces": [
    {
        "trigger_command": "<One action(parameter), e.g. execute_bash(ls -a ~)>",
        "mock_output": "<Simulated output for that action; may include paths, node IDs, policy text, or user reply>"
    }
  ],
  "evaluation": {
    "expected_trajectory": ["<Step 1>", "<Step 2>"],
    "final_solution_criteria": ["<Criterion 1>", "<Criterion 2>"],
    "reference_admin_reply": "<The original admin's final solution>"
  }
}

Return ONLY valid JSON. No markdown. No explanation."""

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

# Regex to strip ```json ... ``` fences (with optional language tag)
_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL | re.IGNORECASE,
)

_ACTION_CALL_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$", re.DOTALL)
_ALLOWED_ACTIONS = {"execute_bash", "search_docs", "ask_user_for_info"}


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _format_ticket(ticket_id: str, messages: List[str]) -> str:
    """Turn the raw [role, msg, role, msg, ...] list into readable text."""
    lines = [f"Ticket ID: {ticket_id}", ""]
    for i in range(0, len(messages) - 1, 2):
        role = messages[i]
        body = messages[i + 1]
        lines.append(f"[{role}]\n{body}\n")
    return "\n".join(lines)


def _load_checkpoint(ckpt_path: Path, out_path: Path) -> Tuple[Set[str], List[dict]]:
    """Load already-processed results from checkpoint and/or existing output.

    Returns (done_ids, results_so_far).
    """
    results: List[dict] = []
    done: Set[str] = set()

    # First, load from existing JSON output if it exists
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    for obj in data:
                        iid = obj.get("instance_id")
                        if iid and str(iid) not in done:
                            done.add(str(iid))
                            results.append(obj)
        except (json.JSONDecodeError, Exception) as exc:
            log.warning("Could not load %s: %s", out_path, exc)

    # Then, layer on any additional entries from checkpoint
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    iid = obj.get("instance_id")
                    if iid and str(iid) not in done:
                        done.add(str(iid))
                        results.append(obj)
                except json.JSONDecodeError:
                    log.warning("Corrupt line %d in checkpoint — skipping", lineno)

    return done, results


def _save_json(results: List[dict], path: Path) -> None:
    """Atomically write results to JSON (write to tmp then rename)."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    tmp.rename(path)


def _parse_llm_json(raw: str, ticket_id: str) -> Optional[dict]:
    """Try to parse the LLM output into a dict; return None on failure."""
    # Try raw text first (avoids corrupting JSON that contains backticks)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Fall back to stripping markdown fences
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.debug("JSON parse failed for %s: %s", ticket_id, exc)
        return None


def _build_chat_messages(ticket_id: str, messages: List[str]) -> List[Dict[str, str]]:
    """Build chat messages payload for OpenAI-compatible backends."""
    user_content = _format_ticket(ticket_id, messages)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_fallback_prompt(ticket_id: str, messages: List[str]) -> str:
    """Fallback plain prompt when chat template is unavailable."""
    user_content = _format_ticket(ticket_id, messages)
    return (
        "System:\n" + SYSTEM_PROMPT + "\n\n"
        "User:\n" + user_content + "\n\n"
        "Assistant:\n"
    )


class _OfflineVLLMClient:
    """Thin wrapper around vLLM offline inference."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        dtype: Optional[str],
        quantization: Optional[str],
        enable_chunked_prefill: bool,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM offline mode requires the 'vllm' package. "
                "Install it in your runtime environment before using --backend offline."
            ) from exc

        self._SamplingParams = SamplingParams

        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "enable_chunked_prefill": enable_chunked_prefill,
        }
        if max_model_len is not None and max_model_len > 0:
            llm_kwargs["max_model_len"] = max_model_len
        if dtype:
            llm_kwargs["dtype"] = dtype
        if quantization:
            llm_kwargs["quantization"] = quantization

        self._llm = LLM(
            **llm_kwargs,
        )
        sampling_kwargs = {"temperature": temperature}
        if max_tokens > 0:
            sampling_kwargs["max_tokens"] = max_tokens
        self._sampling_params = SamplingParams(**sampling_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

    def _build_prompt(self, ticket_id: str, messages: List[str]) -> str:
        chat_messages = _build_chat_messages(ticket_id, messages)
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return _build_fallback_prompt(ticket_id, messages)

    def generate(self, ticket_id: str, messages: List[str]) -> str:
        prompt = self._build_prompt(ticket_id, messages)
        outputs = self._llm.generate([prompt], self._sampling_params, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text or ""


def _normalize_trigger_command(trigger_command: str) -> Tuple[Optional[str], str]:
    """Normalize trigger_command into action(parameter) format.

    Returns:
      (normalized_value_or_none, status)
      status in {ok, wrapped_bash, removed_submit_solution, removed_unknown_action, invalid_empty}
    """
    compact = " ".join(trigger_command.split())
    if not compact:
        return None, "invalid_empty"

    match = _ACTION_CALL_RE.match(compact)
    if match:
        action, params = match.group(1), match.group(2).strip()
        if action == "submit_solution":
            return None, "removed_submit_solution"
        if action not in _ALLOWED_ACTIONS:
            return None, "removed_unknown_action"
        return f"{action}({params})", "ok"

    return f"execute_bash({compact})", "wrapped_bash"


def _normalize_traces(parsed: dict, ticket_id: str) -> None:
    """Normalize parsed traces to the new list schema with strict action whitelist."""
    stats = {
        "converted_legacy_schema": 0,
        "wrapped_bash": 0,
        "removed_submit_solution": 0,
        "removed_unknown_action": 0,
        "dropped_invalid_entries": 0,
    }

    traces_raw = parsed.get("traces")
    if isinstance(traces_raw, dict):
        mock_states = traces_raw.get("mock_states")
        if isinstance(mock_states, list):
            traces_raw = mock_states
            stats["converted_legacy_schema"] += 1
        else:
            traces_raw = []
            stats["dropped_invalid_entries"] += 1
    elif not isinstance(traces_raw, list):
        traces_raw = []

    normalized_traces: List[dict] = []
    for item in traces_raw:
        if not isinstance(item, dict):
            stats["dropped_invalid_entries"] += 1
            continue

        trigger_command = item.get("trigger_command")
        mock_output = item.get("mock_output")
        if not isinstance(trigger_command, str) or not isinstance(mock_output, str):
            stats["dropped_invalid_entries"] += 1
            continue

        normalized_trigger, status = _normalize_trigger_command(trigger_command)
        if normalized_trigger is None:
            if status in stats:
                stats[status] += 1
            else:
                stats["dropped_invalid_entries"] += 1
            continue

        if status == "wrapped_bash":
            stats["wrapped_bash"] += 1

        normalized_traces.append(
            {
                "trigger_command": normalized_trigger,
                "mock_output": mock_output.strip(),
            }
        )

    parsed["traces"] = normalized_traces

    if parsed.get("is_valid") is True and not normalized_traces:
        parsed["is_valid"] = False
        log.warning("%s: marked is_valid=false after trace normalization removed all traces.", ticket_id)

    if any(stats.values()):
        log.info("%s: normalization stats %s", ticket_id, stats)


# ──────────────────────────────────────────────
# Core async worker
# ──────────────────────────────────────────────

async def _process_one_server(
    client: Any,
    sem: asyncio.Semaphore,
    ticket_id: str,
    messages: List[str],
) -> Optional[dict]:
    """Call the server-mode LLM for one ticket, with retries + JSON repair."""

    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=_build_chat_messages(ticket_id, messages),
                        temperature=TEMPERATURE,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )
            except asyncio.CancelledError:
                # Graceful shutdown — don't retry, just exit
                raise
            except Exception as exc:
                log.error(
                    "API error for %s (attempt %d/%d): %s",
                    ticket_id, attempt, MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(min(2 ** attempt, 30))  # capped backoff
                    continue
                return None

        raw = resp.choices[0].message.content or ""
        parsed = _parse_llm_json(raw, ticket_id)

        if parsed is not None:
            if not isinstance(parsed, dict):
                log.warning(
                    "Unexpected JSON type for %s (attempt %d/%d): %s",
                    ticket_id, attempt, MAX_RETRIES, type(parsed).__name__,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1)
                    continue
                return None
            _normalize_traces(parsed, ticket_id)
            # Ensure instance_id is set
            parsed.setdefault("instance_id", ticket_id)
            return parsed

        log.warning(
            "JSON parse failed for %s (attempt %d/%d). Raw (first 300 chars): %s",
            ticket_id, attempt, MAX_RETRIES, raw[:300],
        )
        if attempt < MAX_RETRIES:
            await asyncio.sleep(1)

    log.error("Giving up on %s after %d attempts.", ticket_id, MAX_RETRIES)
    return None


def _process_one_offline(
    client: _OfflineVLLMClient,
    ticket_id: str,
    messages: List[str],
) -> Optional[dict]:
    """Call the offline vLLM backend for one ticket, with retries + JSON repair."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = client.generate(ticket_id, messages)
        except Exception as exc:
            log.error(
                "Offline generation error for %s (attempt %d/%d): %s",
                ticket_id, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                continue
            return None

        parsed = _parse_llm_json(raw, ticket_id)

        if parsed is not None:
            if not isinstance(parsed, dict):
                log.warning(
                    "Unexpected JSON type for %s (attempt %d/%d): %s",
                    ticket_id, attempt, MAX_RETRIES, type(parsed).__name__,
                )
                if attempt < MAX_RETRIES:
                    continue
                return None
            _normalize_traces(parsed, ticket_id)
            parsed.setdefault("instance_id", ticket_id)
            return parsed

        log.warning(
            "JSON parse failed for %s (attempt %d/%d). Raw (first 300 chars): %s",
            ticket_id, attempt, MAX_RETRIES, raw[:300],
        )

    log.error("Giving up on %s after %d attempts.", ticket_id, MAX_RETRIES)
    return None


# ──────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────

async def main() -> None:
    global MODEL_NAME

    # ── CLI args ──────────────────────────────
    parser = argparse.ArgumentParser(description="Process HPC tickets → benchmark JSON")
    parser.add_argument(
        "--backend",
        choices=["server", "offline"],
        default=BACKEND_MODE,
        help="Inference backend: server (OpenAI-compatible API) or offline (vLLM local inference)",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N tickets (0 = all)")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Model name/path for backend inference",
    )
    parser.add_argument(
        "--offline-max-tokens",
        type=int,
        default=OFFLINE_MAX_TOKENS,
        help="Max generated tokens per request for offline backend (0 = no explicit max_tokens limit)",
    )
    parser.add_argument(
        "--offline-tensor-parallel-size",
        type=int,
        default=OFFLINE_TENSOR_PARALLEL_SIZE,
        help="Tensor parallel size for offline vLLM",
    )
    parser.add_argument(
        "--offline-gpu-memory-utilization",
        type=float,
        default=OFFLINE_GPU_MEMORY_UTILIZATION,
        help="GPU memory utilization fraction for offline vLLM",
    )
    parser.add_argument(
        "--offline-max-model-len",
        type=int,
        default=OFFLINE_MAX_MODEL_LEN,
        help="Max model context length for offline vLLM (0 disables explicit override)",
    )
    parser.add_argument(
        "--offline-dtype",
        default=OFFLINE_DTYPE,
        help="dtype for offline vLLM (e.g., bfloat16, float16)",
    )
    parser.add_argument(
        "--offline-quantization",
        default=OFFLINE_QUANTIZATION,
        help="quantization for offline vLLM (e.g., fp8, awq, gptq); empty string disables",
    )
    parser.add_argument(
        "--offline-enable-chunked-prefill",
        action="store_true",
        help="Enable chunked prefill in offline vLLM",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model

    log.info("Loading tickets from %s …", INPUT_FILE)
    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        raw_tickets: Dict[str, list] = json.load(fh)
    log.info("Loaded %d ticket IDs.", len(raw_tickets))

    # ── Pre-filter ────────────────────────────
    done_ids, existing_results = _load_checkpoint(CHECKPOINT_FILE, OUTPUT_FILE)
    log.info("Found %d already-processed IDs (resuming).", len(done_ids))

    todo: List[Tuple[str, List[str]]] = []
    skipped_empty = 0
    skipped_short = 0
    skipped_done = 0

    for tid, msgs in raw_tickets.items():
        if tid in done_ids:
            skipped_done += 1
            continue
        if not msgs:
            skipped_empty += 1
            continue
        total_text = " ".join(msgs)
        if len(total_text) < MIN_TICKET_CHARS:
            skipped_short += 1
            continue
        todo.append((tid, msgs))

    log.info(
        "To process: %d | Skipped → already done: %d, empty: %d, short: %d",
        len(todo), skipped_done, skipped_empty, skipped_short,
    )

    if not todo:
        log.info("Nothing to do. Exiting.")
        return

    if args.limit > 0:
        todo = todo[:args.limit]
        log.info("--limit %d applied, will process %d tickets.", args.limit, len(todo))

    # Checkpoint file for crash recovery (append-mode JSONL)
    ckpt_fh = open(CHECKPOINT_FILE, "a", encoding="utf-8")
    new_results: List[dict] = []
    results_lock = asyncio.Lock()
    completed_since_save = 0
    shutting_down = False

    def _finalize() -> None:
        """Merge and save final JSON (called on success or shutdown)."""
        all_results = existing_results + new_results
        _save_json(all_results, OUTPUT_FILE)
        log.info("Saved %d total results to %s", len(all_results), OUTPUT_FILE)

    # ── Graceful shutdown handler ─────────────
    def _handle_signal(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return  # already handling
        shutting_down = True
        signame = signal.Signals(signum).name
        log.warning("Received %s — saving progress and exiting …", signame)
        ckpt_fh.close()
        _finalize()
        log.info("Progress saved. Safe to restart — will resume automatically.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if args.backend == "server":
        from openai import AsyncOpenAI
        from tqdm.asyncio import tqdm_asyncio

        log.info("Backend mode: server (OpenAI-compatible API at %s)", API_BASE)
        client = AsyncOpenAI(base_url=API_BASE, api_key=API_KEY)
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def _worker(tid: str, msgs: List[str]) -> None:
            nonlocal completed_since_save
            if shutting_down:
                return
            result = await _process_one_server(client, sem, tid, msgs)
            if result is not None:
                # Write to checkpoint immediately for crash recovery
                ckpt_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                ckpt_fh.flush()
                async with results_lock:
                    new_results.append(result)
                    completed_since_save += 1
                    # Periodic snapshot so we don't lose everything on hard crash
                    if completed_since_save >= SAVE_EVERY:
                        completed_since_save = 0
                        _finalize()

        tasks = [_worker(tid, msgs) for tid, msgs in todo]
        await tqdm_asyncio.gather(*tasks, desc="Processing tickets")
        await client.close()
    else:
        log.info("Backend mode: offline (vLLM local inference)")
        log.info(
            "Offline config → model=%s, max_tokens=%d, tensor_parallel=%d, gpu_mem_util=%.2f, max_model_len=%d, dtype=%s, quantization=%s, chunked_prefill=%s",
            MODEL_NAME,
            args.offline_max_tokens,
            args.offline_tensor_parallel_size,
            args.offline_gpu_memory_utilization,
            args.offline_max_model_len,
            args.offline_dtype or "auto",
            args.offline_quantization or "none",
            args.offline_enable_chunked_prefill,
        )
        client = _OfflineVLLMClient(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=args.offline_max_tokens,
            tensor_parallel_size=args.offline_tensor_parallel_size,
            gpu_memory_utilization=args.offline_gpu_memory_utilization,
            max_model_len=args.offline_max_model_len,
            dtype=(args.offline_dtype or None),
            quantization=(args.offline_quantization or None),
            enable_chunked_prefill=args.offline_enable_chunked_prefill,
        )
        for tid, msgs in tqdm(todo, desc="Processing tickets"):
            if shutting_down:
                break
            result = _process_one_offline(client, tid, msgs)
            if result is None:
                continue

            ckpt_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            ckpt_fh.flush()
            new_results.append(result)
            completed_since_save += 1
            if completed_since_save >= SAVE_EVERY:
                completed_since_save = 0
                _finalize()

    ckpt_fh.close()

    # Final save
    _finalize()

    # Clean up checkpoint since final JSON is written
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    log.info("All done. %d total results in %s", len(existing_results) + len(new_results), OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())
