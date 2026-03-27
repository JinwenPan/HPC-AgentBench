#!/usr/bin/env python3
"""
Multi-stage pipeline for building HPC-AgentBench v2 from raw tickets.

Stages:
  1. preclassify
  2. generate_candidate
  3. validate_candidate
  4. repair_or_filter

The pipeline uses local vLLM inference, with offline in-process mode as the
primary execution path. It writes stage outputs as JSONL and exports both
canonical and runtime benchmark artifacts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark_semantics import audit_cleaned_dataset
from dataset_schema import (
    apply_validation_verdict,
    deterministic_validate_canonical,
    extract_reference_admin_reply,
    format_ticket_conversation,
    merge_validation_verdict,
    normalize_canonical_candidate,
    preclassify_ticket,
    project_runtime_record,
    summarize_canonical_dataset,
)
from llm_config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OFFLINE_DTYPE,
    DEFAULT_OFFLINE_ENABLE_CHUNKED_PREFILL,
    DEFAULT_OFFLINE_GPU_MEMORY_UTILIZATION,
    DEFAULT_OFFLINE_MAX_MODEL_LEN,
    DEFAULT_OFFLINE_QUANTIZATION,
    DEFAULT_OFFLINE_TENSOR_PARALLEL_SIZE,
    DEFAULT_SERVER_API_KEY,
    DEFAULT_SERVER_BASE_URL,
)
from local_llm import LocalLLMClient
from tqdm import tqdm

INPUT_FILE = Path(__file__).parent / "tickets_combined.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "pipeline_outputs"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_RETRIES = 3
GENERATOR_MAX_TOKENS = 2048
VALIDATOR_MAX_TOKENS = 768
REPAIR_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0

PRECLASSIFY_FILE = "preclassify.jsonl"
CANDIDATE_FILE = "candidate.jsonl"
VALIDATED_FILE = "validated.jsonl"
FINAL_CANONICAL_FILE = "canonical.jsonl"
RUNTIME_GROUNDED_FILE = "runtime.grounded.json"
RUNTIME_RECONSTRUCTED_FILE = "runtime.reconstructed.json"
QUALITY_REPORT_FILE = "quality_report.json"

GENERATOR_SYSTEM_PROMPT = """\
You are building a publishable benchmark dataset for an HPC support agent.

Your task is to transform one raw human/admin support ticket into a canonical
candidate record for a trace-based benchmark.

Important rules:
- The output must be valid JSON only.
- Tools are limited to execute_bash, search_docs, ask_user_for_info.
- If the raw ticket contains any assistant reply, `evaluation.reference_admin_reply`
  must copy the final raw assistant reply verbatim. Do not rewrite, polish,
  summarize, expand, or idealize it.
- If the raw ticket lacks assistant replies, you MUST NOT fabricate a grounded
  reference_admin_reply. Such samples should usually be reconstructed.
- Grounded samples must stay faithful to the raw ticket. Do not invent a
  detailed failure path, job ID, path, or policy step unless it is directly
  grounded in the ticket or introduced via ask_user_for_info.
- Do not turn support links, portal URLs, or generic admin instructions into
  synthetic curl/wget page checks unless the raw ticket explicitly contains
  those commands or their observed outputs.
- If the raw ticket already quotes commands, shell values, file paths,
  hostnames, or observed command/error output, prefer extracting those into
  grounded execute_bash/search_docs traces instead of replacing them with
  ask_user_for_info.
- If the raw ticket only reports a high-level request plus a final admin-side
  resolution, do not invent a long execute_bash/search_docs workflow to fill
  in the missing internal support steps. That belongs in reconstructed, not
  grounded.
- If the initial user request is vague or underspecified and there is no later
  user follow-up clarifying the task, prefer `reconstructed` instead of
  fabricating a very specific grounded benchmark scenario.
- Reconstructed samples may expand the scenario, but they must stay plausible
  for HPC ticket handling and must not masquerade as grounded.
- Any hidden username, job ID, path, or resource value needed by later traces
  must appear first through ask_user_for_info or a prior mock_output.
- Do not add ask_user_for_info if the needed information already appears in the
  raw ticket conversation.

Return exactly one JSON object with this schema:
{
  "instance_id": "<ticket id>",
  "is_valid": true,
  "release_tier": "grounded" or "reconstructed",
  "construction_mode": "extracted" or "reconstructed",
  "instruction": "<initial user-facing problem statement>",
  "traces": [
    {
      "tool": "execute_bash" | "search_docs" | "ask_user_for_info",
      "argument": "<tool argument>",
      "trigger_command": "<tool(argument)>",
      "mock_output": "<simulated observation>",
      "observation_source": "bash" | "docs" | "user",
      "grounding": "grounded" | "inferred",
      "required": true
    }
  ],
  "evaluation": {
    "expected_trajectory": ["..."],
    "final_solution_criteria": ["..."],
    "reference_admin_reply": "<empty string if none>",
    "has_reference_admin_reply": true or false
  }
}

`trigger_command` must always use the exact canonical form `tool(argument)`.
Do not output bare strings like `search_docs foo bar` or `execute_bash ls -l`.

If the ticket is clearly unusable as a benchmark item, return:
{"instance_id":"<ticket id>","is_valid":false}
"""

VALIDATOR_SYSTEM_PROMPT = """\
You are a QA verifier for a benchmark dataset built from raw HPC support tickets.

You will receive:
1. the raw ticket conversation,
2. a preclassification summary,
3. a candidate canonical record,
4. deterministic QA flags.

Decide whether the candidate should be valid, whether it belongs in grounded or
reconstructed release tier, whether the reference_admin_reply is truly grounded
in the raw ticket, and whether repair is needed.

Be strict about these failure modes:
- The candidate rewrites or embellishes the final assistant reply instead of
  copying it verbatim.
- A grounded sample starts from a vague user request and invents a much more
  specific benchmark problem without user-side evidence.
- A grounded sample invents a multi-step execute_bash/search_docs workflow even
  though the raw ticket only contains a high-level request or a final
  resolution update.
- A grounded sample converts assistant-provided URLs or portal instructions
  into synthetic curl/wget verification traces that never appeared in the raw
  conversation.
- The candidate inserts ask_user_for_info even though the required information
  is already present in the raw ticket.
- The raw ticket already contains explicit commands, shell values, paths, or
  observed failures, but the candidate fails to extract them into traces and
  instead leaves the trace set too thin.

Return exactly one JSON object:
{
  "is_valid": true or false,
  "release_tier": "grounded" or "reconstructed",
  "construction_mode": "extracted" or "reconstructed",
  "has_reference_admin_reply": true or false,
  "qa_flags": ["..."],
  "qa_notes": "<short note>",
  "repair_needed": true or false,
  "repair_instructions": "<empty string if no repair is needed>"
}
"""

REPAIR_SYSTEM_PROMPT = """\
You are repairing a benchmark candidate record for an HPC agent dataset.

Use the raw ticket and QA verdict to output a corrected canonical candidate.

Rules:
- Output valid JSON only.
- Preserve the overall issue and solution intent of the raw ticket.
- Fix information-gating problems by inserting ask_user_for_info before hidden
  entities appear downstream.
- When a raw assistant reply exists, set `evaluation.reference_admin_reply` to
  the exact final assistant message from the raw ticket.
- If the user-side grounding is too weak for a grounded sample, downgrade it to
  `reconstructed` instead of inventing extra grounded detail.
- If the raw ticket already contains explicit commands, shell values, file
  paths, hostnames, or observed command output, prefer extracting them into
  grounded traces instead of redundant ask_user_for_info.
- Do not preserve curl/wget traces that merely verify assistant-provided URLs
  or portal pages unless the raw ticket explicitly included those commands or
  the resulting outputs.
- If the raw ticket only provides a high-level request plus a final resolution,
  remove invented internal execute_bash/search_docs tool chains rather than
  preserving them as grounded.
- Remove fabricated grounded reference_admin_reply content.
- If the sample cannot be repaired into a valid benchmark item, return:
  {"instance_id":"<ticket id>","is_valid":false}
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = [
        line
        for line in cleaned.splitlines()
        if not line.strip().startswith("```")
    ]
    return "\n".join(lines).strip()


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    for candidate in (text.strip(), _strip_fences(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_done_ids(path: Path) -> set[str]:
    done_ids: set[str] = set()
    for record in _iter_jsonl(path):
        instance_id = str(record.get("instance_id", "")).strip()
        if instance_id:
            done_ids.add(instance_id)
    return done_ids


def _append_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _chunked(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _resolve_default_output(stage: str, output: str) -> Path:
    if output:
        return Path(output)
    if stage in {"repair_or_filter", "all"}:
        return DEFAULT_OUTPUT_DIR
    filename_map = {
        "preclassify": PRECLASSIFY_FILE,
        "generate_candidate": CANDIDATE_FILE,
        "validate_candidate": VALIDATED_FILE,
    }
    return DEFAULT_OUTPUT_DIR / filename_map[stage]


def _resolve_input_path(stage: str, input_path: str, output_root: Path) -> Path:
    if input_path:
        return Path(input_path)
    if stage == "preclassify":
        return INPUT_FILE
    if stage == "generate_candidate":
        return output_root / PRECLASSIFY_FILE
    if stage == "validate_candidate":
        return output_root / CANDIDATE_FILE
    if stage == "repair_or_filter":
        return output_root / VALIDATED_FILE
    raise ValueError(f"Unsupported stage {stage!r}")


def _build_generator_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    default_candidate = record.get("preclassification", {}).get("candidate_class", "")
    source_meta = record.get("source_meta", {})
    user_content = (
        f"Default candidate class: {default_candidate}\n"
        f"Source meta: {json.dumps(source_meta, ensure_ascii=False)}\n\n"
        f"{format_ticket_conversation(record['instance_id'], record.get('messages', []))}"
    )
    return [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_validator_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    candidate = record.get("candidate", {})
    deterministic = record.get("deterministic_validation", {})
    raw_reference = extract_reference_admin_reply(record.get("messages", []))
    user_content = (
        f"Preclassification: {json.dumps(record.get('preclassification', {}), ensure_ascii=False)}\n"
        f"Source meta: {json.dumps(record.get('source_meta', {}), ensure_ascii=False)}\n"
        f"Raw reference admin reply (if any): {raw_reference}\n"
        f"Deterministic QA: {json.dumps(deterministic, ensure_ascii=False)}\n\n"
        "Candidate canonical record:\n"
        f"{json.dumps(candidate, ensure_ascii=False, indent=2)}\n\n"
        "Raw ticket:\n"
        f"{format_ticket_conversation(record['instance_id'], record.get('messages', []))}"
    )
    return [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_repair_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    user_content = (
        f"Validation verdict: {json.dumps(record.get('validation', {}), ensure_ascii=False)}\n\n"
        "Candidate canonical record:\n"
        f"{json.dumps(record.get('candidate', {}), ensure_ascii=False, indent=2)}\n\n"
        "Raw ticket:\n"
        f"{format_ticket_conversation(record['instance_id'], record.get('messages', []))}"
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _process_batch_with_retries(
    records: Sequence[Dict[str, Any]],
    build_messages: Callable[[Dict[str, Any]], List[Dict[str, str]]],
    handle_parsed: Callable[[Dict[str, Any], Optional[Dict[str, Any]], str, int], Optional[Dict[str, Any]]],
    handle_failure: Callable[[Dict[str, Any], int, str], Dict[str, Any]],
    client: LocalLLMClient,
    batch_size: int,
    max_retries: int,
    max_tokens: int,
    desc: str,
) -> List[Dict[str, Any]]:
    indexed_records = list(enumerate(records))
    completed: Dict[int, Dict[str, Any]] = {}
    pending = indexed_records

    for attempt in range(1, max_retries + 1):
        if not pending:
            break
        next_pending: List[Tuple[int, Dict[str, Any]]] = []
        for chunk in tqdm(list(_chunked(pending, batch_size)), desc=f"{desc} (attempt {attempt})"):
            chunk_indices = [index for index, _ in chunk]
            chunk_records = [record for _, record in chunk]
            try:
                raw_responses = client.generate_batch(
                    [build_messages(record) for record in chunk_records],
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                log.error(
                    "%s batch failed on attempt %d for %d record(s): %s",
                    desc,
                    attempt,
                    len(chunk_records),
                    exc,
                )
                if attempt < max_retries:
                    next_pending.extend(chunk)
                    continue
                raw_responses = [""] * len(chunk_records)
            if len(raw_responses) != len(chunk_records):
                log.error(
                    "%s batch returned %d responses for %d records on attempt %d; padding missing outputs.",
                    desc,
                    len(raw_responses),
                    len(chunk_records),
                    attempt,
                )
                raw_responses = list(raw_responses[: len(chunk_records)])
                raw_responses.extend([""] * (len(chunk_records) - len(raw_responses)))
            for index, record, raw_response in zip(chunk_indices, chunk_records, raw_responses):
                parsed = _parse_json(raw_response)
                handled = handle_parsed(record, parsed, raw_response, attempt)
                if handled is not None:
                    completed[index] = handled
                elif attempt < max_retries:
                    next_pending.append((index, record))
                else:
                    completed[index] = handle_failure(record, attempt, raw_response)
        pending = next_pending

    for index, record in pending:
        completed[index] = handle_failure(record, max_retries, "")

    return [completed[index] for index in range(len(records))]


def run_preclassify(
    raw_path: Path,
    output_path: Path,
    limit: int,
    resume: bool,
) -> Path:
    if output_path.exists() and not resume:
        output_path.unlink()

    with raw_path.open("r", encoding="utf-8") as handle:
        raw_tickets = json.load(handle)

    done_ids = _load_done_ids(output_path) if resume and output_path.exists() else set()
    items = list(raw_tickets.items())
    if limit > 0:
        items = items[:limit]

    pending_records = []
    for ticket_id, messages in items:
        ticket_id = str(ticket_id)
        if ticket_id in done_ids:
            continue
        pending_records.append(preclassify_ticket(ticket_id, messages))

    if pending_records:
        _append_jsonl(output_path, pending_records)
    log.info("Preclassified %d ticket(s) into %s", len(pending_records), output_path)
    return output_path


def run_generate_candidate(
    input_path: Path,
    output_path: Path,
    client: LocalLLMClient,
    model: str,
    batch_size: int,
    max_retries: int,
    max_tokens: int,
    resume: bool,
) -> Path:
    if output_path.exists() and not resume:
        output_path.unlink()

    records = list(_iter_jsonl(input_path))
    done_ids = _load_done_ids(output_path) if resume and output_path.exists() else set()
    pending: List[Dict[str, Any]] = []
    immediate_results: List[Dict[str, Any]] = []

    for record in records:
        instance_id = str(record.get("instance_id", ""))
        if instance_id in done_ids:
            continue
        candidate_class = record.get("preclassification", {}).get("candidate_class")
        if candidate_class == "invalid":
            invalid_candidate = normalize_canonical_candidate(
                {"instance_id": instance_id, "is_valid": False},
                record,
                generator_model=model,
            )
            invalid_candidate["quality"]["qa_flags"].append("preclassified_invalid")
            immediate_results.append(
                {
                    **record,
                    "candidate": invalid_candidate,
                    "generator_raw_response": "",
                    "generator_attempts": 0,
                }
            )
        else:
            pending.append(record)

    def handle_parsed(
        record: Dict[str, Any],
        parsed: Optional[Dict[str, Any]],
        raw_response: str,
        attempt: int,
    ) -> Optional[Dict[str, Any]]:
        if parsed is None:
            return None
        candidate = normalize_canonical_candidate(parsed, record, generator_model=model)
        return {
            **record,
            "candidate": candidate,
            "generator_raw_response": raw_response,
            "generator_attempts": attempt,
        }

    def handle_failure(record: Dict[str, Any], attempt: int, raw_response: str) -> Dict[str, Any]:
        candidate = normalize_canonical_candidate(
            {"instance_id": record["instance_id"], "is_valid": False},
            record,
            generator_model=model,
        )
        candidate["quality"]["qa_flags"].append("generation_failed")
        return {
            **record,
            "candidate": candidate,
            "generator_raw_response": raw_response,
            "generator_attempts": attempt,
        }

    staged_results = _process_batch_with_retries(
        pending,
        build_messages=_build_generator_messages,
        handle_parsed=handle_parsed,
        handle_failure=handle_failure,
        client=client,
        batch_size=batch_size,
        max_retries=max_retries,
        max_tokens=max_tokens,
        desc="generate_candidate",
    )

    if immediate_results:
        _append_jsonl(output_path, immediate_results)
    if staged_results:
        _append_jsonl(output_path, staged_results)
    log.info(
        "Generated %d candidate record(s) into %s",
        len(immediate_results) + len(staged_results),
        output_path,
    )
    return output_path


def run_validate_candidate(
    input_path: Path,
    output_path: Path,
    client: LocalLLMClient,
    batch_size: int,
    max_retries: int,
    max_tokens: int,
    resume: bool,
) -> Path:
    if output_path.exists() and not resume:
        output_path.unlink()

    records = list(_iter_jsonl(input_path))
    done_ids = _load_done_ids(output_path) if resume and output_path.exists() else set()

    pending: List[Dict[str, Any]] = []
    immediate_results: List[Dict[str, Any]] = []
    for record in records:
        instance_id = str(record.get("instance_id", ""))
        if instance_id in done_ids:
            continue
        candidate = record.get("candidate", {})
        deterministic = deterministic_validate_canonical(candidate)
        record = {**record, "deterministic_validation": deterministic}
        if not candidate.get("is_valid", True):
            immediate_results.append(
                {
                    **record,
                    "validation": deterministic,
                    "validator_raw_response": "",
                    "validator_attempts": 0,
                }
            )
        else:
            pending.append(record)

    def handle_parsed(
        record: Dict[str, Any],
        parsed: Optional[Dict[str, Any]],
        raw_response: str,
        attempt: int,
    ) -> Optional[Dict[str, Any]]:
        merged = merge_validation_verdict(record.get("candidate", {}), parsed)
        return {
            **record,
            "validation": merged,
            "validator_raw_response": raw_response,
            "validator_attempts": attempt,
        }

    def handle_failure(record: Dict[str, Any], attempt: int, raw_response: str) -> Dict[str, Any]:
        merged = merge_validation_verdict(record.get("candidate", {}), None)
        merged["qa_flags"] = sorted(set(merged["qa_flags"]) | {"validator_failed"})
        return {
            **record,
            "validation": merged,
            "validator_raw_response": raw_response,
            "validator_attempts": attempt,
        }

    staged_results = _process_batch_with_retries(
        pending,
        build_messages=_build_validator_messages,
        handle_parsed=handle_parsed,
        handle_failure=handle_failure,
        client=client,
        batch_size=batch_size,
        max_retries=max_retries,
        max_tokens=max_tokens,
        desc="validate_candidate",
    )

    if immediate_results:
        _append_jsonl(output_path, immediate_results)
    if staged_results:
        _append_jsonl(output_path, staged_results)
    log.info(
        "Validated %d candidate record(s) into %s",
        len(immediate_results) + len(staged_results),
        output_path,
    )
    return output_path


def run_repair_or_filter(
    input_path: Path,
    output_dir: Path,
    client: LocalLLMClient,
    model: str,
    batch_size: int,
    max_retries: int,
    max_tokens: int,
    tier: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(_iter_jsonl(input_path))

    final_records_by_index: Dict[int, Dict[str, Any]] = {}
    pending_repair: List[Tuple[int, Dict[str, Any]]] = []
    for record_index, record in enumerate(records):
        verdict = record.get("validation", {})
        candidate = record.get("candidate", {})
        if verdict.get("repair_needed") and candidate.get("is_valid", True):
            pending_repair.append((record_index, record))
        else:
            final_records_by_index[record_index] = (
                apply_validation_verdict(candidate, verdict, validator_model=model)
            )

    def _fallback_preserving_candidate(record: Dict[str, Any], extra_flags: Sequence[str]) -> Dict[str, Any]:
        candidate = record.get("candidate", {})
        baseline = deterministic_validate_canonical(candidate)
        prior_validation = record.get("validation", {})

        if prior_validation.get("release_tier") == "reconstructed":
            baseline["release_tier"] = "reconstructed"
            baseline["construction_mode"] = "reconstructed"
        if isinstance(prior_validation.get("has_reference_admin_reply"), bool):
            baseline["has_reference_admin_reply"] = (
                prior_validation["has_reference_admin_reply"]
                and bool(candidate.get("evaluation", {}).get("reference_admin_reply", "").strip())
            )

        baseline["qa_flags"] = sorted(
            set(baseline.get("qa_flags", []))
            | set(prior_validation.get("qa_flags", []))
            | set(extra_flags)
        )
        baseline["qa_notes"] = str(prior_validation.get("qa_notes", "")).strip()
        baseline["repair_needed"] = False
        baseline["repair_instructions"] = ""
        return apply_validation_verdict(
            candidate,
            baseline,
            validator_model=model,
        )

    def handle_parsed(
        record: Dict[str, Any],
        parsed: Optional[Dict[str, Any]],
        raw_response: str,
        attempt: int,
    ) -> Optional[Dict[str, Any]]:
        if parsed is None:
            return None
        repaired_candidate = normalize_canonical_candidate(parsed, record, generator_model=model)
        post_repair = deterministic_validate_canonical(repaired_candidate)
        post_repair["qa_flags"] = sorted(set(post_repair["qa_flags"]) | {"repair_applied"})
        repaired_final = apply_validation_verdict(
            repaired_candidate,
            post_repair,
            validator_model=model,
        )
        if repaired_final.get("is_valid", True):
            return repaired_final
        fallback = _fallback_preserving_candidate(
            record,
            extra_flags={"repair_applied", "repair_regressed_to_invalid"},
        )
        if fallback.get("is_valid", False):
            return fallback
        return repaired_final

    def handle_failure(record: Dict[str, Any], attempt: int, raw_response: str) -> Dict[str, Any]:
        fallback = dict(record.get("validation", {}))
        fallback["qa_flags"] = sorted(set(fallback.get("qa_flags", [])) | {"repair_failed"})
        failed_result = apply_validation_verdict(record.get("candidate", {}), fallback, validator_model=model)
        if failed_result.get("is_valid", True):
            return failed_result
        preserved = _fallback_preserving_candidate(
            record,
            extra_flags={"repair_failed", "repair_preserved_candidate"},
        )
        return preserved if preserved.get("is_valid", False) else failed_result

    repaired_records = [record for _, record in pending_repair]
    repaired_results = _process_batch_with_retries(
        repaired_records,
        build_messages=_build_repair_messages,
        handle_parsed=handle_parsed,
        handle_failure=handle_failure,
        client=client,
        batch_size=batch_size,
        max_retries=max_retries,
        max_tokens=max_tokens,
        desc="repair_or_filter",
    )
    for (record_index, _), repaired_record in zip(pending_repair, repaired_results):
        final_records_by_index[record_index] = repaired_record

    final_records = [
        final_records_by_index[index]
        for index in range(len(records))
        if index in final_records_by_index
    ]

    canonical_path = output_dir / FINAL_CANONICAL_FILE
    _write_jsonl(canonical_path, final_records)

    runtime_grounded = [
        project_runtime_record(record)
        for record in final_records
        if record.get("is_valid", True) and record.get("release_tier") == "grounded"
    ]
    runtime_reconstructed = [
        project_runtime_record(record)
        for record in final_records
        if record.get("is_valid", True) and record.get("release_tier") == "reconstructed"
    ]

    if tier in {"all", "grounded"}:
        _write_json(output_dir / RUNTIME_GROUNDED_FILE, runtime_grounded)
    if tier in {"all", "reconstructed"}:
        _write_json(output_dir / RUNTIME_RECONSTRUCTED_FILE, runtime_reconstructed)

    quality_report = summarize_canonical_dataset(final_records)
    _write_json(output_dir / QUALITY_REPORT_FILE, quality_report)

    log.info(
        "Exported %d canonical record(s), %d grounded runtime record(s), and %d reconstructed runtime record(s) into %s",
        len(final_records),
        len(runtime_grounded),
        len(runtime_reconstructed),
        output_dir,
    )
    return canonical_path


def _run_cleaned_audit(dataset_path: Path, output_path: Optional[Path]) -> None:
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    report = audit_cleaned_dataset(dataset)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build HPC-AgentBench v2 artifacts from raw tickets.")
    parser.add_argument(
        "--stage",
        choices=["preclassify", "generate_candidate", "validate_candidate", "repair_or_filter", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Stage input path. Defaults to the standard previous-stage artifact.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Stage output path for single stages, or output directory for repair_or_filter/all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume single-stage JSONL outputs by skipping already-written instance_ids.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of prompts per vLLM batch.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for generation/validation/repair per record.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N raw tickets during preclassify.",
    )
    parser.add_argument(
        "--tier",
        choices=["all", "grounded", "reconstructed"],
        default="all",
        help="Which runtime tier projections to export during repair_or_filter/all.",
    )
    parser.add_argument(
        "--backend",
        choices=["server", "offline"],
        default="offline",
        help="Inference backend. Offline in-process is the primary mode.",
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Generator/validator/repair model.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for all pipeline LLM calls.",
    )
    parser.add_argument(
        "--generator-max-tokens",
        type=int,
        default=GENERATOR_MAX_TOKENS,
        help="Maximum generated tokens for the candidate generation stage.",
    )
    parser.add_argument(
        "--validator-max-tokens",
        type=int,
        default=VALIDATOR_MAX_TOKENS,
        help="Maximum generated tokens for the validator stage.",
    )
    parser.add_argument(
        "--repair-max-tokens",
        type=int,
        default=REPAIR_MAX_TOKENS,
        help="Maximum generated tokens for the repair stage.",
    )
    parser.add_argument("--server-base-url", default=DEFAULT_SERVER_BASE_URL)
    parser.add_argument("--server-api-key", default=DEFAULT_SERVER_API_KEY)
    parser.add_argument(
        "--offline-tensor-parallel-size",
        type=int,
        default=DEFAULT_OFFLINE_TENSOR_PARALLEL_SIZE,
    )
    parser.add_argument(
        "--offline-gpu-memory-utilization",
        type=float,
        default=DEFAULT_OFFLINE_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--offline-max-model-len",
        type=int,
        default=DEFAULT_OFFLINE_MAX_MODEL_LEN,
    )
    parser.add_argument("--offline-dtype", default=DEFAULT_OFFLINE_DTYPE)
    parser.add_argument("--offline-quantization", default=DEFAULT_OFFLINE_QUANTIZATION)
    parser.add_argument(
        "--offline-enable-chunked-prefill",
        action="store_true",
        default=DEFAULT_OFFLINE_ENABLE_CHUNKED_PREFILL,
    )
    parser.add_argument(
        "--audit-cleaned",
        default="",
        help="Audit an existing runtime-format cleaned dataset instead of running the pipeline.",
    )
    parser.add_argument(
        "--audit-output",
        default="",
        help="Optional path to save the audit JSON.",
    )
    return parser


def _build_client(args: argparse.Namespace) -> LocalLLMClient:
    return LocalLLMClient(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.generator_max_tokens,
        server_base_url=args.server_base_url,
        server_api_key=args.server_api_key,
        offline_tensor_parallel_size=args.offline_tensor_parallel_size,
        offline_gpu_memory_utilization=args.offline_gpu_memory_utilization,
        offline_max_model_len=args.offline_max_model_len,
        offline_dtype=args.offline_dtype,
        offline_quantization=args.offline_quantization,
        offline_enable_chunked_prefill=args.offline_enable_chunked_prefill,
    )


def main() -> None:
    args = build_parser().parse_args()

    if args.audit_cleaned:
        _run_cleaned_audit(
            Path(args.audit_cleaned),
            Path(args.audit_output) if args.audit_output else None,
        )
        return

    stage = args.stage
    output_root = _resolve_default_output(stage, args.output)
    if stage in {"preclassify", "generate_candidate", "validate_candidate"} and not output_root.suffix:
        filename_map = {
            "preclassify": PRECLASSIFY_FILE,
            "generate_candidate": CANDIDATE_FILE,
            "validate_candidate": VALIDATED_FILE,
        }
        output_root = output_root / filename_map[stage]
    if stage == "all":
        output_root.mkdir(parents=True, exist_ok=True)

    if stage == "preclassify":
        input_path = _resolve_input_path(stage, args.input, DEFAULT_OUTPUT_DIR)
        output_path = output_root
        run_preclassify(input_path, output_path, args.limit, args.resume)
        return

    client = _build_client(args)
    try:
        if stage == "generate_candidate":
            input_path = _resolve_input_path(stage, args.input, output_root.parent if output_root.suffix else output_root)
            run_generate_candidate(
                input_path=input_path,
                output_path=output_root,
                client=client,
                model=args.model,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                max_tokens=args.generator_max_tokens,
                resume=args.resume,
            )
            return

        if stage == "validate_candidate":
            input_path = _resolve_input_path(stage, args.input, output_root.parent if output_root.suffix else output_root)
            run_validate_candidate(
                input_path=input_path,
                output_path=output_root,
                client=client,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                max_tokens=args.validator_max_tokens,
                resume=args.resume,
            )
            return

        if stage == "repair_or_filter":
            output_dir = output_root
            input_path = _resolve_input_path(stage, args.input, output_dir)
            run_repair_or_filter(
                input_path=input_path,
                output_dir=output_dir,
                client=client,
                model=args.model,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                max_tokens=args.repair_max_tokens,
                tier=args.tier,
            )
            return

        if stage == "all":
            preclassify_path = output_root / PRECLASSIFY_FILE
            candidate_path = output_root / CANDIDATE_FILE
            validated_path = output_root / VALIDATED_FILE

            run_preclassify(INPUT_FILE, preclassify_path, args.limit, args.resume)
            run_generate_candidate(
                input_path=preclassify_path,
                output_path=candidate_path,
                client=client,
                model=args.model,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                max_tokens=args.generator_max_tokens,
                resume=args.resume,
            )
            run_validate_candidate(
                input_path=candidate_path,
                output_path=validated_path,
                client=client,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                max_tokens=args.validator_max_tokens,
                resume=args.resume,
            )
            run_repair_or_filter(
                input_path=validated_path,
                output_dir=output_root,
                client=client,
                model=args.model,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                max_tokens=args.repair_max_tokens,
                tier=args.tier,
            )
            return
    finally:
        client.close()


if __name__ == "__main__":
    main()
