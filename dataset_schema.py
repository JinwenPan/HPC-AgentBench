import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from benchmark_semantics import (
    audit_ticket_information_flow,
    normalize_trigger_command,
    parse_trigger_command,
)

VALID_RELEASE_TIERS = {"grounded", "reconstructed"}
VALID_CONSTRUCTION_MODES = {"extracted", "reconstructed"}
VALID_TOOLS = {"execute_bash", "search_docs", "ask_user_for_info"}
VALID_OBSERVATION_SOURCES = {"bash", "docs", "user"}
VALID_TRACE_GROUNDING = {"grounded", "inferred"}

TOOL_TO_OBSERVATION_SOURCE = {
    "execute_bash": "bash",
    "search_docs": "docs",
    "ask_user_for_info": "user",
}

GREETING_ONLY_RE = re.compile(
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|thanks|thank you)"
    r"[\s,!.?]*$",
    re.IGNORECASE,
)
TECHNICAL_HINT_RE = re.compile(
    r"\b(error|fail|failed|failing|crash|crashed|hang|stuck|pending|slow|"
    r"quota|permission|denied|job|queue|node|slurm|module|ssh|login|"
    r"account|disk|memory|cpu|gpu|runtime|port|compile|install|path|"
    r"idev|jupyter|conda|python|mpi|cuda|oom|killed|timeout|maverick2|"
    r"stampede2|lonestar6|ls6|cluster|batch|interactive|allocation)\b",
    re.IGNORECASE,
)
EXECUTION_EVIDENCE_RE = re.compile(
    r"(\$[A-Z_]+|/[\w./:@-]+|--[\w-]+|\b(?:scp|rsync|ssh|cp|mv|cat|ls|grep|"
    r"squeue|scontrol|sbatch|sacct|chmod|module|conda|python|bash|zsh|"
    r"login\d*\.|data\.tacc|echo \$shell|connection closed|password:|"
    r"tacc_token|token:)\b)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9_./$@:-]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)>\"]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "confirm",
    "current",
    "exact",
    "for",
    "from",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "please",
    "request",
    "requested",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "user",
    "using",
    "what",
}


def iter_ticket_pairs(messages: Sequence[Any]) -> Iterable[Tuple[str, str]]:
    for index in range(0, max(len(messages) - 1, 0), 2):
        role = messages[index]
        body = messages[index + 1]
        if isinstance(role, str) and isinstance(body, str):
            yield role, body


def format_ticket_conversation(ticket_id: str, messages: Sequence[Any]) -> str:
    lines = [f"Ticket ID: {ticket_id}", ""]
    for role, body in iter_ticket_pairs(messages):
        lines.append(f"[{role}]\n{body}\n")
    return "\n".join(lines).strip()


def _human_messages(messages: Sequence[Any]) -> List[str]:
    return [body for role, body in iter_ticket_pairs(messages) if role == "Human"]


def _assistant_messages(messages: Sequence[Any]) -> List[str]:
    return [body for role, body in iter_ticket_pairs(messages) if role == "Assistant"]


def extract_source_meta(messages: Sequence[Any]) -> Dict[str, Any]:
    role_pattern = [role for role, _ in iter_ticket_pairs(messages)]
    human_count = sum(1 for role in role_pattern if role == "Human")
    assistant_count = sum(1 for role in role_pattern if role == "Assistant")
    return {
        "turn_count": len(role_pattern),
        "role_pattern": ">".join(role_pattern),
        "has_assistant_reply": assistant_count > 0,
        "has_user_followup": human_count > 1,
    }


def extract_reference_admin_reply(messages: Sequence[Any]) -> str:
    assistant_messages = _assistant_messages(messages)
    return assistant_messages[-1].strip() if assistant_messages else ""


def first_human_instruction(messages: Sequence[Any]) -> str:
    for body in _human_messages(messages):
        stripped = body.strip()
        if stripped:
            return stripped
    return ""


def _is_obviously_invalid(messages: Sequence[Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    human_messages = _human_messages(messages)
    first_human = first_human_instruction(messages)

    if not list(iter_ticket_pairs(messages)):
        reasons.append("empty_ticket")
    if not human_messages:
        reasons.append("no_human_message")
    if first_human and GREETING_ONLY_RE.match(first_human):
        reasons.append("greeting_only")
    if first_human and len(first_human) < 12 and not TECHNICAL_HINT_RE.search(first_human):
        reasons.append("too_short_nontechnical")

    return bool(reasons), reasons


def preclassify_ticket(ticket_id: str, messages: Sequence[Any]) -> Dict[str, Any]:
    source_meta = extract_source_meta(messages)
    first_human = first_human_instruction(messages)
    all_human_text = "\n".join(_human_messages(messages))
    obvious_invalid, reasons = _is_obviously_invalid(messages)

    if obvious_invalid:
        candidate_class = "invalid"
    elif (
        not TECHNICAL_HINT_RE.search(first_human)
        and not source_meta["has_user_followup"]
        and source_meta["has_assistant_reply"]
    ):
        candidate_class = "reconstructed_candidate"
        reasons = reasons + ["underspecified_initial_user_request"]
    elif (
        not TECHNICAL_HINT_RE.search(all_human_text)
        and source_meta["has_assistant_reply"]
    ):
        candidate_class = "reconstructed_candidate"
        reasons = reasons + ["weak_user_side_grounding"]
    elif source_meta["has_assistant_reply"]:
        candidate_class = "grounded_candidate"
    else:
        candidate_class = "reconstructed_candidate"

    return {
        "instance_id": str(ticket_id),
        "source_ticket_id": str(ticket_id),
        "messages": list(messages),
        "source_meta": source_meta,
        "instruction_seed": first_human,
        "preclassification": {
            "candidate_class": candidate_class,
            "reasons": reasons,
        },
    }


def _default_release_tier(source_meta: Dict[str, Any], candidate_class: str) -> str:
    if candidate_class == "grounded_candidate" and source_meta.get("has_assistant_reply"):
        return "grounded"
    return "reconstructed"


def _default_construction_mode(release_tier: str, source_meta: Dict[str, Any]) -> str:
    if release_tier == "grounded" and source_meta.get("has_assistant_reply"):
        return "extracted"
    return "reconstructed"


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _default_trace_grounding(tool: str, release_tier: str) -> str:
    if tool == "ask_user_for_info" and release_tier == "grounded":
        return "inferred"
    if release_tier == "grounded":
        return "grounded"
    return "inferred"


def _text_has_execution_evidence(text: str) -> bool:
    return bool(EXECUTION_EVIDENCE_RE.search(text or ""))


def _significant_tokens(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }
    return tokens


def _token_overlap_ratio(source_text: str, context_text: str) -> float:
    source_tokens = _significant_tokens(source_text)
    context_tokens = _significant_tokens(context_text)
    if not source_tokens:
        return 0.0
    return len(source_tokens & context_tokens) / len(source_tokens)


def _is_redundant_ask_user_trace(trace: Dict[str, Any], canonical: Dict[str, Any]) -> bool:
    if trace.get("tool") != "ask_user_for_info":
        return False
    context_parts = [
        str(canonical.get("instruction", "")),
        str(canonical.get("evaluation", {}).get("reference_admin_reply", "")),
    ]
    context_text = "\n".join(part for part in context_parts if part.strip())
    question_overlap = _token_overlap_ratio(str(trace.get("argument", "")), context_text)
    answer_overlap = _token_overlap_ratio(str(trace.get("mock_output", "")), context_text)
    return answer_overlap >= 0.55 or (question_overlap >= 0.45 and answer_overlap >= 0.35)


def _is_synthetic_url_verification_trace(trace: Dict[str, Any], canonical: Dict[str, Any]) -> bool:
    if trace.get("tool") != "execute_bash":
        return False

    argument = str(trace.get("argument", "")).strip()
    lowered = argument.lower()
    if not (lowered.startswith("curl ") or lowered.startswith("wget ")):
        return False

    urls = URL_RE.findall(argument)
    if not urls:
        return False

    evidence_text = "\n".join(
        [
            str(canonical.get("instruction", "")),
            str(canonical.get("evaluation", {}).get("reference_admin_reply", "")),
        ]
    ).lower()
    if not evidence_text.strip():
        return False

    for url in urls:
        normalized_url = url.rstrip(".,;:").lower()
        if normalized_url in evidence_text:
            return True

    for url in urls:
        normalized_url = url.rstrip(".,;:").lower()
        host = normalized_url.split("://", 1)[-1].split("/", 1)[0]
        if host and host in evidence_text:
            return True
    return False


def _normalize_trace_item(
    item: Dict[str, Any],
    trace_index: int,
    release_tier: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    tool = item.get("tool")
    argument = item.get("argument")
    trigger_command = item.get("trigger_command")
    trigger_parsed = False

    if isinstance(trigger_command, str) and trigger_command.strip():
        parsed = parse_trigger_command(trigger_command)
        if parsed is not None:
            parsed_tool, parsed_argument, normalized_trigger = parsed
            tool = parsed_tool
            if not isinstance(argument, str) or not argument.strip():
                argument = parsed_argument
            trigger_command = normalized_trigger
            trigger_parsed = True

    if (not isinstance(tool, str) or tool not in VALID_TOOLS) and isinstance(trigger_command, str):
        parsed = parse_trigger_command(trigger_command)
        if parsed is not None:
            tool, argument, trigger_command = parsed
            trigger_parsed = True

    if not isinstance(tool, str) or tool not in VALID_TOOLS:
        return None

    if not isinstance(argument, str):
        argument = ""
    argument = argument.strip()

    if not trigger_parsed:
        trigger_command = f"{tool}({argument})"

    normalized_trigger = normalize_trigger_command(
        trigger_command if isinstance(trigger_command, str) and trigger_command.strip() else f"{tool}({argument})"
    )
    if normalized_trigger is None:
        return None

    parsed = parse_trigger_command(normalized_trigger)
    if parsed is None:
        return None
    parsed_tool, parsed_argument, normalized_trigger = parsed
    tool = parsed_tool
    argument = parsed_argument

    mock_output = item.get("mock_output", "")
    if not isinstance(mock_output, str):
        mock_output = str(mock_output)

    observation_source = item.get("observation_source")
    if observation_source not in VALID_OBSERVATION_SOURCES:
        observation_source = TOOL_TO_OBSERVATION_SOURCE[tool]

    grounding = item.get("grounding")
    if grounding not in VALID_TRACE_GROUNDING:
        grounding = _default_trace_grounding(tool, release_tier)

    return {
        "trace_id": f"trace_{trace_index + 1:03d}",
        "tool": tool,
        "argument": argument,
        "trigger_command": normalized_trigger,
        "mock_output": mock_output.strip(),
        "observation_source": observation_source,
        "grounding": grounding,
        "required": True,
    }


def normalize_canonical_candidate(
    parsed: Any,
    preclassified_record: Dict[str, Any],
    generator_model: str,
) -> Dict[str, Any]:
    source_meta = dict(preclassified_record.get("source_meta", {}))
    source_ticket_id = str(preclassified_record.get("source_ticket_id", ""))
    candidate_class = preclassified_record.get("preclassification", {}).get(
        "candidate_class",
        "reconstructed_candidate",
    )
    release_tier = _default_release_tier(source_meta, candidate_class)
    construction_mode = _default_construction_mode(release_tier, source_meta)
    instruction_seed = str(preclassified_record.get("instruction_seed", "")).strip()

    record: Dict[str, Any] = parsed if isinstance(parsed, dict) else {}

    is_valid = bool(record.get("is_valid", True))
    if record == {"is_valid": False}:
        is_valid = False

    requested_tier = record.get("release_tier")
    if requested_tier in VALID_RELEASE_TIERS:
        release_tier = requested_tier

    requested_mode = record.get("construction_mode")
    if requested_mode in VALID_CONSTRUCTION_MODES:
        construction_mode = requested_mode
    else:
        construction_mode = _default_construction_mode(release_tier, source_meta)

    instruction = record.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = instruction_seed

    traces_raw = record.get("traces", [])
    normalized_traces: List[Dict[str, Any]] = []
    if isinstance(traces_raw, list):
        for trace_index, item in enumerate(traces_raw):
            normalized_item = _normalize_trace_item(item, trace_index, release_tier)
            if normalized_item is not None:
                normalized_traces.append(normalized_item)

    evaluation_raw = record.get("evaluation", {})
    if not isinstance(evaluation_raw, dict):
        evaluation_raw = {}

    quality_raw = record.get("quality", {})
    if not isinstance(quality_raw, dict):
        quality_raw = {}

    reference_admin_reply = evaluation_raw.get("reference_admin_reply")
    if not isinstance(reference_admin_reply, str):
        reference_admin_reply = ""
    raw_reference_admin_reply = extract_reference_admin_reply(
        preclassified_record.get("messages", [])
    )

    has_reference_admin_reply = evaluation_raw.get("has_reference_admin_reply")
    if not isinstance(has_reference_admin_reply, bool):
        has_reference_admin_reply = bool(reference_admin_reply.strip())

    quality_flags = _coerce_string_list(quality_raw.get("qa_flags", []))
    quality_notes = str(quality_raw.get("qa_notes", "")).strip()

    if raw_reference_admin_reply.strip():
        if reference_admin_reply.strip() and reference_admin_reply.strip() != raw_reference_admin_reply.strip():
            quality_flags.append("reference_admin_reply_reset_to_raw")
        reference_admin_reply = raw_reference_admin_reply.strip()
        has_reference_admin_reply = True
    else:
        reference_admin_reply = ""
        has_reference_admin_reply = False

    canonical = {
        "instance_id": str(record.get("instance_id") or source_ticket_id),
        "source_ticket_id": source_ticket_id,
        "is_valid": is_valid,
        "release_tier": release_tier,
        "construction_mode": construction_mode,
        "instruction": instruction.strip(),
        "source_meta": {
            "turn_count": int(source_meta.get("turn_count", 0)),
            "role_pattern": str(source_meta.get("role_pattern", "")),
            "has_assistant_reply": bool(source_meta.get("has_assistant_reply", False)),
            "has_user_followup": bool(source_meta.get("has_user_followup", False)),
        },
        "traces": normalized_traces,
        "evaluation": {
            "expected_trajectory": _coerce_string_list(
                evaluation_raw.get("expected_trajectory", [])
            ),
            "final_solution_criteria": _coerce_string_list(
                evaluation_raw.get("final_solution_criteria", [])
            ),
            "reference_admin_reply": reference_admin_reply.strip(),
            "has_reference_admin_reply": has_reference_admin_reply,
        },
        "quality": {
            "generator_model": generator_model,
            "validator_model": str(quality_raw.get("validator_model", "")).strip(),
            "qa_flags": quality_flags,
            "qa_notes": quality_notes,
        },
    }

    if canonical["is_valid"] and not canonical["traces"]:
        canonical["is_valid"] = False
        canonical["quality"]["qa_flags"].append("no_traces_after_normalization")

    return canonical


def deterministic_validate_canonical(canonical: Dict[str, Any]) -> Dict[str, Any]:
    qa_flags = list(canonical.get("quality", {}).get("qa_flags", []))
    source_meta = canonical.get("source_meta", {})
    evaluation = canonical.get("evaluation", {})
    traces = canonical.get("traces", [])
    is_valid = bool(canonical.get("is_valid", True))
    release_tier = canonical.get("release_tier", "reconstructed")
    construction_mode = canonical.get("construction_mode", "reconstructed")

    if release_tier not in VALID_RELEASE_TIERS:
        qa_flags.append("invalid_release_tier")
        release_tier = "reconstructed"
    if construction_mode not in VALID_CONSTRUCTION_MODES:
        qa_flags.append("invalid_construction_mode")
        construction_mode = _default_construction_mode(release_tier, source_meta)

    if not canonical.get("instruction", "").strip():
        qa_flags.append("empty_instruction")
        is_valid = False

    if is_valid and not traces:
        qa_flags.append("empty_valid_traces")
        is_valid = False

    for trace in traces:
        tool = trace.get("tool")
        if tool not in VALID_TOOLS:
            qa_flags.append("invalid_trace_tool")
            is_valid = False
        trigger = trace.get("trigger_command", "")
        parsed = parse_trigger_command(trigger)
        if parsed is None:
            qa_flags.append("invalid_trigger_command")
            is_valid = False
            continue
        parsed_tool, parsed_argument, _ = parsed
        if parsed_tool != tool or parsed_argument != trace.get("argument", ""):
            qa_flags.append("trace_argument_trigger_mismatch")
            is_valid = False
        if trace.get("observation_source") not in VALID_OBSERVATION_SOURCES:
            qa_flags.append("invalid_observation_source")
            is_valid = False
        if trace.get("grounding") not in VALID_TRACE_GROUNDING:
            qa_flags.append("invalid_trace_grounding")
            is_valid = False

    has_reference_admin_reply = bool(evaluation.get("has_reference_admin_reply", False))
    reference_admin_reply = str(evaluation.get("reference_admin_reply", "")).strip()
    if has_reference_admin_reply and not reference_admin_reply:
        qa_flags.append("missing_reference_admin_reply_text")
        has_reference_admin_reply = False
    if reference_admin_reply and not has_reference_admin_reply:
        qa_flags.append("reference_admin_reply_without_flag")
        has_reference_admin_reply = True

    if release_tier == "grounded" and not source_meta.get("has_assistant_reply", False):
        qa_flags.append("grounded_without_assistant_reply")
        release_tier = "reconstructed"
        construction_mode = "reconstructed"

    if (
        release_tier == "grounded"
        and not TECHNICAL_HINT_RE.search(str(canonical.get("instruction", "")))
        and not source_meta.get("has_user_followup", False)
    ):
        qa_flags.append("underspecified_grounded_instruction")
        release_tier = "reconstructed"
        construction_mode = "reconstructed"

    if (
        not source_meta.get("has_assistant_reply", False)
        and has_reference_admin_reply
    ):
        qa_flags.append("fabricated_reference_admin_reply")
        has_reference_admin_reply = False
        reference_admin_reply = ""
        release_tier = "reconstructed"
        construction_mode = "reconstructed"

    if release_tier == "grounded":
        unsupported_inferred = [
            trace
            for trace in traces
            if trace.get("grounding") == "inferred"
            and trace.get("tool") != "ask_user_for_info"
        ]
        if unsupported_inferred:
            qa_flags.append("unsupported_inferred_trace_in_grounded")
            release_tier = "reconstructed"
            construction_mode = "reconstructed"

    if release_tier == "grounded" and source_meta.get("has_assistant_reply", False) and not reference_admin_reply:
        qa_flags.append("missing_raw_reference_admin_reply")
        release_tier = "reconstructed"
        construction_mode = "reconstructed"

    instruction_text = str(canonical.get("instruction", ""))
    non_user_traces = [
        trace for trace in traces if trace.get("tool") in {"execute_bash", "search_docs"}
    ]
    if (
        release_tier == "grounded"
        and non_user_traces
        and not source_meta.get("has_user_followup", False)
        and not _text_has_execution_evidence("\n".join([instruction_text, reference_admin_reply]))
    ):
        qa_flags.append("unsupported_grounded_tool_chain_without_ticket_evidence")
        release_tier = "reconstructed"
        construction_mode = "reconstructed"

    redundant_ask_user = [
        trace for trace in traces if _is_redundant_ask_user_trace(trace, canonical)
    ]
    if redundant_ask_user:
        qa_flags.append("redundant_ask_user_trace")

    if (
        release_tier == "grounded"
        and _text_has_execution_evidence(instruction_text)
        and not non_user_traces
    ):
        qa_flags.append("under_extracted_grounded_trace_set")

    synthetic_url_verification = [
        trace for trace in traces if _is_synthetic_url_verification_trace(trace, canonical)
    ]
    if release_tier == "grounded" and synthetic_url_verification:
        qa_flags.append("synthetic_url_verification_in_grounded")
        release_tier = "reconstructed"
        construction_mode = "reconstructed"

    info_issues = audit_ticket_information_flow(canonical)
    if info_issues:
        qa_flags.append("information_gating_violation")
        if release_tier == "grounded":
            release_tier = "reconstructed"
            construction_mode = "reconstructed"

    repairable_flags = {
        "information_gating_violation",
        "reference_admin_reply_without_flag",
        "missing_reference_admin_reply_text",
        "unsupported_inferred_trace_in_grounded",
        "unsupported_grounded_tool_chain_without_ticket_evidence",
        "redundant_ask_user_trace",
        "under_extracted_grounded_trace_set",
        "synthetic_url_verification_in_grounded",
    }
    fatal_flags = {
        "empty_instruction",
        "empty_valid_traces",
        "invalid_trace_tool",
        "invalid_trigger_command",
        "trace_argument_trigger_mismatch",
        "invalid_observation_source",
        "invalid_trace_grounding",
    }
    repair_needed = bool(set(qa_flags) & repairable_flags) and is_valid
    if set(qa_flags) & fatal_flags:
        is_valid = False

    instructions = []
    if "information_gating_violation" in qa_flags:
        instructions.append(
            "Insert ask_user_for_info traces before any downstream trace that uses hidden usernames, job IDs, paths, or resources."
        )
    if "fabricated_reference_admin_reply" in qa_flags:
        instructions.append(
            "Remove fabricated reference_admin_reply content and mark has_reference_admin_reply=false."
        )
    if "unsupported_inferred_trace_in_grounded" in qa_flags:
        instructions.append(
            "Either downgrade the sample to reconstructed or replace inferred non-user traces with evidence grounded in the raw ticket."
        )
    if "unsupported_grounded_tool_chain_without_ticket_evidence" in qa_flags:
        instructions.append(
            "Do not invent a grounded execute_bash/search_docs workflow when the raw ticket only states a high-level request or resolution outcome; downgrade to reconstructed or keep only directly evidenced facts."
        )
    if "redundant_ask_user_trace" in qa_flags or "under_extracted_grounded_trace_set" in qa_flags:
        instructions.append(
            "Replace ask_user_for_info traces that merely restate facts already present in the raw ticket with grounded execute_bash/search_docs traces only when explicit commands, shell values, file paths, or observed outputs are already quoted in the conversation."
        )
    if "synthetic_url_verification_in_grounded" in qa_flags:
        instructions.append(
            "Do not convert support links or portal URLs from the assistant reply into synthetic curl/wget verification traces unless the raw ticket explicitly includes those commands or their outputs."
        )

    return {
        "is_valid": is_valid,
        "release_tier": release_tier,
        "construction_mode": construction_mode,
        "has_reference_admin_reply": has_reference_admin_reply,
        "qa_flags": sorted(set(qa_flags)),
        "qa_notes": "",
        "repair_needed": repair_needed,
        "repair_instructions": " ".join(instructions).strip(),
        "info_issue_count": len(info_issues),
    }


def merge_validation_verdict(
    canonical: Dict[str, Any],
    llm_verdict: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    deterministic = deterministic_validate_canonical(canonical)
    if not isinstance(llm_verdict, dict):
        return deterministic

    merged = dict(deterministic)

    if isinstance(llm_verdict.get("is_valid"), bool):
        merged["is_valid"] = llm_verdict["is_valid"] and deterministic["is_valid"]

    release_tier = llm_verdict.get("release_tier")
    if release_tier in VALID_RELEASE_TIERS:
        merged["release_tier"] = release_tier
    construction_mode = llm_verdict.get("construction_mode")
    if construction_mode in VALID_CONSTRUCTION_MODES:
        merged["construction_mode"] = construction_mode

    if isinstance(llm_verdict.get("has_reference_admin_reply"), bool):
        merged["has_reference_admin_reply"] = (
            llm_verdict["has_reference_admin_reply"]
            and bool(canonical.get("evaluation", {}).get("reference_admin_reply", "").strip())
        )

    qa_flags = list(merged["qa_flags"])
    qa_flags.extend(_coerce_string_list(llm_verdict.get("qa_flags", [])))
    merged["qa_flags"] = sorted(set(qa_flags))
    merged["qa_notes"] = str(llm_verdict.get("qa_notes", "")).strip()
    if isinstance(llm_verdict.get("repair_needed"), bool):
        merged["repair_needed"] = llm_verdict["repair_needed"] or merged["repair_needed"]
    repair_instructions = str(llm_verdict.get("repair_instructions", "")).strip()
    if repair_instructions:
        merged["repair_instructions"] = repair_instructions

    enforced = deterministic_validate_canonical(
        {
            **canonical,
            "is_valid": merged["is_valid"],
            "release_tier": merged["release_tier"],
            "construction_mode": merged["construction_mode"],
            "evaluation": {
                **canonical.get("evaluation", {}),
                "has_reference_admin_reply": merged["has_reference_admin_reply"],
            },
            "quality": {
                **canonical.get("quality", {}),
                "qa_flags": merged["qa_flags"],
            },
        }
    )
    if not enforced["is_valid"]:
        merged["is_valid"] = False
    if enforced["release_tier"] != merged["release_tier"]:
        merged["release_tier"] = enforced["release_tier"]
        merged["construction_mode"] = enforced["construction_mode"]
    merged["qa_flags"] = sorted(set(merged["qa_flags"]) | set(enforced["qa_flags"]))
    merged["repair_needed"] = merged["repair_needed"] or enforced["repair_needed"]
    if not merged["repair_instructions"]:
        merged["repair_instructions"] = enforced["repair_instructions"]
    merged["has_reference_admin_reply"] = enforced["has_reference_admin_reply"]
    return merged


def apply_validation_verdict(
    canonical: Dict[str, Any],
    verdict: Dict[str, Any],
    validator_model: str,
) -> Dict[str, Any]:
    finalized = {
        **canonical,
        "is_valid": bool(verdict.get("is_valid", canonical.get("is_valid", True))),
        "release_tier": verdict.get("release_tier", canonical.get("release_tier", "reconstructed")),
        "construction_mode": verdict.get(
            "construction_mode",
            canonical.get("construction_mode", "reconstructed"),
        ),
    }
    evaluation = dict(finalized.get("evaluation", {}))
    has_reference_admin_reply = bool(
        verdict.get(
            "has_reference_admin_reply",
            evaluation.get("has_reference_admin_reply", False),
        )
    )
    evaluation["has_reference_admin_reply"] = has_reference_admin_reply
    if not has_reference_admin_reply:
        evaluation["reference_admin_reply"] = ""
    finalized["evaluation"] = evaluation

    quality = dict(finalized.get("quality", {}))
    qa_flags = sorted(
        set(_coerce_string_list(quality.get("qa_flags", [])))
        | set(_coerce_string_list(verdict.get("qa_flags", [])))
    )
    quality["validator_model"] = validator_model
    quality["qa_flags"] = qa_flags
    quality["qa_notes"] = str(verdict.get("qa_notes", quality.get("qa_notes", ""))).strip()
    finalized["quality"] = quality
    return finalized


def project_runtime_record(canonical: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = canonical.get("evaluation", {})
    return {
        "instance_id": canonical.get("instance_id"),
        "source_ticket_id": canonical.get("source_ticket_id"),
        "is_valid": canonical.get("is_valid", True),
        "release_tier": canonical.get("release_tier", "reconstructed"),
        "construction_mode": canonical.get("construction_mode", "reconstructed"),
        "instruction": canonical.get("instruction", ""),
        "traces": [
            {
                "trigger_command": trace.get("trigger_command", ""),
                "mock_output": trace.get("mock_output", ""),
            }
            for trace in canonical.get("traces", [])
            if isinstance(trace, dict)
        ],
        "evaluation": {
            "expected_trajectory": _coerce_string_list(
                evaluation.get("expected_trajectory", [])
            ),
            "final_solution_criteria": _coerce_string_list(
                evaluation.get("final_solution_criteria", [])
            ),
            "reference_admin_reply": str(
                evaluation.get("reference_admin_reply", "")
            ).strip(),
            "has_reference_admin_reply": bool(
                evaluation.get("has_reference_admin_reply", False)
            ),
        },
    }


def summarize_canonical_dataset(canonical_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_records = [record for record in canonical_records if record.get("is_valid", True)]
    grounded_records = [
        record for record in valid_records if record.get("release_tier") == "grounded"
    ]
    reconstructed_records = [
        record for record in valid_records if record.get("release_tier") == "reconstructed"
    ]
    first_action_counter: Counter[str] = Counter()
    gated_issue_count = 0

    for record in valid_records:
        traces = record.get("traces", [])
        if traces:
            first_action_counter[traces[0].get("tool", "unknown")] += 1
        if audit_ticket_information_flow(record):
            gated_issue_count += 1

    return {
        "grounded_count": len(grounded_records),
        "reconstructed_count": len(reconstructed_records),
        "invalid_count": len(canonical_records) - len(valid_records),
        "missing_info_audit_rate": (
            gated_issue_count / len(valid_records) if valid_records else 0.0
        ),
        "first_action_distribution": dict(sorted(first_action_counter.items())),
    }
