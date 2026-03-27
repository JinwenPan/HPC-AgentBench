import re
import shlex
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ACTION_CALL_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$", re.DOTALL)
TOOL_PARAM_KEYS = {
    "execute_bash": "command",
    "search_docs": "query",
    "ask_user_for_info": "question",
}
PARAM_MATCH_THRESHOLDS = {
    "execute_bash": 0.85,
    "search_docs": 0.70,
    "ask_user_for_info": 0.70,
}
PATH_RE = re.compile(r"(/[\w.\-]+(?:/[\w.\-]+)+)")
URL_RE = re.compile(r"https?://[^\s)'\"]+")
GENERIC_ID_RE = re.compile(r"\b\d{3,}\b")
JOB_ID_PATTERNS = [
    re.compile(r"\bjob(?:id)?[=\s:]+(\d{3,})", re.IGNORECASE),
    re.compile(r"\bshow job\s+(\d{3,})", re.IGNORECASE),
    re.compile(r"\b-j\s*(\d{3,})", re.IGNORECASE),
]
USER_PATTERNS = [
    re.compile(r"(?:^|\s)-u\s+([a-z][a-z0-9_]{2,})", re.IGNORECASE),
    re.compile(r"\busername\s*(?:is|=|:)\s*['\"]?([a-z][a-z0-9_]{2,})", re.IGNORECASE),
    re.compile(r"\buser(?:id)?\s*(?:is|=|:)\s*['\"]?([a-z][a-z0-9_]{2,})", re.IGNORECASE),
    re.compile(r"\buserid=([a-z][a-z0-9_]{2,})", re.IGNORECASE),
    re.compile(r"\bname=([a-z][a-z0-9_]{2,})", re.IGNORECASE),
    re.compile(r"\b([a-z][a-z0-9_]{2,})@[a-z0-9.\-]+\b", re.IGNORECASE),
]
RESOURCE_PATTERNS = [
    re.compile(r"\b\d+\s*(?:cpus?|nodes?|tasks?)\b", re.IGNORECASE),
    re.compile(r"(?<![%./])\b\d+\s*(?:g|gb|m|mb|t|tb)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(?:hours?|hrs?|minutes?|mins?)\b", re.IGNORECASE),
    re.compile(r"\b\d+:\d{2}:\d{2}\b"),
]
PATH_USER_RE = re.compile(r"/(?:work|work2|home|home1|scratch|scratch1|scratch2)/\d*/?([A-Za-z0-9_]+)/")
GENERIC_USERNAME_TOKENS = {
    "account",
    "activation",
    "admin",
    "associated",
    "deactivation",
    "email",
    "name",
    "portal",
    "user",
    "username",
    "where",
}
PHRASE_REPLACEMENTS = (
    (re.compile(r"\bmax job configuration error\b", re.IGNORECASE), "maxjobconfig"),
    (re.compile(r"\bmax job configuration\b", re.IGNORECASE), "maxjobconfig"),
    (re.compile(r"\bmax job config\b", re.IGNORECASE), "maxjobconfig"),
    (re.compile(r"\bjob configuration error\b", re.IGNORECASE), "jobconfigerror"),
    (re.compile(r"\bidev session\b", re.IGNORECASE), "idev"),
    (re.compile(r"\bjob request\b", re.IGNORECASE), "jobrequest"),
)
SEMANTIC_TOKEN_RE = re.compile(r"[a-z0-9_./-]+")
SEMANTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "exact",
    "for",
    "from",
    "help",
    "hello",
    "hi",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "let",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "out",
    "please",
    "provide",
    "received",
    "regarding",
    "resolve",
    "resolution",
    "session",
    "so",
    "solution",
    "that",
    "the",
    "their",
    "this",
    "to",
    "trying",
    "use",
    "want",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
    "your",
}


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_trigger_command(trigger_command: str) -> Optional[str]:
    """Canonicalize a trigger command into ``action(argument)`` form."""
    if not isinstance(trigger_command, str):
        return None

    compact = normalize_whitespace(trigger_command)
    if not compact:
        return None

    match = ACTION_CALL_RE.match(compact)
    if not match:
        return None

    action, argument = match.group(1), match.group(2).strip()
    if action not in TOOL_PARAM_KEYS:
        return None

    return f"{action}({argument})"


def parse_trigger_command(trigger_command: str) -> Optional[Tuple[str, str, str]]:
    normalized = normalize_trigger_command(trigger_command)
    if normalized is None:
        return None

    match = ACTION_CALL_RE.match(normalized)
    if match is None:
        return None
    return match.group(1), match.group(2), normalized


def extract_action_invocation(
    action_dict: Any,
) -> Optional[Tuple[str, str, str, bool]]:
    """Best-effort extraction of tool action, argument, and normalized trigger."""
    if not isinstance(action_dict, dict):
        return None

    action_name = action_dict.get("action")
    if action_name not in TOOL_PARAM_KEYS:
        return None

    params = action_dict.get("params", {})
    if not isinstance(params, dict):
        params = {}

    expected_param_key = TOOL_PARAM_KEYS[action_name]
    schema_valid = set(params.keys()) == {expected_param_key}

    if expected_param_key in params and isinstance(params.get(expected_param_key), str):
        argument = params[expected_param_key]
    else:
        string_values = [value for value in params.values() if isinstance(value, str)]
        if len(string_values) == 1:
            argument = string_values[0]
        elif string_values:
            argument = " ".join(string_values)
        else:
            argument = ""

    normalized_trigger = normalize_trigger_command(f"{action_name}({argument})")
    if normalized_trigger is None:
        return None

    return action_name, argument, normalized_trigger, schema_valid


def _normalize_semantic_text(text: str) -> str:
    normalized = text.lower()
    for pattern, replacement in PHRASE_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    normalized = normalized.replace('"', " ").replace("'", " ")
    return normalized


def _stem_token(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def semantic_tokens(text: str) -> Set[str]:
    normalized = _normalize_semantic_text(text)
    tokens = set()
    for raw_token in SEMANTIC_TOKEN_RE.findall(normalized):
        if PATH_RE.fullmatch(raw_token):
            continue
        token = _stem_token(raw_token.strip("._-"))
        if not token or token in SEMANTIC_STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.add(token)
    return tokens


def _extract_job_ids(text: str) -> Set[str]:
    job_ids = set()
    for pattern in JOB_ID_PATTERNS:
        for job_id in pattern.findall(text):
            job_ids.add(job_id)
    return job_ids


def _extract_paths(text: str, filesystem_only: bool = False) -> List[str]:
    url_spans = [match.span() for match in URL_RE.finditer(text)] if filesystem_only else []
    paths: List[str] = []

    for match in PATH_RE.finditer(text):
        start, end = match.span()
        if filesystem_only and any(url_start <= start and end <= url_end for url_start, url_end in url_spans):
            continue
        paths.append(match.group(1))

    return paths


def _extract_usernames(text: str, include_paths: bool = True) -> Set[str]:
    usernames = set()
    for pattern in USER_PATTERNS:
        for username in pattern.findall(text):
            normalized = username.lower()
            if normalized in GENERIC_USERNAME_TOKENS:
                continue
            usernames.add(normalized)
    if include_paths:
        for path in _extract_paths(text):
            match = PATH_USER_RE.search(path)
            if match:
                normalized = match.group(1).lower()
                if normalized not in GENERIC_USERNAME_TOKENS:
                    usernames.add(normalized)
    return usernames


def _normalize_resource(resource: str) -> str:
    return normalize_whitespace(resource.lower()).replace(" ", "")


def _extract_resources(text: str) -> Set[str]:
    resources = set()
    for pattern in RESOURCE_PATTERNS:
        for resource in pattern.findall(text):
            resources.add(_normalize_resource(resource))
    return resources


def semantic_entities(text: str) -> Set[str]:
    normalized = _normalize_semantic_text(text)
    entities = set()
    for path in _extract_paths(normalized):
        entities.add(f"path:{path}")
    for job_id in _extract_job_ids(normalized):
        entities.add(f"job:{job_id}")
    for username in _extract_usernames(normalized):
        entities.add(f"user:{username}")
    for resource in _extract_resources(normalized):
        entities.add(f"resource:{resource}")
    for token in semantic_tokens(normalized):
        if any(char.isdigit() for char in token) or "_" in token or token in {
            "maverick2",
            "stampede2",
            "lonestar6",
            "ls6",
            "maxjobconfig",
            "debug",
            "normal",
            "queue",
            "partition",
        }:
            entities.add(f"term:{token}")
    return entities


def _overlap_metrics(expected: Set[str], candidate: Set[str]) -> Tuple[float, float, float]:
    if not expected:
        return 1.0, 1.0, 1.0
    if not candidate:
        return 0.0, 0.0, 0.0

    overlap = len(expected & candidate)
    recall = overlap / len(expected)
    precision = overlap / len(candidate)
    if recall == 0.0 or precision == 0.0:
        return recall, precision, 0.0
    f1 = 2 * recall * precision / (recall + precision)
    return recall, precision, f1


def score_textual_semantics(candidate: str, reference: str) -> float:
    candidate_norm = _normalize_semantic_text(candidate)
    reference_norm = _normalize_semantic_text(reference)
    if candidate_norm == reference_norm:
        return 1.0

    candidate_tokens = semantic_tokens(candidate_norm)
    reference_tokens = semantic_tokens(reference_norm)
    token_recall, _, token_f1 = _overlap_metrics(reference_tokens, candidate_tokens)

    candidate_entities = semantic_entities(candidate_norm)
    reference_entities = semantic_entities(reference_norm)
    entity_recall, _, _ = _overlap_metrics(reference_entities, candidate_entities)

    score = 0.50 * token_recall + 0.35 * entity_recall + 0.15 * token_f1
    return _clamp(score)


def _split_shell_prefix(command: str) -> str:
    return re.split(r"\s*(?:&&|\|\||\||;)\s*", command, maxsplit=1)[0].strip()


def _shell_tokens(command: str) -> List[str]:
    prefix = _split_shell_prefix(command)
    if not prefix:
        return []
    try:
        return shlex.split(prefix)
    except ValueError:
        return prefix.split()


def _command_template(command: str) -> str:
    tokens = _shell_tokens(command)
    if not tokens:
        return ""

    lowered = [token.lower() for token in tokens]
    main = lowered[0]
    if main == "scontrol" and len(lowered) >= 3 and lowered[1] == "show":
        return f"scontrol show {lowered[2]}"
    if main in {"squeue", "sacct"}:
        if "-u" in lowered:
            return f"{main} -u"
        if "-j" in lowered:
            return f"{main} -j"
        return main
    if main == "module" and len(lowered) >= 2:
        return f"module {lowered[1]}"
    if main == "cat":
        return "cat"
    return main


def _command_keywords(command: str) -> Set[str]:
    tokens = _shell_tokens(command)
    keywords = set()
    for token in tokens:
        lowered = token.lower()
        if lowered.startswith("-"):
            continue
        if PATH_RE.fullmatch(lowered):
            continue
        if GENERIC_ID_RE.fullmatch(lowered):
            continue
        stemmed = _stem_token(lowered.strip("._-"))
        if stemmed and stemmed not in SEMANTIC_STOPWORDS:
            keywords.add(stemmed)
    return keywords


def _command_flags(command: str) -> Set[str]:
    return {
        token.lower()
        for token in _shell_tokens(command)
        if token.startswith("-")
    }


def _path_similarity(candidate_paths: Sequence[str], reference_paths: Sequence[str]) -> float:
    if not reference_paths:
        return 1.0
    if not candidate_paths:
        return 0.0

    best = 0.0
    for reference_path in reference_paths:
        ref_segments = [segment for segment in reference_path.split("/") if segment]
        for candidate_path in candidate_paths:
            if candidate_path == reference_path:
                return 1.0
            candidate_segments = [segment for segment in candidate_path.split("/") if segment]
            overlap = len(set(ref_segments) & set(candidate_segments))
            if ref_segments:
                best = max(best, 0.5 * (overlap / len(ref_segments)))
    return _clamp(best)


def _entity_recall(candidate: Set[str], reference: Set[str]) -> float:
    recall, _, _ = _overlap_metrics(reference, candidate)
    return recall


def score_execute_bash(candidate: str, reference: str) -> float:
    candidate_norm = normalize_whitespace(candidate)
    reference_norm = normalize_whitespace(reference)
    if candidate_norm == reference_norm:
        return 1.0

    candidate_template = _command_template(candidate_norm)
    reference_template = _command_template(reference_norm)
    candidate_main = candidate_template.split()[0] if candidate_template else ""
    reference_main = reference_template.split()[0] if reference_template else ""
    if candidate_template and candidate_template == reference_template:
        template_score = 1.0
    elif candidate_main and candidate_main == reference_main:
        template_score = 0.5
    else:
        template_score = 0.0

    candidate_keywords = _command_keywords(candidate_norm) | _command_flags(candidate_norm)
    reference_keywords = _command_keywords(reference_norm) | _command_flags(reference_norm)
    _, _, keyword_f1 = _overlap_metrics(reference_keywords, candidate_keywords)

    candidate_paths = PATH_RE.findall(candidate_norm)
    reference_paths = PATH_RE.findall(reference_norm)
    path_score = _path_similarity(candidate_paths, reference_paths)

    candidate_job_ids = {f"job:{job_id}" for job_id in _extract_job_ids(candidate_norm)}
    reference_job_ids = {f"job:{job_id}" for job_id in _extract_job_ids(reference_norm)}
    job_score = _entity_recall(candidate_job_ids, reference_job_ids) if reference_job_ids else 1.0

    candidate_users = {f"user:{user}" for user in _extract_usernames(candidate_norm, include_paths=False)}
    reference_users = {f"user:{user}" for user in _extract_usernames(reference_norm, include_paths=False)}
    user_score = _entity_recall(candidate_users, reference_users) if reference_users else 1.0

    candidate_resources = {f"resource:{resource}" for resource in _extract_resources(candidate_norm)}
    reference_resources = {f"resource:{resource}" for resource in _extract_resources(reference_norm)}
    resource_score = (
        _entity_recall(candidate_resources, reference_resources)
        if reference_resources
        else 1.0
    )

    entity_components = []
    if reference_paths:
        entity_components.append(path_score)
    if reference_job_ids:
        entity_components.append(job_score)
    if reference_users:
        entity_components.append(user_score)
    if reference_resources:
        entity_components.append(resource_score)

    if entity_components:
        entity_score = sum(entity_components) / len(entity_components)
    else:
        entity_score = score_textual_semantics(candidate_norm, reference_norm)

    score = 0.45 * template_score + 0.35 * entity_score + 0.20 * keyword_f1
    return _clamp(score)


def score_tool_arguments(action_name: str, candidate_argument: str, reference_argument: str) -> float:
    if action_name == "execute_bash":
        return score_execute_bash(candidate_argument, reference_argument)
    return score_textual_semantics(candidate_argument, reference_argument)


def param_match_threshold(action_name: str) -> float:
    return PARAM_MATCH_THRESHOLDS[action_name]


def audit_information_entities(text: str) -> Set[str]:
    entities = set()
    normalized = _normalize_semantic_text(text)
    for path in _extract_paths(normalized, filesystem_only=True):
        entities.add(f"path:{path}")
    for job_id in _extract_job_ids(normalized):
        entities.add(f"job:{job_id}")
    for username in _extract_usernames(normalized):
        entities.add(f"user:{username}")
    for resource in _extract_resources(normalized):
        entities.add(f"resource:{resource}")
    return entities


def audit_ticket_information_flow(ticket: Dict[str, Any]) -> List[Dict[str, Any]]:
    instruction = ticket.get("instruction", "")
    known_entities = set(audit_information_entities(instruction))
    issues: List[Dict[str, Any]] = []

    traces = ticket.get("traces", [])
    if not isinstance(traces, list):
        return issues

    for trace_index, trace in enumerate(traces):
        if not isinstance(trace, dict):
            continue
        trigger_command = trace.get("trigger_command", "")
        parsed = parse_trigger_command(trigger_command)
        if parsed is None:
            continue

        action_name, argument, normalized_trigger = parsed
        trigger_entities = audit_information_entities(argument)
        missing_entities = sorted(trigger_entities - known_entities)
        if missing_entities:
            issues.append(
                {
                    "trace_index": trace_index,
                    "action": action_name,
                    "trigger_command": normalized_trigger,
                    "missing_entities": missing_entities,
                    "missing_categories": sorted(
                        {entity.split(":", 1)[0] for entity in missing_entities}
                    ),
                }
            )
        known_entities.update(trigger_entities)

        mock_output = trace.get("mock_output", "")
        if isinstance(mock_output, str):
            known_entities.update(audit_information_entities(mock_output))

    return issues


def audit_cleaned_dataset(dataset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    flagged_tickets = []
    category_counts: Dict[str, int] = {}

    for ticket in dataset:
        issues = audit_ticket_information_flow(ticket)
        if not issues:
            continue

        for issue in issues:
            for category in issue["missing_categories"]:
                category_counts[category] = category_counts.get(category, 0) + 1

        flagged_tickets.append(
            {
                "instance_id": ticket.get("instance_id"),
                "issue_count": len(issues),
                "issues": issues,
            }
        )

    return {
        "total_tickets": len(dataset),
        "flagged_ticket_count": len(flagged_tickets),
        "missing_category_counts": dict(sorted(category_counts.items())),
        "flagged_tickets": flagged_tickets,
    }
