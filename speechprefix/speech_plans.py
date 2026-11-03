"""Validation for human-reviewed speech theme/section/talking-point plans."""

from __future__ import annotations


def validate_speech_plan(plan: dict, event: dict, *, require_reviewed: bool = False) -> None:
    """Validate plan provenance, hierarchy, and target-speaker span references."""
    if not isinstance(plan, dict) or type(plan.get("schema_version")) is not int:
        raise ValueError("speech plan must be an object with an integer schema_version")
    if not isinstance(event, dict):
        raise ValueError("source event must be an object")
    if plan["schema_version"] != 1:
        raise ValueError("unsupported speech-plan schema_version")
    if plan.get("event_id") != event.get("event_id"):
        raise ValueError("speech plan event_id must match its source event")
    status = plan.get("review_status")
    if not isinstance(status, str) or status not in {"draft", "reviewed"}:
        raise ValueError("review_status must be draft or reviewed")
    if status == "reviewed" and not (isinstance(plan.get("reviewed_by"), str) and plan["reviewed_by"].strip()):
        raise ValueError("reviewed plans require reviewed_by")
    if require_reviewed and status != "reviewed":
        raise ValueError("only reviewed speech plans may be used as training labels")

    turns = event.get("turns")
    if not isinstance(turns, list):
        raise ValueError("source event must contain a turns list")
    by_id = {turn.get("turn_id"): turn for turn in turns if isinstance(turn, dict)}
    if len(by_id) != len(turns) or None in by_id:
        raise ValueError("source event turn IDs must be present and unique")
    positions = {turn_id: index for index, turn_id in enumerate(by_id)}
    reviewed = status == "reviewed" or require_reviewed
    ids: set[str] = set()

    def check_node(node: dict, path: str, child_key: str | None) -> set[str]:
        if not isinstance(node, dict):
            raise ValueError(f"{path} must be an object")
        node_id = node.get("id")
        summary = node.get("summary")
        if not isinstance(node_id, str) or not node_id.strip() or node_id in ids:
            raise ValueError(f"{path}.id must be a unique nonempty string")
        ids.add(node_id)
        if not isinstance(summary, str) or (reviewed and not summary.strip()):
            raise ValueError(f"{path}.summary must be nonempty for reviewed plans")
        turn_ids = node.get("turn_ids")
        if not isinstance(turn_ids, list) or any(not isinstance(turn_id, str) for turn_id in turn_ids):
            raise ValueError(f"{path}.turn_ids must be a list of turn IDs")
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError(f"{path}.turn_ids must not contain duplicates")
        if reviewed and not turn_ids:
            raise ValueError(f"{path}.turn_ids must ground reviewed plan nodes in the transcript")
        if any(turn_id not in by_id for turn_id in turn_ids):
            raise ValueError(f"{path}.turn_ids reference an unknown source turn")
        if any(by_id[turn_id].get("speaker_role") != "target_speaker" for turn_id in turn_ids):
            raise ValueError(f"{path}.turn_ids must reference target-speaker turns only")
        order = [positions[turn_id] for turn_id in turn_ids]
        if order != sorted(order):
            raise ValueError(f"{path}.turn_ids must follow source transcript order")

        children = node.get(child_key, []) if child_key is not None else []
        if not isinstance(children, list):
            raise ValueError(f"{path}.{child_key} must be a list")
        if reviewed and child_key and not children:
            raise ValueError(f"{path}.{child_key} cannot be empty in a reviewed plan")
        child_turns = set()
        child_starts = []
        for index, child in enumerate(children):
            nested_key = "talking_points" if child_key == "sections" else None
            nested = check_node(child, f"{path}.{child_key}[{index}]", nested_key)
            child_turns.update(nested)
            child_starts.append(min((positions[turn_id] for turn_id in nested), default=-1))
        if child_starts != sorted(child_starts):
            raise ValueError(f"{path}.{child_key} must be in source transcript order")
        if not child_turns.issubset(set(turn_ids)):
            raise ValueError(f"{path}.turn_ids must include all turns referenced by child nodes")
        return set(turn_ids)

    themes = plan.get("themes")
    if not isinstance(themes, list):
        raise ValueError("speech plan themes must be a list")
    if reviewed and not themes:
        raise ValueError("reviewed speech plans require at least one theme")
    theme_starts = []
    for index, theme in enumerate(themes):
        covered = check_node(theme, f"themes[{index}]", "sections")
        theme_starts.append(min((positions[turn_id] for turn_id in covered), default=-1))
    if theme_starts != sorted(theme_starts):
        raise ValueError("themes must be in source transcript order")
