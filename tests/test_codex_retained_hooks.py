import json

from mempalace.hooks_cli import _count_human_messages, _extract_recent_messages
from mempalace.normalize import _try_codex_jsonl


def response(role, text, kinds=None, phase=None):
    payload = {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
        "internal_chat_message_metadata_passthrough": {"content_item_kinds": kinds or []},
    }
    if phase:
        payload["phase"] = phase
    return {"type": "response_item", "payload": payload}


def jsonl(rows):
    return "\n".join(json.dumps(row) for row in rows)


def test_retained_codex_conversation_excludes_harness_and_analysis(tmp_path):
    rows = [
        {"type": "session_meta"},
        response("user", "private harness injection", ["agents_md.instructions"]),
        response("user", "remember cobalt heron", ["user.text"]),
        response("assistant", "private reasoning", phase="analysis"),
        response("assistant", "heron remembered", phase="final_answer"),
    ]
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(jsonl(rows))
    assert _count_human_messages(str(transcript)) == 1
    assert _extract_recent_messages(str(transcript)) == ["remember cobalt heron"]
    normalized = _try_codex_jsonl(transcript.read_text())
    assert "remember cobalt heron" in normalized
    assert "heron remembered" in normalized
    assert "private harness injection" not in normalized
    assert "private reasoning" not in normalized


def test_old_codex_event_stream_remains_canonical(tmp_path):
    rows = [
        {"type": "session_meta"},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "old prompt"}},
        {"type": "event_msg", "payload": {"type": "agent_message", "message": "old answer"}},
        response("user", "old prompt", ["user.text"]),
        response("assistant", "old answer", phase="final_answer"),
    ]
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(jsonl(rows))
    assert _count_human_messages(str(transcript)) == 1
    assert _extract_recent_messages(str(transcript)) == ["old prompt"]
    assert _try_codex_jsonl(transcript.read_text()).count("old prompt") == 1


def test_incomplete_retained_conversation_is_not_imported():
    rows = [{"type": "session_meta"}, response("user", "only prompt", ["user.text"])]
    assert _try_codex_jsonl(jsonl(rows)) is None


def test_claude_user_messages_still_count(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(jsonl([{"message": {"role": "user", "content": "claude prompt"}}]))
    assert _count_human_messages(str(transcript)) == 1
    assert _extract_recent_messages(str(transcript)) == ["claude prompt"]
