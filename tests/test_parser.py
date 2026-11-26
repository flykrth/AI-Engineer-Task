"""Unit tests for the multi-format WhatsApp parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.parser import parse_chat_file, parse_chat_text

TEAM_CHAT_PATH = Path(__file__).resolve().parent.parent / "data" / "team_chat.txt"


def test_parse_team_chat_completeness_and_spans() -> None:
    chat = parse_chat_file(TEAM_CHAT_PATH)

    assert chat.date_order == "DD/MM"
    assert chat.total_source_lines == 107
    assert len(chat.messages) == 102

    # Every line from 1 to 107 must be covered contiguously without gaps or overlaps.
    assert chat.messages[0].line_start == 1
    assert chat.messages[-1].line_end == 107
    for prev, curr in zip(chat.messages, chat.messages[1:]):
        assert curr.line_start == prev.line_end + 1
        assert curr.timestamp_iso >= prev.timestamp_iso


def test_parse_team_chat_multiline_and_flags() -> None:
    chat = parse_chat_file(TEAM_CHAT_PATH)

    multiline = [m for m in chat.messages if m.line_start == 17]
    assert len(multiline) == 1
    task_split = multiline[0]
    assert task_split.sender == "Aarav"
    assert task_split.line_start == 17
    assert task_split.line_end == 22
    assert task_split.kind == "chat"
    assert "Rohan - auth (login/signup)" in task_split.text
    assert "Main - deployment" in task_split.text

    system_msgs = [m for m in chat.messages if m.kind == "system"]
    assert len(system_msgs) == 3
    assert all(m.sender is None for m in system_msgs)
    assert [m.line_start for m in system_msgs] == [1, 2, 3]

    media_msgs = [m for m in chat.messages if m.kind == "media_omitted"]
    assert [m.line_start for m in media_msgs] == [29, 46, 61, 104]

    edited_msgs = [m for m in chat.messages if m.is_edited]
    assert len(edited_msgs) == 1
    edited = edited_msgs[0]
    assert edited.line_start == 41
    assert edited.sender == "Meera"
    assert edited.kind == "chat"
    assert edited.text == "Rs 800 theek hai. 5 log hain, split kar lenge"

    deleted_msgs = [m for m in chat.messages if m.kind == "deleted"]
    assert len(deleted_msgs) == 1
    assert deleted_msgs[0].line_start == 47
    assert deleted_msgs[0].sender == "Aarav"
    assert deleted_msgs[0].text == "This message was deleted"


def test_parse_ios_bracket_formats() -> None:
    raw = (
        "[03/10/26, 9:14:00 PM] Aarav created group \"Launch\"\n"
        "[03/10/26, 9:15:30 PM] Aarav: welcome everyone\n"
        "[03/10/2026, 21:16:05] Priya: image omitted\n"
        "[03/10/2026, 21:17:00] Meera: You deleted this message\n"
    )
    chat = parse_chat_text(raw)

    assert chat.date_order == "DD/MM"
    assert chat.total_source_lines == 4
    assert len(chat.messages) == 4

    assert chat.messages[0].kind == "system"
    assert chat.messages[0].sender is None
    assert chat.messages[0].timestamp_iso == "2026-10-03T21:14:00"

    assert chat.messages[1].kind == "chat"
    assert chat.messages[1].sender == "Aarav"
    assert chat.messages[1].timestamp_iso == "2026-10-03T21:15:30"

    assert chat.messages[2].kind == "media_omitted"
    assert chat.messages[2].sender == "Priya"

    assert chat.messages[3].kind == "deleted"
    assert chat.messages[3].sender == "Meera"


def test_parse_android_24h_and_12h_midnight_noon() -> None:
    raw = (
        "03/10/2026, 00:05 - Aarav: early morning check\n"
        "03/10/2026, 21:14 - Priya: night check\n"
        "04/10/26, 12:05 AM - Rohan: 12am check\n"
        "04/10/26, 12:05 PM - Meera: 12pm check\n"
    )
    chat = parse_chat_text(raw)

    assert [m.timestamp_iso for m in chat.messages] == [
        "2026-10-03T00:05:00",
        "2026-10-03T21:14:00",
        "2026-10-04T00:05:00",
        "2026-10-04T12:05:00",
    ]


def test_parse_invisible_unicode_markers() -> None:
    raw = (
        "\ufeff\u200e[03/10/26, 9:14:00\u202fPM] \u200eAarav: \u200fhello\u00a0team\n"
        "\u200e03/10/26, 9:15\u202fPM\u00a0-\u00a0Priya: \u200eimage omitted\n"
    )
    chat = parse_chat_text(raw)

    assert len(chat.messages) == 2
    assert chat.messages[0].sender == "Aarav"
    assert chat.messages[0].text == "hello team"
    assert chat.messages[0].timestamp_raw == "03/10/26, 9:14:00 PM"
    assert chat.messages[1].sender == "Priya"
    assert chat.messages[1].kind == "media_omitted"


def test_infer_mm_dd_from_values_over_12() -> None:
    raw = (
        "10/13/26, 9:00 AM - Aarav: start\n"
        "10/25/26, 10:30 AM - Priya: update\n"
    )
    chat = parse_chat_text(raw)

    assert chat.date_order == "MM/DD"
    assert chat.messages[0].timestamp_iso == "2026-10-13T09:00:00"
    assert chat.messages[1].timestamp_iso == "2026-10-25T10:30:00"


def test_infer_mm_dd_from_chronological_monotonicity() -> None:
    # Under DD/MM these dates would go May 3 -> Jun 3 -> Jan 4 (non-monotonic).
    # Under MM/DD they are Mar 5 -> Mar 6 -> Apr 1 (monotonic).
    raw = (
        "03/05/26, 9:00 AM - Aarav: day one\n"
        "03/06/26, 9:00 AM - Priya: day two\n"
        "04/01/26, 9:00 AM - Rohan: month two\n"
    )
    chat = parse_chat_text(raw)

    assert chat.date_order == "MM/DD"
    assert [m.timestamp_iso for m in chat.messages] == [
        "2026-03-05T09:00:00",
        "2026-03-06T09:00:00",
        "2026-04-01T09:00:00",
    ]


def test_ambiguous_dates_default_to_dd_mm() -> None:
    raw = (
        "03/10/26, 9:00 AM - Aarav: one\n"
        "05/10/26, 9:00 AM - Priya: two\n"
    )
    chat = parse_chat_text(raw)

    assert chat.date_order == "DD/MM"
    assert chat.messages[0].timestamp_iso == "2026-10-03T09:00:00"
    assert chat.messages[1].timestamp_iso == "2026-10-05T09:00:00"


def test_multiline_edited_message_and_system_colon() -> None:
    raw = (
        "03/10/26, 9:00 PM - Aarav changed the subject from \"Draft: v1\" to \"Final\"\n"
        "03/10/26, 9:05 PM - Meera: line one\n"
        "line two <This message was edited>\n"
    )
    chat = parse_chat_text(raw)

    assert len(chat.messages) == 2
    assert chat.messages[0].kind == "system"
    assert chat.messages[0].sender is None

    assert chat.messages[1].kind == "chat"
    assert chat.messages[1].sender == "Meera"
    assert chat.messages[1].is_edited is True
    assert chat.messages[1].line_start == 2
    assert chat.messages[1].line_end == 3
    assert chat.messages[1].text == "line one\nline two"


def test_empty_and_invalid_inputs() -> None:
    empty = parse_chat_text("")
    assert empty.messages == []
    assert empty.total_source_lines == 0

    with pytest.raises(ValueError, match="No valid WhatsApp timestamp headers"):
        parse_chat_text("not a whatsapp export\njust random text\n")


def test_parse_unseen_stress_test_fixtures() -> None:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    ios_path = fixtures_dir / "unseen_ios_24h_chat.txt"
    noisy_path = fixtures_dir / "unseen_noisy_large_chat.txt"

    ios_chat = parse_chat_file(ios_path)
    assert 100 <= ios_chat.total_source_lines <= 130
    assert ios_chat.date_order == "DD/MM"
    assert ios_chat.messages[0].line_start == 1
    assert ios_chat.messages[-1].line_end == ios_chat.total_source_lines
    assert any(m.kind == "deleted" for m in ios_chat.messages)
    assert any(m.kind == "media_omitted" for m in ios_chat.messages)
    assert any(m.is_edited for m in ios_chat.messages)
    for prev, curr in zip(ios_chat.messages, ios_chat.messages[1:]):
        assert curr.line_start == prev.line_end + 1
        assert curr.timestamp_iso >= prev.timestamp_iso

    noisy_chat = parse_chat_file(noisy_path)
    assert 550 <= noisy_chat.total_source_lines <= 650
    assert 260 <= len(noisy_chat.messages) <= 330
    assert noisy_chat.date_order == "DD/MM"
    assert noisy_chat.messages[0].line_start == 1
    assert noisy_chat.messages[-1].line_end == noisy_chat.total_source_lines
    assert any(m.kind == "deleted" for m in noisy_chat.messages)
    assert any(m.kind == "media_omitted" for m in noisy_chat.messages)
    assert any(m.is_edited for m in noisy_chat.messages)
    for prev, curr in zip(noisy_chat.messages, noisy_chat.messages[1:]):
        assert curr.line_start == prev.line_end + 1
        assert curr.timestamp_iso >= prev.timestamp_iso
