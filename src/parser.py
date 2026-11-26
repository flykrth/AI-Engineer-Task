"""Multi-format WhatsApp chat export parser."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MessageKind = Literal["chat", "system", "media_omitted", "deleted"]
DateOrder = Literal["DD/MM", "MM/DD"]

_INVISIBLE_CHARS_TRANS = str.maketrans(
    {
        "\ufeff": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u200e": "",
        "\u200f": "",
        "\u202a": "",
        "\u202b": "",
        "\u202c": "",
        "\u202d": "",
        "\u202e": "",
        "\u2066": "",
        "\u2067": "",
        "\u2068": "",
        "\u2069": "",
        "\u202f": " ",
        "\u00a0": " ",
        "\u2007": " ",
        "\u2009": " ",
    }
)

_DATE_TOKEN = (
    r"(?:(?P<iso_year>\d{4})[./-](?P<iso_month>\d{1,2})[./-](?P<iso_day>\d{1,2})"
    r"|(?P<d1>\d{1,2})[./-](?P<d2>\d{1,2})[./-](?P<year>\d{2,4}))"
)
_TIME_TOKEN = (
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
    r"(?:\s*(?P<ampm>[AaPp]\.?[Mm]\.?))?"
)

_ANDROID_HEADER_RE = re.compile(
    rf"^(?P<ts>{_DATE_TOKEN},?\s+{_TIME_TOKEN})\s+-\s+(?P<body>.*)$"
)
_IOS_HEADER_RE = re.compile(
    rf"^\[(?P<ts>{_DATE_TOKEN},?\s+{_TIME_TOKEN})\]\s*(?P<body>.*)$"
)
_SENDER_SPLIT_RE = re.compile(r"^(?P<sender>[^:\n]+?):(?:\s+(?P<text>.*)|)$")
_EDITED_MARKER_RE = re.compile(r"(?:\s*<This message was edited>\s*)+$", re.IGNORECASE)
_DELETED_TEXT_RE = re.compile(
    r"^(?:this message was deleted|you deleted this message)\.?$", re.IGNORECASE
)
_MEDIA_OMITTED_RE = re.compile(
    r"^(?:<\s*media\s+omitted\s*>|(?:image|video|audio|sticker|gif|document|contact card|media)\s+omitted)$",
    re.IGNORECASE,
)

_SYSTEM_SENDER_PHRASES = (
    "messages and calls are end-to-end encrypted",
    "created group ",
    "added ",
    "removed ",
    "joined using ",
    "changed the subject",
    "changed this group",
    "changed the group",
    "deleted this group",
    "security code",
    "pinned a message",
    "turned on disappearing messages",
    "turned off disappearing messages",
)


class ParsedMessage(BaseModel):
    """Normalized representation of one WhatsApp message or system event."""

    msg_id: str
    line_start: int
    line_end: int
    timestamp_raw: str
    timestamp_iso: str
    sender: str | None
    text: str
    kind: MessageKind
    is_edited: bool = False


class ParsedChat(BaseModel):
    """Container for parsed messages and file-level metadata."""

    messages: list[ParsedMessage] = Field(default_factory=list)
    date_order: DateOrder = "DD/MM"
    total_source_lines: int = 0
    source_path: str | None = None


class _RawHeaderMatch(BaseModel):
    """Intermediate header parse before date order resolution."""

    line_no: int
    timestamp_raw: str
    body: str
    iso_year: int | None = None
    iso_month: int | None = None
    iso_day: int | None = None
    d1: int | None = None
    d2: int | None = None
    year: int | None = None
    hour: int
    minute: int
    second: int
    ampm: str | None = None


class _PendingMessage:
    """Mutable accumulator used while merging multi-line continuations."""

    def __init__(self, header: _RawHeaderMatch, timestamp_iso: str) -> None:
        self.header = header
        self.timestamp_iso = timestamp_iso
        self.line_start = header.line_no
        self.line_end = header.line_no
        self.lines: list[str] = [header.body]

    def append_continuation(self, line: str, line_no: int) -> None:
        self.lines.append(line)
        self.line_end = line_no


def clean_invisible_unicode(text: str) -> str:
    """Normalize invisible WhatsApp formatting marks and special spaces."""
    return text.translate(_INVISIBLE_CHARS_TRANS)


def _match_header(cleaned_line: str, line_no: int) -> _RawHeaderMatch | None:
    match = _ANDROID_HEADER_RE.match(cleaned_line) or _IOS_HEADER_RE.match(cleaned_line)
    if not match:
        return None
    groups = match.groupdict()
    ampm_raw = groups["ampm"]
    ampm = ampm_raw.replace(".", "").upper() if ampm_raw else None
    return _RawHeaderMatch(
        line_no=line_no,
        timestamp_raw=groups["ts"].strip(),
        body=groups["body"],
        iso_year=int(groups["iso_year"]) if groups["iso_year"] else None,
        iso_month=int(groups["iso_month"]) if groups["iso_month"] else None,
        iso_day=int(groups["iso_day"]) if groups["iso_day"] else None,
        d1=int(groups["d1"]) if groups["d1"] else None,
        d2=int(groups["d2"]) if groups["d2"] else None,
        year=int(groups["year"]) if groups["year"] else None,
        hour=int(groups["hour"]),
        minute=int(groups["minute"]),
        second=int(groups["second"]) if groups["second"] else 0,
        ampm=ampm,
    )


def _normalize_year(year: int) -> int:
    return 2000 + year if year < 100 else year


def _build_datetime(header: _RawHeaderMatch, date_order: DateOrder) -> datetime:
    if header.iso_year is not None:
        year = header.iso_year
        month = header.iso_month or 0
        day = header.iso_day or 0
    else:
        assert header.d1 is not None and header.d2 is not None and header.year is not None
        year = _normalize_year(header.year)
        if date_order == "DD/MM":
            day, month = header.d1, header.d2
        else:
            month, day = header.d1, header.d2

    hour = header.hour
    if header.ampm is not None:
        if hour < 1 or hour > 12:
            raise ValueError(f"Invalid 12-hour value {hour} at line {header.line_no}")
        if header.ampm == "AM":
            hour = 0 if hour == 12 else hour
        elif header.ampm == "PM":
            hour = 12 if hour == 12 else hour + 12

    return datetime(year, month, day, hour, header.minute, header.second)


def _score_date_order(
    headers: list[_RawHeaderMatch], date_order: DateOrder
) -> tuple[int, int]:
    """Return (invalid_count, chronological_inversions) for a candidate date order."""
    invalid_count = 0
    inversions = 0
    prev_dt: datetime | None = None

    for header in headers:
        try:
            dt = _build_datetime(header, date_order)
        except ValueError:
            invalid_count += 1
            continue
        if prev_dt is not None and dt < prev_dt:
            inversions += 1
        prev_dt = dt

    return invalid_count, inversions


def infer_date_order(headers: list[_RawHeaderMatch]) -> DateOrder:
    """Infer DD/MM vs MM/DD across all headers, defaulting to DD/MM on ambiguity."""
    numeric_headers = [h for h in headers if h.iso_year is None]
    if not numeric_headers:
        return "DD/MM"

    dmy_score = _score_date_order(numeric_headers, "DD/MM")
    mdy_score = _score_date_order(numeric_headers, "MM/DD")

    if mdy_score < dmy_score:
        return "MM/DD"
    return "DD/MM"


def _looks_like_system_prefix(candidate_sender: str) -> bool:
    if '"' in candidate_sender or len(candidate_sender) > 80:
        return True
    lowered = candidate_sender.lower()
    return any(phrase in lowered for phrase in _SYSTEM_SENDER_PHRASES)


def _classify_and_clean_body(
    full_body: str,
) -> tuple[str | None, str, MessageKind, bool]:
    """Split sender from body, strip edit markers, and classify message kind."""
    first_line, has_newline, rest_lines = full_body.partition("\n")
    sender: str | None = None
    body_text = full_body

    sender_match = _SENDER_SPLIT_RE.match(first_line)
    if sender_match:
        candidate_sender = sender_match.group("sender").strip()
        if candidate_sender and not _looks_like_system_prefix(candidate_sender):
            sender = candidate_sender
            first_body = sender_match.group("text") or ""
            body_text = f"{first_body}\n{rest_lines}" if has_newline else first_body

    is_edited = False
    cleaned_body, edit_count = _EDITED_MARKER_RE.subn("", body_text.rstrip())
    if edit_count > 0:
        is_edited = True
    cleaned_body = cleaned_body.strip()

    if _DELETED_TEXT_RE.match(cleaned_body):
        kind: MessageKind = "deleted"
    elif _MEDIA_OMITTED_RE.match(cleaned_body):
        kind = "media_omitted"
    elif sender is None:
        kind = "system"
    else:
        kind = "chat"

    return sender, cleaned_body, kind, is_edited


def parse_chat_text(content: str, *, source_path: str | None = None) -> ParsedChat:
    """Parse raw WhatsApp export text into normalized chronological messages."""
    raw_lines = content.splitlines()
    total_source_lines = len(raw_lines)
    if total_source_lines == 0:
        return ParsedChat(
            messages=[],
            date_order="DD/MM",
            total_source_lines=0,
            source_path=source_path,
        )

    cleaned_lines = [clean_invisible_unicode(line) for line in raw_lines]
    matched_headers: list[_RawHeaderMatch | None] = [
        _match_header(line, idx) for idx, line in enumerate(cleaned_lines, start=1)
    ]
    headers_only = [h for h in matched_headers if h is not None]
    if not headers_only:
        raise ValueError("No valid WhatsApp timestamp headers found in chat text.")

    date_order = infer_date_order(headers_only)

    pending_messages: list[_PendingMessage] = []
    for line_no, (cleaned_line, header) in enumerate(
        zip(cleaned_lines, matched_headers), start=1
    ):
        if header is not None:
            try:
                dt = _build_datetime(header, date_order)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid timestamp '{header.timestamp_raw}' at line {line_no} "
                    f"for detected date order {date_order}."
                ) from exc
            pending_messages.append(_PendingMessage(header, dt.isoformat()))
        elif pending_messages:
            if cleaned_line.strip() == "" and line_no == total_source_lines:
                continue
            pending_messages[-1].append_continuation(cleaned_line, line_no)

    messages: list[ParsedMessage] = []
    for idx, pending in enumerate(pending_messages, start=1):
        full_body = "\n".join(pending.lines)
        sender, text, kind, is_edited = _classify_and_clean_body(full_body)
        messages.append(
            ParsedMessage(
                msg_id=f"M{idx:03d}",
                line_start=pending.line_start,
                line_end=pending.line_end,
                timestamp_raw=pending.header.timestamp_raw,
                timestamp_iso=pending.timestamp_iso,
                sender=sender,
                text=text,
                kind=kind,
                is_edited=is_edited,
            )
        )

    return ParsedChat(
        messages=messages,
        date_order=date_order,
        total_source_lines=total_source_lines,
        source_path=source_path,
    )


def parse_chat_file(path: str | Path) -> ParsedChat:
    """Read and parse a WhatsApp chat export file from disk."""
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")
    return parse_chat_text(content, source_path=str(file_path))
