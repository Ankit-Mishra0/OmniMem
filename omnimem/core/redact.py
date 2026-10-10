"""
Secret scrubbing, applied before text is stored.

Browser capture reads whatever is on screen, which regularly includes a key
someone pasted into a chat. Redaction happens on the way *in* -- once a secret
is in index.db it is also in a rendered brief that gets injected into three
other tools.

This is a safety net, not a guarantee. It catches the shapes that actually
show up in chat transcripts; it cannot catch an opaque credential that looks
like an ordinary word.
"""

import re
from typing import List, Pattern, Tuple

_RULES: List[Tuple[Pattern[str], str]] = [
    # Ordered: the most specific shapes first, so a generic assignment rule
    # cannot swallow a token we could have labelled precisely.
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "[redacted:private-key]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[redacted:aws-key]"),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), "[redacted:aws-key]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[redacted:github-token]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "[redacted:api-key]"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), "[redacted:api-key]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), "[redacted:slack-token]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[redacted:google-key]"),
    (re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
     "[redacted:jwt]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[redacted:email]"),
    # Credentials embedded in a URL.
    (re.compile(r"://[^\s/:@]+:[^\s/@]+@"), "://[redacted:userinfo]@"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{16,}"), r"\1 [redacted:token]"),
    # key = value assignments, the catch-all.
    (re.compile(
        r"(?i)\b(api[_\-]?key|secret|secret[_\-]?key|password|passwd|pwd|token|"
        r"access[_\-]?token|refresh[_\-]?token|client[_\-]?secret|auth)"
        r"(\s*[:=]\s*)(['\"]?)([^\s'\"]{8,})\3"),
     r"\1\2\3[redacted]\3"),
]


def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for pattern, replacement in _RULES:
        out = pattern.sub(replacement, out)
    return out


def has_secret(text: str) -> bool:
    return redact(text) != text
