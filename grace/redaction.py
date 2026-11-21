"""Secret patterns and a logging filter that redacts them.

The same patterns drive tests/test_no_secrets.py, so "what counts as a key" lives in one place.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

REDACTED = "[REDACTED]"

# Name -> pattern. The look-behinds stop matches inside ordinary words ("task-runner-...").
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    # OpenAI, OpenRouter (sk-or-v1-...), Stability and other sk- style keys.
    "sk_key": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{20,}"),
    "huggingface_token": re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}"),
    "groq_api_key": re.compile(r"(?<![A-Za-z0-9])gsk_[A-Za-z0-9]{40,}"),
}

# Configured key values shorter than this are not redacted literally (too many false hits).
_MIN_LITERAL_LEN = 8


def find_secrets(text: str) -> list[tuple[str, int]]:
    """Return (pattern name, offset) for every key-like match. Never returns the match itself."""
    return sorted(
        (name, m.start())
        for name, pattern in SECRET_PATTERNS.items()
        for m in pattern.finditer(text)
    )


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Replace key-like substrings and any of `extra_secrets` with REDACTED."""
    for secret in extra_secrets:
        if secret and len(secret) >= _MIN_LITERAL_LEN:
            text = text.replace(secret, REDACTED)
    for pattern in SECRET_PATTERNS.values():
        text = pattern.sub(REDACTED, text)
    return text


class RedactingFilter(logging.Filter):
    """Rewrites a record's message, traceback and stack so no key reaches a handler.

    Attach it to *handlers* (not loggers): logger filters do not see records propagated from
    child loggers, handler filters see everything that handler emits.
    """

    def __init__(self, extra_secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self.extra_secrets = tuple(s for s in extra_secrets if s)

    def add_secrets(self, secrets: Iterable[str]) -> None:
        self.extra_secrets += tuple(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not crash logging
            message = str(record.msg)
        record.msg = redact(message, self.extra_secrets)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text, self.extra_secrets)
        if record.stack_info:
            record.stack_info = redact(record.stack_info, self.extra_secrets)
        return True


def install_redaction(
    extra_secrets: Iterable[str] = (), logger: logging.Logger | None = None
) -> None:
    """Add a RedactingFilter to every handler of `logger` (default: root).

    Idempotent: a handler that already has one just learns the extra secrets.
    """
    logger = logger or logging.getLogger()
    secrets = tuple(extra_secrets)
    for handler in logger.handlers:
        existing = next((f for f in handler.filters if isinstance(f, RedactingFilter)), None)
        if existing is None:
            handler.addFilter(RedactingFilter(secrets))
        else:
            existing.add_secrets(secrets)


def configure_logging(level: int = logging.INFO, extra_secrets: Iterable[str] = ()) -> None:
    """Basic stderr logging with redaction on every root handler."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    install_redaction(extra_secrets)
