"""grace.redaction: key patterns and the logging filter."""

from __future__ import annotations

import io
import logging

import pytest

from grace.redaction import (
    REDACTED,
    RedactingFilter,
    find_secrets,
    install_redaction,
    redact,
)

# Fake keys in each supported shape, assembled at runtime so the repo scan never sees them.
FAKES = {
    "google_api_key": "AI" + "za" + "SyA" + "b" * 32,
    "sk_key": "sk" + "-" + "proj-" + "Z9" * 20,
    "openrouter": "sk" + "-or-v1-" + "0f" * 32,
    "huggingface_token": "hf" + "_" + "Qw" * 17,
    "groq_api_key": "gsk" + "_" + "Mn" * 26,
}


@pytest.mark.parametrize("name", FAKES)
def test_each_key_shape_is_found_and_redacted(name: str) -> None:
    text = f"calling api with key={FAKES[name]} now"
    assert find_secrets(text)
    assert redact(text) == f"calling api with key={REDACTED} now"


@pytest.mark.parametrize(
    "text",
    [
        "task-runner-for-the-whole-pipeline-please",  # 'sk-' inside a word
        "dask-expr==1.1.21",
        "the hf_ prefix and gsk_ prefix alone",
        "AIza short",
        "sk-short",
        "ctrl+alt+m",
    ],
)
def test_ordinary_text_is_not_flagged(text: str) -> None:
    assert find_secrets(text) == []
    assert redact(text) == text


def test_find_secrets_never_returns_the_value() -> None:
    hits = find_secrets(FAKES["groq_api_key"])
    assert hits == [("groq_api_key", 0)]


def test_extra_literal_secrets_are_redacted() -> None:
    assert redact("token=plainsecretvalue", ["plainsecretvalue"]) == f"token={REDACTED}"
    assert redact("a b", ["", "b"]) == "a b"  # short / empty literals are ignored


def _logger(name: str) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger = logging.getLogger(name)
    logger.handlers[:] = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    install_redaction(logger=logger)
    return logger, stream


def test_filter_redacts_message_args_and_traceback() -> None:
    logger, stream = _logger("grace.test.redact1")
    key = FAKES["google_api_key"]

    logger.info("direct %s", key)
    logger.warning("dict %(k)s", {"k": FAKES["sk_key"]})
    try:
        raise RuntimeError(f"bad key {FAKES['groq_api_key']}")
    except RuntimeError:
        logger.exception("failed")
    logger.info("stack", stack_info=True, extra={})

    out = stream.getvalue()
    for fake in FAKES.values():
        assert fake not in out
    assert out.count(REDACTED) >= 3
    assert "RuntimeError: bad key [REDACTED]" in out


def test_filter_on_handler_covers_child_loggers() -> None:
    parent, stream = _logger("grace.test.redact2")
    child = logging.getLogger("grace.test.redact2.child")
    child.setLevel(logging.DEBUG)
    child.error("child leaked %s", FAKES["huggingface_token"])
    assert FAKES["huggingface_token"] not in stream.getvalue()
    assert REDACTED in stream.getvalue()


def test_install_is_idempotent_and_adds_literals() -> None:
    logger, stream = _logger("grace.test.redact3")
    install_redaction(["configured-value-123"], logger=logger)
    [handler] = logger.handlers
    assert sum(isinstance(f, RedactingFilter) for f in handler.filters) == 1
    logger.info("value configured-value-123")
    assert "configured-value-123" not in stream.getvalue()


def test_bad_format_string_does_not_crash() -> None:
    logger, stream = _logger("grace.test.redact4")
    logger.info("needs two %s %s", "one")
    assert "needs two" in stream.getvalue()
