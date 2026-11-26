"""LLM router: what the grammar can't parse -> validated Intents, through a provider chain.

    final transcript -> grammar.parse() -> follow-up resolver -> Router.route() -> Intents

Grammar hits never reach this module. For everything else:

- Providers are tried in settings order (groq -> gemini -> openrouter). Requests time out
  after 3 s. HTTP 429, a timeout or any other provider error puts that provider in a 60 s
  cooldown (grace/core/cooldown.py) and the next one is tried. A provider whose free quota is
  used up (grace/core/quota.py) is skipped. Nothing raises out of ``route``; when every
  provider is down GRACE says so and fixed commands keep working.
- The LLM is a stateless orchestrator: the router owns the dialogue (last 6 exchanges + the
  last action) and sends it with the tool list on every call, so any provider can take over
  mid-conversation. No server-side conversation ids.
- The model must answer with tool calls (or plain text, taken as a spoken answer). Every call
  is validated with pydantic into an Intent (grace/intent/tools.py). An invalid reply (bad JSON,
  unknown tool, bad arguments) gets ONE retry on the same provider with the error explained,
  then the next provider is tried; only when all of them fail does GRACE ask to rephrase.
- Multi-step requests: a call with ``more: true`` means another step follows. The router shows
  the model its calls as queued and asks again (at most MAX_ROUNDS calls per request). Nothing
  runs until ``route`` returns, so a provider failing half-way just restarts on the next one.
- The system prompt and tool list are built once and are byte-identical on every call (keeps
  provider prompt caches warm). What changes per call (history, last action, time) goes after
  them, in the messages.
- Risky actions are never offered as tools. If a model invents one anyway it fails validation.
- ``complete`` (code analysis) runs the same chain without tools and can exclude providers:
  screen text never goes to Gemini.

Runs on the intent thread (grace-intent); it blocks only that thread.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from grace.core.cooldown import Cooldowns
from grace.core.quota import NullQuota, Quota
from grace.intent.dialogue import Dialogue
from grace.intent.providers import (
    ChatProvider,
    ChatReply,
    ProviderAuthError,
    ProviderError,
    QuotaExceededError,
)
from grace.intent.tools import ANSWER_TOOL, ToolCallError, ToolSet
from grace.intent.types import Intent
from grace.metrics.log import Metrics, NullMetrics

log = logging.getLogger(__name__)

# Byte-identical on every call (see module doc). Short: every token counts against free quotas.
SYSTEM_PROMPT = (
    "You are GRACE, a voice assistant that controls the user's Windows computer. Every reply "
    "is a tool call. To do something on the computer, call the matching tool; never claim you "
    "did something without calling a tool. To talk (answers, questions, small talk, or when "
    "you can't help), call answer_user with at most two short spoken sentences of plain text: "
    "no markdown, lists, emoji or URLs. For several requests in one sentence, call one tool per "
    "step in order and set more=true on every step except the last. If the request is "
    "unclear, ask a short question with "
    "answer_user. 'Last action' in the user's message is what GRACE just did: use it for "
    "follow-ups like 'a bit more', 'undo that' or 'now the brightness'. You cannot shut down, "
    "restart or sleep the computer: tell the user to say 'shut down the computer' themselves, "
    "which asks for confirmation."
)
REPHRASE = "Sorry, I didn't get that. Could you say it another way?"
UNAVAILABLE = "I can't reach an assistant right now. Fixed commands still work."
NO_PROVIDER = "I'm not sure how to help with that. Say help for the commands I know."
ANALYSIS_INSTRUCTION = "Answer briefly in plain spoken English, no markdown."
MAX_INTENTS = 3
MAX_ROUNDS = 3  # model calls per request when it asks for more steps (excluding the retry)

_MARKDOWN = re.compile(r"[*_`#>|~]+|^\s*(?:[-•]|\d+[.)])\s+", re.M)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def spoken(text: str, max_sentences: int = 2) -> str:
    """Model text -> what TTS says: no markdown, at most ``max_sentences`` sentences. Pure."""
    plain = " ".join(_MARKDOWN.sub(" ", text or "").split())
    sentences = [s for s in _SENTENCE_END.split(plain) if s]
    return " ".join(sentences[:max_sentences]).strip()


@dataclass
class RouteResult:
    intents: list[Intent] = field(default_factory=list)
    reply: str = ""  # spoken when there are no intents (rephrase / unavailable)
    provider: str | None = None
    ok: bool = True
    latency_ms: float = 0.0
    attempts: list[dict[str, str]] = field(default_factory=list)  # [{provider, error}]
    retried: bool = False


class Router:
    def __init__(
        self,
        providers: Sequence[ChatProvider],
        toolset: ToolSet,
        *,
        cooldown_s: float = 60.0,
        max_tokens: int = 300,
        temperature: float = 0.2,
        max_intents: int = MAX_INTENTS,
        analysis_timeout_s: float = 15.0,
        quota: Quota | None = None,
        metrics: Metrics | None = None,
        on_provider: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.providers = list(providers)
        self.toolset = toolset
        self.cooldowns = Cooldowns(cooldown_s, clock)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_intents = max_intents
        self.analysis_timeout_s = analysis_timeout_s
        self.quota = quota or NullQuota()
        self.metrics = metrics or NullMetrics()
        self._on_provider = on_provider or (lambda _name: None)
        self._clock = clock
        self._now = now
        self._system = {"role": "system", "content": SYSTEM_PROMPT}
        self._tools = toolset.tools  # built once: byte-identical across calls

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    # ------------------------------------------------------------------ prompt
    def user_message(self, text: str, dialogue: Dialogue | None) -> str:
        """The per-call context goes here, after the cached system prompt + tools."""
        lines = [f"Time: {self._now().strftime('%A %d %B %Y, %H:%M')}"]
        last = dialogue.last() if dialogue is not None else None
        if last is not None:
            lines.append(f"Last action: {self.toolset.describe(last.intent)}")
        lines.append(f"User: {text}")
        return "\n".join(lines)

    def messages(self, text: str, dialogue: Dialogue | None = None) -> list[dict[str, Any]]:
        history = dialogue.messages() if dialogue is not None else []
        return [
            self._system,
            *history,
            {"role": "user", "content": self.user_message(text, dialogue)},
        ]

    # ------------------------------------------------------------------ routing
    def route(self, text: str, dialogue: Dialogue | None = None) -> RouteResult:
        """Never raises. Intents to run, or a reply to speak when there are none."""
        started = self._clock()
        result = self._route(text, dialogue)
        result.latency_ms = round((self._clock() - started) * 1000, 1)
        self._on_provider(result.provider or "offline")
        self.metrics.write(
            "router",
            provider=result.provider,
            ok=result.ok,
            latency_ms=result.latency_ms,
            intents=[i.name for i in result.intents],
            attempts=result.attempts,
            retried=result.retried,
        )  # no utterance text: it can be private
        log.info(
            "router %s ok=%s in %.0f ms -> %s",
            result.provider or "-",
            result.ok,
            result.latency_ms,
            [i.name for i in result.intents] or result.reply,
        )
        return result

    def _route(self, text: str, dialogue: Dialogue | None) -> RouteResult:
        if not self.providers:
            return RouteResult(reply=NO_PROVIDER, ok=False)
        messages = self.messages(text, dialogue)  # the SAME history goes to every provider
        attempts: list[dict[str, str]] = []
        invalid = None
        for provider in self.providers:
            skip = self._skip(provider)
            if skip:
                attempts.append({"provider": provider.name, "error": skip})
                continue
            try:
                result = self._converse(provider, messages, text, attempts)
            except ProviderError as exc:
                self._fail(provider.name, exc)
                attempts.append({"provider": provider.name, "error": type(exc).__name__})
                continue
            if result.ok:
                return result
            invalid = result  # bad output twice: try the next provider (no cooldown)
        if invalid is not None:
            invalid.attempts = attempts
            return invalid
        return RouteResult(reply=UNAVAILABLE, ok=False, attempts=attempts)

    def _skip(self, provider: ChatProvider) -> str | None:
        if self.cooldowns.active(provider.name):
            return "cooldown"
        if not self.quota.allow(_quota_key(provider)):
            return "quota"
        return None

    def _converse(
        self,
        provider: ChatProvider,
        messages: list[dict[str, Any]],
        text: str,
        attempts: list[dict[str, str]],
    ) -> RouteResult:
        """One provider: up to MAX_ROUNDS calls, one validation retry. Raises ProviderError."""
        intents: list[Intent] = []
        retried = False
        convo = list(messages)
        for _ in range(MAX_ROUNDS):
            reply = self._chat(provider, convo)
            got, more, error = self._parse(reply, text)
            if error is not None:
                attempts.append({"provider": provider.name, "error": f"invalid: {error}"})
                if retried:  # one retry per request
                    break
                retried = True
                log.info(
                    "router: %s gave an invalid reply (%s); retrying once", provider.name, error
                )
                reply = self._chat(provider, [*convo, *self._correction(reply, error)])
                got, more, error = self._parse(reply, text)
                if error is not None:
                    attempts.append({"provider": provider.name, "error": f"invalid: {error}"})
                    break
            intents += got
            if not more or len(intents) >= self.max_intents:
                break
            convo += self._continuation(reply)
        if not intents:
            return RouteResult(
                reply=REPHRASE, provider=provider.name, ok=False, attempts=attempts, retried=retried
            )
        return RouteResult(
            intents[: self.max_intents], provider=provider.name, attempts=attempts, retried=retried
        )

    def _chat(self, provider: ChatProvider, messages: list[dict[str, Any]]) -> ChatReply:
        try:
            reply = provider.chat(
                messages, self._tools, max_tokens=self.max_tokens, temperature=self.temperature
            )
        except ProviderError:
            self.quota.record(_quota_key(provider))  # a failed request still counts
            raise
        self.quota.record(_quota_key(provider), tokens=_tokens(reply))
        return reply

    def _fail(self, name: str, exc: ProviderError) -> None:
        self.cooldowns.trip(name)
        level = (
            logging.WARNING
            if isinstance(exc, (ProviderAuthError, QuotaExceededError))
            else logging.INFO
        )
        log.log(level, "router: %s failed (%s: %s); cooling down %.0f s", name,
                type(exc).__name__, exc, self.cooldowns.cooldown_s)  # fmt: skip

    def _parse(self, reply: ChatReply, text: str) -> tuple[list[Intent], bool, str | None]:
        """Reply -> (Intents, another step follows, error message for the model). Pure."""
        if not reply.tool_calls:
            answer = spoken(reply.content)
            if not answer:
                return [], False, "the reply was empty: call a tool"
            return [Intent("app.answer", {"text": answer}, text, "router")], False, None
        intents, more = [], False
        for call in reply.tool_calls[: self.max_intents]:
            try:
                intent, call_more = self.toolset.parse_call(call.name, call.arguments, text)
            except ToolCallError as exc:
                return [], False, str(exc)
            if call.name == ANSWER_TOOL:
                answer = spoken(intent.args.get("text", ""))
                if not answer:
                    return [], False, "answer_user: empty text"
                intent = Intent(intent.name, {"text": answer}, text, "router")
            more = more or call_more
            intents.append(intent)
        return intents, more, None

    @staticmethod
    def _continuation(reply: ChatReply) -> list[dict[str, Any]]:
        """The model's calls as an assistant turn, each answered "queued", to ask for the next
        step (standard tool-call messages)."""
        calls = [
            {
                **call.extra,  # e.g. Gemini's thought signature
                "id": call.id or f"call_{n}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments or "{}"},
            }
            for n, call in enumerate(reply.tool_calls)
        ]
        results = [
            {"role": "tool", "tool_call_id": c["id"], "content": "Queued. Next step, if any."}
            for c in calls
        ]
        return [
            {"role": "assistant", "content": reply.content or None, "tool_calls": calls},
            *results,
        ]

    @staticmethod
    def _correction(reply: ChatReply, error: str) -> list[dict[str, str]]:
        said = "; ".join(f"{c.name}({c.arguments})" for c in reply.tool_calls) or reply.content
        return [
            {"role": "assistant", "content": said[:500] or "(empty)"},
            {
                "role": "user",
                "content": f"That reply was invalid: {error}. Reply again with valid tool calls.",
            },
        ]

    # ------------------------------------------------------------------ code analysis
    def complete(
        self, prompt: str, *, instruction: str | None = None, exclude: frozenset[str] = frozenset()
    ) -> str:
        """One-shot text, no tools, no history (code analysis). ``exclude`` names providers
        that must not see the text (screen text never goes to Gemini). Never raises."""
        messages = [
            {"role": "system", "content": instruction or ANALYSIS_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        for provider in self.providers:
            if provider.name in exclude or self._skip(provider):
                continue
            try:
                reply = provider.chat(
                    messages, [], max_tokens=400, temperature=self.temperature,
                    timeout_s=self.analysis_timeout_s,
                )  # fmt: skip
            except ProviderError as exc:
                self.quota.record(_quota_key(provider))
                self._fail(provider.name, exc)
                continue
            self.quota.record(_quota_key(provider), tokens=_tokens(reply))
            if reply.content:
                self._on_provider(provider.name)
                return reply.content
        allowed = [p.name for p in self.providers if p.name not in exclude]
        if not allowed:
            return "Code analysis needs a Groq or OpenRouter key: Gemini never sees screen text."
        return UNAVAILABLE


def _quota_key(provider: ChatProvider) -> str:
    return f"{provider.name}:{getattr(provider, 'model', '')}"


def _tokens(reply: ChatReply) -> int:
    try:
        return int(reply.usage.get("total_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def build_router(
    settings: Any,
    registry: Any,
    *,
    search: bool = False,
    images: bool = False,
    quota: Quota | None = None,
    metrics: Metrics | None = None,
    on_provider: Callable[[str], None] | None = None,
    providers: Sequence[ChatProvider] | None = None,
) -> Router:
    """Router from settings. ``search`` / ``images`` offer those tools (their executors are
    configured). ``providers`` overrides the real clients (tests, eval)."""
    from grace.intent.providers import build_chat_providers
    from grace.intent.tools import build_toolset

    r = settings.router
    toolset = build_toolset(registry, search=search, images=images)
    return Router(
        build_chat_providers(settings) if providers is None else providers,
        toolset,
        cooldown_s=r.cooldown_s,
        max_tokens=r.max_tokens,
        temperature=r.temperature,
        analysis_timeout_s=r.analysis_timeout_s,
        quota=quota,
        metrics=metrics,
        on_provider=on_provider,
    )
