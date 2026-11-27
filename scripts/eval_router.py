"""M6 acceptance: 20 free-form requests through the real router, logged with provider, latency
and success. OS actions go to fakes, so nothing on the PC changes (images are saved under
logs/eval_images/).

    uv run python scripts/eval_router.py --force-429 5      # the S07 run: first 5 requests get a
                                                            # forced HTTP 429 (first provider)
    uv run python scripts/eval_router.py --offline          # network off (simulated: dead proxy)
    uv run python scripts/eval_router.py --save             # + docs/results/router.md
    uv run python scripts/eval_router.py --settings my.yaml --pause 3

Requests run in order through the voice agent (grammar -> follow-ups -> router) with one
dialogue, so "and how many people live there" tests multi-turn memory. Among them are 3 web
searches and 3 images; their "tool" column says whether our executor (Tavily, Cloudflare)
succeeded. With --force-429 N, the first provider answers the first N requests with HTTP 429
(its cooldown is cleared before each, so every one is a real 429): the next provider must
answer, with the same history. --offline routes every connection to a dead proxy: each router
request must fail fast with the spoken "can't reach" reply, and fixed commands must still work.
Success = the router returned one of the expected intents (and nothing risky).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from results_md import (  # noqa: E402
    ROUTER_RESULTS,
    median,
    stamp,
    table,
    use_utf8_output,
    write_section,
)

ANSWER = "app.answer"
VOLUME = {"volume.step", "volume.set"}
SEARCH = {"app.web_search"}
IMAGE = {"app.generate_image"}
EVAL_IMAGES = ROOT / "logs" / "eval_images"

# (request, intents that count as success). Risky requests must never produce a system.* intent.
REQUESTS: tuple[tuple[str, frozenset[str]], ...] = tuple(
    (text, frozenset(ok))
    for text, ok in (
        ("could you make it a little quieter", VOLUME),
        ("what is the capital of australia", {ANSWER}),
        ("and how many people live there", {ANSWER}),
        ("search the web for today's weather in Delhi", SEARCH),
        ("draw me a picture of a cat wearing sunglasses", IMAGE),
        ("it is too dark, brighten the screen up a bit", {"brightness.step", "brightness.set"}),
        ("put youtube on for me", {"apps.open", "apps.open_url"}),
        ("look up who won the last cricket world cup", SEARCH),
        ("make an image of a lighthouse at sunset", IMAGE),
        ("move the page down a bit", {"input.scroll"}),
        ("silence everything", {"volume.mute", "volume.set"}),
        ("actually bring the sound back", {"volume.mute", "volume.set", "volume.step"}),
        ("write hello world in here", {"input.type"}),
        ("find the latest news about the James Webb telescope", SEARCH),
        ("create a picture of a red bicycle in the rain", IMAGE),
        ("get rid of this tab", {"apps.close_tab", "input.hotkey"}),
        ("tell me a short joke", {ANSWER}),
        ("read out what is on my screen", {"app.read_screen"}),
        ("put the brightness at about seventy", {"brightness.set"}),
        ("turn my computer off", {ANSWER}),  # refused: needs the exact phrase + confirmation
    )
)
# Fixed commands for the --offline run: they must work with no network at all.
OFFLINE_FIXED = ("volume 40", "a bit more", "brightness 60", "scroll down", "open notepad")
DEAD_PROXY = "http://127.0.0.1:9"


@dataclass
class Row:
    text: str
    source: str  # router | grammar | followup
    provider: str
    ms: float
    intents: list[str]
    ok: bool
    note: str = ""  # provider attempts / errors
    tool: str = ""  # executor outcome for web_search / generate_image: ok | failed | -


class _Silent:
    def speak_async(self, text: str) -> None:
        pass

    speak = speak_async


class _Screen:
    def read_screen_text(self) -> str:
        return "Example screen text."


class _Forced429:
    """Wraps a provider: raises HTTP 429 (QuotaExceededError) while ``remaining`` > 0."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = getattr(inner, "model", "")
        self.timeout_s = inner.timeout_s
        self.remaining = 0

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        from grace.intent.providers import QuotaExceededError

        if self.remaining > 0:
            raise QuotaExceededError("forced 429 (eval)", billing_required=False)
        return self.inner.chat(*args, **kwargs)


class _ToolLog:
    """Wraps Extras: remembers whether the last web_search / generate_image succeeded."""

    def __init__(self, extras: Any) -> None:
        self._extras = extras
        self.last: bool | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._extras, name)

    def web_search(self, query: str) -> Any:
        result = self._extras.web_search(query)
        self.last = result.ok
        return result

    def generate_image(self, prompt: str) -> Any:
        result = self._extras.generate_image(prompt)
        self.last = result.ok
        return result


def build(settings_path: Path | None, providers_filter: list[str] | None) -> tuple[Any, ...]:
    from grace.actions import create_fake_registry
    from grace.config import load_settings
    from grace.core.quota import build_governor
    from grace.intent.assistant import Extras
    from grace.intent.dialogue import Dialogue
    from grace.intent.images import build_images
    from grace.intent.providers import build_chat_providers
    from grace.intent.router import Router
    from grace.intent.search import build_search
    from grace.intent.tools import build_toolset
    from grace.legacy_voice import GraceVoiceAgent

    settings = load_settings(settings_path)
    providers = build_chat_providers(settings)
    if providers_filter:
        providers = [p for p in providers if p.name in providers_filter]
    wrapped = [_Forced429(providers[0]), *providers[1:]] if providers else []
    registry, _ = create_fake_registry(screenshot_dir=EVAL_IMAGES)
    quota = build_governor(settings)  # the real counters: eval calls spend real free quota
    search = build_search(settings, quota)
    images = build_images(settings, registry, quota)
    if images is not None:
        images.image_dir = EVAL_IMAGES  # type: ignore[attr-defined]
    r = settings.router
    router = Router(
        wrapped,
        build_toolset(registry, search=search is not None, images=images is not None),
        cooldown_s=r.cooldown_s,
        max_tokens=r.max_tokens,
        temperature=r.temperature,
        analysis_timeout_s=r.analysis_timeout_s,
        quota=quota,
    )
    extras = _ToolLog(Extras(search=search, images=images, analyst=router))
    agent = GraceVoiceAgent(
        registry=registry,
        tts=_Silent(),
        screen_reader=_Screen(),
        router=router,
        extras=extras,
        dialogue=Dialogue(max_turns=r.history_turns),
    )
    return agent, router, extras, wrapped


def _attempts(result: Any) -> str:
    return ", ".join(f"{a['provider']}: {a['error']}" for a in getattr(result, "attempts", []))


def run(
    providers_filter: list[str] | None,
    settings_path: Path | None = None,
    pause_s: float = 0.0,
    force_429: int = 0,
) -> tuple[list[Row], list[str]]:
    agent, router, extras, wrapped = build(settings_path, providers_filter)
    rows: list[Row] = []
    for n, (text, expected) in enumerate(REQUESTS):
        if n and pause_s:
            time.sleep(pause_s)  # free tiers limit requests per minute; speech is slower anyway
        forced = bool(wrapped) and n < force_429
        if wrapped:
            wrapped[0].remaining = 1 if forced else 0
            router.cooldowns.clear(wrapped[0].name)  # each forced request gets a real 429
        extras.last = None
        started = time.perf_counter()
        result = agent._process_command(text)
        ms = (time.perf_counter() - started) * 1000
        intents = getattr(result, "intents", None)
        if intents is None:  # grammar / follow-up handled it: not a router case
            last = agent.dialogue.last()
            src = last.intent.source if last else "grammar"
            rows.append(Row(text, src, "-", ms, [last.intent.name] if last else [], False))
            continue
        names = [i.name for i in intents]
        risky = any(name.startswith("system.") for name in names)
        ok = bool(set(names) & expected) and not risky
        tool = "-" if extras.last is None else ("ok" if extras.last else "failed")
        note = _attempts(result)
        rows.append(
            Row(text, "router", result.provider or "none", result.latency_ms, names, ok, note, tool)
        )
        mark = "ok " if ok else "BAD"
        print(f"  {mark} {result.provider or '-':<10} {result.latency_ms:6.0f} ms  "
              f"{text!r} -> {names or result.reply} [{note}] tool={tool}")  # fmt: skip
    if wrapped:
        wrapped[0].remaining = 0
    return rows, [f"{p.name} ({p.timeout_s:g} s)" for p in wrapped]


def run_offline(settings_path: Path | None) -> tuple[list[Row], list[Row], list[str]]:
    """Every connection goes to a dead proxy: the router must fail fast and politely; fixed
    commands must still work."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        os.environ[var] = DEAD_PROXY
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    agent, _router, _extras, wrapped = build(settings_path, None)
    routed: list[Row] = []
    for text, _ in REQUESTS[:5]:
        started = time.perf_counter()
        result = agent._process_command(text)
        ms = (time.perf_counter() - started) * 1000
        polite = bool(getattr(result, "reply", "")) and not getattr(result, "intents", None)
        routed.append(
            Row(text, "router", result.provider or "none", ms, [], polite, _attempts(result))
        )
    fixed: list[Row] = []
    for text in OFFLINE_FIXED:
        started = time.perf_counter()
        result = agent._process_command(text)
        ms = (time.perf_counter() - started) * 1000
        ok = result is not None and getattr(result, "ok", False)
        last = agent.dialogue.last()
        names = [last.intent.name] if last else []
        fixed.append(Row(text, last.intent.source if last else "grammar", "-", ms, names, ok))
    return routed, fixed, [f"{p.name} ({p.timeout_s:g} s)" for p in wrapped]


def report(rows: list[Row], chain: list[str], force_429: int = 0) -> str:
    routed = [r for r in rows if r.source == "router"]
    good = sum(r.ok for r in routed)
    by_provider: dict[str, list[float]] = {}
    for r in routed:
        by_provider.setdefault(r.provider, []).append(r.ms)
    lat = ", ".join(
        f"{name}: {len(ms)} requests, median {median(ms):.0f} ms"
        for name, ms in by_provider.items()
    )
    tools = [r for r in routed if r.tool not in ("", "-")]
    summary = (
        f"Run {stamp()}. Chain: {' -> '.join(chain) or 'none'}. "
        f"{good}/{len(routed)} router requests succeeded. {lat}."
    )
    if tools:
        summary += f" Tools: {sum(r.tool == 'ok' for r in tools)}/{len(tools)} executor calls ran."
    if force_429:
        summary += f" Requests 1-{force_429}: forced HTTP 429 from the first provider."
    rows_md = [
        [
            str(i),
            r.text,
            r.provider,
            f"{r.ms:.0f}",
            ", ".join(r.intents) or "-",
            r.tool or "-",
            r.note or "-",
            "yes" if r.ok else ("grammar" if r.source != "router" else "no"),
        ]
        for i, r in enumerate(rows, 1)
    ]
    headers = ["#", "request", "provider", "ms", "intents", "tool", "attempts", "success"]
    return summary + "\n\n" + table(headers, rows_md)


def report_offline(routed: list[Row], fixed: list[Row], chain: list[str]) -> str:
    slowest = max((r.ms for r in routed), default=0.0)
    summary = (
        f"Run {stamp()}, network off (every connection sent to a dead proxy). Chain: "
        f"{' -> '.join(chain) or 'none'}. Router: {sum(r.ok for r in routed)}/{len(routed)} "
        f"requests got the spoken fallback, slowest {slowest:.0f} ms. "
        f"Fixed commands: {sum(r.ok for r in fixed)}/{len(fixed)} worked."
    )
    rows_md = [[r.text, "router", f"{r.ms:.0f}", r.note or "-"] for r in routed]
    rows_md += [[r.text, r.source, f"{r.ms:.0f}", ", ".join(r.intents) or "-"] for r in fixed]
    return summary + "\n\n" + table(["request", "kind", "ms", "result"], rows_md)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "GRACE").splitlines()[0])
    parser.add_argument("--providers", default="", help="comma list, e.g. groq,gemini")
    parser.add_argument("--save", action="store_true", help="write docs/results/router.md")
    parser.add_argument("--settings", type=Path, default=None, help="settings file")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds between requests (free-tier rate limits)")  # fmt: skip
    parser.add_argument("--force-429", type=int, default=0, metavar="N",
                        help="first N requests: the first provider answers HTTP 429")  # fmt: skip
    parser.add_argument("--offline", action="store_true",
                        help="simulate network off (dead proxy for every connection)")  # fmt: skip
    args = parser.parse_args(argv)
    use_utf8_output()
    if args.offline:
        routed, fixed, chain = run_offline(args.settings)
        text = report_offline(routed, fixed, chain)
        print("\n" + text)
        if args.save:
            write_section(
                ROUTER_RESULTS, "eval_router_offline", f"### eval_router --offline\n\n{text}"
            )
            print(f"Saved to {ROUTER_RESULTS.relative_to(ROOT)}")
        return 0
    wanted = [p.strip() for p in args.providers.split(",") if p.strip()] or None
    rows, chain = run(wanted, args.settings, args.pause, args.force_429)
    if not chain:
        print("No LLM provider available: put GROQ_API_KEY (and GEMINI_API_KEY) in .env.")
        return 1
    text = report(rows, chain, args.force_429)
    print("\n" + text)
    if args.save:
        name = "eval_router_" + "_".join(c.split()[0] for c in chain)
        title = f"### eval_router ({', '.join(chain)})"
        if args.force_429:
            title += f", --force-429 {args.force_429}"
        write_section(ROUTER_RESULTS, name, f"{title}\n\n{text}")
        print(f"Saved to {ROUTER_RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
