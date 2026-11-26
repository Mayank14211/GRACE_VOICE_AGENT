"""M6 acceptance: 20 free-form requests through the real router, logged with provider, latency
and success. OS actions go to fakes, so nothing on the PC changes.

    uv run python scripts/eval_router.py                    # provider chain from settings.yaml
    uv run python scripts/eval_router.py --providers local  # network off: llama-server only
    uv run python scripts/eval_router.py --save             # + docs/results/router.md
    uv run python scripts/eval_router.py --settings my.yaml # another settings file

Requests run in order through the voice agent (grammar -> follow-ups -> router) with one
dialogue, so the memory cases ("and how many people live there") test multi-turn context.
None of them is a fixed command: a request the grammar catches is reported as such. Success =
the router returned one of the expected intents.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from results_md import ROUTER_RESULTS, median, stamp, table, use_utf8_output  # noqa: E402

ANSWER = "app.answer"
VOLUME = {"volume.step", "volume.set"}

# (request, intents that count as success). Risky requests must never produce a system.* intent.
REQUESTS: tuple[tuple[str, frozenset[str]], ...] = tuple(
    (text, frozenset(ok))
    for text, ok in (
        ("could you make it a little quieter", VOLUME),
        ("it is too dark, brighten the screen up a bit", {"brightness.step", "brightness.set"}),
        ("put youtube on for me", {"apps.open", "apps.open_url"}),
        ("what is the capital of australia", {ANSWER}),
        ("and how many people live there", {ANSWER}),
        ("move the page down a bit", {"input.scroll"}),
        ("silence everything", {"volume.mute", "volume.set"}),
        ("actually bring the sound back", {"volume.mute", "volume.set", "volume.step"}),
        ("write hello world in here", {"input.type"}),
        ("copy whatever I have selected", {"clipboard.copy", "input.hotkey"}),
        ("get rid of this tab", {"apps.close_tab", "input.hotkey"}),
        ("I need the calculator", {"apps.open"}),
        ("what sort of things can you help me with", {ANSWER}),
        ("tell me a short joke", {ANSWER}),
        ("read out what is on my screen", {"app.read_screen"}),
        ("grab a picture of my screen", {"screen.screenshot"}),
        ("bring up the task manager", {"apps.open", "input.hotkey"}),
        ("how much is fifteen percent of two hundred", {ANSWER}),
        ("put the brightness at about seventy", {"brightness.set"}),
        ("turn my computer off", {ANSWER}),  # refused: needs the exact phrase + confirmation
    )
)


@dataclass
class Row:
    text: str
    source: str  # router | grammar | followup
    provider: str
    ms: float
    intents: list[str]
    ok: bool
    note: str = ""  # provider errors when it failed


class _Silent:
    def speak_async(self, text: str) -> None:
        pass

    speak = speak_async


class _Screen:
    def read_screen_text(self) -> str:
        return "Example screen text."


def run(
    providers_filter: list[str] | None, settings_path: Path | None = None, pause_s: float = 0.0
) -> tuple[list[Row], list[str]]:
    from grace.actions import create_fake_registry
    from grace.config import load_settings
    from grace.intent.dialogue import Dialogue
    from grace.intent.providers import build_chat_providers
    from grace.intent.router import Router
    from grace.intent.tools import build_toolset
    from grace.legacy_voice import GraceVoiceAgent

    settings = load_settings(settings_path)
    providers = build_chat_providers(settings)
    if providers_filter:
        providers = [p for p in providers if p.name in providers_filter]
    registry, _ = create_fake_registry()
    router = Router(
        providers,
        build_toolset(registry, allow_risky=settings.router.allow_risky),
        cooldown_s=settings.router.cooldown_s,
        max_tokens=settings.router.max_tokens,
        temperature=settings.router.temperature,
    )
    agent = GraceVoiceAgent(
        registry=registry,
        tts=_Silent(),
        screen_reader=_Screen(),
        router=router,
        dialogue=Dialogue(max_turns=settings.router.history_turns),
    )
    rows: list[Row] = []
    for n, (text, expected) in enumerate(REQUESTS):
        if n and pause_s:
            time.sleep(pause_s)  # free tiers limit requests per minute; speech is slower anyway
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
        risky = any(n.startswith("system.") for n in names)
        ok = bool(set(names) & expected) and not risky
        note = ", ".join(f"{a['provider']}: {a['error']}" for a in result.attempts)
        rows.append(
            Row(text, "router", result.provider or "none", result.latency_ms, names, ok, note)
        )
        mark = "ok " if ok else "BAD"
        print(f"  {mark} {result.provider or '-':<10} {result.latency_ms:6.0f} ms  "
              f"{text!r} -> {names or result.reply}")  # fmt: skip
    return rows, [f"{p.name} ({p.timeout_s:g} s)" for p in providers]


def report(rows: list[Row], chain: list[str]) -> str:
    routed = [r for r in rows if r.source == "router"]
    good = sum(r.ok for r in routed)
    by_provider: dict[str, list[float]] = {}
    for r in routed:
        by_provider.setdefault(r.provider, []).append(r.ms)
    lat = ", ".join(
        f"{name}: {len(ms)} requests, median {median(ms):.0f} ms"
        for name, ms in by_provider.items()
    )
    lines = [
        f"Run {stamp()}. Chain: {' -> '.join(chain) or 'none'}. "
        f"{good}/{len(routed)} router requests succeeded. {lat}.",
        "",
        table(
            ["#", "request", "provider", "ms", "intents", "success"],
            [
                [
                    str(i),
                    r.text,
                    r.provider,
                    f"{r.ms:.0f}",
                    ", ".join(r.intents) or r.note or "-",
                    "yes" if r.ok else ("grammar" if r.source != "router" else "no"),
                ]
                for i, r in enumerate(rows, 1)
            ],
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "GRACE").splitlines()[0])
    parser.add_argument("--providers", default="", help="comma list, e.g. local (network off)")
    parser.add_argument("--save", action="store_true", help="write docs/results/router.md")
    parser.add_argument("--settings", type=Path, default=None, help="settings file")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds between requests (free-tier rate limits)")  # fmt: skip
    args = parser.parse_args(argv)
    use_utf8_output()
    wanted = [p.strip() for p in args.providers.split(",") if p.strip()] or None
    rows, chain = run(wanted, args.settings, args.pause)
    if not chain:
        print("No provider available (keys in .env, or router.local.enabled + llama-server).")
        return 1
    text = report(rows, chain)
    print("\n" + text)
    if args.save:
        from results_md import write_section

        name = "eval_router_" + "_".join(c.split()[0] for c in chain)
        write_section(ROUTER_RESULTS, name, f"### eval_router ({', '.join(chain)})\n\n{text}")
        print(f"Saved to {ROUTER_RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
