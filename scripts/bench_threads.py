"""Find the fastest inference thread count for this machine (inference.threads in settings.yaml).

    uv run python scripts/bench_threads.py                  # 4 / 6 / 8 threads
    uv run python scripts/bench_threads.py --save           # + docs/results/stt.md
    uv run python scripts/bench_threads.py --llm --save     # + local LLM: docs/results/router.md

faster-whisper (extra [stt-whisper]) runs at each thread count (cpu_threads); Moonshine has no
thread setting, so it is measured with onnxruntime's default and with one thread
(MOONSHINE_ORT_SINGLE_THREAD=1) for reference. Every configuration runs in its own process so
thread pools never mix. Never 12: logical-thread counts make inference slower (docs/PLAN.md 2).

--llm (M6): llama-server is started at each thread count (-t N) on a spare port, warmed up with
the router's system prompt + tools, then sent the same router requests; reported: median
request latency, prompt and generation tokens/s (llama-server's own timings). Needs llama-server
and the GGUF (scripts/run_llama_server.ps1 sets both up). A llama-server already running on
router.local.base_url is left alone.

Run on mains power, Windows power mode "Best performance", nothing else heavy running.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from results_md import (  # noqa: E402
    ROUTER_RESULTS,
    STT_RESULTS,
    median,
    stamp,
    table,
    use_utf8_output,
    write_section,
)

MAX_THREADS = 8
LLM_PORT = 8089  # not router.local.base_url's 8080: a running server is never disturbed
LLM_PROMPTS = (
    "what is the capital of japan",
    "turn the sound down a little",
    "please open notepad for me",
    "what can you do",
    "scroll down a bit on this page",
)


def pick_files(audio_dir: Path | None, limit: int) -> list[Path]:
    default_dirs = [ROOT / "tests" / "audio", ROOT / "data" / "synthetic_eval"]
    for folder in [audio_dir] if audio_dir else default_dirs:
        if folder.is_dir():
            files = sorted(folder.glob("*.wav"))
            if files:
                return files[:limit]
    fix = ROOT / "tests" / "fixtures" / "audio"
    return [fix / "one_command.wav", fix / "two_commands.wav"]


def recommend(results: dict[int, float], tolerance: float = 0.05) -> int:
    """Fewest threads within ``tolerance`` of the fastest median (pure)."""
    best = min(results.values())
    return min(t for t, ms in results.items() if ms <= best * (1 + tolerance))


def row(engine: str, threads: str, r: dict) -> list[str]:
    return [engine, threads, f"{r['latency_ms']:.0f}", f"{r['rtf']:.3f}", f"{r['load_ms']:.0f}"]


# --------------------------------------------------------------------------- child process
def child(engine_name: str, threads: int, files: list[str]) -> dict:
    from grace.config import load_settings
    from grace.input.audio import frames, load_wav
    from grace.input.stt import engine_factories, transcribe

    settings = load_settings()
    engine = engine_factories(settings, threads=threads)[engine_name]()
    started = time.perf_counter()
    engine.load()
    load_ms = (time.perf_counter() - started) * 1000
    audio = [load_wav(f) for f in files]
    transcribe(engine, frames(audio[0]))  # warm-up
    latency, compute, audio_s = [], [], 0.0
    for wav in audio:
        final, _ = transcribe(engine, frames(wav))
        latency.append(final.latency_ms)  # end of audio -> text
        compute.append((final.t_final - final.t_first_audio) * 1000)  # all work, fed at once
        audio_s += final.audio_s
    return {
        "load_ms": load_ms,
        "latency_ms": median(latency),
        "compute_ms": median(compute),
        "rtf": sum(compute) / 1000 / max(audio_s, 1e-9),
    }


def run_child(engine: str, threads: int, files: list[Path], env_extra: dict[str, str]) -> dict:
    env = {**os.environ, **env_extra}
    cmd = [sys.executable, __file__, "--child", engine, str(threads), *map(str, files)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"error": tail[0]}
    return json.loads(lines[-1])


# --------------------------------------------------------------------------- LLM
def llm_summary(samples: list[dict]) -> dict:
    """Per-request samples -> median latency and tokens/s (pure)."""

    def med(key: str) -> float:
        return median([float(s[key]) for s in samples if s.get(key) is not None])

    return {
        "latency_ms": med("latency_ms"),
        "prompt_tps": med("prompt_per_second"),
        "gen_tps": med("predicted_per_second"),
        "ok": sum(1 for s in samples if s.get("ok")),
        "n": len(samples),
    }


def bench_llm_threads(settings, threads: int, prompts: tuple[str, ...]) -> dict:
    """Start llama-server with ``threads``, measure, stop it. {"error": ...} on failure."""
    from grace.actions import create_fake_registry
    from grace.intent.local_llm import command_from_settings, wait_until_up
    from grace.intent.providers import (
        LOCAL_API_KEY,
        OpenAIChatProvider,
        ProviderError,
        make_openai_client,
    )
    from grace.intent.router import Router
    from grace.intent.tools import build_toolset

    base_url = f"http://127.0.0.1:{LLM_PORT}/v1"
    try:
        cmd = command_from_settings(settings, threads=threads, base_url=base_url)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}
    started = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_until_up(base_url, timeout_s=600):
            return {"error": "llama-server did not come up within 10 minutes"}
        load_ms = (time.perf_counter() - started) * 1000
        client = make_openai_client(base_url, LOCAL_API_KEY)
        provider = OpenAIChatProvider("local", client, settings.router.local.model, timeout_s=120)
        router = Router([provider], build_toolset(create_fake_registry()[0]))
        router.warm_up(timeout_s=600)  # prompt cache filled, as in demo mode
        samples = []
        for prompt in prompts:
            try:
                reply = provider.chat(
                    router.messages(prompt),
                    router.toolset.tools,
                    max_tokens=settings.router.max_tokens,
                    temperature=0.0,
                )
            except ProviderError as exc:
                samples.append({"ok": False, "error": str(exc)})
                continue
            samples.append(
                {
                    "ok": bool(reply.tool_calls or reply.content),
                    "latency_ms": reply.latency_ms,
                    "prompt_per_second": reply.timings.get("prompt_per_second"),
                    "predicted_per_second": reply.timings.get("predicted_per_second"),
                }
            )
        return {"load_ms": load_ms, **llm_summary(samples)}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def bench_llm(settings, counts: list[int]) -> tuple[list[str], int | None]:
    rows, latency = [], {}
    for t in counts:
        r = bench_llm_threads(settings, t, LLM_PROMPTS)
        if "error" in r:
            print(f"  llama-server x{t}: {r['error']}")
            continue
        latency[t] = r["latency_ms"]
        print(
            f"  llama-server x{t}: {r['latency_ms']:.0f} ms per request, prompt "
            f"{r['prompt_tps']:.0f} tok/s, generation {r['gen_tps']:.1f} tok/s "
            f"({r['ok']}/{r['n']} answered)"
        )
        rows.append(
            [
                settings.router.local.gguf.name,
                str(t),
                f"{r['latency_ms']:.0f}",
                f"{r['prompt_tps']:.0f}",
                f"{r['gen_tps']:.1f}",
                f"{r['ok']}/{r['n']}",
                f"{r['load_ms'] / 1000:.0f}",
            ]
        )
    if not rows:
        return [], None
    best = recommend(latency)
    lines = [
        f"Run {stamp()}. {len(LLM_PROMPTS)} router requests per thread count after a warm-up "
        "(system prompt + tools cached). Latency = request sent -> full reply (non-streaming).",
        "",
        table(
            ["model", "threads", "median ms", "prompt tok/s", "gen tok/s", "answered", "load s"],
            rows,
        ),
        "",
        f"**Recommendation:** `inference.threads: {best}` for llama-server on this machine.",
    ]
    return lines, best


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--child"]:
        print(json.dumps(child(argv[1], int(argv[2]), argv[3:])))
        return 0
    parser = argparse.ArgumentParser(description=(__doc__ or "GRACE").splitlines()[0])
    parser.add_argument("--threads", default="4,6,8", help="comma list, each 1-8")
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20, help="recordings per configuration")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--llm", action="store_true", help="also llama-server (local LLM)")
    args = parser.parse_args(argv)
    use_utf8_output()

    counts = sorted({int(t) for t in args.threads.split(",")})
    if any(not 1 <= t <= MAX_THREADS for t in counts):
        parser.error(f"thread counts must be 1-{MAX_THREADS} (never the logical count, 12)")
    files = pick_files(args.audio_dir, args.limit)
    from grace.config import load_settings
    from grace.input.stt import whisper_installed

    settings = load_settings()
    print(f"{len(files)} recordings from {files[0].parent}; current inference.threads = "
          f"{settings.inference.threads}")  # fmt: skip
    rows: list[list[str]] = []
    whisper: dict[int, float] = {}
    if whisper_installed():
        for t in counts:
            r = run_child("whisper_local", t, files, {})
            if "error" in r:
                print(f"  faster-whisper {settings.stt.whisper_model} x{t}: {r['error']}")
                continue
            whisper[t] = r["latency_ms"]
            print(f"  faster-whisper {settings.stt.whisper_model} x{t}: {r['latency_ms']:.0f} ms "
                  f"per command (RTF {r['rtf']:.2f})")  # fmt: skip
            rows.append(row(f"faster-whisper {settings.stt.whisper_model} int8", str(t), r))
    else:
        print("  faster-whisper not installed (uv sync --extra stt-whisper): skipped")
    arch = settings.stt.moonshine_model
    for label, env in (("ORT default", {}), ("1", {"MOONSHINE_ORT_SINGLE_THREAD": "1"})):
        r = run_child("moonshine", settings.inference.threads, files, env)
        if "error" in r:
            print(f"  moonshine {arch} ({label}): {r['error']}")
            continue
        print(f"  moonshine {arch} threads {label}: {r['latency_ms']:.0f} ms after end of audio, "
              f"RTF {r['rtf']:.3f}")  # fmt: skip
        rows.append(row(f"moonshine {arch}", label, r))
    llm_lines: list[str] = []
    if args.llm:
        llm_lines, best_llm = bench_llm(settings, counts)
        if best_llm is not None:
            print(f"Recommended inference.threads for llama-server: {best_llm}")

    lines = [
        f"Run {stamp()}. {len(files)} recordings from "
        f"`{files[0].parent.relative_to(ROOT).as_posix()}`, fed as fast as possible (no pacing); "
        "latency = end of audio → final text (for faster-whisper this is the whole "
        "transcription), RTF = compute time / audio time.",
        "",
        table(["engine", "threads", "median ms", "RTF", "load ms"], rows),
    ]
    if whisper:
        best = recommend(whisper)
        lines += [
            "",
            f"**Recommendation:** `inference.threads: {best}` on this machine (fewest threads "
            "within 5 % of the fastest faster-whisper run).",
        ]
        print(f"Recommended inference.threads: {best}")
    text = "\n".join(lines)
    if args.save:
        write_section(STT_RESULTS, "bench_threads", f"### bench_threads\n\n{text}")
        print(f"Saved to {STT_RESULTS.relative_to(ROOT)}")
        if llm_lines:
            body = "### bench_threads --llm\n\n" + "\n".join(llm_lines)
            write_section(ROUTER_RESULTS, "bench_llm", body)
            print(f"Saved to {ROUTER_RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
