"""Temporary manual-test runner (replaced by the S05 runtime):

    uv run python -m grace.input.gestures              # real mouse / volume / brightness
    uv run python -m grace.input.gestures --dry-run    # print intents, touch nothing
    uv run python -m grace.input.gestures --log docs/results/gesture_run.jsonl

No preview window. The console shows the recognised gesture and every action. Ctrl+C quits.
Settings come from settings.yaml (``gestures:`` section).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import IO

from grace.actions.registry import ActionResult
from grace.config import ConfigError, load_settings
from grace.input.gestures import build_gesture_loop
from grace.intent.types import Intent

QUIET_ACTIONS = {"input.move_to"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grace.input.gestures", description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="fake OS backends; print intents")
    parser.add_argument("--camera", type=int, help="override gestures.camera_index")
    parser.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run)")
    parser.add_argument("--log", type=Path, help="append gesture and action events as JSONL")
    parser.add_argument("--show-moves", action="store_true", help="also print cursor moves")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    gs = settings.gestures
    if args.camera is not None:
        gs = gs.model_copy(update={"camera_index": args.camera})

    from grace.actions import create_fake_registry, create_registry
    from grace.actions.executor import ActionExecutor
    from grace.actions.input_ctl import primary_screen_size

    if args.dry_run:
        registry, _ = create_fake_registry()
        screen = (1920, 1080)
    else:
        registry = create_registry()
        screen = primary_screen_size()

    log_file: IO[str] | None = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log.open("a", encoding="utf-8")
    lock = threading.Lock()
    t0 = time.monotonic()

    def record(kind: str, **fields: object) -> None:
        if log_file is None:
            return
        with lock:
            row = {"t": round(time.time(), 3), "kind": kind, **fields}
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()

    def on_gesture(name: str) -> None:
        print(f"[{time.monotonic() - t0:7.2f}s] Gesture: {name}", flush=True)
        record("gesture", name=name)

    def on_result(intent: Intent, result: ActionResult) -> None:
        if intent.name not in QUIET_ACTIONS or args.show_moves:
            print(f"           -> {intent.name} {intent.args} ok={result.ok}", flush=True)
        if intent.name not in QUIET_ACTIONS:
            record("action", name=intent.name, args=intent.args, ok=result.ok)

    executor = ActionExecutor(registry, on_result).start()
    loop = build_gesture_loop(gs, screen, executor.submit, on_gesture)
    mode = "DRY RUN (nothing on the OS changes)" if args.dry_run else "LIVE"
    print(f"GRACE gestures, {mode}. Screen {screen[0]}x{screen[1]}. Ctrl+C to quit.")
    print("Index up = move | pinch = click (hold = drag) | thumb-middle pinch = right click")
    print("Two fingers = scroll | three = brightness | thumb+two = volume | fist 1 s = pause")
    loop.start()
    try:
        deadline = t0 + args.seconds if args.seconds else None
        while deadline is None or time.monotonic() < deadline:
            threading.Event().wait(0.25)  # main thread idles; the camera loop never sleeps
    except KeyboardInterrupt:
        pass
    finally:
        loop.stop()
        executor.stop()
        s = loop.stats
        elapsed = max(time.monotonic() - t0, 1e-6)
        share = s.hand_frames / s.frames if s.frames else 0.0
        print(
            f"\n{s.frames} frames ({s.frames / elapsed:.1f} fps incl. camera start), "
            f"hand visible in {share:.0%}, {s.intents} intents, "
            f"classifier + controller {s.mean_process_ms:.2f} ms per frame"
        )
        if log_file is not None:
            log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
