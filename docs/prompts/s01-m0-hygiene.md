# Session S01 — M0: Repo hygiene and baseline

## 1. Project context
You are a senior Python engineer working on GRACE, a multimodal (voice + hand gesture) desktop
accessibility assistant for Windows 10/11, built as a college minor project. Repo: GRACE_VOICE_AGENT.
docs/PLAN.md is the source of truth for scope, architecture, stack, models, gesture map and gates.

PHASE 1 GOAL
A reliable, demo-ready build at zero running cost.
- Base machine (development + all performance targets): Intel Core i5-12450H, 8 cores (4P + 4E),
  12 threads, AVX2, 16 GB RAM, CPU-only inference.
- Demo laptop (secondary): AMD Ryzen 5 5500U, 6 cores / 12 threads, AVX2, 16 GB RAM, built-in mic.
- Inference thread count is a setting (inference.threads): default 4 on the i5 (P-cores), 6 on the
  Ryzen. Never use 12. No GPU code paths.
- If the machine you run on is not Windows, run only non-hardware tests and say so in the report.

NON-NEGOTIABLE REQUIREMENTS
- Voice and gesture control run at the same time, in one process.
- Global hotkeys (configurable in settings.yaml) are the PRIMARY toggles:
    Ctrl+Alt+M mic on/off | Ctrl+Alt+G gestures on/off | Ctrl+Alt+Space push-to-talk (hold)
    Ctrl+Alt+Q panic stop (stop all input, release mouse, app stays alive)
    Ctrl+Alt+Y confirm risky action | Ctrl+Alt+H history panel | Ctrl+Alt+X mark last action wrong
  Voice "gesture on/off" may exist but is never relied on.
- No camera preview window. The only UI is a transparent, click-through, always-on-top overlay:
    * subtitle bar, bottom-centre of the primary screen: what GRACE heard and replied, fades out;
    * status chip, bottom-right: mic on/off, gesture on/off, the gesture currently recognised
      (e.g. "Gesture: SCROLL"), provider in use.
- Offline-first: wake word, English STT, gestures and all fixed commands work without network.
  Cloud APIs (Groq, Gemini, OpenRouter) are optional accelerators with automatic fallback.
- Risky actions (shutdown, restart, close-all, sending messages) need exact phrasing AND
  confirmation. No substring/keyword-contains matching may trigger any action.
- API keys only from .env via python-dotenv. Never hard-code, log or commit keys.

TARGET LAYOUT (create incrementally; keep old files only until their replacement lands)
grace/
  __main__.py            # `python -m grace` and `python -m grace doctor`
  config.py              # .env + settings.yaml -> pydantic Settings
  core/bus.py            # thread-safe event bus (queue.Queue + Qt signal bridge)
  core/state.py          # mic_on, gesture_on, mode (command|dictation), last N dialogue turns
  input/hotkeys.py
  input/audio.py         # sounddevice stream + Silero VAD
  input/wakeword.py      # openWakeWord, custom model
  input/stt/             # base.py (interface), moonshine.py, whisper_local.py, groq.py
  input/gestures/        # tracker.py (MediaPipe), classifier.py (pure), controller.py (state machine)
  intent/grammar.py      # deterministic command grammar -> Intent
  intent/router.py       # LLM fallback chain -> Intent
  intent/providers.py    # OpenAI-compatible clients: groq, gemini, openrouter, local llama-server
  actions/registry.py    # name -> handler; the ONLY layer that touches the OS
  actions/system.py, actions/input_ctl.py, actions/apps.py, actions/screen.py,
  actions/notifications.py, actions/uia.py
  ui/overlay.py          # PySide6 subtitle bar + status chip
  feedback/tts.py, feedback/earcons.py
  metrics/log.py         # JSONL events for accuracy/latency results
scripts/  doctor.py, eval_stt.py, bench_threads.py, report.py, demo_check.py, run_llama_server.ps1
tests/

THREADING MODEL
Qt owns the main thread (overlay). Audio, gesture camera loop, STT and the intent router each run
in their own worker thread and talk only through core/bus.py. Overlay updates use Qt signals.
All OS actions run in one action-executor thread so voice and gesture never race.

CODING STANDARDS
- Type hints; small pure functions (classifier, grammar, router decisions) that test without hardware.
- Every hardware-touching class has an interface and a Fake used in tests.
- Read library docs/examples before using an API you are unsure of (moonshine-voice, openwakeword,
  winsdk, mediapipe, pywinauto). Do not guess signatures. If mediapipe's legacy `solutions.hands` is
  missing in the pinned version, use the Tasks API HandLandmarker.
- Scope discipline: implement only this session's milestone. Put anything else under
  "Follow-ups" in the report.

## 2. Stability rules (apply to every change)
1. Before adding any package: confirm a cp311 win_amd64 wheel exists for the exact version, add it
   with `uv add <pkg>==<version>`, run `uv lock`, then `uv pip check`. Report the version.
2. Never upgrade or remove an existing pin unless this milestone says so. If a new package forces a
   different version of an existing one, STOP and report the conflict instead of resolving it silently.
3. Known conflict risks to check after every install:
   - numpy: mediapipe, opencv-python, onnxruntime and faster-whisper must agree on one numpy.
   - protobuf: mediapipe pins a protobuf range; other Google libraries may want newer.
   - onnxruntime: openwakeword, moonshine-voice and Silero VAD must share ONE onnxruntime
     (CPU build; never onnxruntime-gpu alongside it).
   - comtypes: every thread that touches pycaw calls comtypes.CoInitialize() first and
     CoUninitialize() on exit.
4. Heavy or optional features live behind extras ([stt-whisper], [llm-local], [uia]) and lazy imports,
   so a missing optional package disables one feature instead of crashing startup.
5. The doctor (scripts/doctor.py, later `python -m grace doctor`) must pass before the session ends.
   It checks: Python version, `uv pip check`, every import, mic and camera open, model files present
   with expected hashes, .env keys present (values never printed), llama-server reachable if enabled,
   Windows build number. Checks for features not built yet print "not configured", not failures.
6. All tests pass: `uv run pytest -m "not hardware and not network"`. Hardware and network tests are
   marked and run manually on the base machine.
7. No secrets in code, tests, logs or commits. Logs redact anything that looks like a key.
8. Every external call (LLM, STT API) has a timeout, a retry limit and a fallback. Nothing blocks the
   audio, gesture or UI threads.
9. Small commits, one concern each, with clear messages.

## 3. Session start
1. Read docs/PLAN.md fully. If it is missing, STOP and ask the user for it.
2. Confirm you are on branch `s01-m0-hygiene` (create it from main if needed).
3. There is no uv project yet. Record the baseline instead: list files, note what main.py imports,
   and check whether `python main.py --help` runs. Report findings before changing anything.
4. Restate this milestone's tasks and acceptance criteria in 5-10 lines, then start.

## 4. This session — M0: Repo hygiene and baseline
GOAL: clean the repo and create a reproducible, testable baseline without breaking the old app.

TASKS
1. Delete: __pycache__/, "tempCodeRunnerFile (1).py", voice_agent.py, Gen_images/, the screenshot PNGs
   in the repo root, and all image-generation code and CLI arguments in main.py.
2. Update .gitignore: __pycache__/, .venv/, .env, models/, logs/, *.gguf, downloaded *.onnx, .pytest_cache/.
3. Create pyproject.toml managed by uv, Python 3.11. Pin only what the CURRENT code imports and still
   needs (follow stability rule 1 for each). Export requirements.txt from the lock.
4. Add scripts/doctor.py implementing stability rule 5; features not built yet print "not configured".
5. Add pytest + ruff config; tests/test_imports.py (every remaining module imports on Windows without
   hardware; mock what must be mocked); pytest markers: hardware, network, windows.
6. Ensure docs/PLAN.md and docs/prompts/ are committed.
7. README: update install steps to uv; keep the old usage section until replaced.

ACCEPTANCE
- `uv sync` works on a clean machine; `uv run pytest` green; `uv run python scripts/doctor.py` runs.
- `python main.py` still starts with the old behaviour (the team always has a working build).

## 5. Session end
Finish with this report and nothing after it:

### Session S01 report — M0 Repo hygiene and baseline
- Summary (3-5 lines)
- Files changed / added / deleted
- Dependencies added or changed (exact versions, licence)
- `uv pip check` output
- Test output (counts; failures with reasons)
- Doctor output
- Acceptance criteria: each one marked met / not met / needs manual check
- Manual checks the team must run on the base machine (numbered, with expected result)
- Known limitations and risks
- Follow-ups (out of scope for this session)
