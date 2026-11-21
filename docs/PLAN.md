# GRACE — Project Plan

> Source of truth for anyone (human or coding agent) working on this repo.
> Coding agents: read this file fully before starting any session.
> Last updated: 2026-11-20

## 1. What GRACE is

GRACE (General-purpose Responsive Assistant for Computer Environment) is a multimodal desktop
accessibility assistant for Windows 10/11. Users control the computer by **voice** and by **hand
gestures** (webcam), at the same time, with **global hotkeys** as the reliable fallback.

| Phase | Goal | Audience |
|---|---|---|
| Phase 1 — Minor project | Reliable, demo-ready build on one machine, zero running cost | Examiners, demo day |
| Phase 2 — Publishable package | Installable, configurable, documented product | Real users |

## 2. Hardware baseline

| Role | Machine | Notes |
|---|---|---|
| **Base: development and testing** | Intel Core i5-12450H — 8 cores (4 Performance + 4 Efficiency), 12 threads, AVX2, 16 GB RAM, CPU-only inference | All performance targets are measured here |
| Demo laptop (secondary) | AMD Ryzen 5 5500U — 6 cores / 12 threads, AVX2, 16 GB RAM, built-in mic | Re-run `scripts/eval_stt.py` and `scripts/demo_check.py` here before demo day |

Rules that follow from the hardware:

- **Inference threads are a setting** (`inference.threads` in `settings.yaml`).
  Defaults: 4 on the i5-12450H (P-cores only; E-cores often slow llama.cpp down),
  6 on the Ryzen 5 5500U. Benchmark 4 / 6 / 8 once with `scripts/bench_threads.py` and keep the fastest.
- Never use logical-thread counts (12) for inference.
- Run performance tests on mains power with Windows power mode set to "Best performance".
- No GPU code paths in Phase 1. ONNX Runtime CPU build only.
- **OS assumption:** the base machine runs Windows 10/11. If it runs Linux, only non-hardware tests run
  there; Windows-only modules (pycaw, winsdk, pywinauto, hotkeys, overlay) are covered by fakes, and
  hardware checks move to the demo laptop.

## 3. Hard requirements (Phase 1)

1. Voice and gesture control run **simultaneously in one process**.
2. **Global hotkeys** (configurable) are the primary toggles:

   | Hotkey | Action |
   |---|---|
   | `Ctrl+Alt+M` | Mic listening on/off |
   | `Ctrl+Alt+G` | Gesture control on/off |
   | `Ctrl+Alt+Space` (hold) | Push-to-talk |
   | `Ctrl+Alt+Q` | Panic stop: stop all input, release mouse, app stays alive |
   | `Ctrl+Alt+Y` | Confirm a pending risky action |
   | `Ctrl+Alt+H` | Show/hide conversation history panel |
   | `Ctrl+Alt+X` | Mark last action as wrong (metrics) |

   Reason: in testing, the wake word needed repeating on the built-in laptop mic with no noise
   cancellation. Voice "gesture on/off" may exist but is never relied on.
3. **No camera preview.** The only UI is a transparent, click-through overlay:
   - subtitle bar, bottom-centre of the primary screen: what GRACE heard and replied;
   - status chip, bottom-right: mic, gesture, current recognised gesture name, provider in use.
4. **Offline-first.** Wake word, English STT, gestures and all fixed commands work with no network.
5. **Risky actions** (shutdown, restart, close-all, send message) need exact phrasing **and**
   confirmation. No substring matching may trigger them.
6. **No secrets in code.** Keys come only from `.env` (python-dotenv).
7. **Zero running cost.** Only free tiers and local models.

## 4. Architecture

```
            ┌──────────── Input (worker threads) ─────────────┐
 Hotkeys ──►│ hotkeys.py                                       │
 Mic ──────►│ audio.py (sounddevice + Silero VAD)              │
            │   ├─ wakeword.py (openWakeWord, custom word)     │
            │   └─ stt/ moonshine | whisper_local | groq       │
 Webcam ───►│ gestures/ tracker → classifier → controller      │
            └───────────────────────┬──────────────────────────┘
                                    │  events
                              core/bus.py  (+ core/state.py)
                                    │
            ┌───────────────────────▼──────────────────────────┐
            │ intent/grammar.py  ── match? ──► Intent           │
            │        │ no                                       │
            │        ▼                                          │
            │ intent/router.py → providers: groq → gemini →     │
            │                    openrouter → local llama-server│
            └───────────────────────┬──────────────────────────┘
                                    │ Intent (validated)
            ┌───────────────────────▼──────────────────────────┐
            │ actions/registry.py  (ONLY layer that touches OS) │
            │ system | input_ctl | apps | screen |              │
            │ notifications | uia                               │
            └───────────────────────┬──────────────────────────┘
                                    │ results
            ┌───────────────────────▼──────────────────────────┐
            │ ui/overlay.py (Qt main thread) · feedback/tts.py  │
            │ feedback/earcons.py · metrics/log.py (JSONL)      │
            └───────────────────────────────────────────────────┘
```

**Threading:** Qt owns the main thread. Audio, gesture loop, STT and router run in worker
threads. Workers communicate only via `core/bus.py`; overlay updates via Qt signals. All OS actions
run in one action-executor thread so voice and gesture never race. Any thread using pycaw calls
`comtypes.CoInitialize()` first.

**Target layout:**

```
grace/
  __main__.py            # python -m grace ; python -m grace doctor
  config.py              # .env + settings.yaml -> pydantic Settings
  core/bus.py, core/state.py
  input/hotkeys.py, input/audio.py, input/wakeword.py
  input/stt/{base,moonshine,whisper_local,groq}.py
  input/gestures/{tracker,classifier,controller}.py
  intent/{grammar,router,providers}.py
  actions/{registry,system,input_ctl,apps,screen,notifications,uia}.py
  ui/overlay.py
  feedback/{tts,earcons}.py
  metrics/log.py
scripts/  doctor.py, eval_stt.py, bench_threads.py, report.py, demo_check.py, run_llama_server.ps1
models/   (git-ignored; downloaded with hash checks)
tests/    unit tests + fixtures (gestures/*.json, audio/*.wav)
docs/     PLAN.md, wakeword.md, demo_script.md, results/
```

## 5. Tech stack (decided)

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.11, 64-bit | Widest Windows wheel coverage (MediaPipe, PySide6, ONNX Runtime, winsdk) |
| Env | `uv` + `uv.lock`; exported `requirements.txt` | Reproducible installs |
| Config | `python-dotenv` + `settings.yaml` (pydantic) | Keys out of code; tunables without code edits |
| Overlay | PySide6 | Translucent click-through window, safe cross-thread signals |
| Hotkeys | `keyboard` | Global hotkeys on Windows |
| Audio + VAD | `sounddevice` + Silero VAD (ONNX) | Low latency; ends utterance on silence |
| Wake word | `openwakeword`, custom .onnx | Offline, tiny, trainable |
| STT local | `moonshine-voice` Small Streaming (English) | ~165 ms vs ~3.4 s for Whisper Small on x86 CPU |
| STT fallback | `faster-whisper` base int8; Groq `whisper-large-v3-turbo` | Hindi/Hinglish, accurate dictation |
| LLM client | `openai` SDK for Groq, Gemini (OpenAI-compatible endpoint), OpenRouter, local `llama-server` | One code path; provider swap by config |
| Local LLM | llama.cpp `llama-server` + Gemma-4-E4B Q4_K_M | Best CPU tool-calling accuracy in research; separate process |
| Gestures | `mediapipe` + `opencv-python` | Real-time on CPU |
| Volume / brightness | `pycaw` + `comtypes`; `screen-brightness-control` | Exact control; replaces WMI/key hacks |
| Notifications | `winsdk` UserNotificationListener | Real toast data, no OCR |
| UI automation | `pywinauto` (UIA) — stretch | Click/read controls by name |
| TTS | `pyttsx3` (SAPI); optional `piper-tts` | Instant, free |
| Tests / lint | `pytest`, `pytest-mock`, `pytest-qt`, `pytest-timeout`, `ruff` | Fast feedback; hardware tests marked |

**Removed from the old code:** `google-generativeai` (deprecated; `gemini-pro` no longer exists),
`SpeechRecognition` + Google web STT, `PyAudio`, image generation, all hard-coded keys.

## 6. Models and providers (as of 2026-11-20 — re-check before Phase 2)

**Routing rule:** fixed commands are parsed by `intent/grammar.py` and never call an LLM.
Only unmatched or conversational input goes to the router.

| Provider (order) | Model | Free limits | Use |
|---|---|---|---|
| Groq | `openai/gpt-oss-20b` (LLM), `whisper-large-v3-turbo` (STT) | LLM: 30 req/min, 1,000/day, 200K tokens/day. Whisper: 20 req/min, 2,000/day, 28,800 audio-s/day. No card | Primary cloud |
| Google Gemini | Flash-Lite (current ID from docs) | Per-project, visible only in AI Studio; free-tier inputs may be used for training | Backup LLM; never send private screen text |
| OpenRouter | a `:free` model | 20 req/min, 50/day | Last resort |
| Local | Gemma-4-E4B Q4_K_M via llama-server | Unlimited; ~9–10 tokens/s on a 4-core AVX2 CPU (benchmark, not our machine) | Offline + demo mode |

**Added early (2026-11-21, on the team's request, before S07):** a Gemini assistant in
`grace/intent/assistant.py` + `providers.py` answers unmatched speech in the legacy `main.py` loop:
questions with memory, tool calling over the registry (risky actions never exposed), web search
(Google Search grounding) and image generation (`gemini-3.1-flash-image`). It calls the Gemini
**Interactions API** (`/v1beta/interactions`) with the pinned httpx; no new dependency. Verified:
text + tool calling are free on `gemini-3.5-flash-lite`; search grounding and image generation
have **no free tier** (HTTP 429 `limit: 0`), so they conflict with "zero running cost" unless the
team uses a billing-enabled key. S07 should fold this into the router's provider chain rather than
build a second Gemini client. Image generation re-enters scope by team decision (section 5 had
removed it).

Router behaviour: 3 s timeout (cloud), 15 s (local); HTTP 429/error → provider cooldown 60 s → next.
Keep system prompt + tool list byte-identical across calls so llama-server reuses its prompt cache.
Expose 8–10 grouped tools, not one per command. Demo mode pins provider order to `[local]`.

## 7. Gesture map (default; editable in settings.yaml)

| Pose | Action |
|---|---|
| Index up | Move cursor (inner active box of frame → full screen, One Euro smoothing, mirrored) |
| Thumb–index pinch | Left click once per pinch; hold > 400 ms = drag |
| Thumb–middle pinch | Right click |
| Index + middle up | Scroll by offset from entry point, deadzone |
| Index + middle + ring up | Brightness by offset from entry point, deadzone |
| Thumb + index + middle up | Volume by offset from entry point, deadzone |
| Fist held 1 s | Pause / resume gestures |
| Open palm | Neutral; resets entry points |

A pose must hold for N frames (default 4) to activate. No `time.sleep()` in the camera loop.
The active gesture name is shown on the overlay status chip.

## 8. Milestones, sessions and gates

One milestone = one coding session = one prompt file in `docs/prompts/`.
A session starts only when the previous gate is green.

| Session | Milestone | Gate (all must pass) |
|---|---|---|
| S01 | M0 Repo hygiene + baseline | Clean `uv sync`; tests + doctor run; old app still starts |
| S02 | M1 Secrets + config | Secret scan passes; starts with empty `.env` |
| S03 | M2 Action layer + safe grammar | ≥60 grammar cases incl. negatives; no risky action without confirmation; exact volume/brightness |
| S04 | M3 Gesture engine | One click per pinch; deadzone; per-gesture success ≥ bar (TBD) over 10 tries |
| S05 | M4 Runtime + hotkeys + overlay v1 | Voice + gesture together 10 min; toggles < 200 ms |
| S06 | M5 Audio: VAD, wake word, local STT | Median end-of-speech → text < 500 ms (TBD) on base machine |
| S07 | M6 Intent router + conversation | No crash on any provider failure; 20 free-form requests logged |
| S08 | M7 Overlay polish + feedback | State always visible; 1-min capture |
| S09 | M8 Notification reader | Test toast + WhatsApp Desktop message announced |
| S10 | M9 Metrics + demo mode | 3 rehearsals logged; `report.md`; `demo_check` green on demo laptop |
| S11 | M10 UI Automation (stretch) | Click-by-name in Notepad, Explorer, Save dialog |
| S12–S18 | Phase 2: P1–P7 | Phase 1 suite stays green; clean-VM install |

**Phase 2 scope:** P1 package + platform layer + plugins · P2 tray app, settings UI, installer ·
P3 full UI Automation navigation · P4 Hindi/English + offline mode · P5 accessibility modes
(dwell-click, gesture-only, push-to-talk-only) · P6 WhatsApp read/reply via Baileys sidecar with
confirmation · P7 CI, docs, PyPI release. **Out of scope:** head/face tracking.

## 9. Session workflow (humans)

1. `git checkout main && git pull`, then `git checkout -b sNN-<milestone>`.
2. Paste the whole `docs/prompts/sNN-*.md` file into the coding agent. Nothing else is needed.
3. Agent ends with the report template. Run the manual hardware checks yourself.
4. Merge, tag `mN-done`, update the checkbox in section 10.

## 10. Status

- [ ] S01 M0 · [ ] S02 M1 · [ ] S03 M2 · [ ] S04 M3 · [ ] S05 M4 · [ ] S06 M5
- [ ] S07 M6 · [ ] S08 M7 · [ ] S09 M8 · [ ] S10 M9 · [ ] S11 M10 (stretch)

## 11. Open decisions

- Pass bars for gestures and STT latency (team to confirm before S04 and S06).
- Custom wake word: the word itself, and who records training samples.
- Whether to purge leaked keys from git history (`git filter-repo`) — keys must be revoked either way.
- Base machine OS (Windows assumed).

## 12. References

- Research: models, free APIs, benchmarks — https://claude.ai/artifact/5sCgoq9DQMEs9rQqR7WjR3
- Build prompts doc — https://claude.ai/artifact/KLE7j8Nc3FjNgKuRKBkNk5
