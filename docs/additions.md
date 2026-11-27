# GRACE additions register

Suggestions from the team that go beyond (or change) the milestone prompts in `docs/prompts/`.
Every suggestion is recorded here **when it is made**, even if it is built later.

## Why this file exists

Planned additions must not force a rebuild of components that already exist. So:

1. **At the start of every session**, after `docs/PLAN.md`, read this file.
2. For each entry that is `Pending` or `Planned`, check the **Constraints** line against what the
   session is about to build. Build in a way that the addition can be added later *without
   changing* the new component's public interface. If the session cannot honour a constraint,
   stop and report it before building.
3. When the session implements an entry, set its status to `Done (SNN)` and record where it lives.
4. New suggestions from the team are appended as the next `A-NNN`, with the date.

Statuses: `Pending` (wanted, not scheduled) · `Planned (SNN)` (scheduled for a session) ·
`Done (SNN)` · `Dropped` (with the reason).

## Entries

### A-001: On-screen microphone button

- **Date / status:** 2026-11-23 · Done (S05)
- **Request:** a button on the screen that toggles the mic, so the user does not need the
  Ctrl+Alt+M shortcut. Convenience.
- **Decision:** the hotkey stays (PLAN section 3 makes hotkeys the primary fallback, and panic
  stop must not need a mouse). The button is a second route to the same toggle.
- **Where:** `MicButton` / `FloatingButton` in `grace/ui/overlay.py`; clicks call
  `Runtime.toggle_mic(t, source="button")`; setting `runtime.mic_button` (default on).
- **Deviation from PLAN section 3 item 3** ("only UI is a click-through overlay"): the button is a
  small window that accepts mouse clicks but never takes keyboard focus or activation. It is the
  only overlay window that is not click-through.
- **Constraints for other work:**
  - Any further on-screen control (gesture toggle, panic, history) subclasses `FloatingButton`;
    it does not add a new window type.
  - Controls call `Runtime` methods, never state or workers directly, so hotkey, button and voice
    share one code path. Runtime methods that a user can trigger take `source`
    (`hotkey` | `button` | `voice`) so S10 can report which route is used.
  - The overlay must stay usable with `--no-overlay` and without PySide6; nothing outside
    `grace/ui/` may import Qt.
  - Positions are computed from the primary screen's available area, so S08 (polish) can move or
    make buttons draggable without touching callers.

### A-002: On-screen gesture (camera) button

- **Date / status:** 2026-11-23 · Done (S05)
- **Request:** the same convenience as A-001 for gesture control, which also switches the camera
  hardware on and off, so the user does not need Ctrl+Alt+G.
- **Decision:** the hotkey stays, as for A-001. The button is a second route to
  `Runtime.toggle_gestures(t, source="button")`.
- **Where:** `GestureButton` in `grace/ui/overlay.py`, placed left of the mic button; setting
  `runtime.gesture_button` (default on). Green camera = on, red slash = off, red ring = gestures
  are on but the camera is missing or busy (`NO CAMERA`).
- **Constraints for other work:**
  - Buttons line up leftwards from the chip in `Overlay._place_buttons`; a new control is added
    to `Overlay._buttons()` and nothing else about the layout changes.
  - The camera is only open while gestures are on (`GestureLoop`); a button must never open or
    close it directly, only call `Runtime`.

### A-003: Cloud-first router: no local LLM, our own tools, quota governor

- **Date / status:** 2026-11-26 · Done (S07) for the router; the STT side (Groq Whisper default,
  Moonshine as the optional `[stt-local]` extra) belongs to S06b.
- **Request:** the team's architecture decision of 2026-11-26. No local LLM (llama.cpp / Gemma).
  The LLM chain is Groq gpt-oss-20b -> Gemini Flash-Lite -> OpenRouter `:free`. The router is a
  stateless orchestrator that keeps the dialogue itself. `web_search` (one free search API) and
  `generate_image` (Cloudflare Workers AI FLUX.1 schnell) are our own tools; Gemini search
  grounding and image generation are out (billing). Risky actions are never tools. A quota
  governor tracks requests and tokens per provider and model, and warns at 80 %.
- **Where:** `grace/intent/router.py`, `tools.py`, `search.py` (Tavily, choice in
  `docs/results/search.md`), `images.py`, `assistant.py` (`Extras`), `grace/core/cooldown.py`,
  `grace/core/quota.py`; settings `router:`, `images:`, `search:`; keys `CLOUDFLARE_ACCOUNT_ID`,
  `CLOUDFLARE_API_TOKEN`, `TAVILY_API_KEY`.
- **Deviations:** Gemini chat uses its OpenAI-compatible endpoint, not the S03 Interactions
  client (team choice in the S07 continuation). Gemini's timeout is 6 s, not 3 s (measured
  free-tier tail).
- **Constraints for other work:**
  - New router tools are executed by our code. The model only emits calls; add the executor
    to `Extras` and a single-action tool in `grace/intent/tools.py`.
  - Registry actions named `notifications.*` (S09) appear in the `notifications` tool group
    automatically; don't add a separate tool.
  - Every cloud call goes through `grace/core/quota.py` (`allow` before, `record` after) and
    has a timeout. S06b's `grace/input/stt/budget.py` must share this store (TODO in quota.py)
    rather than build a second one.
  - Screen text and selected code never go to Gemini: use `Router.complete(...,
    exclude={"gemini"})`.
  - Nothing may store conversation state on a provider (no server-side ids); history lives in
    `grace/intent/dialogue.py`.

<!-- Add new entries above this line. Next ID: A-004 -->
