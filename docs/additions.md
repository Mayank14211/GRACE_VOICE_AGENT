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

<!-- Add new entries above this line. Next ID: A-003 -->
