# Runtime results (M4 acceptance)

Gate (docs/PLAN.md section 8): voice + gesture together for 10 minutes without freezing; toggles
respond within 200 ms; CPU usage recorded.

## How to run

1. Base machine (i5-12450H), mains power, Windows power mode "Best performance", normal room light,
   built-in mic and webcam.
2. `uv run python -m grace --gestures --seconds 660` (11 minutes; quits by itself).
   GRACE writes a metrics file to `logs/metrics/session-<date>-<time>.jsonl` and prints a summary
   on exit: number of toggles shown, median / max toggle latency, mean / max CPU.
3. During the run, keep **both** inputs busy:
   - every ~30 s a voice command ("listen, volume 40", "listen, open notepad", "listen, scroll
     down"), and a few with push-to-talk (hold Ctrl+Alt+Space, say "volume 30", release);
   - in between, gestures: move the cursor, click, scroll, change volume / brightness;
   - at least 10 presses each of Ctrl+Alt+M and Ctrl+Alt+G, 3 of Ctrl+Alt+H, 2 of Ctrl+Alt+Q
     (then turn mic and gestures back on), 1 of Ctrl+Alt+X.
4. Open Task Manager > Details, add the "CPU" column, and note the `python.exe` value at 2, 5 and
   9 minutes (cross-check of the metrics file).
5. Fill in the table. "Freeze" = the overlay stops updating, the cursor stops following the hand,
   or a hotkey does nothing for more than 1 s.

Toggle latency in the metrics file is measured from the key press (keyboard hook callback) to the
status chip being repainted on screen. It does not include the keyboard hook's own delivery time.

## Run 1

Date: ____  Tester: ____  Machine: ____  Power mode: ____  Duration: ____ min

| # | Check | Expected | Result | Notes |
|---|---|---|---|---|
| 1 | 10 min with voice + gestures | no freeze, no crash, overlay keeps updating | | |
| 2 | Toggle latency (summary line) | median and max < 200 ms | median ___ ms, max ___ ms | |
| 3 | Ctrl+Alt+M x10 | chip flips each time, tone plays, no double toggle when held | | |
| 4 | Ctrl+Alt+G x10 | chip flips; camera LED on only while gestures are on | | |
| 5 | Push-to-talk x5 | command runs without the wake word, also with the mic off | | |
| 6 | Ctrl+Alt+Q during a drag | button released, mic + gestures off, app still running | | |
| 7 | Ctrl+Alt+H | history panel shows / hides | | |
| 8 | Ctrl+Alt+X | "Marked ... as wrong" shown; `marked_wrong` row in metrics | | |
| 9 | Overlay | click-through, never takes focus, not in taskbar / Alt+Tab | | |
| 10 | CPU (metrics summary) | recorded | mean ___ %, max ___ % of the machine | |
| 11 | CPU (Task Manager, 2/5/9 min) | recorded | ___ / ___ / ___ % | |
| 12 | Mic button x10 (click) | flips mic, tone plays, typing app keeps focus (caret keeps blinking) | | |
| 13 | Mic button vs Ctrl+Alt+M | same result; metrics `toggle` rows show `source` button / hotkey | | |
| 14 | Mic button latency | click-to-chip median < 200 ms in the summary line | | |
| 15 | Camera button x10 (click) | gestures and camera LED flip together, tone plays, typing app keeps focus | | |
| 16 | Camera button with the camera in use elsewhere | red ring and chip says `NO CAMERA`; recovers when the other app closes it | | |
