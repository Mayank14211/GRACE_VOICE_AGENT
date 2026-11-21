# Gesture results (M3 acceptance)

Gate (docs/PLAN.md section 8): one click per pinch; deadzone holds; per-gesture success rate at or
above the team's pass bar over 10 tries. **Pass bar: TBD by the team** (PLAN section 11). Record
it here before the run: `pass bar = __ / 10`.

## How to run

1. Base machine (i5-12450H), mains power, "Best performance" power mode, normal room light,
   sitting at about arm's length from the built-in webcam.
2. `uv run python -m grace.input.gestures --log docs/results/gesture_run.jsonl`
   (add `--dry-run` for the recognition-only rows if you don't want the mouse to move).
3. For each row, do the gesture 10 times from the open-palm pose. A try **succeeds** when the
   console shows the expected `Gesture:` name and the expected action happens **exactly once**
   (for continuous gestures: the level/scroll moves the right way and stops in the deadzone), with
   no unintended click. Write any failure in Notes (wrong pose shown, double click, lag...).
4. Do the table once with the right hand and once with the left hand.

Settings used (copy the `gestures:` section of settings.yaml if it differs from the defaults):

```yaml
# defaults
```

## Run 1

Date: ____  Tester: ____  Machine: ____  Camera: ____  Light: ____  Hand: Right / Left

| # | Gesture | Expected | Successes / 10 | Notes |
|---|---|---|---|---|
| 1 | Index up | `MOVE`, cursor follows smoothly, no clicks | | |
| 2 | Cursor reach | point at each screen corner area; cursor reaches all four edges | | |
| 3 | Thumb–index pinch (quick) | exactly one left click per pinch | | |
| 4 | Pinch and hold > 0.4 s, move, release | `DRAG`: a desktop icon / window moves, released on un-pinch | | |
| 5 | Thumb–middle pinch | exactly one right click (context menu) | | |
| 6 | Two fingers up, move up / down | `SCROLL`, page scrolls the matching way | | |
| 7 | Two fingers, hold still near entry | no scrolling (deadzone) | | |
| 8 | Three fingers, move up / down | `BRIGHTNESS` changes the matching way | | |
| 9 | Thumb + two fingers, move up / down | `VOLUME` changes the matching way | | |
| 10 | Fist held 1 s, then fist 1 s again | `PAUSED`, then resumes | | |
| 11 | Open palm | `READY`, nothing happens | | |

Frames per second / hand-visible share printed at the end of the run: ____
