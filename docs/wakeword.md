# Custom wake word (openWakeWord)

GRACE listens for a wake word with [openWakeWord](https://github.com/dscripka/openWakeWord)
(offline, about 1 % of one core). The team trains its own word; until that model exists GRACE
uses the spoken word in `runtime.wake_word` (default "listen"), checked in the offline transcript.

The wake word is **optional**. `Ctrl+Alt+M` (mic on/off) and holding `Ctrl+Alt+Space`
(push-to-talk) always work, with or without it. With `wakeword.enabled: false`, everything said
while the mic is on is treated as a command.

| Setting (settings.yaml) | Default | Meaning |
|---|---|---|
| `wakeword.enabled` | `true` | `false` = no wake word: every utterance is a command while the mic is on |
| `wakeword.model_path` | `models/wakeword/grace.onnx` | the team's model (relative to the repo root) |
| `wakeword.threshold` | `0.5` | score (0–1) a frame needs; lower = more sensitive, more false wakes |
| `wakeword.patience` | `2` | 80 ms frames in a row that must reach the threshold |
| `wakeword.follow_up_s` | `10` | after a command, keep listening this long without the wake word |

## 1. Pick the word

- 2–4 syllables, with sounds that do not occur in everyday speech in the room
  ("hey grace" is better than "grace"; avoid words the team says all the time).
- Write it the way it sounds. The Colab notebook generates thousands of synthetic examples from
  the text, so the spelling drives the pronunciation (`hey grayce` if `grace` comes out wrong).
- Record the decision in `docs/PLAN.md` section 11 (open decision: "the word itself").

## 2. Collect the team's recordings

Record on the **demo laptop's built-in mic** in the **demo room** where possible: that is where
it has to work. Use 16 kHz mono WAV; `scripts/record_audio.py` does this:

```powershell
# Positives: every team member, 20+ takes each, normal / fast / quiet / from 1-2 m away.
uv run python scripts/record_audio.py --out data/wakeword/positive --prefix mayank --count 20
# Negatives: 10+ minutes of the room (fans, typing, people talking, the demo itself).
uv run python scripts/record_audio.py --out data/wakeword/negative --prefix room --seconds 60 --count 10
```

`data/` is git-ignored (voices are personal data; share the folder privately, not on GitHub).

## 3. Train in the official Colab

openWakeWord's README links two Google Colab notebooks:

1. **Simple** (<1 hour, no code):
   <https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing>
   Type the word, run all cells, download the `.onnx`. Good for a first model.
2. **Detailed** (`automatic_model_training.ipynb`, more control):
   <https://colab.research.google.com/drive/1yyFH-fpguX2BTAW8wSQxTrJnJTM-0QAd?usp=sharing>
   Use this one to add the team's data:
   - upload `data/wakeword/positive/*.wav` and add them to the **positive** training clips
     (alongside the synthetic ones the notebook generates);
   - upload `data/wakeword/negative/*.wav` and add them to the **negative / background**
     audio, so the model learns that room noise and the team's normal talk are not the word;
   - export **ONNX** (GRACE runs openWakeWord on onnxruntime, not tflite).

Save the result as `models/wakeword/grace.onnx` (or point `wakeword.model_path` at it), and set
`runtime.wake_word` to the word as it is written (e.g. `grace`): when someone says the command in
the same breath, any leftover of the word at the start of the transcript is removed using that
text. Model files are git-ignored; keep the trained file in the team drive and note its sha256:

```powershell
Get-FileHash models\wakeword\grace.onnx -Algorithm SHA256
```

Optional, if false wakes stay high: openWakeWord can also train a **custom verifier**
(a second-stage check on the team's voices, `openwakeword.train_custom_verifier`). GRACE does not
load verifiers yet; add it as a follow-up if the threshold alone is not enough.

## 4. Tune the threshold on demo-room recordings

Record **new** clips for testing (not the ones used for training): 20+ wake words per person
and 10+ minutes of room audio, then:

```powershell
uv run python scripts/tune_wakeword.py --model models/wakeword/grace.onnx `
    --positives data/wakeword/test_positive --negatives data/wakeword/test_negative
```

It prints, for thresholds 0.2–0.9, the **false-reject rate** (wake words missed) and **false
accepts per hour** (wakes from room audio), with the current `patience`. Its suggestion: among
thresholds that keep false accepts under the bar (`--max-false-per-hour`, default 0.5, which is
openWakeWord's own target alongside < 5 % false rejects), those that miss the fewest wake words,
and of those the middle one, leaving margin both ways. Set it in `settings.yaml`:

```yaml
wakeword:
  threshold: 0.6
  patience: 2
```

Re-run on the demo laptop before demo day (`scripts/demo_check.py` from S10 will include it).

## 5. Check it

```powershell
uv run python -m grace doctor     # "wake word model": loads it and reports threshold/patience
uv run python -m grace            # say the word: a short tone, the chip shows LISTENING
```

To test the whole path before the team's model exists, use openWakeWord's "hey jarvis" model
(non-commercial licence, testing only):

```powershell
uv run python scripts/fetch_models.py --demo-wakeword
# settings.yaml:  wakeword: {model_path: models/openwakeword/hey_jarvis_v0.1.onnx}
```

## How GRACE uses it

`grace/input/voice.py` feeds every 32 ms microphone frame to the detector while waiting for the
wake word. A detection plays a tone and opens the follow-up window: the next utterance (found by
Silero VAD) is transcribed and run as a command, then GRACE keeps listening without the wake word
for `follow_up_s`. Say "stop listening" to close the window early. Speaking the command in the
same breath ("hey grace, open notepad") works: the command starts after the detection, and any
leftover of the wake word (`runtime.wake_word`) at the start of the transcript is dropped.
