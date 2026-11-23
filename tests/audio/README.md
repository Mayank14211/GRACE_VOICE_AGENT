# STT test recordings

The team's real voice recordings for `scripts/eval_stt.py` (accuracy and latency per engine).
Target: **30–50 commands per person**, recorded on the base machine (and again on the demo
laptop before demo day).

```powershell
uv run python scripts/fetch_models.py                       # once: Silero VAD for recording
uv run python scripts/record_audio.py --phrases tests/audio/phrases.txt --prefix <yourname>
uv run python scripts/eval_stt.py --save                    # results -> docs/results/stt.md
```

`record_audio.py` shows each phrase, records until you stop talking, saves
`<yourname>_NNN.wav` (16 kHz mono) and appends a row to `transcripts.csv`:

```csv
file,text,speaker
mayank_001.wav,open notepad,mayank
```

If you said something different from the prompt, fix the `text` column: it must be what is
actually in the recording. Delete a bad take's `.wav` and its row, and record it again.

**Consent.** Voices are personal data. Only commit recordings with the speaker's agreement; if
someone prefers not to, keep their files out of git (`git update-index --skip-worktree` or a
private folder passed with `eval_stt.py --audio-dir`).

Before the team's recordings exist, `scripts/make_audio_fixtures.py --eval-set data/synthetic_eval`
builds the same phrase list with the Windows SAPI voices; `eval_stt.py --audio-dir
data/synthetic_eval` gives a first (synthetic) number. Synthetic speech is cleaner than a real
voice on a laptop mic, so it is not the acceptance result.
