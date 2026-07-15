# Archangel

A Linux PDF reader that reads to you in a natural voice, highlights the
current sentence as it goes, and lets you click any word to jump there.
Built on [Piper TTS](https://github.com/rhasspy/piper) for offline neural
speech, PyMuPDF for the PDF layer, and PyQt6 for the GUI.

![status](https://img.shields.io/badge/status-alpha-orange)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

## What it does

- **Reads PDFs aloud** using Piper voice models (fully offline once installed).
- **Highlights the current sentence** in the PDF as it's spoken.
- **Click any word** to start reading from that sentence.
- **Play / Pause / Stop / Prev / Next sentence** — real pause (`QMediaPlayer`),
  not `aplay` kill-and-restart.
- **All Piper knobs in the sidebar**: voice model, speed (`length_scale`),
  noise scale, phoneme-duration variation (`noise_w`), sentence silence,
  speaker id for multi-speaker models.
- Zoom, page navigation, volume.
- **Synthesis runs in-process on a worker thread**, with the voice model
  loaded once and the next sentence prefetched while the current one plays.
  The GUI never blocks.

## Performance

Measured on the primer PDF (8 pages, 4.6k words, 287 sentences),
`en_GB-alan-medium`, CPU only:

| | Time |
|---|---|
| Voice model load (once, at first play) | ~3 s |
| Synthesis per sentence thereafter | ~0.3–0.9 s |

Because the next sentence is synthesized while the current one is playing,
the gap between sentences is normally inaudible.

> An earlier version of this app spawned the `piper` CLI once per sentence,
> which reloaded the 63 MB model every time and cost ~2.4 s of dead air
> between sentences. If you fork from that revision, this is the thing to fix.

## Install

Archangel expects Piper installed in a Python venv. Recommended layout:

```bash
mkdir -p ~/.local/piper && cd ~/.local/piper
python3 -m venv venv
source venv/bin/activate
pip install piper-tts PyQt6 PyMuPDF
```

Grab at least one voice model into `~/.local/piper/voices/` from
[huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) —
each voice is a pair of files:

```bash
mkdir -p ~/.local/piper/voices && cd ~/.local/piper/voices
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium"
curl -L -o en_GB-alan-medium.onnx "$BASE/en_GB-alan-medium.onnx"
curl -L -o en_GB-alan-medium.onnx.json "$BASE/en_GB-alan-medium.onnx.json"
```

Any of the [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
models works. Popular English picks: `en_US-lessac-medium`,
`en_US-hfc_female-medium`, `en_GB-alan-medium`.

## Run

```bash
~/.local/piper/venv/bin/python archangel.py /path/to/document.pdf
```

Or without an argument, then *File → Open PDF…*.

## Configuration

Archangel discovers voices from `~/.local/piper/voices/*.onnx` and the
piper CLI from `~/.local/piper/venv/bin/piper`. If your paths differ,
edit the two constants at the top of `archangel.py`:

```python
PIPER_BIN = str(Path.home() / ".local/piper/venv/bin/piper")
VOICES_DIR = Path.home() / ".local/piper/voices"
```

## Piper knobs, briefly

| Control | Piper flag | Default | What it does |
|---|---|---|---|
| Speed | `--length_scale` | 1.0 | Lower = faster (0.85 is a sensible faster-reading value) |
| Noise scale | `--noise_scale` | 0.667 | Prosody variability — higher is more expressive, less predictable |
| Noise-W | `--noise_w` | 0.8 | Per-phoneme duration variability — higher gives natural rhythm |
| Sentence silence | `--sentence_silence` | 0.2 s | Pause between sentences |
| Speaker | `--speaker` | 0 | For multi-speaker models only |

## Limits

- **Sentence-level highlighting**, not word-by-word — for now. Piper's
  Python API does expose `synthesize(..., include_alignments=True)`
  returning per-phoneme timing, so word-level karaoke is a real
  possibility rather than a research problem; it simply isn't wired up
  yet. See the roadmap.
- **PDF quality**: text extraction depends on the PDF having a real text
  layer. Scanned PDFs need OCR first (e.g. `ocrmypdf`).
- **Rotated pages** are not handled in the click-to-position hit test.
- Synthesized WAVs accumulate in a temp directory for the session; they're
  cleaned up on exit, not pruned as you go.
- Sentence splitting leans on PyMuPDF's block structure, which is reliable
  for ordinary prose PDFs. Multi-column layouts, tables, and figure
  captions may still group oddly.

## Controls

| | |
|---|---|
| **Space** | Play / pause |
| **Shift + →** | Next sentence |
| **Shift + ←** | Previous sentence |
| **Click a word** | Read from that sentence |
| Hover a word | Faint underline preview; word shown in the status bar |

## Roadmap

- Word-level karaoke highlighting via `synthesize(include_alignments=True)`
- Interval index for sentence→word mapping (unblocks book-length PDFs)
- Configurable highlight colour and shape
- Bookmarks and reading position memory
- Export selected pages to WAV/MP3

## Licence

MIT. See `LICENSE`.
