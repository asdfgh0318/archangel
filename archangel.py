#!/usr/bin/env python3
"""Archangel — a PDF reader that reads to you with Piper TTS, with sentence-level
highlighting, click-to-position, and full Piper knob control.

Depends on: PyQt6, PyMuPDF, piper-tts (already in venv). Voices in ~/.local/piper/voices.
"""
from __future__ import annotations

import re
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from piper import PiperVoice, SynthesisConfig
from PyQt6.QtCore import (QObject, QRectF, Qt, QThread, QUrl, pyqtSignal)
from PyQt6.QtGui import (QAction, QBrush, QColor, QImage, QPainter, QPen, QPixmap)
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSlider, QSpinBox, QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

VOICES_DIR = Path.home() / ".local/piper/voices"
RENDER_ZOOM = 1.6

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])|(?<=[.!?])\n+")


@dataclass
class WordBox:
    """One word extracted from a PDF page, with its bounding box."""
    page: int          # 0-indexed page number
    x0: float          # PDF coords (points)
    y0: float
    x1: float
    y1: float
    text: str
    char_start: int    # offset into the concatenated document string


@dataclass
class Sentence:
    """A contiguous run of words forming one sentence."""
    text: str
    word_indices: list[int]  # into WordBox list


class PdfScene(QGraphicsScene):
    """Displays one PDF page + a highlight overlay for the current sentence.
    Emits a signal when a word is clicked."""
    word_clicked = pyqtSignal(int)  # word global index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._highlight_items: list[QGraphicsRectItem] = []
        self._word_rects: list[tuple[QRectF, int]] = []  # (rect on scene, word idx)
        self._zoom = RENDER_ZOOM

    def show_page(self, page: fitz.Page, page_words: list[tuple[WordBox, int]]):
        """Render a page. Caller passes (WordBox, global_index) pairs already
        filtered to this page."""
        self.clear()
        self._highlight_items.clear()
        self._word_rects.clear()
        mat = fitz.Matrix(self._zoom, self._zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride,
                     QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(img.copy())
        self._pixmap_item = self.addPixmap(pm)
        self._pixmap_item.setZValue(0)
        self.setSceneRect(0, 0, pm.width(), pm.height())
        for w, gi in page_words:
            r = QRectF(w.x0 * self._zoom, w.y0 * self._zoom,
                       (w.x1 - w.x0) * self._zoom, (w.y1 - w.y0) * self._zoom)
            self._word_rects.append((r, gi))

    def highlight_words(self, global_indices: set[int]):
        for it in self._highlight_items:
            self.removeItem(it)
        self._highlight_items.clear()
        pen = QPen(Qt.PenStyle.NoPen)
        brush = QBrush(QColor(255, 235, 60, 110))  # translucent yellow
        for rect, gi in self._word_rects:
            if gi in global_indices:
                pad = 1.5
                item = QGraphicsRectItem(rect.adjusted(-pad, -pad, pad, pad))
                item.setPen(pen)
                item.setBrush(brush)
                item.setZValue(1)
                self.addItem(item)
                self._highlight_items.append(item)

    def mousePressEvent(self, event):
        p = event.scenePos()
        for rect, gi in self._word_rects:
            if rect.contains(p):
                self.word_clicked.emit(gi)
                return
        super().mousePressEvent(event)

    def set_zoom(self, z: float):
        self._zoom = z


class Voices:
    """Discovers .onnx voice models in VOICES_DIR."""
    @staticmethod
    def list() -> list[Path]:
        return sorted(VOICES_DIR.glob("*.onnx"))


class SynthWorker(QObject):
    """Synthesizes sentences on a background thread using an in-process
    PiperVoice. The voice model is loaded once and reused, which is the
    whole point: spawning the piper CLI per sentence costs ~2.4 s of model
    loading every time, which is audible dead air between sentences."""
    done = pyqtSignal(int, str)     # sentence index, wav path
    failed = pyqtSignal(int, str)   # sentence index, error message
    voice_loaded = pyqtSignal(str)  # voice name

    def __init__(self, tempdir: str):
        super().__init__()
        self._voice: PiperVoice | None = None
        self._voice_path: str | None = None
        self._tempdir = tempdir

    def ensure_voice(self, path: str):
        """Load a voice if it isn't the one already loaded. Runs on the worker
        thread; the ~2 s cost is paid once per voice, not once per sentence."""
        if self._voice_path == path and self._voice is not None:
            return
        self._voice = PiperVoice.load(path)
        self._voice_path = path
        self.voice_loaded.emit(Path(path).name)

    def synthesize(self, index: int, text: str, voice_path: str,
                   length_scale: float, noise_scale: float, noise_w: float,
                   sentence_silence: float, speaker_id: int | None):
        try:
            self.ensure_voice(voice_path)
            cfg = SynthesisConfig(
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w_scale=noise_w,
                speaker_id=speaker_id,
            )
            out = Path(self._tempdir) / f"s_{index}.wav"
            with wave.open(str(out), "wb") as wf:
                self._voice.synthesize_wav(text, wf, syn_config=cfg)
            self.done.emit(index, str(out))
        except Exception as e:  # noqa: BLE001 - surface any synth failure to the UI
            self.failed.emit(index, str(e))


class MainWindow(QMainWindow):
    # index, text, voice_path, length_scale, noise_scale, noise_w,
    # sentence_silence, speaker_id
    synth_requested = pyqtSignal(int, str, str, float, float, float, float, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Archangel")
        self.resize(1400, 900)

        self.doc: fitz.Document | None = None
        self.words: list[WordBox] = []
        self.text: str = ""
        self.sentences: list[Sentence] = []
        self.current_sentence: int = -1
        self.current_page: int = 0
        self.speakers: dict[str, int] = {}
        self._tempdir = tempfile.TemporaryDirectory(prefix="archangel_")

        # synthesis state
        self._wav_cache: dict[int, str] = {}   # sentence index -> wav path
        self._pending: set[int] = set()        # sentence indices in flight
        self._want_play = False                # play as soon as current arrives

        # audio
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.audio_out.setVolume(0.9)

        # synthesis worker on its own thread — the voice model is loaded once
        # there and reused, so sentences cost synthesis time only.
        self._thread = QThread(self)
        self._worker = SynthWorker(self._tempdir.name)
        self._worker.moveToThread(self._thread)
        self._worker.done.connect(self._on_synth_done)
        self._worker.failed.connect(self._on_synth_failed)
        self._worker.voice_loaded.connect(
            lambda n: self.statusBar().showMessage(f"Voice loaded: {n}", 3000))
        self.synth_requested.connect(self._worker.synthesize)
        self._thread.start()

        self._build_ui()
        self._refresh_voice_list()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # --- left: PDF viewer -----------------------------------
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        self.scene = PdfScene()
        self.scene.word_clicked.connect(self._on_word_clicked)
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.Antialiasing)
        lv.addWidget(self.view, 1)

        # nav bar
        nav = QHBoxLayout()
        self.open_btn = QPushButton("Open PDF…")
        self.open_btn.clicked.connect(self.open_pdf)
        self.prev_page_btn = QPushButton("◀ Page")
        self.prev_page_btn.clicked.connect(lambda: self._goto_page(self.current_page - 1))
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.valueChanged.connect(lambda v: self._goto_page(v - 1))
        self.next_page_btn = QPushButton("Page ▶")
        self.next_page_btn.clicked.connect(lambda: self._goto_page(self.current_page + 1))
        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.5, 4.0)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setValue(RENDER_ZOOM)
        self.zoom_spin.valueChanged.connect(self._on_zoom)
        nav.addWidget(self.open_btn)
        nav.addStretch()
        nav.addWidget(self.prev_page_btn)
        nav.addWidget(QLabel("Page"))
        nav.addWidget(self.page_spin)
        nav.addWidget(self.next_page_btn)
        nav.addStretch()
        nav.addWidget(QLabel("Zoom"))
        nav.addWidget(self.zoom_spin)
        lv.addLayout(nav)

        # transport
        transport = QHBoxLayout()
        self.prev_btn = QPushButton("⏮ Prev")
        self.prev_btn.clicked.connect(lambda: self._jump_sentence(-1))
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.next_btn = QPushButton("Next ⏭")
        self.next_btn.clicked.connect(lambda: self._jump_sentence(+1))
        for b in (self.prev_btn, self.play_btn, self.stop_btn, self.next_btn):
            b.setMinimumWidth(90)
            transport.addWidget(b)
        transport.addStretch()
        self.sentence_lbl = QLabel("–")
        self.sentence_lbl.setStyleSheet("color:#666;")
        self.sentence_lbl.setMaximumWidth(600)
        self.sentence_lbl.setWordWrap(False)
        transport.addWidget(self.sentence_lbl, 1)
        lv.addLayout(transport)

        splitter.addWidget(left)

        # --- right: Piper controls sidebar ----------------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 8, 8, 8)

        # voice
        gb_voice = QGroupBox("Voice model")
        gv = QVBoxLayout(gb_voice)
        self.voice_combo = QComboBox()
        gv.addWidget(self.voice_combo)
        self.reload_voices_btn = QPushButton("Reload voice list")
        self.reload_voices_btn.clicked.connect(self._refresh_voice_list)
        gv.addWidget(self.reload_voices_btn)
        self.speaker_lbl = QLabel("Speaker: (single)")
        self.speaker_spin = QSpinBox()
        self.speaker_spin.setMinimum(0)
        self.speaker_spin.setMaximum(0)
        self.speaker_spin.setEnabled(False)
        gv.addWidget(self.speaker_lbl)
        gv.addWidget(self.speaker_spin)
        rv.addWidget(gb_voice)
        self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)

        # piper knobs
        gb_knobs = QGroupBox("Synthesis knobs")
        gl = QVBoxLayout(gb_knobs)
        self.speed = self._add_knob(gl, "Speed (length_scale)", 0.5, 2.0, 0.05, 1.0,
                                    "Lower = faster.  Piper default 1.0")
        self.noise_scale = self._add_knob(gl, "Noise scale", 0.0, 1.5, 0.05, 0.667,
                                          "Prosody variation.  Piper default 0.667")
        self.noise_w = self._add_knob(gl, "Noise-W (phoneme dur.)", 0.0, 1.5, 0.05, 0.8,
                                      "Phoneme duration variability.  Piper default 0.8")
        self.sentence_silence = self._add_knob(gl, "Sentence silence (s)", 0.0, 3.0, 0.05, 0.2,
                                                "Piper default 0.2")
        self.reset_knobs_btn = QPushButton("Reset synthesis knobs")
        self.reset_knobs_btn.clicked.connect(self._reset_knobs)
        gl.addWidget(self.reset_knobs_btn)
        rv.addWidget(gb_knobs)

        # Changing any knob makes already-rendered audio stale.
        for knob in (self.speed, self.noise_scale, self.noise_w,
                     self.sentence_silence):
            knob.valueChanged.connect(self._invalidate_cache)
        self.speaker_spin.valueChanged.connect(self._invalidate_cache)

        # audio out
        gb_audio = QGroupBox("Audio")
        ga = QVBoxLayout(gb_audio)
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume"))
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(90)
        self.vol_slider.valueChanged.connect(lambda v: self.audio_out.setVolume(v / 100.0))
        vol_row.addWidget(self.vol_slider)
        ga.addLayout(vol_row)
        rv.addWidget(gb_audio)

        rv.addStretch()

        # about the voices dir
        info = QLabel(f"<small>Voices dir: <code>{VOICES_DIR}</code><br>"
                      "Synthesis: in-process Piper (model loaded once)</small>")
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setWordWrap(True)
        info.setStyleSheet("color:#666;")
        rv.addWidget(info)

        splitter.addWidget(right)
        splitter.setSizes([1000, 400])

        self.setStatusBar(QStatusBar())

        # menu
        mb = self.menuBar()
        f = mb.addMenu("&File")
        act_open = QAction("Open PDF…", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.open_pdf)
        f.addAction(act_open)
        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        f.addAction(act_quit)

    def _add_knob(self, layout, label, lo, hi, step, default, tip) -> QDoubleSpinBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setToolTip(tip)
        row.addWidget(lbl)
        row.addStretch()
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(3)
        sb.setValue(default)
        sb.setToolTip(tip)
        sb.setMinimumWidth(90)
        row.addWidget(sb)
        layout.addLayout(row)
        return sb

    def _reset_knobs(self):
        self.speed.setValue(1.0)
        self.noise_scale.setValue(0.667)
        self.noise_w.setValue(0.8)
        self.sentence_silence.setValue(0.2)

    # ---- voice discovery ----
    def _refresh_voice_list(self):
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        voices = Voices.list()
        for v in voices:
            self.voice_combo.addItem(v.name, str(v))
        self.voice_combo.blockSignals(False)
        if voices:
            self._on_voice_changed(0)

    def _on_voice_changed(self, idx: int):
        path = self.voice_combo.currentData()
        if not path:
            return
        self._invalidate_cache()  # different voice, different audio
        # Multi-speaker check via .onnx.json
        try:
            import json
            with open(path + ".json") as f:
                cfg = json.load(f)
            n = int(cfg.get("num_speakers", 1))
            if n > 1:
                self.speaker_lbl.setText(f"Speaker (0–{n-1})")
                self.speaker_spin.setMaximum(n - 1)
                self.speaker_spin.setEnabled(True)
            else:
                self.speaker_lbl.setText("Speaker: (single)")
                self.speaker_spin.setValue(0)
                self.speaker_spin.setMaximum(0)
                self.speaker_spin.setEnabled(False)
        except Exception:
            pass

    # ---- PDF ----
    def open_pdf(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open PDF", str(Path.home()), "PDF (*.pdf)")
        if not path:
            return
        self._stop()
        self.doc = fitz.open(path)
        self._extract_words_and_sentences()
        self.current_page = 0
        self.current_sentence = -1
        self.page_spin.blockSignals(True)
        self.page_spin.setMaximum(len(self.doc))
        self.page_spin.setValue(1)
        self.page_spin.blockSignals(False)
        self._render_current_page()
        self.setWindowTitle(f"Archangel — {Path(path).name}")

    def _extract_words_and_sentences(self):
        self.words = []
        chunks: list[str] = []
        char_ptr = 0
        for pi, page in enumerate(self.doc):
            page_words = page.get_text("words")  # x0 y0 x1 y1 w block line word
            page_words.sort(key=lambda t: (t[5], t[6], t[7]))
            for (x0, y0, x1, y1, w, blk, ln, wi) in page_words:
                # append text with separators; we track exact char_start of each word
                if chunks and not chunks[-1].endswith(("\n", " ")):
                    chunks.append(" ")
                    char_ptr += 1
                self.words.append(WordBox(
                    page=pi, x0=x0, y0=y0, x1=x1, y1=y1,
                    text=w, char_start=char_ptr))
                chunks.append(w)
                char_ptr += len(w)
            if pi < len(self.doc) - 1:
                chunks.append("\n\n")
                char_ptr += 2
        self.text = "".join(chunks)
        # sentence split — assign each sentence its word indices
        self.sentences = []
        # find sentence spans on the concatenated string
        sent_texts = SENTENCE_SPLIT.split(self.text)
        pos = 0
        for st in sent_texts:
            if not st.strip():
                pos = self.text.find(st, pos) + len(st)
                continue
            start = self.text.find(st, pos)
            end = start + len(st)
            pos = end
            # find word indices whose char_start ∈ [start, end)
            widx = [i for i, w in enumerate(self.words)
                    if start <= w.char_start < end]
            if widx:
                self.sentences.append(Sentence(text=st.strip(), word_indices=widx))
        self.statusBar().showMessage(
            f"{len(self.doc)} pages · {len(self.words)} words · {len(self.sentences)} sentences")

    def _render_current_page(self):
        if not self.doc:
            return
        page = self.doc[self.current_page]
        # scene wants (WordBox, global_idx) filtered to this page
        page_pairs = [(w, i) for i, w in enumerate(self.words) if w.page == self.current_page]
        self.scene.show_page(page, page_pairs)
        self._render_highlight()
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(self.current_page + 1)
        self.page_spin.blockSignals(False)

    def _render_highlight(self):
        if self.current_sentence < 0 or self.current_sentence >= len(self.sentences):
            self.scene.highlight_words(set())
            self.sentence_lbl.setText("–")
            return
        s = self.sentences[self.current_sentence]
        self.scene.highlight_words(set(s.word_indices))
        preview = s.text.replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "…"
        self.sentence_lbl.setText(preview)

    def _goto_page(self, page_index: int):
        if not self.doc:
            return
        page_index = max(0, min(len(self.doc) - 1, page_index))
        self.current_page = page_index
        self._render_current_page()

    def _on_zoom(self, z: float):
        self.scene.set_zoom(z)
        self._render_current_page()

    def _on_word_clicked(self, gi: int):
        # find sentence containing this word
        for si, s in enumerate(self.sentences):
            if gi in s.word_indices:
                self._go_to_sentence(si, autoplay=True)
                return

    # ---- playback ----
    def _toggle_play(self):
        if not self.sentences:
            return
        st = self.player.playbackState()
        if st == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_btn.setText("▶ Play")
        elif st == QMediaPlayer.PlaybackState.PausedState:
            self.player.play()
            self.play_btn.setText("⏸ Pause")
        else:
            # stopped: start at current sentence (or first)
            if self.current_sentence < 0:
                self.current_sentence = 0
                self._render_highlight()
                self._maybe_flip_page_to(0)
            self._play_current_sentence()

    def _stop(self):
        self.player.stop()
        self.play_btn.setText("▶ Play")

    def _jump_sentence(self, delta: int):
        if not self.sentences:
            return
        if self.current_sentence < 0:
            self.current_sentence = 0
        else:
            self.current_sentence = max(0, min(len(self.sentences) - 1,
                                                self.current_sentence + delta))
        self._go_to_sentence(self.current_sentence, autoplay=True)

    def _go_to_sentence(self, si: int, autoplay: bool):
        self.current_sentence = si
        self._maybe_flip_page_to(si)
        self._render_highlight()
        if autoplay:
            self._play_current_sentence()

    def _maybe_flip_page_to(self, si: int):
        s = self.sentences[si]
        # first word's page
        pg = self.words[s.word_indices[0]].page
        if pg != self.current_page:
            self.current_page = pg
            self._render_current_page()

    def _request_synth(self, index: int):
        """Ask the worker thread to synthesize a sentence. Non-blocking."""
        if index < 0 or index >= len(self.sentences):
            return
        if index in self._wav_cache or index in self._pending:
            return
        voice = self.voice_combo.currentData()
        if not voice:
            self.statusBar().showMessage("No voice model selected.")
            return
        self._pending.add(index)
        self.synth_requested.emit(
            index, self.sentences[index].text, voice,
            self.speed.value(), self.noise_scale.value(), self.noise_w.value(),
            self.sentence_silence.value(),
            self.speaker_spin.value() if self.speaker_spin.isEnabled() else None,
        )

    def _invalidate_cache(self):
        """Synthesis knobs changed — previously rendered audio is stale."""
        self._wav_cache.clear()

    def _on_synth_done(self, index: int, wav_path: str):
        self._pending.discard(index)
        self._wav_cache[index] = wav_path
        # If we're waiting on this one, play it now.
        if self._want_play and index == self.current_sentence:
            self._want_play = False
            self._start_wav(wav_path)
            self._request_synth(index + 1)  # prefetch next while this plays

    def _on_synth_failed(self, index: int, err: str):
        self._pending.discard(index)
        self.statusBar().showMessage(f"Synthesis failed on sentence {index}: {err}")
        if self._want_play and index == self.current_sentence:
            self._want_play = False
            self.play_btn.setText("▶ Play")

    def _start_wav(self, wav_path: str):
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(wav_path))
        self.player.play()
        self.play_btn.setText("⏸ Pause")

    def _play_current_sentence(self):
        if self.current_sentence < 0 or self.current_sentence >= len(self.sentences):
            return
        self._render_highlight()
        idx = self.current_sentence
        cached = self._wav_cache.get(idx)
        if cached:
            self._start_wav(cached)
            self._request_synth(idx + 1)  # prefetch next
        else:
            self._want_play = True
            self.play_btn.setText("… synthesizing")
            self._request_synth(idx)

    def _on_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self.current_sentence + 1 < len(self.sentences):
                self.current_sentence += 1
                self._maybe_flip_page_to(self.current_sentence)
                self._play_current_sentence()
            else:
                self.play_btn.setText("▶ Play")
                self.statusBar().showMessage("End of document.")

    def closeEvent(self, event):
        self.player.stop()
        self._thread.quit()
        self._thread.wait(3000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    # if a PDF was passed on the command line, open it
    if len(sys.argv) > 1 and sys.argv[1].endswith(".pdf"):
        w.open_pdf(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
