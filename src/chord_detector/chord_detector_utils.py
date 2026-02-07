from __future__ import annotations

import numpy as np
import librosa


class ChordDetector:
    """
    Streaming chord detector.

    Feed it audio blocks with .process_block(x) and it will occasionally return:
      - a chord name (e.g. "Am", "C#") when stable
      - None if nothing new to report

    Example:
        det = ChordDetector(sr=44100, chord_window_s=1.5, chord_hz=10, smoothing=2)
        chord = det.process_block(x)  # x is 1D float32 audio block
        if chord is not None:
            print(chord)
    """

    def __init__(
        self,
        sr: int = 44100,
        update_s: float = 0.02,
        chord_window_s: float = 1.5,
        chord_hz: float = 10.0,
        smoothing: int = 2,
        # Major, minor, or all (preferred)
        chord_type: str = "all",
        silence_thresh: float = 0.01,
    ):
        self.sr = int(sr)
        self.update_s = float(update_s)

        self.chord_window_s = float(chord_window_s)
        self.chord_hz = float(chord_hz)
        self.smoothing = int(smoothing)
        self.chord_type = chord_type
        self.silence_thresh = float(silence_thresh)

        # How many update calls between detections
        self.every_n_updates = max(1, int(round((1.0 / self.chord_hz) / self.update_s)))

        # Rolling buffer of audio to analyze
        self.buf_len = int(round(self.chord_window_s * self.sr))
        self.audio_buf = np.zeros(self.buf_len, dtype=np.float32)

        # Templates
        self.templates, self.chord_names = self._get_chord_templates(self.chord_type)

        # State for debouncing / stability
        self.update_count = 0
        self.last_reported = None
        self.last_candidate = None
        self.candidate_count = 0

    def reset(self):
        self.audio_buf[:] = 0
        self.update_count = 0
        self.last_reported = None
        self.last_candidate = None
        self.candidate_count = 0

    def process_block(self, x: np.ndarray) -> str | None:
        """
        Push one block of audio (1D float array). Returns:
          - chord string when a *new stable chord* is detected
          - None otherwise
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        # update rolling buffer
        n = len(x)
        if n >= self.buf_len:
            self.audio_buf[:] = x[-self.buf_len :]
        else:
            self.audio_buf[:-n] = self.audio_buf[n:]
            self.audio_buf[-n:] = x

        # run detection occasionally
        self.update_count += 1
        if self.update_count % self.every_n_updates != 0:
            return None

        chord = self._recognize_chord(self.audio_buf)

        if chord == self.last_candidate:
            self.candidate_count += 1
        else:
            self.last_candidate = chord
            self.candidate_count = 1

        if self.candidate_count >= self.smoothing and chord != self.last_reported:
            self.last_reported = chord
            return chord

        return None

    def _recognize_chord(self, y: np.ndarray) -> str:
        if np.max(np.abs(y)) < self.silence_thresh:
            return "silence"

        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=self.sr)
        mean_chroma = self._normalize_chroma(np.mean(chroma_cq, axis=1))
        return self._get_match(self.templates, mean_chroma, self.chord_names)

    @staticmethod
    def _normalize_chroma(chroma: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(chroma)
        return chroma / np.maximum(n, 1e-9)

    @staticmethod
    def _get_match(templates: np.ndarray, chroma_vector: np.ndarray, chords: list[str]) -> str:
        sims = templates @ chroma_vector
        return chords[int(np.argmax(sims))]

    # ---------- Templates ----------
    @staticmethod
    def _get_major_templates():
        chords = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        arr = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # C
                [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # C#
                [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # D
                [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # D#
                [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],  # E
                [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],  # F
                [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],  # F#
                [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],  # G
                [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],  # G#
                [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],  # A
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],  # A#
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1],  # B
            ],
            dtype=np.float32,
        )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9), chords

    @staticmethod
    def _get_minor_templates():
        chords = ["Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm"]
        arr = np.array(
            [
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # Cm
                [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  # C#m
                [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # Dm
                [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # D#m
                [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # Em
                [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],  # Fm
                [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],  # F#m
                [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],  # Gm
                [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],  # G#m
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # Am
                [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # A#m
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1],  # Bm
            ],
            dtype=np.float32,
        )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9), chords

    @classmethod
    def _get_chord_templates(cls, chord_type: str = "all"):
        chord_type = (chord_type or "all").lower()
        if chord_type == "major":
            return cls._get_major_templates()
        if chord_type == "minor":
            return cls._get_minor_templates()
        maj, maj_names = cls._get_major_templates()
        min_, min_names = cls._get_minor_templates()
        return np.concatenate([maj, min_], axis=0), (maj_names + min_names)
