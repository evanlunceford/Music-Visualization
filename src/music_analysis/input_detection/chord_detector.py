import time
import numpy as np
import librosa


class ChordDetector:
    def __init__(
        self,
        sr: int = 44100,
        update_s: float = 0.02,
        chord_window_s: float = 1.5,
        chord_hz: float = 10.0,
        smoothing: int = 2,
        # "maj", "min", "all"
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

        self.every_n_updates = max(1, int(round((1.0 / self.chord_hz) / self.update_s)))

        self.buf_len = int(round(self.chord_window_s * self.sr))
        self.audio_buf = np.zeros(self.buf_len, dtype=np.float32)

        self.templates, self.chord_names = self._build_chord_templates(self.chord_type)

        self.key_templates, self.key_names = self._build_key_templates()

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
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        n = len(x)
        if n >= self.buf_len:
            self.audio_buf[:] = x[-self.buf_len :]
        else:
            self.audio_buf[:-n] = self.audio_buf[n:]
            self.audio_buf[-n:] = x

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

    def live_chords(self, max_seconds: float = 999, device: int | None = None):
        from music_analysis.utils.audio_stream import AudioStream

        start_time = time.time()

        blocksize = max(1, int(round(self.update_s * self.sr)))
        effective_update_s = blocksize / self.sr
        self.every_n_updates = max(1, int(round((1.0 / self.chord_hz) / effective_update_s)))

        key_energy = np.zeros(12, dtype=np.float32)
        decay = 0.92

        last_chord = None
        last_key = None
        last_key_print_t = 0.0

        print("Listening... (Ctrl+C to stop)")
        with AudioStream(sr=self.sr, block_s=self.update_s, device=device) as stream:
            while (time.time() - start_time <= max_seconds):
                x = stream.get_block(timeout=0.1)
                if x is None:
                    continue
                chord = self.process_block(x)
                if chord is None or chord == last_chord:
                    continue

                print(chord)
                last_chord = chord

                vec = self._chord_to_vec(chord)
                if vec is None:
                    continue

                key_energy *= decay
                key_energy += vec

                key_name, conf = self.detect_key(key_energy)

                now = time.time()
                if (key_name != last_key) and (now - last_key_print_t > 0.5):
                    print(f"-> Key: {key_name}  (conf={conf:.3f})")
                    last_key = key_name
                    last_key_print_t = now

    def detect_key(self, key_energy: np.ndarray) -> tuple[str, float]:
        if key_energy is None or np.all(key_energy == 0):
            return ("unknown", 0.0)

        v = key_energy.astype(np.float32)
        v /= np.maximum(np.linalg.norm(v), 1e-9)

        scores = self.key_templates @ v
        best = int(np.argmax(scores))

        s_sorted = np.sort(scores)
        conf = float(s_sorted[-1] - s_sorted[-2]) if len(s_sorted) >= 2 else 0.0

        return (self.key_names[best], conf)

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

    @staticmethod
    def _build_chord_templates(chord_type: str):
        """
        Returns:
          templates: (N, 12) normalized
          chord_names: list[str]
        Major triad: [0,4,7]
        Minor triad: [0,3,7]
        """
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

        def triad(root_idx: int, intervals):
            v = np.zeros(12, dtype=np.float32)
            for i in intervals:
                v[(root_idx + i) % 12] = 1.0
            v /= np.maximum(np.linalg.norm(v), 1e-9)
            return v

        templates = []
        names = []

        chord_type = chord_type.lower().strip()
        use_maj = chord_type in ("all", "maj", "major")
        use_min = chord_type in ("all", "min", "minor")

        if not (use_maj or use_min):
            use_maj = use_min = True

        for i, root in enumerate(note_names):
            if use_maj:
                templates.append(triad(i, (0, 4, 7)))
                names.append(root)
            if use_min:
                templates.append(triad(i, (0, 3, 7)))
                names.append(f"{root}m")

        T = np.stack(templates, axis=0).astype(np.float32)
        return T, names

    # ---------- Key templates ----------
    @staticmethod
    def _major_key_template():
        v = np.zeros(12, dtype=np.float32)
        v[[0, 2, 4, 5, 7, 9, 11]] = 1.0
        return v / np.linalg.norm(v)

    @staticmethod
    def _minor_key_template():
        v = np.zeros(12, dtype=np.float32)
        v[[0, 2, 3, 5, 7, 8, 10]] = 1.0
        return v / np.linalg.norm(v)

    @staticmethod
    def _rotate(v: np.ndarray, k: int) -> np.ndarray:
        return np.roll(v, k)

    @classmethod
    def _build_key_templates(cls):
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        maj0 = cls._major_key_template()
        min0 = cls._minor_key_template()

        templates = []
        names = []
        for i, root in enumerate(note_names):
            templates.append(cls._rotate(maj0, i))
            names.append(f"{root} major")
        for i, root in enumerate(note_names):
            templates.append(cls._rotate(min0, i))
            names.append(f"{root} minor")

        T = np.stack(templates, axis=0).astype(np.float32)
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        return T, names

    def _chord_to_vec(self, chord: str) -> np.ndarray | None:
        if chord == "silence":
            return None
        try:
            idx = self.chord_names.index(chord)
        except ValueError:
            return None
        return self.templates[idx]
