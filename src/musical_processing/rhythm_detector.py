import time
import queue
import numpy as np
import sounddevice as sd
import librosa


class RhythmDetector:
    def __init__(
        self,
        sr=44100,
        block_s=0.02,
        # Window for stored "tempo" values
        window_s=8.0,
        # How long until we update the queue of values
        update_s=0.5,
        n_fft=2048,
        hop_length=512,
        bpm_min=60,
        bpm_max=200,
        smooth_n=5,
    ):
        self.sr = sr
        self.blocksize = int(round(block_s * sr))
        self.window_len = int(round(window_s * sr))
        self.update_s = update_s

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max

        self.audio_buf = np.zeros(self.window_len, dtype=np.float32)
        self._last_update_t = 0.0
        self._bpm_hist = []
        self.smooth_n = smooth_n

    def push_audio(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = len(x)
        if n >= self.window_len:
            self.audio_buf[:] = x[-self.window_len:]
        else:
            self.audio_buf[:-n] = self.audio_buf[n:]
            self.audio_buf[-n:] = x

    def _onset_envelope(self, y: np.ndarray) -> np.ndarray:
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )
        # normalize to reduce level dependence
        onset_env = onset_env.astype(np.float32)
        onset_env -= onset_env.min()
        denom = max(onset_env.max(), 1e-6)
        onset_env /= denom
        return onset_env

    def _tempo_from_autocorr(self, onset_env: np.ndarray) -> float | None:
        if onset_env.size < 4 or np.all(onset_env == 0):
            return None

        # remove DC
        x = onset_env - np.mean(onset_env)

        ac = np.correlate(x, x, mode="full")[len(x)-1:]
        if ac[0] <= 0:
            return None
        
        # Lag is how much time you shift a signal to see if it lines up with itself again

        # Convert BPM range to lag range (in frames)
        fps = self.sr / self.hop_length
        lag_min = int(np.floor((60.0 / self.bpm_max) * fps))
        lag_max = int(np.ceil((60.0 / self.bpm_min) * fps))
        lag_min = max(lag_min, 1)
        lag_max = min(lag_max, len(ac) - 1)

        if lag_max <= lag_min:
            return None

        # Find best lag in range
        segment = ac[lag_min:lag_max+1]
        best_i = int(np.argmax(segment))
        best_lag = lag_min + best_i

        bpm = 60.0 * fps / best_lag
        return float(bpm)

    def update_bpm(self) -> float | None:
        now = time.time()
        if (now - self._last_update_t) < self.update_s:
            return None
        self._last_update_t = now

        onset_env = self._onset_envelope(self.audio_buf)
        bpm = self._tempo_from_autocorr(onset_env)
        if bpm is None:
            return None

        self._bpm_hist.append(bpm)
        if len(self._bpm_hist) > self.smooth_n:
            self._bpm_hist.pop(0)

        return float(np.median(self._bpm_hist))

    # Main BPM detection function
    def listen_bpm(self, max_seconds: float = 999):

        start_time = time.time()

        q = queue.Queue()

        def callback(indata, frames, time_info, status):
            """
            Used in both chord_detector and rhythm detector
            to pass new data into a queue if status is false

            Frames and time_info are required for the sd.InputStream
            callback method
            """
            if status:
                pass
            q.put(indata[:, 0].copy())

        print("Listening for BPM... (Ctrl+C to stop)")
        with sd.InputStream(
            channels=1,
            samplerate=self.sr,
            blocksize=self.blocksize,
            dtype="float32",
            callback=callback,
        ):
            last_print = None
            while (time.time() - start_time <= max_seconds):
                x = q.get()
                self.push_audio(x)
                bpm = self.update_bpm()
                if bpm is not None:
                    bpm_r = int(round(bpm))
                    if bpm_r != last_print:
                        print(f"BPM: {bpm:.1f}")
                        last_print = bpm_r

