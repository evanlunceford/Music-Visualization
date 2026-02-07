from __future__ import annotations

import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


class LiveSpectrogram:
    """
    Uses connected microphone as input for the spectogram
    Device parameter can modify the input device

    Usage:
        from live_spectrogram import LiveSpectrogram
        LiveSpectrogram().run()   # blocks until window closes
    """

    def __init__(
        self,
        sr: int = 44100,
        nfft: int = 2048,
        update_s: float = 0.02,
        seconds_visible: float = 20.0,
        fmax: float | None = None,
        levels: tuple[float, float] = (-120.0, 0.0),
        title: str = "Rolling Spectrogram (Mic)",
        window_title: str = "Live Spectrogram",
        device: int | str | None = None,
    ):
        self.SR = sr
        self.NFFT = nfft
        self.UPDATE_S = float(update_s)
        self.SECONDS_VISIBLE = float(seconds_visible)
        self.FMAX = (sr / 2) if fmax is None else float(fmax)

        self.BLOCK = int(self.SR * self.UPDATE_S)
        self.WINDOW = np.hanning(self.NFFT).astype(np.float32)

        # FFT bins
        freqs = np.fft.rfftfreq(self.NFFT, d=1 / self.SR)
        max_bin = int(np.searchsorted(freqs, self.FMAX, side="right") - 1)
        max_bin = max(1, min(max_bin, len(freqs) - 1))
        self.BINS = max_bin + 1
        self.freq_max_shown = float(freqs[max_bin])

        # spectrogram buffer: (freq_bins, time_columns)
        self.COLUMNS = int(self.SECONDS_VISIBLE / self.UPDATE_S)
        self.spec = np.full((self.BINS, self.COLUMNS), levels[0], dtype=np.float32)

        self.levels = levels
        self.title = title
        self.window_title = window_title
        self.device = device

        # runtime objects
        self.app = None
        self.win = None
        self.plot = None
        self.img = None
        self.timer = None
        self.stream = None

    @staticmethod
    def to_db(mags: np.ndarray) -> np.ndarray:
        return 20.0 * np.log10(np.maximum(mags, 1e-10))

    def _build_ui(self):
        self.app = pg.mkQApp(self.window_title)

        self.win = pg.GraphicsLayoutWidget(show=True, title=self.title)
        self.plot = self.win.addPlot(title=self.title)
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setLabel("left", "Frequency (Hz)")
        self.plot.setXRange(-self.SECONDS_VISIBLE, 0)
        self.plot.setYRange(0, self.freq_max_shown)

        self.img = pg.ImageItem()
        self.plot.addItem(self.img)

        # Map image coords to real units
        self.img.setRect(
            QtCore.QRectF(-self.SECONDS_VISIBLE, 0, self.SECONDS_VISIBLE, self.freq_max_shown)
        )
        self.img.setLevels(self.levels)

        # Colormap
        try:
            cmap = pg.colormap.get("inferno")
            self.img.setLookupTable(cmap.getLookupTable())
        except Exception:
            pass

        # Cleanup on exit
        self.app.aboutToQuit.connect(self.stop)

    def _start_audio(self):
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.SR,
            blocksize=self.BLOCK,
            dtype="float32",
            device=self.device,
        )
        self.stream.start()

    def _tick(self):
        """Timer callback: read audio, compute FFT column, roll spectrogram, update image."""
        if self.stream is None or self.img is None:
            return

        audio, _ = self.stream.read(self.BLOCK)
        x = audio[:, 0].astype(np.float32)

        # pad/trim to NFFT
        if len(x) < self.NFFT:
            x_fft = np.pad(x, (0, self.NFFT - len(x)))
        else:
            x_fft = x[: self.NFFT]

        X = np.fft.rfft(x_fft * self.WINDOW)
        db_col = self.to_db(np.abs(X)).astype(np.float32)[: self.BINS]

        # shift left, append newest
        self.spec[:, :-1] = self.spec[:, 1:]
        self.spec[:, -1] = db_col

        # display expects (time, freq) so transpose
        self.img.setImage(self.spec.T, autoLevels=False)

    def start(self):
        """Start window + audio + timer (non-blocking)."""
        if self.app is None:
            self._build_ui()

        if self.stream is None:
            self._start_audio()

        if self.timer is None:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self._tick)
            self.timer.start(int(self.UPDATE_S * 1000))

    def stop(self):
        """Stop timer and audio stream safely."""
        try:
            if self.timer is not None:
                self.timer.stop()
        except Exception:
            pass
        finally:
            self.timer = None

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None

    def run(self):
        """Start and block until the window closes."""
        self.start()
        self.app.exec()


def run_live_spectrogram(**kwargs):
    """Convenience function."""
    LiveSpectrogram(**kwargs).run()


if __name__ == "__main__":
    run_live_spectrogram()
