import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

SR = 44100
NFFT = 2048
UPDATE_S = 0.02
SECONDS_VISIBLE = 20
FMAX = SR / 2

BLOCK = int(SR * UPDATE_S)
WINDOW = np.hanning(NFFT).astype(np.float32)

# For chord detection
CHORD_WINDOW_S = 1.5
CHORD_HZ = 10 
CHORD_EVERY_N_UPDATES = max(1, int(round((1.0 / CHORD_HZ) / UPDATE_S)))
CHORD_SMOOTHING = 2 

# FFT frequency bins
freqs = np.fft.rfftfreq(NFFT, d=1 / SR)
max_bin = int(np.searchsorted(freqs, FMAX, side="right") - 1)
max_bin = max(1, min(max_bin, len(freqs) - 1))
freq_max_shown = float(freqs[max_bin])

BINS = max_bin + 1
COLUMNS = int(SECONDS_VISIBLE / UPDATE_S)

# Spectrogram buffer
spec = np.full((BINS, COLUMNS), -120.0, dtype=np.float32)

def to_db(mags: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(mags, 1e-10))

app = pg.mkQApp("Live Spectrogram")
win = pg.GraphicsLayoutWidget(show=True, title="Rolling Spectrogram")
plot = win.addPlot(title="Rolling Spectrogram (Mic)")
plot.setLabel("bottom", "Time (s)")
plot.setLabel("left", "Frequency (Hz)")
plot.setXRange(-SECONDS_VISIBLE, 0)
plot.setYRange(0, freq_max_shown)

img = pg.ImageItem()
plot.addItem(img)
img.setRect(QtCore.QRectF(-SECONDS_VISIBLE, 0, SECONDS_VISIBLE, freq_max_shown))
img.setLevels((-120, 0))
try:
    cmap = pg.colormap.get("inferno")
    img.setLookupTable(cmap.getLookupTable())
except Exception:
    pass

# ---------- Audio stream ----------
stream = sd.InputStream(
    channels=1,
    samplerate=SR,
    blocksize=BLOCK,
    dtype="float32",
)
stream.start()

def update():
    global spec

    audio, _ = stream.read(BLOCK)
    x = audio[:, 0].astype(np.float32)

    x_fft = x
    if len(x_fft) < NFFT:
        x_fft = np.pad(x_fft, (0, NFFT - len(x_fft)))
    else:
        x_fft = x_fft[:NFFT]

    X = np.fft.rfft(x_fft * WINDOW)
    db_col = to_db(np.abs(X)).astype(np.float32)[:BINS]

    spec[:, :-1] = spec[:, 1:]
    spec[:, -1] = db_col
    img.setImage(spec.T, autoLevels=False)


# Acts as the main function execution
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(UPDATE_S * 1000))

def cleanup():
    try: timer.stop()
    except Exception: pass
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass

app.aboutToQuit.connect(cleanup)
app.exec()
cleanup()
