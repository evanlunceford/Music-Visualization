import numpy as np
import sounddevice as sd

DEFAULT_UPDATE_TIME_SECONDS = 0.05
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_NFFT = 4096

def spectrum_from_mic(
    duration_s: float = DEFAULT_UPDATE_TIME_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    nfft: int = DEFAULT_NFFT,
    device=None
):
    """
    Generator yielding (freqs_hz, magnitudes) repeatedly.
    freqs_hz: shape (nfft//2 + 1,)
    magnitudes: linear amplitude (not dB), same shape
    """
    blocksize = int(duration_s * sample_rate)
    window = np.hanning(nfft).astype(np.float32)

    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=blocksize,
        device=device,
        dtype="float32",
    ) as stream:
        while True:
            audio, _ = stream.read(blocksize)
            x = audio[:, 0]

            if len(x) < nfft:
                x = np.pad(x, (0, nfft - len(x)))
            else:
                x = x[:nfft]

            xw = x * window
            fft = np.fft.rfft(xw)
            mags = np.abs(fft)
            freqs = np.fft.rfftfreq(nfft, d=1 / sample_rate)

            # Hz, amplitude
            yield freqs, mags

if __name__ == "__main__":
    gen = spectrum_from_mic()
    # Will run for a few seconds, printing out the highest frequency
    # and related magnitude
    for _ in range(100):
        freqs, mags = next(gen)
        peak_i = int(np.argmax(mags))
        print(f"Peak bin: {freqs[peak_i]:8.1f} Hz  magnitude={mags[peak_i]:.3f}")
