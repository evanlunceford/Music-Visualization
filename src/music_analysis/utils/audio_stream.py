import queue
import sounddevice as sd


class AudioStream:
    """
    A utility class to handle audio blocks

    Usage:
        with AudioStream(sr=44100, block_s=0.02) as stream:
            for audio_block in stream.blocks(timeout=0.1):
                # process audio_block (numpy array)
                pass
    """

    def __init__(self, sr: int = 44100, block_s: float = 0.02, device: int | None = None):
        self.sr = sr
        self.block_s = block_s
        self.blocksize = max(1, int(round(block_s * sr)))
        self.device = device
        self._queue = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if not status:
            self._queue.put(indata[:, 0].copy())

    def __enter__(self):
        self._stream = sd.InputStream(
            channels=1,
            samplerate=self.sr,
            blocksize=self.blocksize,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return False

    def get_block(self, timeout: float = 0.1):
        """
        Get a single audio block from the queue.
        Returns None if timeout is reached.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def blocks(self, timeout: float = 0.1):
        """
        Generator that yields audio blocks indefinitely.
        Yields None on timeout (caller should handle).
        """
        while True:
            yield self.get_block(timeout=timeout)
