from collections import deque
import time


DEFAULT_WINDOW_SECONDS = 600 # 10 minutes

class SoundCache:
    """
    Caches snapshots of sound
    from a given moment to reference when analyzing/visualizing
    
    SCHEMA:
        chord: str = "none"
        frequency: float = 0
        bpm: float = 0
        timestamp: float = time.time()
    """

    def __init__(self, window_seconds: int = DEFAULT_WINDOW_SECONDS):
        self.data = deque()
        self.window_seconds = window_seconds

    def _prune(self, now):
        cutoff = now - self.window_seconds
        while self.data and self.data[0]["timestamp"] < cutoff:
            self.data.popleft()

    def add(self, chord: str = "none", frequency: float = 0, bpm: float = 0):
        now = time.time()

        chord_entry = {
            "chord": chord,
            "frequency": frequency,
            "bpm": bpm,
            "timestamp": now
        }

        self.data.append(chord_entry)
        self._prune(now)

    # ---------------------------
    # Query Methods
    # ---------------------------

    def chord_count(self, chord: str) -> int:
        """
        Count how many times a given chord appears
        within the active time window.
        """
        now = time.time()
        self._prune(now)
        return sum(1 for entry in self.data if entry["chord"] == chord)

    def max_frequency(self) -> dict | None:
        """
        Return the entry with the highest frequency
        within the active window
        Returns None if empty
        """
        now = time.time()
        self._prune(now)

        if not self.data:
            return None

        return max(self.data, key=lambda x: x["frequency"])

    def top_freqencies(self, n: int) -> list[dict]:
        """
        Return top n entries sorted by frequency (descending)
        """
        now = time.time()
        self._prune(now)

        return sorted(self.data, key=lambda x: x["frequency"], reverse=True)[:n]

    def since(self, seconds: float) -> list[dict] | None:
        """
        Return entries from the last `seconds`
        Returns None if seconds is greater or equal to window_seconds
        """
        now = time.time()
        self._prune(now)

        if seconds >= self.window_seconds:
            return None

        cutoff = now - seconds
        return [entry for entry in self.data if entry["timestamp"] >= cutoff]

    def last_n(self, n: int) -> list[dict]:
        """
        Return the last n entries added (most recent first)
        """
        now = time.time()
        self._prune(now)

        return list(self.data)[-n:]


    def __str__(self) -> str:
        """
        Pretty-print summary of the cache for quick console inspection
        """
        now = time.time()
        self._prune(now)

        size = len(self.data)
        if size == 0:
            return f"SoundCache(window={self.window_seconds}s) → EMPTY"

        oldest = self.data[0]["timestamp"]
        newest = self.data[-1]["timestamp"]
        span = newest - oldest

        # Basic stats
        chords = {}
        max_freq = 0.0
        for entry in self.data:
            chord = entry["chord"]
            chords[chord] = chords.get(chord, 0) + 1
            if entry["frequency"] > max_freq:
                max_freq = entry["frequency"]

        top_chords = sorted(chords.items(), key=lambda x: x[1], reverse=True)[:3]

        summary_lines = [
            f"SoundCache(window={self.window_seconds}s)",
            f"Entries: {size}",
            f"Time span: {span:.2f}s",
            f"Max frequency: {max_freq:.2f} Hz",
            f"Top chords: {top_chords}",
            "Last 5 entries:"
        ]

        for entry in list(self.data)[-5:]:
            summary_lines.append(
                f"  {entry['chord']:6} | "
                f"{entry['frequency']:7.2f} Hz | "
                f"{entry['bpm']:6.1f} BPM"
            )

        return "\n".join(summary_lines)
