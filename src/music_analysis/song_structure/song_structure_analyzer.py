from collections import Counter
import math

from cache.sound_cache import SoundCache

class SongStructureAnalyzer:
    """
    Provides several ways to analyze
    a cache of chords to derive conclusions
    about the structure of a song
    
    Ex: Detecting a change from a chorus to a verse
    """
    def __init__(self, cache: SoundCache, bin_seconds: float = 2.0):
        self.cache = cache
        self.bin_seconds = bin_seconds

    def analyze_structure(self, lookback_seconds: float = 300) -> list[dict]:
        """
        Looks for repeated parts of the last [lookback_seconds]
        seconds of a SoundCache

        Returns a list of detected "parts" of a song.        
        """
        entries = self.cache.since(lookback_seconds)
        if not entries:
            # Nothing to analyze
            return []

        # Split entries into bins
        bins = self._bin_entries(entries)

        # Find the features for each bin
        features = [self._features_for_bin(b) for b in bins]

        # Look for boundaries (sections of the song)
        boundaries = self._boundaries(features)

        # See if we can find a chorus
        labels = self._label_sections(bins, features, boundaries)

        return labels

    
    def _bin_entries(self, entries: list[dict]) -> list[list[dict]]:
        
        start = entries[0]["timestamp"]
        bs = self.bin_seconds
        bins = []
        cur = []
        cur_start = start

        for e in entries:
            if e["timestamp"] < cur_start + bs:
                cur.append(e)
            else:
                bins.append(cur)
                cur = [e]
                # Advance cur_start to the bin that contains e (exclusive)
                cur_start = start + math.floor((e["timestamp"] - start) / bs) * bs
        # Append anything that's left if it exists
        if cur:
            bins.append(cur)
        return bins

    def _features_for_bin(self, bin_entries: list[dict]):
        chords = [e["chord"] for e in bin_entries if e["chord"] and e["chord"] != "none"]
        bpms = [e["bpm"] for e in bin_entries if e["bpm"] and e["bpm"] > 0]

        chord_counts = Counter(chords)
        # Finding the dominate (or mode for the snobby music theorists) for this bin
        dominant = chord_counts.most_common(1)[0][0] if chord_counts else "none"

        # Calculating chord change rate within the bin
        changes = 0
        last = None
        for c in chords:
            if last is not None and c != last:
                changes += 1
            last = c
        change_rate = changes / max(1, len(chords) - 1) if chords else 0.0

        bpm_mean = sum(bpms) / len(bpms) if bpms else 0.0
        bpm_var = (sum((b - bpm_mean) ** 2 for b in bpms) / len(bpms)) if bpms else 0.0

        return {
            "dominant_chord": dominant,
            "chord_hist": chord_counts,
            "change_rate": change_rate,
            "bpm_mean": bpm_mean,
            "bpm_var": bpm_var,
        }

    def _boundaries(self, features, chord_weight=1.0, bpm_weight=0.3, rate_weight=0.5):
        # Using Jaccard over the chord sets
        def jaccard(a: Counter, b: Counter):
            sa, sb = set(a.keys()), set(b.keys())
            if not sa and not sb:
                return 0.0
            return 1.0 - (len(sa & sb) / len(sa | sb))

        diffs = [0.0]
        for i in range(1, len(features)):
            d_chord = jaccard(features[i-1]["chord_hist"], features[i]["chord_hist"])
            d_bpm = abs(features[i-1]["bpm_mean"] - features[i]["bpm_mean"]) / 200.0
            d_rate = abs(features[i-1]["change_rate"] - features[i]["change_rate"])
            diffs.append(chord_weight*d_chord + bpm_weight*d_bpm + rate_weight*d_rate)

        # Pick peaks as boundaries (simple threshold: mean + 1 std. dev.)
        mean = sum(diffs) / len(diffs)
        std = math.sqrt(sum((x-mean)**2 for x in diffs) / len(diffs))
        thresh = mean + std

        boundaries = [0]
        for i, d in enumerate(diffs):
            if i != 0 and d >= thresh:
                boundaries.append(i)
        boundaries.append(len(features))
        boundaries = sorted(set(boundaries))
        return boundaries

    def _label_sections(self, bins, features, boundaries):
        # Basic heuristic labeling:
        # - Find repeated dominant-chord sequences across segments; the most repeated becomes "chorus"
        segments = []
        for a, b in zip(boundaries, boundaries[1:]):
            dom_seq = [features[i]["dominant_chord"] for i in range(a, b)]
            start_ts = bins[a][0]["timestamp"]
            end_ts = bins[b-1][-1]["timestamp"]
            segments.append({"start": start_ts, "end": end_ts, "dom_seq": dom_seq})

        # Fingerprint each segment by collapsing repeats: 
        # Ex: ["C","C","G","G","Am"] -> ["C","G","Am"]
        def collapse(seq):
            out = []
            last = None
            for x in seq:
                if x != last and x != "none":
                    out.append(x)
                last = x
            return tuple(out)

        fps = [collapse(s["dom_seq"]) for s in segments]
        counts = Counter(fps)
        # Time to choose a chorus candidate to nominate! 
        # Criteria is if it appears >=2 times and has decent length
        chorus_fp = None
        for fp, ct in counts.most_common():
            if ct >= 2 and len(fp) >= 3:
                chorus_fp = fp
                break

        labeled = []
        for s, fp in zip(segments, fps):
            label = "section"
            if chorus_fp and fp == chorus_fp:
                label = "chorus"
            labeled.append({**s, "label": label})
        # Optionally label in-between parts as verse/bridge based on position
        for i in range(len(labeled)):
            if labeled[i]["label"] == "section":
                labeled[i]["label"] = "verse" if (chorus_fp is None or i < 2) else "bridge"
        return labeled


