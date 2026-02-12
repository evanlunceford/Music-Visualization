import json
import os
import sys
import sounddevice as sd
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.json"

_DEFAULT_CONFIG = {
    "audio": {
        "mode": "mic",
        "device_match": {"name": "", "backend": ""},
        "sample_rate": 44100,
    }
}


def load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return _DEFAULT_CONFIG


def _get_host_api_name(hostapi_id: int) -> str:
    return sd.query_hostapis(hostapi_id)["name"]


def _find_device_by_match(name_sub: str, backend_sub: str = "") -> int | None:
    """Find first input device matching name/backend substrings (case-insensitive)."""
    devices = sd.query_devices()
    name_lower = name_sub.lower()
    backend_lower = backend_sub.lower()

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        if name_lower and name_lower not in dev["name"].lower():
            continue
        if backend_lower:
            api_name = _get_host_api_name(dev["hostapi"])
            if backend_lower not in api_name.lower():
                continue
        return i
    return None


def _find_system_loopback() -> int | None:
    """Try to find a system audio loopback device (Stereo Mix, etc.)."""
    loopback_names = ["stereo mix", "loopback", "what u hear", "wave out"]
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        name_lower = dev["name"].lower()
        for pattern in loopback_names:
            if pattern in name_lower:
                return i
    return None


def resolve_device() -> int | None:
    """
    Resolve the audio input device index from config + env vars.

    Returns an int device index, or None for the sounddevice default.
    Priority: AUDIO_DEVICE env var > AUDIO_MODE env var > config.json > default (mic).
    """
    config = load_config()
    audio_cfg = config.get("audio", {})

    mode = os.environ.get("AUDIO_MODE", audio_cfg.get("mode", "mic")).lower().strip()
    env_device = os.environ.get("AUDIO_DEVICE", "").strip()

    if env_device:
        result = _find_device_by_match(env_device)
        if result is not None:
            return result
        print(
            f"WARNING: AUDIO_DEVICE='{env_device}' matched no input device. "
            f"Falling back to mode='{mode}'.",
            file=sys.stderr,
        )

    if mode == "mic":
        return None

    if mode == "system":
        dev = _find_system_loopback()
        if dev is not None:
            return dev
        print(
            "WARNING: No system loopback device found. Falling back to default mic.",
            file=sys.stderr,
        )
        return None

    if mode == "device":
        match_cfg = audio_cfg.get("device_match", {})
        name = match_cfg.get("name", "")
        backend = match_cfg.get("backend", "")
        if not name:
            print(
                "WARNING: mode='device' but device_match.name is empty. Using default.",
                file=sys.stderr,
            )
            return None
        dev = _find_device_by_match(name, backend)
        if dev is not None:
            return dev
        print(
            f"WARNING: No device matched name='{name}', backend='{backend}'. "
            f"Falling back to default.",
            file=sys.stderr,
        )
        return None

    print(f"WARNING: Unknown audio mode '{mode}'. Using default.", file=sys.stderr)
    return None


def list_devices() -> str:
    """Return a formatted string of all audio input devices."""
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    lines = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        api_name = _get_host_api_name(dev["hostapi"])
        marker = " [DEFAULT]" if i == default_in else ""
        lines.append(
            f"  {i:>3}  {dev['name']:<55} {api_name:<25} "
            f"(in={dev['max_input_channels']}){marker}"
        )
    header = f"{'idx':>5}  {'Device Name':<55} {'Backend':<25} Channels"
    return header + "\n" + "\n".join(lines)


def pick_device_interactive() -> int:
    """Interactive terminal device selector. Returns chosen device index."""
    print("\nAvailable INPUT devices:\n")
    print(list_devices())
    print()

    devices = sd.query_devices()
    input_indices = [i for i, d in enumerate(devices) if d["max_input_channels"] >= 1]

    while True:
        choice = input("Enter device index (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        try:
            idx = int(choice)
            if idx in input_indices:
                dev = devices[idx]
                print(f"Selected: {dev['name']}")
                return idx
            print(f"Index {idx} is not a valid input device. Try again.")
        except ValueError:
            print("Please enter a number.")
