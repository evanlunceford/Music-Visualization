THIS TOOL IS IN DEVELOPMENT

This tool is aiming to allow for visualization of audio. Similar to Apple's old iTunes feature, the goal is to create live visualizations based on the
sound being inputted. This can be in the form of music, conversation, or just static sound around your room.



Backend is mainly object-oriented in order to increase modularity based on what you want to use this tool for.

## Audio Device Configuration

By default the app uses your system's default microphone. To capture system audio (what your speakers are playing) or pick a specific device, you have a few options.

### Quick Start

List available input devices:
```
python src/main.py --list-devices
```

Interactively pick a device:
```
python src/main.py --pick-device
```

### Config File

Copy the template and edit it:
```
cp config.example.json config.json
```

`config.json` supports three modes:

| Mode | Description |
|------|-------------|
| `"mic"` | Default microphone (no config needed) |
| `"system"` | Auto-finds a loopback device like Stereo Mix for capturing desktop audio |
| `"device"` | Match a specific device by name/backend substring |

Example — capture system audio:
```json
{
  "audio": {
    "mode": "system",
    "device_match": { "name": "", "backend": "" },
    "sample_rate": 44100
  }
}
```

Example — match a specific device:
```json
{
  "audio": {
    "mode": "device",
    "device_match": { "name": "Stereo Mix", "backend": "WDM-KS" },
    "sample_rate": 44100
  }
}
```

### Environment Variable Override

For quick one-off runs without editing config:
```
AUDIO_DEVICE="Stereo Mix" python src/main.py
AUDIO_MODE=mic python src/main.py
```

### Enabling Stereo Mix on Windows

If `"system"` mode reports no loopback device found:
1. Open **Sound Settings** > **Recording** tab
2. Right-click > **Show Disabled Devices**
3. Enable **Stereo Mix**
4. (Optional) Set it as default recording device