# Configuration Guide

## Overview

The Emotion-Based Music Player uses a `config.json` file for all configurable parameters. This allows you to customize the behavior without modifying code.

## Configuration File Location

The configuration file should be placed in the project root directory:
```
Parasseri-Techies/
├── config.json          ← Configuration file
├── main.py
├── Modules/
└── Music/
```

## Configuration Sections

### 1. Emotion Detection

Controls how emotions are detected and validated:

```json
"emotion_detection": {
  "confidence_threshold": 0.6,
  "minimum_emotion_duration": 5.0,
  "smoothing_window_size": 15,
  "buffer_duration": 10
}
```

**Parameters:**

- **confidence_threshold** (0.0 - 1.0): Minimum confidence required to change emotions
  - Higher = more stable, less responsive (recommended: 0.6 - 0.8)
  - Lower = more responsive, may jump frequently (not recommended < 0.5)
  
- **minimum_emotion_duration** (seconds): How long an emotion must persist before changing
  - Higher = prevents rapid changes (recommended: 3 - 7 seconds)
  - Lower = more responsive to mood changes
  
- **smoothing_window_size** (frames): Number of recent detections to analyze
  - Larger = smoother but slower to respond (recommended: 10 - 20)
  - Smaller = faster response but less stable
  
- **buffer_duration** (seconds): How often to evaluate emotion changes
  - Recommended: 8 - 12 seconds

### 2. Camera Settings

Controls camera input and face detection:

```json
"camera": {
  "device_index": 0,
  "frame_width": 640,
  "frame_height": 480,
  "scale_factor": 1.3,
  "min_neighbors": 5
}
```

**Parameters:**

- **device_index**: Camera device number (0 = default camera, 1 = external camera)
- **frame_width/height**: Video resolution (lower = faster, higher = better quality)
- **scale_factor**: Face detection sensitivity (1.1 - 1.5)
  - Lower = more sensitive, may detect false faces
  - Higher = less sensitive, may miss faces
- **min_neighbors**: Minimum neighbors for face detection (3 - 7)
  - Lower = more detections, more false positives
  - Higher = fewer detections, more accurate

### 3. Audio Settings

Controls music playback:

```json
"audio": {
  "supported_formats": [".mp3", ".wav", ".ogg", ".flac"],
  "fade_duration_ms": 1000,
  "default_volume": 0.7
}
```

**Parameters:**

- **supported_formats**: Audio file extensions to load
- **fade_duration_ms**: Fade out duration when changing tracks (milliseconds)
- **default_volume** (0.0 - 1.0): Initial volume level

### 4. UI Settings

Controls the user interface:

```json
"ui": {
  "window_width": 512,
  "window_height": 900,
  "show_confidence": true,
  "theme": "dark"
}
```

**Parameters:**

- **window_width/height**: UI window size in pixels
- **show_confidence**: Show confidence percentage and progress bar
- **theme**: UI color theme (currently only "dark" supported)

### 5. Logging

Controls application logging:

```json
"logging": {
  "enabled": true,
  "level": "INFO",
  "log_to_file": true,
  "log_file": "emotion_music_player.log"
}
```

**Parameters:**

- **enabled**: Enable/disable logging
- **level**: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - DEBUG: Detailed information for debugging
  - INFO: General information (recommended)
  - WARNING: Only warnings and errors
- **log_to_file**: Save logs to file
- **log_file**: Log file path

## Tuning for Different Use Cases

### Stable, Calm Experience
```json
{
  "emotion_detection": {
    "confidence_threshold": 0.75,
    "minimum_emotion_duration": 7.0,
    "smoothing_window_size": 20
  }
}
```

### Responsive, Dynamic Experience
```json
{
  "emotion_detection": {
    "confidence_threshold": 0.55,
    "minimum_emotion_duration": 3.0,
    "smoothing_window_size": 10
  }
}
```

### Performance Mode (Lower-End Hardware)
```json
{
  "camera": {
    "frame_width": 480,
    "frame_height": 360,
    "scale_factor": 1.5
  }
}
```

## Troubleshooting

### Emotion Changes Too Frequently
- Increase `confidence_threshold` (try 0.7 or 0.75)
- Increase `minimum_emotion_duration` (try 6 or 7 seconds)
- Increase `smoothing_window_size` (try 18 or 20)

### Emotion Doesn't Change When It Should
- Decrease `confidence_threshold` (try 0.5 or 0.55)
- Decrease `minimum_emotion_duration` (try 3 or 4 seconds)
- Decrease `smoothing_window_size` (try 10 or 12)

### Camera Not Detected
- Try different `device_index` values (0, 1, 2, etc.)
- Check camera permissions in system settings

### Face Not Detected
- Decrease `scale_factor` (try 1.1 or 1.2)
- Decrease `min_neighbors` (try 3 or 4)
- Ensure good lighting
- Face camera directly

### Poor Performance
- Reduce `frame_width` and `frame_height`
- Increase `scale_factor`
- Close other applications

## Platform-Specific Notes

### Windows
All default settings should work well.

### macOS
- Camera permission required (System Preferences → Security & Privacy → Camera)
- May need to adjust `device_index` if using external camera

### Linux
- May need to install additional packages (see requirements.txt)
- Camera permissions may require user to be in `video` group

## Advanced: Runtime Configuration

You can modify configuration at runtime through the UI settings panel (coming in future update) or by editing `config.json` and restarting the application.
