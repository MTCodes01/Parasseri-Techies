# Music Player Using Emotion Detection

## Project Description

The **Emotion-Detecting Music Player** is an intelligent application that uses facial emotion recognition to create a personalized music experience. By detecting the user's emotional state through a camera, the player automatically selects and plays music that matches or enhances the detected mood.

### ✨ New Features (v2.0)

- **🎯 Emotion Stability System**: Prevents rapid emotion jumping with confidence-based validation
- **⚙️ Configurable Settings**: Adjust sensitivity and behavior via `config.json`
- **🎵 Multi-Format Support**: Plays MP3, WAV, OGG, and FLAC files
- **📊 Confidence Display**: Real-time emotion confidence visualization
- **🔊 Volume Control**: Adjustable volume slider in UI
- **🌍 Cross-Platform**: Works on Windows, macOS, and Linux
- **📝 Enhanced Logging**: Detailed logs for debugging and analytics
- **🚀 Improved Training**: Better model training with data augmentation and callbacks
- **☁️ Colab Notebook**: Train models in the cloud with GPU acceleration

## Features

- **Real-Time Emotion Detection:** 
  - Uses camera to capture and analyze facial expressions
  - Supports 7 emotions: Happy, Sad, Fear, Disgust, Surprise, Angry, and Neutral
  - Confidence-based validation prevents false detections
  
- **Smart Emotion Smoothing:**
  - Prevents rapid emotion changes with configurable thresholds
  - Requires minimum confidence (default 60%) before changing emotion
  - Enforces minimum duration (default 5 seconds) for stable experience
  
- **Mood-Based Music Playback:** 
  - Automatically selects music matching detected emotion
  - Smart track selection avoids recent repeats
  - Smooth fade transitions between tracks
  
- **Enhanced User Interface:** 
  - Confidence percentage and progress bar
  - Current track display
  - Volume control slider
  - Manual emotion selection
  - Pause/play and skip controls

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera
- Audio output device

### Steps

1. Navigate to the desired directory using the shell/command prompt.

2. Clone the repository:
    ```sh
    git clone https://github.com/MTCodes01/Parasseri-Techies.git
    ```

3. Navigate to the project directory:
    ```sh
    cd Parasseri-Techies
    ```

4. Install the required dependencies:
    
    **Windows:**
    ```sh
    pip install -r requirements.txt
    ```
    
    **macOS:**
    ```sh
    pip install -r requirements.txt
    # If pygame installation fails:
    brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
    pip install pygame --pre
    ```
    
    **Linux:**
    ```sh
    sudo apt-get install python3-opencv python3-pygame
    pip install -r requirements.txt
    ```

5. Download the pre-trained model: [Click here](https://drive.google.com/file/d/1M52BHKj0jHqzLb_hWR32xqRU-RADDikV/view?usp=sharing)
   - Place `model.weights.h5` in the project root directory
   - **OR** train your own model (see Training section below)

6. Download the music library: [Click here](https://drive.google.com/drive/folders/1vqBbMODw54qFj-yE4OlC9n_98m6mT7_L?usp=sharing)
   - Extract the `Music` folder to the project root directory
   - **OR** add your own music (see Music Organization section)

## Usage

1. Ensure `model.weights.h5` is in the project directory

2. Run the main script:
    ```sh
    python main.py
    ```

3. The application will:
   - Open a camera window showing real-time emotion detection
   - Display the UI with emotion and confidence information
   - Automatically play music based on detected emotion

4. Controls:
   - **Play/Pause**: Toggle music playback
   - **Next**: Skip to next track in current emotion
   - **Volume Slider**: Adjust playback volume
   - **Emotion Dropdown**: Manually select emotion
   - **Quit**: Close application
   - **Press 'Q'** in camera window to exit

## Music Organization

Organize your music files in emotion-based folders:

```
Music/
├── Angry/      # Intense, aggressive music
├── Disgust/    # Dark, heavy music
├── Fear/       # Suspenseful, tense music
├── Happy/      # Upbeat, cheerful music
├── Sad/        # Melancholic, slow music
├── Surprise/   # Energetic, unexpected music
└── Neutral/    # Calm, ambient music
```

**Supported formats:** MP3, WAV, OGG, FLAC

## Configuration

Customize the application behavior by editing `config.json`:

```json
{
  "emotion_detection": {
    "confidence_threshold": 0.6,      // Adjust emotion sensitivity
    "minimum_emotion_duration": 5.0,  // Prevent rapid changes
    "smoothing_window_size": 15       // Detection smoothing
  },
  "audio": {
    "default_volume": 0.7,            // Initial volume (0.0 - 1.0)
    "fade_duration_ms": 1000          // Fade transition time
  }
}
```

See [CONFIGURATION.md](CONFIGURATION.md) for detailed configuration guide.

## Training Your Own Model

### Option 1: Local Training

1. Ensure you have the training dependencies:
    ```sh
    pip install matplotlib deeplake
    ```

2. Run the training script:
    ```sh
    python train.py
    ```

3. The script will:
   - Download the FER2013 dataset automatically
   - Train with data augmentation and callbacks
   - Save the best model as `model.weights.h5`
   - Generate training plots

### Option 2: Google Colab (Recommended)

1. Upload `emotion_training.ipynb` to Google Colab

2. Enable GPU acceleration:
   - Runtime → Change runtime type → GPU

3. Run all cells to train the model

4. Download `model.weights.h5` when training completes

**Benefits of Colab:**
- Free GPU acceleration (faster training)
- No local setup required
- Interactive visualization
- Easy experimentation

## Directory Structure

```
Parasseri-Techies/
├── Modules/
│   ├── __init__.py
│   ├── music.py              # Music playback engine
│   ├── UI.py                 # User interface
│   ├── emotion_smoother.py   # Emotion stability system
│   ├── config_manager.py     # Configuration handler
│   └── logger.py             # Logging system
├── Music/                    # Emotion-based music folders
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Sad/
│   ├── Surprise/
│   └── Neutral/
├── config.json               # Configuration file
├── main.py                   # Main application
├── train.py                  # Local training script
├── emotion_training.ipynb    # Colab training notebook
├── model.weights.h5          # Trained model weights
├── requirements.txt          # Python dependencies
├── CONFIGURATION.md          # Configuration guide
└── README.md                 # This file
```

## Troubleshooting

### Emotion Changes Too Frequently
- Increase `confidence_threshold` in `config.json` (try 0.7)
- Increase `minimum_emotion_duration` (try 7 seconds)

### Emotion Doesn't Change
- Decrease `confidence_threshold` (try 0.5)
- Ensure good lighting on your face
- Face the camera directly

### Camera Not Working
- Check camera permissions in system settings
- Try different `device_index` in `config.json` (0, 1, 2, etc.)

### No Music Playing
- Verify music files are in correct folders
- Check supported formats (MP3, WAV, OGG, FLAC)
- Check volume slider in UI

### Performance Issues
- Reduce camera resolution in `config.json`
- Close other applications
- Use a simpler model (reduce epochs during training)

See [CONFIGURATION.md](CONFIGURATION.md) for more troubleshooting tips.

## Technical Details

### Emotion Detection Model
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 48x48 grayscale images
- **Output**: 7 emotion classes
- **Dataset**: FER2013 (35,887 training images)
- **Accuracy**: ~65-70% on validation set

### Emotion Smoothing Algorithm
- **Rolling Window**: Analyzes last N detections
- **Confidence Calculation**: Percentage of window matching emotion
- **Transition Dampening**: Minimum duration requirement
- **History Tracking**: Logs all emotion changes

## Performance Tips

1. **Better Accuracy**:
   - Ensure good, even lighting
   - Face camera directly
   - Maintain neutral background
   - Avoid extreme angles

2. **Faster Response**:
   - Lower `minimum_emotion_duration`
   - Reduce `smoothing_window_size`
   - Lower `confidence_threshold`

3. **More Stability**:
   - Increase `confidence_threshold`
   - Increase `minimum_emotion_duration`
   - Increase `smoothing_window_size`

## Contributing

Contributions are welcome! Areas for improvement:
- Additional emotion classes
- Better model architectures
- UI themes and customization
- Spotify/streaming service integration
- Mobile app version

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FER2013 dataset for emotion recognition training
- OpenCV for computer vision capabilities
- TensorFlow/Keras for deep learning
- Pygame for audio playback

---

<div align="center">

**Made with ❤️ for a project expo**

[Report Bug](https://github.com/MTCodes01/Parasseri-Techies/issues) · [Request Feature](https://github.com/MTCodes01/Parasseri-Techies/issues)

</div>
