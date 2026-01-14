"""
Emotion-Based Music Player
Main application that integrates emotion detection with music playback.
"""

import threading
import numpy as np
import cv2
import tensorflow as tf
import time
from collections import Counter
from Modules.music import EmotionMusicPlayer, emotion_dict, music_library
from Modules.UI import EmotionMusicUI
from Modules.emotion_smoother import EmotionSmoother
from Modules.config_manager import get_config
from Modules.logger import init_logger, get_logger

# Load configuration
config = get_config()

# Initialize logger
logger = init_logger(
    log_file=config.get('logging', 'log_file'),
    level=config.get('logging', 'level', default='INFO'),
    log_to_console=True,
    log_to_file=config.get('logging', 'log_to_file', default=True)
)

logger.info("="*50)
logger.info("Emotion-Based Music Player Starting")
logger.info("="*50)

# Initialize the EmotionMusicPlayer with config
player = EmotionMusicPlayer(
    emotion_dict,
    music_library,
    fade_duration=config.get('audio', 'fade_duration_ms', default=1000),
    default_volume=config.get('audio', 'default_volume', default=0.7)
)

# Create the UI and pass the player
ui = EmotionMusicUI(
    player,
    window_width=config.get('ui', 'window_width', default=512),
    window_height=config.get('ui', 'window_height', default=900),
    show_confidence=config.get('ui', 'show_confidence', default=True)
)

# Initialize emotion smoother with config
emotion_smoother = EmotionSmoother(
    confidence_threshold=config.get('emotion_detection', 'confidence_threshold', default=0.6),
    minimum_duration=config.get('emotion_detection', 'minimum_emotion_duration', default=5.0),
    window_size=config.get('emotion_detection', 'smoothing_window_size', default=15)
)

# Buffer to hold recognized emotions over a short period
emotion_buffer = []
buffer_duration = config.get('emotion_detection', 'buffer_duration', default=10)
start_time = time.time()

# Camera configuration
camera_index = config.get('camera', 'device_index', default=0)
frame_width = config.get('camera', 'frame_width', default=640)
frame_height = config.get('camera', 'frame_height', default=480)
scale_factor = config.get('camera', 'scale_factor', default=1.3)
min_neighbors = config.get('camera', 'min_neighbors', default=5)

# Start the webcam feed
logger.info(f"Initializing camera (device index: {camera_index})")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    logger.critical("Error: Could not open video capture")
    print("Error: Could not open video capture.")
    print("Please check your camera connection and try again.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
logger.info(f"Camera initialized: {frame_width}x{frame_height}")

# Load face cascade classifier
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
logger.info("Face cascade classifier loaded")

def emotion_detection():
    """Main emotion detection loop running in separate thread."""
    logger.info("Starting emotion detection thread")
    
    # Build model (must match training architecture in train.py)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Second convolutional block
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    
    logger.info("Loading model weights...")
    try:
        model.load_weights('model.weights.h5')
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model weights: {e}")
        print(f"Error: Could not load model weights: {e}")
        print("Please ensure 'model.weights.h5' is in the current directory.")
        cap.release()
        cv2.destroyAllWindows()
        return

    global start_time, buffer_duration

    frame_count = 0
    last_emotion = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            # Resize and normalize (must match training preprocessing)
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = np.expand_dims(cropped_img, axis=-1) / 255.0  # Normalize to [0, 1]
            cropped_img = np.expand_dims(cropped_img, axis=0)  # Add batch dimension
            
            # Predict emotion
            prediction = model.predict(cropped_img, verbose=0)
            maxindex = int(np.argmax(prediction))
            
            # Debug: Print prediction probabilities every 30 frames
            if frame_count % 30 == 0:
                pred_str = " | ".join([f"{emotion_dict[i]}: {prediction[0][i]*100:.1f}%" for i in range(7)])
                print(f"Predictions: {pred_str}")
            
            # Add to emotion smoother
            emotion_smoother.add_detection(maxindex)
            
            # Get current stable emotion and confidence
            stable_emotion = emotion_smoother.get_current_emotion()
            confidence = emotion_smoother.get_confidence()
            
            # Display on frame
            if stable_emotion is not None:
                emotion_text = emotion_dict[stable_emotion]
                cv2.putText(
                    frame,
                    f"{emotion_text} ({confidence:.0%})",
                    (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Store recognized emotion in the buffer
            emotion_buffer.append(maxindex)
        
        try:
            # Check if the buffer duration has elapsed
            if time.time() - start_time >= buffer_duration:
                if emotion_buffer:
                    # Get most common emotion in buffer
                    most_common_emotion = Counter(emotion_buffer).most_common(1)[0][0]
                    
                    # Check if emotion smoother validates this change
                    stable_emotion = emotion_smoother.get_current_emotion()
                    confidence = emotion_smoother.get_confidence()
                    
                    if stable_emotion is not None:
                        # Update UI with emotion and confidence
                        ui.update_emotion_display(stable_emotion, confidence)
                        
                        # Only change music if emotion actually changed
                        if stable_emotion != last_emotion:
                            logger.emotion_changed(
                                emotion_dict.get(last_emotion, "None"),
                                emotion_dict[stable_emotion],
                                confidence
                            )
                            player.play_music(stable_emotion)
                            ui.update_track_display(player.current_track)
                            last_emotion = stable_emotion
                    
                    emotion_buffer.clear()
                else:
                    # No face detected
                    logger.debug("No face detected in buffer period")
                    ui.update_not_found()
                
                # Reset the timer
                start_time = time.time()
                
        except IndexError:
            logger.warning("No one in view!")
            ui.update_not_found()
            time.sleep(2)
            continue
        except RuntimeError as r:
            if str(r) == 'main thread is not in main loop':
                logger.info("Main thread exited, stopping detection")
                break
            else:
                logger.error(f"Runtime error: {r}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error in detection loop: {e}")
            continue

        # Display the video frame
        cv2.imshow('Emotion Detection - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("User pressed 'q' to quit")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Emotion detection thread stopped")

# Run the emotion detection in a separate thread
logger.info("Starting detection thread")
detection_thread = threading.Thread(target=emotion_detection, daemon=True)
detection_thread.start()

# Run the UI in the main thread
try:
    ui.run()
except KeyboardInterrupt:
    logger.info("Keyboard interrupt received")
except Exception as e:
    logger.error(f"Error in UI: {e}")
finally:
    logger.info("Application shutting down")
    cap.release()
    cv2.destroyAllWindows()

# Wait for the detection thread to finish
detection_thread.join(timeout=2)
logger.info("Application closed")
