import threading
import numpy as np
import cv2
import tensorflow as tf
import time
from collections import Counter
from Modules.music import EmotionMusicPlayer, emotion_dict, music_library
from Modules.UI import EmotionMusicUI

# Initialize the EmotionMusicPlayer
player = EmotionMusicPlayer(emotion_dict, music_library)

# Create the UI and pass the player
ui = EmotionMusicUI(player)

# Buffer to hold recognized emotions over a short period
emotion_buffer = []
buffer_duration = 10 # seconds
start_time = time.time()

# Start the webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def emotion_detection():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.load_weights('model.weights.h5')

    global start_time, buffer_duration

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Store recognized emotion in the buffer
            emotion_buffer.append(maxindex)
        
        try:
            # Check if the buffer is full
            if time.time() - start_time >= buffer_duration:
                # Most common emotion in buffer
                most_common_emotion = Counter(emotion_buffer).most_common(1)[0][0]
                emotion_buffer.clear()
                ui.update_emotion_display(most_common_emotion)
                player.play_music(most_common_emotion)

                # Reset the timer
                start_time = time.time()
        except IndexError as i:
            buffer_duration = 2
            print(f"No one in view!")
            ui.update_not_found()
            time.sleep(5)
            continue
        except RuntimeError as r:
            if str(r) == 'main thread is not in main loop':
                print("Sayonara!")
                break
            else: raise
        except Exception as e:
            print(f"Error: {e}")
            continue

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the emotion detection in a separate thread
detection_thread = threading.Thread(target=emotion_detection)
detection_thread.start()

# Run the UI in the main thread
ui.run()

# Wait for the detection thread to finish
detection_thread.join()
