import cv2

def check_camera():
    # Attempt to access the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera is open. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame reading is unsuccessful, break the loop
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the frame in a window named 'Camera'
        cv2.imshow('Camera', frame)

        # Wait for the 'q' key to be pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Uncomment the line below to run the camera check function
check_camera()
