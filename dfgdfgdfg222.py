
import cv2
import time

# Load the Haar cascade XML file for face detection
cascade_path = 'opencv-master\data\haarcascades\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize variables for fps calculation
frame_count = 0
start_time = time.time()

# Initialize the video capture object
video = cv2.VideoCapture(0)  # Change the parameter to the camera index if necessary

# Check if the camera is opened successfully
if not video.isOpened():
    raise Exception("Could not open the camera")

while True:
    # Read a frame from the camera
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection using Haar cascades
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Calculate fps
    frame_count += 1
    if frame_count >= 30:  # Calculate fps every 30 frames
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = end_time

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video.release()
cv2.destroyAllWindows()