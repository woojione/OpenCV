import cv2
import dlib
import time

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

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

    # Perform face detection
    faces = detector(gray)

    # Draw bounding boxes around the detected faces
    for face in faces:
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

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