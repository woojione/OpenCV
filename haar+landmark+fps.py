import cv2
import time
import dlib

# Load the Haar cascade classifier
face_cascade = cv2.CascadeClassifier('opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')

# Load the dlib shape predictor
predictor = dlib.shape_predictor("opencv-master\data\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat")

# Set up the video capture
cap = cv2.VideoCapture(0)

# Set up the FPS calculation
fps = 0
start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the faces and detect landmarks
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = predictor(gray, dlib_rect)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    # Calculate and display the FPS
    fps = fps + 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        print("FPS: ", fps)
        fps = 0
        start_time = time.time()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
