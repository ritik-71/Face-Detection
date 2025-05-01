# Import necessary libraries
import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)

# Main loop to process video frames
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Failed to capture video")
        break
    
    # Convert the frame to grayscale (Haar Cascade works better with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the resulting frame with rectangles around faces
    cv2.imshow('Video', frame)
    
    # Break the loop on 'ESC' key press (ASCII value 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()