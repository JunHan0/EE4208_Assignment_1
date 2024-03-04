import cv2
import os
import sys

def capture_and_preprocess(subfolder_name):
    # Open the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a folder to store the captured images
    folder_name = f'face_data/{subfolder_name}'
    os.makedirs(folder_name, exist_ok=True)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Capture 10 face images
    count = 0
    while count < 10:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If a face is detected, preprocess and save the image
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            resized = cv2.resize(face, (100, 100))
            
            # Apply Gaussian blur
            blurred_face = cv2.GaussianBlur(resized, (5, 5), 0)

            # Apply histogram equalization
            equalized_face = cv2.equalizeHist(blurred_face)

            # Save the processed face image
            cv2.imwrite(f'{folder_name}/{count + 1}.png', equalized_face)
            count += 1

            # Display the frame with the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Video', frame)

            if count >= 10:
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <subfolder_name>")
    else:
        subfolder_name = sys.argv[1]
        capture_and_preprocess(subfolder_name)
