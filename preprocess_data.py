import cv2
import os

def preprocess_images(input_dir, output_dir, target_size=(300, 300)):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # List all the subdirectories in the input directory
    persons = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for person in persons:
        person_input_dir = os.path.join(input_dir, person)
        person_output_dir = os.path.join(output_dir, person)
        os.makedirs(person_output_dir, exist_ok=True)

        # List all image files in the person's directory
        image_files = [f for f in os.listdir(person_input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for i, image_file in enumerate(image_files, start=1):
            # Load the image
            image_path = os.path.join(person_input_dir, image_file)
            image = cv2.imread(image_path)

            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # If a face is detected, crop the face
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Assuming the first detected face is the target face
                cropped_face = gray_image[y:y+h, x:x+w]

                 # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(cropped_face, (5, 5), 0)

                # Apply histogram equalization
                equalized_face = cv2.equalizeHist(blurred_face)

                # Resize the cropped face
                resized_face = cv2.resize(equalized_face, target_size)

                # Save the processed face image with a new name
                output_image_path = os.path.join(person_output_dir, f'{i}.png')
                cv2.imwrite(output_image_path, resized_face)

                print(f"Processed {image_file} for {person}")

    print("Preprocessing completed.")

if __name__ == '__main__':
    input_directory = r'C:\Users\JLING\Documents\EE4208_assignment_1\Jun Han dataset'
    output_directory = r'C:\Users\JLING\Documents\EE4208_assignment_1\face_data'
    preprocess_images(input_directory, output_directory)


if __name__ == '__main__':
    input_directory = r'C:\Users\JLING\Documents\EE4208_assignment_1\Jun Han dataset'
    output_directory = r'C:\Users\JLING\Documents\EE4208_assignment_1\face_data'
    preprocess_images(input_directory, output_directory)
