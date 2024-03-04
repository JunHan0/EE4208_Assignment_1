import numpy as np
import cv2
import os

# Load the .npy file containing the images
npy_file_path = r'C:\Users\JLING\Documents\EE4208_assignment_1\olivetti_faces.npy'
images = np.load(npy_file_path)
#cv2.imshow(images)

if images is None:
    print("Error: Could not load images from .npy file.")
    exit()

# Directory to save the processed images
output_dir = r'C:\Users\JLING\Documents\EE4208_assignment_1\face_data'
os.makedirs(output_dir, exist_ok=True)

# Iterate over the images and save them
for i, image in enumerate(images):
    # Check if the image is valid
    if image is None:
        print(f"Error: Image {i} is None.")
        continue

    # Display the image
    cv2.imshow(f'Image {i}', image)
    cv2.waitKey(1000)  # Wait for 1000 milliseconds (1 second) before displaying the next image

    # Save the image
    #output_file_path = os.path.join(output_dir, f'image_{i}.png')
    #cv2.imwrite(output_file_path, image)

cv2.destroyAllWindows()  # Close all OpenCV windows
print(f'Saved {len(images)} images to {output_dir}')