import cv2
import numpy as np
from numpy.linalg import inv
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
# Base directory where your preprocessed images are stored
base_dir = r'face_data'

# Load and preprocess your dataset (assuming this part is done)
def preprocessing_data():
    # Lists to store the flattened images and labels
    flattened_images = []
    labels = []

    # Assign a numeric label to each person
    label_dict = {}
    label_counter = 0

    # Iterate over each subfolder (person)
    for person in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person)
        
        # Assign a numeric label if not already assigned
        if person not in label_dict:
            label_dict[person] = label_counter
            label_counter += 1

        # Iterate over each image in the subfolder
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming images are already grayscale

            # Flatten the image and append it to the list
            flattened_image = image.flatten()
            flattened_images.append(flattened_image)

            # Append the corresponding label
            labels.append(person)

    # Convert the lists to NumPy arrays
    X_train = np.array(flattened_images)
    y_train = np.array(labels)
    return X_train,y_train

# X_train: array of flattened preprocessed face images
# y_train: array of labels corresponding to X_train

# PCA for dimensionality reduction
X_train, y_train = preprocessing_data()
mean_face = X_train.mean(axis=1, keepdims=True)
X_train = X_train - mean_face
print(y_train)
pca = PCA(n_components=10)  # Adjust the number of components
X_train_pca = pca.fit_transform(X_train)

cov_matrix = np.cov(X_train_pca, rowvar=False)
inv_cov_matrix = inv(cov_matrix)

# Nearest Neighbor classifier
nn = NearestNeighbors(metric='mahalanobis', metric_params={'VI': inv_cov_matrix})
nn.fit(X_train_pca, y_train)

# Real-time face recognition
cap = cv2.VideoCapture(0)
desired_width = 1280
desired_height = 720

# Set the frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (300, 300))  # Resize to match training data
        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(roi_gray_resized, (5, 5), 0)
        # Apply histogram equalization
        equalized_face = cv2.equalizeHist(blurred_face)

        roi_flattened = roi_gray_resized.flatten().reshape(1, -1)
        roi_pca = pca.transform(roi_flattened)
        
        # Find the nearest neighbor
        _, indices = nn.kneighbors(roi_pca, n_neighbors=1)
        label = y_train[indices[0][0]]
        
        # Display the label
        cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
