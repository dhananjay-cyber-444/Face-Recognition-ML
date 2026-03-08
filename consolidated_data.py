import cv2                 # OpenCV for image processing
import os                  # For file path handling
import pickle              # For saving data in binary format
import numpy as np         # For numerical operations

# --------------------------
# 1) CREATE PATHS
# --------------------------

# Directory where processed data will be saved
data_dir = os.path.join(os.getcwd(), 'data')

# Directory where your face images are stored (collected earlier)
img_dir = os.path.join(os.getcwd(), 'images')

# Lists to hold the images and their corresponding labels
image_data = []
labels = []

# --------------------------
# 2) READ ALL IMAGES
# --------------------------

# Loop through every image file inside the 'images' folder
for i in os.listdir(img_dir):

    # Read the image file
    image = cv2.imread(os.path.join(img_dir, i))

    # Resize all images to the same size (100×100 pixels)
    image = cv2.resize(image, (100, 100))

    # Convert image from BGR → Grayscale (used for training)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Store the processed image
    image_data.append(image)

    # Extract label (name) from filename: "prasad_0.jpg" → "prasad"
    labels.append(str(i).split("_")[0])


# Convert lists into Numpy Arrays (faster for machine learning)
image_data = np.array(image_data)
labels = np.array(labels)

# --------------------------
# 3) PRINT TO VERIFY
# --------------------------

print(image_data)     # This prints pixel values
print(labels)         # This prints corresponding names

# --------------------------
# 4) SHOW A SAMPLE IMAGE
# --------------------------

import matplotlib.pyplot as plt
plt.imshow(image_data[60], cmap="gray")   # Show 61st image
plt.show()

# --------------------------
# 5) SAVE DATA USING PICKLE
# --------------------------

# Save image data in binary file "images.p"
with open(os.path.join(data_dir, "images.p"), 'wb') as f:
    pickle.dump(image_data, f)

# Save labels in "labels.p"
with open(os.path.join(data_dir, "labels.p"), 'wb') as f:
    pickle.dump(labels, f)
