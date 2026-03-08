import urllib                      # To read camera image from URL (IP Webcam)
import cv2                        # For face detection, preprocessing, display
import numpy as np                # For array operations
from keras.models import load_model   # To load trained deep learning model

# ----------------------------------------------------
# 1) LOAD FACE DETECTOR AND TRAINED MODEL
# ----------------------------------------------------

# Haarcascade model used to detect face areas in the camera frame
classifier = cv2.CascadeClassifier(
    r'C:/Users/mrp79/Desktop/AIML PROJECT/AIML PROJECT/haarcascade_frontalface_default.xml'
)

# Your trained face-recognition model (CNN)
model = load_model(
    r"C:/Users/mrp79/Desktop/AIML PROJECT/AIML PROJECT/final_model.h5"
)

# URL of your IP Camera (IP Webcam app)
URL = 'http://10.138.142.164:8080/shot.jpg'


# ----------------------------------------------------
# 2) LABEL DECODER FUNCTION
# ----------------------------------------------------

def get_pred_label(pred):
    # The labels that correspond to output classes of your model
    labels = ["DHANIA","SONU"]    # Add more names if you train multiple people
    return labels[pred]


# ----------------------------------------------------
# 3) FACE PREPROCESSING BEFORE PREDICTION
# ----------------------------------------------------

def preprocess(img):
    # Convert face to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to model input size (100×100)
    img = cv2.resize(img, (100, 100))

    # Histogram equalization → improves brightness & contrast
    img = cv2.equalizeHist(img)

    # Reshape to: (1,100,100,1) → required for CNN model
    img = img.reshape(1, 100, 100, 1)

    # Normalize pixel values (0–1)
    img = img / 255

    return img


# ----------------------------------------------------
# 4) START REAL-TIME RECOGNITION LOOP
# ----------------------------------------------------

ret = True
while ret:

    # Read a frame from the phone camera using URL
    img_url = urllib.request.urlopen(URL)

    # Convert into numpy array
    image = np.array(bytearray(img_url.read()), np.uint8)

    # Convert array into actual image
    frame = cv2.imdecode(image, -1)

    # Detect all faces in the current frame
    faces = classifier.detectMultiScale(frame, 1.5, 5)

    # Loop through each detected face
    for x, y, w, h in faces:

        # Crop the face region from the full frame
        face = frame[y:y+h, x:x+w]

        # Draw a blue rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

        # Predict name using trained model
        prediction = np.argmax(model.predict(preprocess(face)))

        # Show predicted name on the screen
        cv2.putText(frame,
                    get_pred_label(prediction),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 0),
                    2)

    # Show video feed in a window
    cv2.imshow("capture", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Close all windows after loop ends
cv2.destroyAllWindows()
