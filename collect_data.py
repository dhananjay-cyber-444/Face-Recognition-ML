import cv2                      # OpenCV for image and video processing
import urllib                  # To read image from an online URL
import numpy as np             # For array operations

# Load the Haarcascade face detection model
# This file helps OpenCV identify human faces in an image
classifier = cv2.CascadeClassifier(
    r"C:/Users/mrp79/Desktop/AIML PROJECT/AIML PROJECT/haarcascade_frontalface_default.xml")

# IP Webcam link (from your mobile camera app)
url = "http://192.168.137.220:8080/shot.jpg"

# List to store the 100 captured face images
data = []

# Loop until 100 face images are collected
while len(data) < 100:

    # Read image from the URL (mobile camera frame)
    image_from_url = urllib.request.urlopen(url)

    # Convert the image data to a numpy array
    frame = np.array(bytearray(image_from_url.read()), np.uint8)

    # Decode the array to convert into an image
    frame = cv2.imdecode(frame, -1)

    # Detect faces (returns x, y, width, height)
    face_points = classifier.detectMultiScale(frame, 1.3, 5)

    # If at least one face is detected
    if len(face_points) > 0:
        for x, y, w, h in face_points:

            # Crop the detected face
            face_frame = frame[y:y+h+1, x:x+w+1]

            # Show only the face
            cv2.imshow("Only face", face_frame)

            # Add face to data list if count is less than 100
            if len(data) <= 100:
                print(len(data) + 1, "/100")
                data.append(face_frame)
                break   # capture only one face per frame

    # Display count on main frame
    cv2.putText(frame, str(len(data)),
                (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                5, (0, 0, 255))

    # Show the full frame with count
    cv2.imshow("frame", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(30) == ord("q"):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()

# If exactly 100 samples collected
if len(data) == 100:
    name = input("Enter Face holder name: ")

    # Save each of the 100 face images to folder
    for i in range(100):
        cv2.imwrite("images/" + name + "_" + str(i) + ".jpg", data[i])

    print("Done")
else:
    print("Need more data")
