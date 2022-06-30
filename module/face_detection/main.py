import cv2 as cv
import tensorflow as tf
import numpy as np
from helper import *

# -------------------------------------
# Load model


class EuclideanLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        diff = tf.math.square(tf.math.subtract(y_true, y_pred))
        euclidean = tf.math.sqrt(tf.reduce_sum(diff))
        return euclidean


fc_loss = tf.keras.losses.CategoricalCrossentropy()
box_loss = EuclideanLoss()

fc_metric = tf.keras.metrics.CategoricalCrossentropy()
box_mertic = tf.keras.metrics.MeanSquaredError()

losses = {
    "face_classification": fc_loss,
    "bounding_box_regression": box_loss
}
metrics = {
    "face_classification": fc_metric,
    "bounding_box_regression": box_mertic
}


Pnet = tf.keras.models.load_model('model/detect_model', compile=False)

Pnet.compile(
    loss=losses,
    optimizer='adam',
    metrics=metrics
)
# -------------------------------------
# Face Tracking

# Create a VideoCapture object and read from input file
cap = cv.VideoCapture('datasets/test.mp4')

if (cap.isOpened() == False):
    print("Error opening video  file")

# Read and predict bounding box for the first frame
ret, prevImage = cap.read()

image_to_detect = cv.cvtColor(prevImage, cv.COLOR_BGR2RGB)
image_to_detect = tf.image.convert_image_dtype(
    image_to_detect, tf.float32).numpy()
prediction = pipeline(image_to_detect, Pnet)[0]
x1, y1, x2, y2 = [int(i) for i in prediction[:4]]


# Convert Image color to Gray and Extract ROI using predicted bounding box
prevImage_gray = cv.cvtColor(prevImage, cv.COLOR_BGR2GRAY)
roi = prevImage_gray[y1:y2, x1:x2]
# Find key points in the ROI
kp0 = cv.goodFeaturesToTrack(roi, 20, 0.1, 15)

for i in kp0:
    i[0, 0] = i[0, 0] + x1
    i[0, 1] = i[0, 1] + y1


# Read until video is completed

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    nextImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculate Optical Flow

    kp1, st, err = cv.calcOpticalFlowPyrLK(
        prevImage_gray, nextImage, kp0, None)

    # Draw Bounding Box Around Key Point
    x, y, w, h = cv.boundingRect(kp1)
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw all found key points
    for point in kp1:
        xp = int(point[0, 0])
        yp = int(point[0, 1])
        cv.circle(frame, (xp, yp), 5, (0, 0, 255), 3)

    # Update Keypoints
    kp0 = kp1
    prevImage_gray = nextImage

    cv.imshow('video', frame)
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

    cv.waitKey(25)

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
