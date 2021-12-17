import numpy as np
import cv2
import dlib

imagePath = r'data/input/image.jpg'
cascPath =r'data/models/haarcascade.xml'

PREDICTOR_PATH =r'data/models/shape_predictor_68_face_landmarks.dat'

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
  #  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

   # Converting the OpenCV rectangle coordinates to Dlib rectangle
   dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

   landmarks = np.matrix([[p.x, p.y]
                          for p in predictor(image, dlib_rect).parts()])

   landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS]

   for idx, point in enumerate(landmarks):
     pos = (point[0, 0], point[0, 1])
     cv2.circle(image, pos, 3, color=(0, 255, 0), thickness=5)

cv2.imwrite("dlib_gan_landmarks.png", image)
#cv2.imshow("Landmarks found", image)
