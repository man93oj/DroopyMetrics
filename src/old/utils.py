import cv2
import dlib
import numpy as np
import tflite_runtime.interpreter as tflite
from scipy.ndimage.morphology import binary_fill_holes
import pandas as pd

EYE_LANDMARKS = {
    "left": (42, 48),
    "right": (36, 42),
}

shape = {}


def get_midpoints(image, shape):
    right_upper = (shape.part(37).y + shape.part(38).y) / 2
    right_lower = (shape.part(40).y + shape.part(41).y) / 2
    left_upper = (shape.part(43).y + shape.part(44).y) / 2
    left_lower = (shape.part(46).y + shape.part(47).y) / 2

    return (right_upper, right_lower, left_upper, left_lower)


def extract_eye_bounds(image):
    """
    Extracts the bounding boxes for both eyes
    """
    # dlib face detector and faical landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        r'data/models/shape_predictor_68_face_landmarks.dat')

    # detect faces in the image
    faces = detector(image, 1)

    # Error handling when no faces detected done training and testing scripts
    if(len(faces) < 1):
        return None, None, None

    if(len(faces) > 1):
        raise RuntimeError("Too many faces detected.")

    face = faces[0]
    shape = predictor(image, face)
    midpoints = get_midpoints(image, shape)
    shape = shape_to_np(shape)

    i, j = EYE_LANDMARKS["left"]
    left_eye = cv2.boundingRect(np.array([shape[i:j]]))

    i, j = EYE_LANDMARKS["right"]
    right_eye = cv2.boundingRect(np.array([shape[i:j]]))

    return left_eye, right_eye, midpoints, shape


def shape_to_np(shape, dtype="int"):
    """
    Converts the dtype shape output by dlib's predictor to numpy array 
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def extend_bounds(height, width, bounds):
    """
    Extend bounds to incoperate surrounding region and make a square. Rough estimate
    """
    (x, y, w, h) = bounds

    x = int(x-(0.5*w))
    y = int(y-w)
    w = int(2*w)
    h = int(w)

    return (x, y, w, h)


def detect_light_reflex(img):
    found = False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = 210.0
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(gray)

    # print(maxVal)
    if maxVal >= threshold:
        found = True

    print(found, maxVal)
    return found, maxLoc


def get_distance(filename, midpoints, clr_coord):
    if 'LE' in filename and not 'RE' in filename:
        _, _, left_upper, left_lower = midpoints
        return abs(left_upper - clr_coord)
    elif 'RE' in filename and not 'LE' in filename:
        right_upper, right_lower, _, _ = midpoints
        return abs(right_upper - clr_coord)


def find_circle(x, y):
    '''
    Copy Pasta'd Code to fit circle.
    '''
    x_m = np.mean(x)
    y_m = np.mean(y)

    u = x - x_m
    v = y - y_m

    Suv = np.sum(u*v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    Ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1 = np.mean(Ri_1)

    return R_1, xc_1, yc_1


def computer_iris_ratio_stupid(iris, poly):
    iris = binary_fill_holes(iris[:, :, 0] > 0)
    poly = binary_fill_holes(poly[:, :, 0] > 0)

    intersection = np.sum((iris+0) * (poly+0))
    denominator = np.sum(iris)
    return float(intersection) / float(denominator)


def iris_extend_bounds(xc, yc, n, img, filename):

    x1 = xc - 20
    x2 = xc + 20
    y1 = yc - 20
    y2 = yc + 20

    x1 *= n/64
    y1 *= n/64
    x2 *= n/64
    y2 *= n/64

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    clr_img = img[y1:y2, x1:x2, :].copy()

    found, maxLoc = detect_light_reflex(clr_img)

    if found:
        cv2.circle(clr_img, maxLoc, 2, color=(0, 0, 255), thickness=3)

        eye_bound_fname2 = filename.split('_IRIS')[0]+str('_detect_CLR.png')

        cv2.imwrite(eye_bound_fname2, clr_img)

    return found, clr_img
    # factor = 5.0
    # blur_img1 = blur_img[y1:y2, x1:x2, :].copy()

    # (h1, w1) = blur_img1.shape[:2]
    # kW = int(w1 / factor)
    # kH = int(h1 / factor)

    # # ensure the width of the kernel is odd
    # if kW % 2 == 0:
    #     kW -= 1
    # # ensure the height of the kernel is odd
    # if kH % 2 == 0:
    #     kH -= 1
    #     # apply a Gaussian blur to the input image using our computed
    #     # kernel size

    # for _ in range(2000):
    #     pup_blur = cv2.GaussianBlur(blur_img1, (kW, kH), 0)

    # blur_img[y1:y2, x1:x2, :] = pup_blur
    # eye_bound_fname2 = filename.split('_IRIS')[0]+str('.png')
    # cv2.imwrite(eye_bound_fname2, blur_img)


def Segment_Eye(filename, img1, midpoints, original_image):
    m, n, _ = img1.shape
    blur_img = img1.copy()

    cv2.imwrite(filename.split('.png')[0]+str('_original.png'), img1)

    img = cv2.resize(img1, (64, 64)).reshape(
        [1, 64, 64, 3]).astype(np.float32) / 127.5 - 1
    interpreter = tflite.Interpreter(
        model_path="data/models/iris_landmark.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    eyelid_data = interpreter.get_tensor(output_details[0]['index'])
    iris_data = interpreter.get_tensor(output_details[1]['index'])

    rad0 = iris_data.reshape([-1, 3])[1:, :2] - \
        iris_data.reshape([-1, 3])[0, :2]
    rad0 = np.mean(np.sqrt(np.sum(rad0 ** 2, axis=1))) * (n/64)
    rad1, x, y = find_circle(iris_data.reshape(
        [-1, 3])[1:, 0], iris_data.reshape([-1, 3])[1:, 1])
    rad1 *= (n/64)
    rad = (rad0 + rad1) / 2
    xc = (x + iris_data.reshape([-1, 3])[0, 0]) / 2
    yc = (y + iris_data.reshape([-1, 3])[0, 1]) / 2

    t = int(np.ceil(n / 80))
    img1 = cv2.circle(img1*1, (int(xc*n/64), int(yc*n/64)),
                      int(rad), color=(255, 0, 255), thickness=t)
    iris = cv2.circle(np.zeros_like(img1), (int(xc*n/64), int(yc*n/64)),
                      int(rad), color=(255, 255, 255), thickness=1)
    eyelids = np.zeros_like(img1)
    for ii in np.arange(8):
        pt1 = (int(eyelid_data.reshape([-1, 3])[ii, 0]*n/64),
               int(eyelid_data.reshape([-1, 3])[ii, 1]*n/64))
        pt2 = (int(eyelid_data.reshape([-1, 3])[ii+1, 0]*n/64),
               int(eyelid_data.reshape([-1, 3])[ii+1, 1]*n/64))
        img1 = cv2.line(img1*1, pt1, pt2, color=(0, 255, 0), thickness=t)
        eyelids = cv2.line(eyelids, pt1, pt2, color=(
            255, 255, 255), thickness=1)
    for ii in np.arange(9, 15):
        pt1 = (int(eyelid_data.reshape([-1, 3])[ii, 0]*n/64),
               int(eyelid_data.reshape([-1, 3])[ii, 1]*n/64))
        pt2 = (int(eyelid_data.reshape([-1, 3])[ii+1, 0]*n/64),
               int(eyelid_data.reshape([-1, 3])[ii+1, 1]*n/64))
        img1 = cv2.line(img1*1, pt1, pt2, color=(0, 255, 0), thickness=t)
        eyelids = cv2.line(eyelids, pt1, pt2, color=(
            255, 255, 255), thickness=1)
    pt1 = (int(eyelid_data.reshape([-1, 3])[15, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[15, 1]*n/64))
    pt2 = (int(eyelid_data.reshape([-1, 3])[8, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[8, 1]*n/64))
    img1 = cv2.line(img1*1, pt1, pt2, color=(0, 255, 0), thickness=t)
    eyelids = cv2.line(eyelids, pt1, pt2, color=(255, 255, 255), thickness=1)
    pt1 = (int(eyelid_data.reshape([-1, 3])[0, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[0, 1]*n/64))
    pt2 = (int(eyelid_data.reshape([-1, 3])[9, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[9, 1]*n/64))
    img1 = cv2.line(img1*1, pt1, pt2, color=(0, 255, 0), thickness=t)
    eyelids = cv2.line(eyelids, pt1, pt2, color=(255, 255, 255), thickness=1)

    clr_coord = (0, 0)

    clr_found, _ = iris_extend_bounds(xc, yc, n, blur_img, filename)

    if clr_found:
        _, clr_coord = detect_light_reflex(img1)

    if clr_found and (abs(int(xc*n/64) - clr_coord[0]) <= 45 and abs(int(yc*n/64) - clr_coord[1]) <= 45):
        # cv2.circle(img1*1, clr_coord, 2, color=(255, 0, 255), thickness=2)
        dists_d = np.sqrt((eyelid_data.reshape([-1, 3])[9:16, 0] - clr_coord[0]*64/n) ** 2 + (
            eyelid_data.reshape([-1, 3])[9:16, 1] - clr_coord[1]*64/n) ** 2)
    else:
        dists_d = np.sqrt((eyelid_data.reshape(
            [-1, 3])[9:16, 0] - xc) ** 2 + (eyelid_data.reshape([-1, 3])[9:16, 1] - yc) ** 2)
        # cv2.circle(img1*1, (int(xc*n/64), int(yc*n/64)),
        #            2, color=(255, 0, 255), thickness=2)

    right_upper, right_lower, left_upper, left_lower = midpoints
    height, width, _ = original_image.shape

    newmidpoints = (int(right_upper*m/height), int(right_lower*m/height),
                    int(left_upper*m/height), int(left_lower*m/height))

    dist_d = np.min(dists_d)*n/64

    ii = np.argmin(dists_d)
    pt2 = (int(eyelid_data.reshape([-1, 3])[ii+9, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[ii+9, 1]*n/64))

    print("absolute differences from (xc, yc) to CLR: ", abs(
        int(xc*n/64) - clr_coord[0]), abs(int(yc*n/64) - clr_coord[1]))

    if clr_found and (abs(int(xc*n/64) - clr_coord[0]) <= 45 and abs(int(yc*n/64) - clr_coord[1]) <= 45):
        img1 = cv2.line(img1*1, clr_coord, pt2,
                        color=(255, 255, 0), thickness=t)
    else:
        img1 = cv2.line(img1*1, (int(xc*n/64), int(yc*n/64)),
                        pt2, color=(255, 255, 0), thickness=t)
        clr_found = False

    return img1, rad, computer_iris_ratio_stupid(iris, eyelids), dist_d, get_distance(filename, newmidpoints, clr_coord[1]), (int(xc*n/64), int(yc*n/64)), clr_found, (img1[:, clr_coord[1], 0], img1[:, clr_coord[1], 1], img1[:, clr_coord[1], 2])


def save_results(results):
    df = pd.DataFrame(results)
    df.to_excel(r'data/output/DroopyEyeMetricsResults.xlsx', index=False)
