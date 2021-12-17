import cv2
import os
import numpy as np
import tflite_runtime.interpreter as tflite
from scipy.ndimage.morphology import binary_fill_holes
import pandas as pd

def find_circle(x, y):
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

    iris_center_x_1 = x_m + uc
    iris_center_y_1 = y_m + vc

    Ri_1 = np.sqrt((x-iris_center_x_1)**2 + (y-iris_center_y_1)**2)
    R_1 = np.mean(Ri_1)

    return R_1, iris_center_x_1, iris_center_y_1


def computer_iris_ratio_stupid(iris, poly):
    iris = binary_fill_holes(iris[:, :, 0] > 0)
    poly = binary_fill_holes(poly[:, :, 0] > 0)

    intersection = np.sum((iris+0) * (poly+0))
    denominator = np.sum(iris)

    if denominator:
        return float(intersection) / float(denominator)
    else:
        return 0

def detect_light_reflex(img):
    found = False
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img[:, :, 0].copy(), (5, 5), 0)

    threshold = 185.0

    _, maxVal, _, maxLoc = cv2.minMaxLoc(gray)
    

    if maxVal >= threshold:
        found = True

    print(found, maxVal)
    return found, maxLoc

def iris_extend_bounds(iris_center_x, iris_center_y, n, img):
    # To get the iris region
    x1 = iris_center_x - 3
    x2 = iris_center_x + 3
    y1 = iris_center_y - 3
    y2 = iris_center_y + 3

    x1 *= n/64
    y1 *= n/64
    x2 *= n/64
    y2 *= n/64

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    iris_img = img[y1:y2, x1:x2, :].copy()
    
    found, maxLoc = detect_light_reflex(iris_img)

    _, n, _ = iris_img.shape

    clr_coord = (int(maxLoc[0]+x1), int(maxLoc[1]+y1))
               
    if found:
        cv2.circle(iris_img, maxLoc, 2, color=(0, 0, 255), thickness=3)

    return found, iris_img, clr_coord 


def Segment_Eye(original_image):
    _, n, _ = original_image.shape
    blur_img = original_image.copy()

    img = cv2.resize(original_image, (64, 64)).reshape(
        [1, 64, 64, 3]).astype(np.float32) / 127.5 - 1

    interpreter = tflite.Interpreter(model_path=os.path.join("data", "models", "iris_landmark.tflite"))
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
    iris_center_x = (x + iris_data.reshape([-1, 3])[0, 0]) / 2
    iris_center_y = (y + iris_data.reshape([-1, 3])[0, 1]) / 2

    t = int(np.ceil(n / 80))
    img1 = cv2.circle(original_image.copy()*1, (int(iris_center_x*n/64), int(iris_center_y*n/64)),
                      int(rad), color=(255, 0, 255), thickness=t)
    iris = cv2.circle(np.zeros_like(img1), (int(iris_center_x*n/64), int(iris_center_y*n/64)),
                      int(rad), color=(255, 255, 255), thickness=t)
    eyelids = np.zeros_like(img1)
    for ii in np.arange(8):
        pt1 = (int(eyelid_data.reshape([-1, 3])[ii, 0]*n/64),
               int(eyelid_data.reshape([-1, 3])[ii, 1]*n/64))
        pt2 = (int(eyelid_data.reshape([-1, 3])[ii+1, 0]*n/64),
               int(eyelid_data.reshape([-1, 3])[ii+1, 1]*n/64))
        img1 = cv2.line(img1*1, pt1, pt2, color=(0, 255, 0), thickness=t)
        eyelids = cv2.line(eyelids, pt1, pt2, color=(
            255, 255, 255), thickness=t)
    for ii in np.arange(9, 15):
        pt1 = (int(eyelid_data.reshape([-1, 3])[ii, 0]*n/64),
               int(eyelid_data.reshape([-1, 3])[ii, 1]*n/64))
        pt2 = (int(eyelid_data.reshape([-1, 3])[ii+1, 0]*n/64),
               int(eyelid_data.reshape([-1, 3])[ii+1, 1]*n/64))
        img1 = cv2.line(img1*1, pt1, pt2, color=(0, 255, 0), thickness=t)
        eyelids = cv2.line(eyelids, pt1, pt2, color=(
            255, 255, 255), thickness=t)
    pt1 = (int(eyelid_data.reshape([-1, 3])[15, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[15, 1]*n/64))
    pt2 = (int(eyelid_data.reshape([-1, 3])[8, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[8, 1]*n/64))
    img1 = cv2.line(img1*1, pt1, pt2, color=(0, 255, 0), thickness=t)
    eyelids = cv2.line(eyelids, pt1, pt2, color=(255, 255, 255), thickness=t)
    pt1 = (int(eyelid_data.reshape([-1, 3])[0, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[0, 1]*n/64))
    pt2 = (int(eyelid_data.reshape([-1, 3])[9, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[9, 1]*n/64))
    img1 = cv2.line(img1*1, pt1, pt2, color=(0, 255, 0), thickness=t)
    eyelids = cv2.line(eyelids, pt1, pt2, color=(255, 255, 255), thickness=t)

    clr_coord = (0, 0)

    clr_found, iris_img, clr_coord = iris_extend_bounds(iris_center_x, iris_center_y, n, blur_img)

    if clr_found and (abs(int(iris_center_x*n/64) - clr_coord[0]) <= 75 and abs(int(iris_center_y*n/64) - clr_coord[1]) <= 75):
        cv2.circle(img1, clr_coord, 2, color=(255, 0, 255), thickness=2)
        dists_d = np.sqrt((eyelid_data.reshape([-1, 3])[9:16, 0] - clr_coord[0]*64/n) ** 2 +
                          (eyelid_data.reshape([-1, 3])[9:16, 1] - clr_coord[1]*64/n) ** 2)
    else:
        dists_d = np.sqrt((eyelid_data.reshape([-1, 3])[9:16, 0] - iris_center_x) ** 2 +
                          (eyelid_data.reshape([-1, 3])[9:16, 1] - iris_center_y) ** 2)
        cv2.circle(img1, (int(iris_center_x*n/64), int(iris_center_y*n/64)),
                   2, color=(255, 0, 255), thickness=2)

    dist_d = np.min(dists_d)*n/64

    ii = np.argmin(dists_d)
    pt2 = (int(eyelid_data.reshape([-1, 3])[ii+9, 0]*n/64),
           int(eyelid_data.reshape([-1, 3])[ii+9, 1]*n/64))

    print("absolute differences from (iris_center_x, iris_center_y) to CLR: ", abs(
        int(iris_center_x*n/64) - clr_coord[0]), abs(int(iris_center_y*n/64) - clr_coord[1]))

    if clr_found and (abs(int(iris_center_x*n/64) - clr_coord[0]) <= 75 and abs(int(iris_center_y*n/64) - clr_coord[1]) <= 75):
        img1 = cv2.line(img1*1, clr_coord, pt2,
                        color=(255, 255, 0), thickness=t)
    else:
        img1 = cv2.line(img1*1, (int(iris_center_x*n/64), int(iris_center_y*n/64)),
                        pt2, color=(255, 255, 0), thickness=t)
        clr_found = False

    return (img1, rad, computer_iris_ratio_stupid(iris, eyelids), dist_d, clr_found, iris_img)


def save_results(destination, results):
    df = pd.DataFrame(results)
    df.to_excel(os.path.join(destination, r"DroopyEyeMetricsResults.xlsx"), index=False)
