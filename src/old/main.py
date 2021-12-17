import os
import cv2
import utils
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


class DroopyEyeMetrics:
    def __init__(self):
        self.INPUT_DIR = os.path.join("data", "input")
        self.OUTPUT_DIR = os.path.join("data", "output")
        self.results = {}
        self.de_Filename = []
        self.rightEyeDist = []
        self.RightIrisRatio = []
        self.LeftEyeDist = []
        self.LeftIrisRatio = []
        self.prediction = []
        self.iris_prediction = []
        self.GroundTruthPred = []
        self.dlib_right_dist = []
        self.dlib_left_dist = []
        self.combinedprediction = []
        # true values and predicted probabilites for roc calculation
        self.dist_roc_y_true = []
        self.dist_roc_y_score = []
        self.irisratio_roc_y_true = []
        self.irisratio_roc_y_score = []
        self.leftfound = []
        self.rightfound = []
        self.left_eye_col_r1 = []
        self.left_eye_col_g2 = []
        self.left_eye_col_b3 = []

        self.right_eye_col_r1 = []
        self.right_eye_col_g2 = []
        self.right_eye_col_b3 = []
        
        import shutil
        if os.path.exists("__pycache__"):
            shutil.rmtree("__pycache__")
        # if os.path.exists(self.OUTPUT_DIR):
        #     shutil.rmtree(self.OUTPUT_DIR)
            
        # creating subfolders for output
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, "both droopy")):
            os.makedirs(os.path.join(self.OUTPUT_DIR, "both droopy"))
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, "left droopy")):
            os.makedirs(os.path.join(self.OUTPUT_DIR, "left droopy"))
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, "right droopy")):
            os.makedirs(os.path.join(self.OUTPUT_DIR, "right droopy"))
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, "not droopy")):
            os.makedirs(os.path.join(self.OUTPUT_DIR, "not droopy"))

    def getmetrics(self):
        for dirName, subdirList, fileList in os.walk(self.INPUT_DIR):
            for image_name in fileList:
                image_path = os.path.join(dirName, image_name)
                output_dirName_path = dirName.replace("input", "output")

                print(image_name, os.path.isfile(image_path) and image_name.split(
                    '.')[1].lower() in ('png', 'jpg', 'jpeg'), output_dirName_path)
                if os.path.isfile(image_path) and image_name.split('.')[1].lower() in ('png', 'jpg', 'jpeg') and not 'train' in dirName:
                    # reading image and converting it to b/w
                    image = cv2.imread(image_path)
                    height, width, _ = image.shape

                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # extracting eyes bounds from image
                    left_eye_bounds, right_eye_bounds, midpoints, shape = utils.extract_eye_bounds(
                        image_gray)
                    # incase face not detected in image
                    if(left_eye_bounds == None or right_eye_bounds == None):
                        print(
                            "No face detected in {}. Skipping this image.".format(image_name))
                        continue

                    x, y, w, h = utils.extend_bounds(
                        height, width, left_eye_bounds)
                    # print("in line 46: ", x, y, w, h)
                    left_eye_bound = image[y:(y+h), x:(x+w), :].copy()

                    x1, y1, w1, h1 = utils.extend_bounds(
                        height, width, right_eye_bounds)
                    # print("in line 50: ", x1, y1, w1, h1)
                    right_eye_bound = image[y1:(y1+h1), x1:(x1+w1), :].copy()
                    
                    cv2.imwrite(os.path.join(output_dirName_path, image_name.split('.')[0]+str('_LE_cropped.png')), cv2.cvtColor(left_eye_bound, cv2.COLOR_BGR2GRAY))
                    cv2.imwrite(os.path.join(output_dirName_path, image_name.split('.')[0]+str('_RE_cropped.png')), cv2.cvtColor(right_eye_bound, cv2.COLOR_BGR2GRAY))
                    
                    # Horizontally flipping the right eye for iris_landmark model
                    # right_eye_bound = cv2.flip(right_eye_bound, 1)

                    # left_eye_bound = left_eye_bound[:, ::-1, :]
                    # left_eye_bound = cv2.flip(left_eye_bound, 1)

                    left_eye_bound = left_eye_bound.copy()
                    right_eye_bound = right_eye_bound.copy()

                    left_eye_radius, right_eye_radius = 0, 0
                    left_iris_img, left_eye_radius, left_iris_ratio, dist1, dlib_left_dist, left_eye_iris_center, left_found, left_eye_col = utils.Segment_Eye(
                        os.path.join(output_dirName_path, image_name.split('.')[0]+str('_LE_IRIS.png')), left_eye_bound, midpoints, image)

                    right_iris_img, right_eye_radius, right_iris_ratio, dist2, dlib_right_dist, right_eye_iris_center, right_found, right_eye_col = utils.Segment_Eye(
                        os.path.join(output_dirName_path, image_name.split('.')[0]+str('_RE_IRIS.png')), right_eye_bound, midpoints, image)

                    cv2.imwrite(os.path.join(output_dirName_path, image_name.split('.')[0]+str('_LE_IRIS.png')), left_iris_img)
                    cv2.imwrite(os.path.join(output_dirName_path, image_name.split('.')[0]+str('_RE_IRIS.png')), right_iris_img)

                    if left_eye_radius * right_eye_radius:
                        rad = (left_eye_radius + right_eye_radius) / 2
                        pix2mm = 4.75 / rad

                        right_eye_distance = dist2*pix2mm
                        left_eye_distance = dist1*pix2mm
                        dlib_left_dist = dlib_left_dist*pix2mm
                        dlib_right_dist = dlib_right_dist*pix2mm

                        # if (9 < right_eye_distance or right_eye_distance < 4.05) and (9 < left_eye_distance or left_eye_distance < 4.05):
                        #     self.prediction.append('both droopy')
                        #     print("both Eye Droopy: ", right_eye_distance,
                        #           left_eye_distance, right_iris_ratio, left_iris_ratio)
                        # elif (9 < right_eye_distance or right_eye_distance < 4.05) or not right_found:
                        #     self.prediction.append('right droopy')
                        #     print("right Eye Droopy",
                        #           right_eye_distance, right_iris_ratio)
                        # elif (9 < left_eye_distance or left_eye_distance < 4.05) or not left_found:
                        #     self.prediction.append('left droopy')
                        #     print("left Eye Droopy",
                        #           left_eye_distance, left_iris_ratio)
                        # else:
                        #     self.prediction.append('not droopy')
                        #     print("not Droopy", right_eye_distance,
                        #           left_eye_distance, right_iris_ratio, left_iris_ratio)

                        # if right_iris_ratio < 0.87 and left_iris_ratio < 0.87:
                        #     self.iris_prediction.append('both droopy')
                        #     print("both Eye Droopy: ", right_eye_distance,
                        #           left_eye_distance, right_iris_ratio, left_iris_ratio)
                        # elif right_iris_ratio < 0.87:
                        #     self.iris_prediction.append('right droopy')
                        #     print("right Eye Droopy",
                        #           right_eye_distance, right_iris_ratio)
                        # elif left_iris_ratio < 0.87:
                        #     self.iris_prediction.append('left droopy')
                        #     print("left Eye Droopy",
                        #           left_eye_distance, left_iris_ratio)
                        # else:
                        #     self.iris_prediction.append('not droopy')
                        #     print("not Droopy", right_eye_distance,
                        #           left_eye_distance, right_iris_ratio, left_iris_ratio)
                        print("left_found, right : ", left_found, right_found)
                        if (right_iris_ratio < 0.92 and left_iris_ratio < 0.92) or (right_eye_distance < 3.48 and left_eye_distance < 3.48) or (not left_found and not right_found):
                            self.combinedprediction.append('both droopy')
                            print("both Eye Droopy: ", right_eye_distance,
                                  left_eye_distance, right_iris_ratio, left_iris_ratio)
                        elif right_iris_ratio < 0.92 or right_eye_distance < 3.48 or not right_found:
                            self.combinedprediction.append('right droopy')
                            print("right Eye Droopy",
                                  right_eye_distance, right_iris_ratio)
                        elif left_iris_ratio < 0.92 or left_eye_distance < 3.48 or not left_found:
                            self.combinedprediction.append('left droopy')
                            print("left Eye Droopy",
                                  left_eye_distance, left_iris_ratio)
                        else:
                            self.combinedprediction.append('not droopy')
                            print("not Droopy", right_eye_distance,
                                  left_eye_distance, right_iris_ratio, left_iris_ratio)

                    groundtruth = dirName.split('/')[-1]

                    # if groundtruth == 'both droopy':
                    #     self.dist_roc_y_true.append(1)
                    #     if self.combinedprediction[-1] == 'both droopy':
                    #         self.dist_roc_y_score.append(1)
                    #     else:
                    #         self.dist_roc_y_score.append(0)

                    #     # self.irisratio_roc_y_true.append(1)

                    # elif groundtruth == 'left droopy':
                    #     self.dist_roc_y_true.append(1)
                    #     if self.combinedprediction[-1] == 'left droopy':
                    #         self.dist_roc_y_score.append(1)
                    #     else:
                    #         self.dist_roc_y_score.append(0)
                    # elif groundtruth == 'right droopy':
                    #     self.dist_roc_y_true.append(1)
                    #     if self.combinedprediction[-1] == 'right droopy':
                    #         self.dist_roc_y_score.append(1)
                    #     else:
                    #         self.dist_roc_y_score.append(0)
                    # elif groundtruth == 'not droopy':
                    #     self.dist_roc_y_true.append(1)
                    #     if self.combinedprediction[-1] == 'not droopy':
                    #         self.dist_roc_y_score.append(1)
                    #     else:
                    #         self.dist_roc_y_score.append(0)

                    self.de_Filename.append(image_name.split('/')[-1])
                    self.rightEyeDist.append(right_eye_distance)
                    self.RightIrisRatio.append(right_iris_ratio)
                    self.LeftEyeDist.append(left_eye_distance)
                    self.LeftIrisRatio.append(left_iris_ratio)
                    self.GroundTruthPred.append(groundtruth)
                    self.dlib_left_dist.append(dlib_left_dist)
                    self.dlib_right_dist.append(dlib_right_dist)
                    self.leftfound.append(left_found)
                    self.rightfound.append(right_found)
                    # print(len(left_eye_col))
                    # left_eye_col_r_img, left_eye_col_g_img, left_eye_col_b_img = left_eye_col
                    # self.left_eye_col_r1.append(left_eye_col_r_img)
                    # self.left_eye_col_g2.append(left_eye_col_g_img)
                    # self.left_eye_col_b3.append(left_eye_col_b_img)

                    # right_eye_col_r_img, right_eye_col_g_img, right_eye_col_b_img = right_eye_col

                    # self.right_eye_col_r1.append(right_eye_col_r_img)
                    # self.right_eye_col_g2.append(right_eye_col_g_img)
                    # self.right_eye_col_b3.append(right_eye_col_b_img)

                    if len(self.de_Filename) != len(self.prediction):
                        l = len(self.de_Filename)
                        self.prediction = self.prediction[:l]

        self.results['de_FileName'] = self.de_Filename
        self.results['Right Found'] = self.rightfound
        self.results['Right_Eye_Distance'] = self.rightEyeDist
        # self.results['DLIB Right Eye Distance'] = self.dlib_right_dist
        self.results['Right_Iris_Ratio'] = self.RightIrisRatio
        self.results['Left Found'] = self.leftfound
        self.results['Left_Eye_Distance'] = self.LeftEyeDist
        # self.results['DLIB Left Eye Distance'] = self.dlib_left_dist
        self.results['Left_Iris_Ratio'] = self.LeftIrisRatio
        # self.results['Prediction'] = self.prediction
        # self.results['Iris Prediction'] = self.iris_prediction
        self.results['Prediction'] = self.combinedprediction
        self.results['Ground_Truth_Prediction'] = self.GroundTruthPred
        # print(self.left_eye_col[6, 10])
        # print(self.left_eye_col[10, 6])
        # print(len(self.left_eye_col_r1), self.left_eye_col_r1[:])

        # self.left_eye_col_r1 = np.array(self.left_eye_col_r1).transpose()
        # self.left_eye_col_g2 = np.array(self.left_eye_col_g2).transpose()
        # self.left_eye_col_b3 = np.array(self.left_eye_col_b3).transpose()

        # self.right_eye_col_r1 = np.array(self.right_eye_col_r1).transpose()
        # self.right_eye_col_g2 = np.array(self.right_eye_col_g2).transpose()
        # self.right_eye_col_b3 = np.array(self.right_eye_col_b3).transpose()

        utils.save_results(self.results)


if __name__ == "__main__":
    dem = DroopyEyeMetrics()
    dem.getmetrics()
