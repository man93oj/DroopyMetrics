import os
import cv2
import shutil
import utils


class DroopyEyeMetrics:
    def __init__(self):
        # self.INPUT_DIR = os.path.join(os.path.join("data", "input"), "train")
        # self.OUTPUT_DIR = os.path.join(os.path.join("data", "output"), "train")
        self.INPUT_DIR = os.path.join(os.path.join("data", "input"), "test_pristine")
        self.OUTPUT_DIR = os.path.join(os.path.join("data", "output"), "test_pristine")
        # self.INPUT_DIR = os.path.join(os.path.join("data", "input"), "test_cropped")
        # self.OUTPUT_DIR = os.path.join(os.path.join("data", "output"), "test_cropped")
        # self.INPUT_DIR = os.path.join(os.path.join("data", "input"), "val")
        # self.OUTPUT_DIR = os.path.join(os.path.join("data", "output"), "val")    
        # self.INPUT_DIR = os.path.join(os.path.join("data", "input"), "val_manoj_abdullah")
        # self.OUTPUT_DIR = os.path.join(os.path.join("data", "output"), "val_manoj_abdullah")       
        self.results = {}
        self.de_Filename = []
        self.clrfound = []
        self.MRD1 = []
        self.IrisRatio = []
        self.MRD1_pred = []
        self.IR_pred = []
        self.GroundTruthPred = []

        if os.path.exists("__pycache__"):
            shutil.rmtree("__pycache__")
        # if os.path.exists(self.OUTPUT_DIR):
        #     shutil.rmtree(self.OUTPUT_DIR)

        if not os.path.exists(os.path.join(self.OUTPUT_DIR, "noptosis")):
            os.makedirs(os.path.join(self.OUTPUT_DIR, "noptosis"))
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, "ptosis")):
            os.makedirs(os.path.join(self.OUTPUT_DIR, "ptosis"))

    def getmetrics(self):
        for dirName, _, fileList in os.walk(self.INPUT_DIR):
            for image_name in fileList:
                image_path = os.path.join(dirName, image_name)
                print(image_path)
                if os.path.isfile(image_path) and image_name.split('.')[1].lower() in ('png', 'jpg', 'jpeg') and not 'droopy' in dirName: 
                    image = cv2.imread(image_path)
                    output_dirName_path = dirName.replace("input", "output")

                    SegmentEyeResults = utils.Segment_Eye(image)

                    iris_img = SegmentEyeResults[0]
                    iris_radius = SegmentEyeResults[1]
                    iris_ratio = SegmentEyeResults[2]

                    MRD1 = 0
                    if iris_radius:
                        pix2mm = 4.75 / iris_radius
                        MRD1 = SegmentEyeResults[3]*pix2mm

                    print('in main1 49', SegmentEyeResults[3], MRD1)
                    clr_found = SegmentEyeResults[4]
                    clr_img = SegmentEyeResults[5]

                    mrd1_pred = 0
                    ir_pred = 0
                    # if (iris_ratio < 0.89 and iris_ratio < 0.89) or (MRD1 < 3.267 and MRD1 > 9.0) or not clr_found:
                    if clr_found:
                        if (MRD1 < 3.267 and MRD1 > 9.0):
                            mrd1_pred = 1
                            print("Ptosis: ", mrd1_pred)
                        elif (iris_ratio < 0.89 and iris_ratio < 0.89) or not clr_found:
                            ir_pred = 1
                            print("Ptosis: ", ir_pred)
                    elif not clr_found:
                        print("Ptosis: ", clr_found)
                        mrd1_pred = 1
                        ir_pred = 1
                        
                    groundtruth = 1

                    if 'noptosis' in dirName:
                        groundtruth = 0

                    self.de_Filename.append(image_name.split('/')[-1])
                    self.clrfound.append(int(not clr_found))
                    self.MRD1.append(MRD1)
                    self.MRD1_pred.append(mrd1_pred)
                    self.IrisRatio.append(iris_ratio)
                    self.IR_pred.append(ir_pred)
                    self.GroundTruthPred.append(groundtruth)

                    cv2.imwrite(os.path.join(output_dirName_path, image_name.split(
                        '.')[0]+str('_IRIS.png')), iris_img)
                    cv2.imwrite(os.path.join(output_dirName_path, image_name.split(
                        '.')[0]+str('_CLR.png')), clr_img)

        self.results['de_FileName'] = self.de_Filename
        self.results['MRD1'] = self.MRD1
        self.results['Iris_Ratio'] = self.IrisRatio
        self.results['CLR Found'] = self.clrfound
        self.results['MRD1_Pred'] = self.MRD1_pred
        self.results['IR_pred'] = self.IR_pred
        self.results['Ground_Truth'] = self.GroundTruthPred

        utils.save_results(self.OUTPUT_DIR, self.results)


if __name__ == "__main__":
    dem = DroopyEyeMetrics()
    dem.getmetrics()
