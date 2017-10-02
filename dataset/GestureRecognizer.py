import pickle
import gzip
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from helper import load_binary_data, get_data, crop, convertToGrayToHOG
from helper import improve_Classifier_using_HNM, non_max_suppression_fast
from helper import dumpclassifier, image_pyramid_step


class GestureRecognizer(object):
    """class to perform gesture recognition"""

    def __init__(self, data_dir, hand_Detector=None, sign_Detector=None):
        self.data_directory = data_dir
        self.handDetector = hand_Detector
        self.signDetector = sign_Detector
        self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F',
                                                 'G', 'H', 'I', 'K', 'L', 'M',
                                                 'N', 'O', 'P', 'Q', 'R', 'S',
                                                 'T', 'U', 'V', 'W', 'X', 'Y'])

    def train(self, train_list):
        """
            train_list : list of users to use for training
            eg ["user_1", "user_2", "user_3"]
            The train function should train all your classifiers
            both binary and multiclass on the given list of users
        """
        print "Train starts"
        # Load data for the binary (hand/not hand) classification task
        imageset, boundbox, hog_list, label_list = load_binary_data(train_list, self.data_directory)

        print "Imageset, boundbox, hog_list,label_list Loaded!"

        # Load data for the multiclass classification task
        X_mul, Y_mul = get_data(train_list, imageset, self.data_directory)

        print "Multiclass data loaded"

        Y_mul = self.label_encoder.fit_transform(Y_mul)

        if self.handDetector is None:
            # Build binary classifier for hand-nothand classification
            self.handDetector = improve_Classifier_using_HNM(
                                    hog_list, label_list, boundbox, imageset,
                                    threshold=10, max_iterations=200)

        print "handDetector trained "

        # Multiclass classification part to classify the various signs/hand gestures CHECK. TODO.

        if self.signDetector is None:
            svcmodel = SVC(kernel='linear', C=0.9, probability=True)
            self.signDetector = svcmodel.fit(X_mul, Y_mul)

        print "sign Detector trained "

        dumpclassifier('handDetector.pkl', self.handDetector)

        dumpclassifier('signDetector.pkl', self.signDetector)

        dumpclassifier('label_encoder.pkl', self.label_encoder)

    def recognize_gesture(self, image):
        """
            image : a 320x240 pixel RGB image in the form of a numpy array

            This function should locate the hand and classify the gesture.
            returns : (position, label)

            position : a tuple of (x1,y1,x2,y2) coordinates of bounding box
                x1,y1 is top left corner, x2,y2 is bottom right

            label : a single character. eg 'A' or 'B'
        """
        # print "In recognize_gesture"
        scales = [1.25,
                  1.015625,
                  0.78125,
                  0.546875,
                  1.5625,
                  1.328125,
                  1.09375,
                  0.859375,
                  0.625,
                  1.40625,
                  1.171875,
                  0.9375,
                  0.703125,
                  1.71875,
                  1.484375]

        detectedBoxes = []  # [x,y,conf,scale]
        for sc in scales:
            detectedBoxes.append(image_pyramid_step(
                                    self.handDetector, image, scale=sc))

        side = [0 for i in xrange(len(scales))]
        for i in xrange(len(scales)):
            side[i] = 128 / scales[i]

        for i in xrange(len(detectedBoxes)):
            detectedBoxes[i][0] = detectedBoxes[i][0] / scales[i]  # x
            detectedBoxes[i][1] = detectedBoxes[i][1] / scales[i]  # y

        nms_lis = []  # [x1,x2,y1,y2]
        for i in xrange(len(detectedBoxes)):
            nms_lis.append([detectedBoxes[i][0],
                            detectedBoxes[i][1],
                            detectedBoxes[i][0] + side[i],
                            detectedBoxes[i][1] + side[i],
                            detectedBoxes[i][2]])
        nms_lis = np.array(nms_lis)

        res = non_max_suppression_fast(nms_lis, 0.4)

        output_det = res[0]
        x_top = output_det[0]
        y_top = output_det[1]
        side = output_det[2]-output_det[0]
        position = [x_top, y_top, x_top+side, y_top+side]

        croppedImage = crop(image, x_top, x_top+side, y_top, y_top+side)
        hogvec = convertToGrayToHOG(croppedImage)

        prediction = self.signDetector.predict_proba([hogvec])[0]

        zi = zip(self.signDetector.classes_, prediction)
        zi.sort(key=lambda x: x[1], reverse=True)

        # To return the top 5 predictions
        final_prediction = []
        for i in range(5):
            final_prediction.append(
                self.label_encoder.inverse_transform(zi[i][0]))
        # print position,final_prediction

        return position, final_prediction

    def save_model(self, **params):

        """
            save your GestureRecognizer to disk.
        """

        self.version = params['version']
        self.author = params['author']

        file_name = params['name']

        pickle.dump(self, gzip.open(file_name, 'wb'))
        # We are using gzip to compress the file
        # If you feel compression is not needed, kindly take lite

    @staticmethod       # similar to static method in Java
    def load_model(**params):
        """
            Returns a saved instance of GestureRecognizer.
            load your trained GestureRecognizer from disk with provided params
            Read - http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-parameters
        """

        file_name = params['name']
        return pickle.load(gzip.open(file_name, 'rb'))

        # People using deep learning need to reinitalize model, load weights here etc.
