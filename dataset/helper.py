from skimage import io
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

import pickle
import numpy as np
import pandas as pd
import random

from sklearn.naive_bayes import MultinomialNB  # GaussianNB


# given a list of filenames return s a dictionary of images
def getfiles(filenames):
    dir_files = {}
    for x in filenames:
        dir_files[x] = io.imread(x)
    return dir_files


# return hog of a particular image vector
def convertToGrayToHOG(imgVector):
    rgbImage = rgb2gray(imgVector)
    return hog(rgbImage)


# takes returns cropped image
def crop(img, x1, x2, y1, y2):
    crp = img[y1:y2, x1:x2]
    crp = resize(crp, ((128, 128)))  # resize
    return crp


# save classifier
def dumpclassifier(filename, model):
    with open(filename, 'wb') as fid:
        pickle.dump(model, fid)


# load classifier
def loadClassifier(picklefile):
    fd = open(picklefile, 'r+')
    model = pickle.load(fd)
    fd.close()
    return model


"""
This function randomly generates bounding boxes
Return: hog vector of those cropped bounding boxes along with label
Label : 1 if hand ,0 otherwise
"""


def buildhandnothand_lis(frame, imgset):
    poslis = []
    neglis = []

    for nameimg in frame.image:
        tupl = frame[frame['image'] == nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0

        dic = [0, 0]

        arg1 = [x_tl, y_tl, conf, side, side]
        poslis.append(convertToGrayToHOG(crop(imgset[nameimg], x_tl, x_tl+side, y_tl, y_tl+side)))
        while dic[0] <= 1 or dic[1] < 1:
            x = random.randint(0, 320-side)
            y = random.randint(0, 240-side)
            crp = crop(imgset[nameimg], x, x+side, y, y+side)
            hogv = convertToGrayToHOG(crp)
            arg2 = [x, y, conf, side, side]

            z = overlapping_area(arg1, arg2)
            if dic[0] <= 1 and z <= 0.5:
                neglis.append(hogv)
                dic[0] += 1
            if dic[0] == 1:
                break
    label_1 = [1 for i in range(0, len(poslis))]
    label_0 = [0 for i in range(0, len(neglis))]
    label_1.extend(label_0)
    poslis.extend(neglis)
    return poslis, label_1


# returns imageset and bounding box for a list of users
def train_binary(train_list, data_directory):
    frame = pd.DataFrame()
    list_ = []
    for user in train_list:
        list_.append(pd.read_csv(data_directory+user+'/'+user+'_loc.csv', index_col=None, header=0))
    frame = pd.concat(list_)
    frame['side'] = frame['bottom_right_x']-frame['top_left_x']
    frame['hand'] = 1

    imageset = getfiles(frame.image.unique())

    # returns actual images and dataframe
    return imageset, frame


# loads data for binary classification (hand/not-hand)
def load_binary_data(user_list, data_directory):
    data1, df = train_binary(user_list, data_directory)  # data 1 - actual images , df is actual bounding box

    # third return, i.e., z is a list of hog vecs, labels
    z = buildhandnothand_lis(df, data1)
    return data1, df, z[0], z[1]


# loads data for multiclass
def get_data(user_list, img_dict, data_directory):
    X = []
    Y = []

    for user in user_list:
        boundingbox_df = pd.read_csv(data_directory+user+'/'+user+'_loc.csv')

        for rows in boundingbox_df.iterrows():
            cropped_img = crop(img_dict[rows[1]['image']],
                               rows[1]['top_left_x'],
                               rows[1]['bottom_right_x'],
                               rows[1]['top_left_y'],
                               rows[1]['bottom_right_y'])
            hogvector = convertToGrayToHOG(cropped_img)
            X.append(hogvector.tolist())
            Y.append(rows[1]['image'].split('/')[1][0])
    return X, Y


# utility funtcion to compute area of overlap
def overlapping_area(detection_1, detection_2):
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


"""
Does hard negative mining and returns list of hog vectos , label list and no_of_false_positives after sliding
"""


def do_hardNegativeMining(cached_window, frame, imgset, model, step_x, step_y):
    lis = []
    no_of_false_positives = 0
    for nameimg in frame.image:
        tupl = frame[frame['image'] == nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0

        arg1 = [x_tl, y_tl, conf, side, side]
        for x in range(0, 320-side, step_x):
            for y in range(0, 240-side, step_y):
                arg2 = [x, y, conf, side, side]
                z = overlapping_area(arg1, arg2)

                prediction = model.predict([cached_window[str(nameimg)+str(x)+str(y)]])[0]

                if prediction == 1 and z <= 0.5:
                    lis.append(cached_window[str(nameimg)+str(x)+str(y)])
                    no_of_false_positives += 1

    label = [0 for i in range(0, len(lis))]
    return lis, label, no_of_false_positives


"""
Modifying to cache image values before hand so as to not redo that again and again
"""


def cacheSteps(imgset, frame, step_x, step_y):
    # print "Cache-ing steps"
    dic = {}
    i = 0
    for img in frame.image:
        tupl = frame[frame['image'] == img].values[0]
        side = tupl[5]
        i += 1
        # if i%10 == 0:
        #     print "{0} images cached ".format(i)
        imaage = imgset[img]
        for x in range(0, 320-side, step_x):
            for y in range(0, 240-side, step_y):
                dic[str(img+str(x)+str(y))] = convertToGrayToHOG(crop(imaage, x, x+side, y, y+side))
    return dic


# frame - bounding boxes-df; yn_df - yes_or_no df
def improve_Classifier_using_HNM(hog_list, label_list, frame, imgset, threshold=50, max_iterations=25):
    # print "Performing HNM :"
    no_of_false_positives = 1000000     # Initialise to some random high value
    i = 0

    step_x = 32
    step_y = 24

    mnb = MultinomialNB()
    cached_wind = cacheSteps(imgset, frame, step_x, step_y)

    while True:
        i += 1
        model = mnb.partial_fit(hog_list, label_list, classes=[0, 1])

        ret = do_hardNegativeMining(cached_wind, frame, imgset, model, step_x=step_x, step_y=step_y)

        hog_list = ret[0]
        label_list = ret[1]
        no_of_false_positives = ret[2]

        if no_of_false_positives == 0:
            return model

        print "Iteration {0} - No_of_false_positives: {1}".format(i, no_of_false_positives)

        if no_of_false_positives <= threshold:
            return model

        if i > max_iterations:
            return model


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # print "Perfmorinf NMS:"
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# Returns the tuple with the highest prediction probability of hand
def image_pyramid_step(model, img, scale=1.0):
    max_confidence_seen = -1
    rescaled_img = rescale(img, scale)
    detected_box = []
    side = 128
    x_border = rescaled_img.shape[1]
    y_border = rescaled_img.shape[0]

    for x in range(0, x_border-side, 32):
        for y in range(0, y_border-side, 24):
            cropped_img = crop(rescaled_img, x, x+side, y, y+side)
            hogvector = convertToGrayToHOG(cropped_img)

            confidence = model.predict_proba([hogvector])

            if confidence[0][1] > max_confidence_seen:
                detected_box = [x, y, confidence[0][1], scale]
                max_confidence_seen = confidence[0][1]

    return detected_box
