import numpy as np

from GestureRecognizer import GestureRecognizer as gr
from PIL import Image, ImageDraw
from os import listdir


def main():
    """
    Tests the classifier using images from the test_images folder and saves the
    images with drawn bounding boxes on output
    """
    new_gr = gr.load_model(name="sign_detector.pkl.gz")  # automatic dict unpacking
    # new_gr = gr('/home/vsant/DevProjects/Sign-Language-and-Static-gesture-recognition-using-sklearn/dataset/',
    #             handDetector, signDetector)
    print "Model loaded"
    test_folder = "test_images/"
    for img in listdir(test_folder):
        test_img = Image.open(test_folder + img)
        test_img = test_img.resize((320, 240), Image.ANTIALIAS)
        img_data = np.asarray(test_img, dtype="int32")
        print "IMG processed"

        pos, pred = new_gr.recognize_gesture(img_data)
        print "Signal predicted"

        draw = ImageDraw.Draw(test_img)
        draw.rectangle(((pos[0], pos[1]), (pos[2], pos[3])), fill=None)
        test_img.save("output/" + img, "JPEG")

        print pos, pred


if __name__ == '__main__':
    main()
