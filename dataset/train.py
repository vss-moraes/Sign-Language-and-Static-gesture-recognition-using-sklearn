from GestureRecognizer import GestureRecognizer as gr

import warnings
warnings.filterwarnings("ignore")


def main():
    gs = gr('/home/vsant/DevProjects/Sign-Language-and-Static-gesture-recognition-using-sklearn/dataset/')
    userlist = ['user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_9', 'user_10']

    gs.train(userlist)
    gs.save_model(name="sign_detector.pkl.gz", version="0.0.1", author='ss')

    print "The GestureRecognizer is saved to disk"


if __name__ == '__main__':
    main()
