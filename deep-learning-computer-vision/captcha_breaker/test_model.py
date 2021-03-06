"""
python3 test_model.py --input downloads --model output/lenet.hdf5
"""
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-m", "--model", required=True, help="path to input model")
args = vars(ap.parse_args())


# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
# randomly sample a few of the input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

# loop over the image paths
for imagePath in imagePaths:
    # load the image and convert it to grayscale, then pad the image
    # to ensure digits caught only the border of the image are retained
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.inRange(
        image,
        # np.array([34, 53, 80]),
        np.array([36, 55, 82]),
        np.array([255, 255, 255])
    )
    gray = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    # threshold the image to reveal the digits
    # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_not(thresh)
    # find contours in the image, keeping only the four largest ones,
    # then sort them from left-to-right
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = contours.sort_contours(cnts)[0]
    # initialize the output image as a "grayscale" image with 3
    # channels along with the output predictions
    output = cv2.merge([gray] * 3)
    predictions = []
    # loop over the contours
    for c in cnts:
        # compute the bounding box for the contour then extract the digit
        (x, y, w, h) = cv2.boundingRect(c)
        roi = gray[y - 5: y + h + 5, x - 5: x + w + 5]
        # pre-process the ROI and classify it then classify it
        roi = preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0]
        predictions.append(str(pred))
        # draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(
            output,
            str(pred),
            (x - 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    # show the output image
    print("[INFO] captcha: {}".format("".join(predictions)))
    cv2.imshow("Output", imutils.resize(output, height=128))
    cv2.waitKey(0)
