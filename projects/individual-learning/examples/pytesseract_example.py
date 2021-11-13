import argparse
import cv2
import os
import pytesseract

from PIL import Image

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--image", required=True, help="Image path")
arg_parser.add_argument("-p", "--preprocess", type=str, default="thresh", help="Type of preprocessing")
args = vars(arg_parser.parse_args())

image = cv2.imread(args["image"])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if args["preprocess"] == "thresh":
	gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif args["preprocess"] == "blur":
	gray_image = cv2.medianBlur(gray_image, 3)

filename = "{}.png".format(os.getpid())

cv2.imwrite(filename, gray_image)

text = pytesseract.image_to_string(Image.open(filename))

os.remove(filename)

print("The text: " + text)

cv2.imshow("Image", image)
cv2.imshow("Output", gray_image)
cv2.waitKey(0)