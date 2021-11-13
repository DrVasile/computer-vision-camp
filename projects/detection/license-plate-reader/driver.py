import argparse
import cv2
import imutils

from imutils import paths
from lpr import LicensePlateReader



def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


arguments = argparse.ArgumentParser()

arguments.add_argument("-i",
                       "--input",
                       required=True,
                       help="Path to the input directory.")

arguments.add_argument("-c",
                       "--clear-border",
                       type=int,
                       default=-1,
                       help="Clear border pixels before OCR.")

arguments.add_argument("-p",
                       "--psm",
                       type=int,
                       default=7,
                       help="The PSM mode for the OCR.")

arguments.add_argument("-d",
                       "--debug",
                       type=int,
                       default=-1,
                       help="Debug mode on(1) or off(-1).")

args = vars(arguments.parse_args())

lpr_instance = LicensePlateReader(debugMode=args["debug"] > 0)
imagePaths = sorted(list(paths.list_images(args["input"])))

for path in imagePaths:
    image = cv2.imread(path)
    image = imutils.resize(image, width=600)
    (licensePlateText, licensePlateContour) = lpr_instance.find_and_extract_text(image, psm=args["psm"], clearBorder=args["clear_border"] > 0)

    if licensePlateText is not None and licensePlateContour is not None:
        box = cv2.boxPoints(cv2.minAreaRect(licensePlateContour))
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

        (x, y, w, h) = cv2.boundingRect(licensePlateContour)
        cv2.putText(image, cleanup_text(licensePlateText), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        print("[INFO] {}".format(licensePlateText))
        cv2.imshow("Output Image", image)
        cv2.waitKey(0)   


# Run with this: python ocr_license_plate.py --input license_plates/group1