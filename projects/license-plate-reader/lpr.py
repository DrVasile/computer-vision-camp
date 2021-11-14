import cv2
import imutils
import numpy
import pytesseract

from skimage.segmentation import clear_border
from lpr_debbuger import LPRDebugger



# The class that contains the methods for reading license plate text from images.
# As parameters to the constructor it accepts values that correspond to the aspect ratio of rectangular plates.
class LicensePlateReader:
    
    
    def __init__(self, minAspectRatio=4, maxAspectRatio=5, debugModeOn=True):
        
        self.minAspectRatio = minAspectRatio # The minimum aspect ratio used to detect and filter license plates.
        self.maxAspectRatio = maxAspectRatio # The maximum aspect ratio used to detect and filter license plates. 
        self.debugModeOn = debugModeOn       # A flag value used to display intermediate results
        self.debbugger = LPRDebugger()
        

    # The method that contains the image processing pipeline.
    # Also, it finds the contour candidates from the image.
    def find_license_plate_candidate_regions(self, grayImage, contoursCnt=5):
        
        rectangleKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackHatImage = cv2.morphologyEx(grayImage, cv2.MORPH_BLACKHAT, rectangleKernel)
        self.debugger.debug_imshow("Black Hat", blackHatImage)
        
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lightRegions = cv2.morphologyEx(grayImage, cv2.MORPH_CLOSE, squareKernel)
        lightRegions = cv2.threshold(lightRegions, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debugger.debug_imshow("Light Regions", lightRegions)
        
        gradientXImage = cv2.Sobel(blackHatImage, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradientXImage = numpy.absolute(gradientXImage)
        minValue = numpy.min(gradientXImage)
        maxValue = numpy.max(gradientXImage)
        gradientXImage = 255 * ((gradientXImage - minValue) / (maxValue - minValue))
        gradientXImage = gradientXImage.asType("uint8")
        self.debugger.debug_imshow("Scharr Filter on X axis", gradientXImage)
        
        gradientXImage = cv2.GaussianBlur(gradientXImage, (5, 5), 0)
        gradientXImage = cv2.morphologyEx(gradientXImage, cv2.MORPH_CLOSE, rectangleKernel)
        thresholdGradientXImage = cv2.threshold(gradientXImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debugger.debug_imshow("Threshold on the Gradient X image", thresholdGradientXImage)

        # TODO substitute with Open
        thresholdGradientXImage = cv2.erode(thresholdGradientXImage, None, iterations=2)
        thresholdGradientXImage = cv2.dilate(thresholdGradientXImage, None, iterations=2)
        self.debugger.debug_imshow("Threshold GradientX Image after Erode & Dilate", thresholdGradientXImage)

        thresholdGradientXImage = cv2.bitwise_and(thresholdGradientXImage, thresholdGradientXImage, mask=lightRegions)
        # TODO substitute with Close
        thresholdGradientXImage = cv2.dilate(thresholdGradientXImage, None, iterations=2)
        thresholdGradientXImage = cv2.erode(thresholdGradientXImage, None, iterations=2)
        self.debugger.debug_imshow("Final Image", thresholdGradientXImage, waitKey=True)

        contours = cv2.findContours(thresholdGradientXImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:contoursCnt]

        return contours


    def locate_license_plate(self, grayImage, candidates, clearBorder=False):

        licensePlateContour = None
        regionOfInterest = None

        for candidate in candidates:

            (x, y, w, h) = cv2.boundingRect(candidate)
            aspectRatio = float(w) / float(h)

            if aspectRatio >= self.minAspectRatio and aspectRatio <= self.maxAspectRatio:

                licensePlateContour = candidate
                licensePlateSnip = grayImage[y:y + h, x:x + w]
                self.debugger.debug_imshow("License Plate Snip", licensePlateSnip)

                regionOfInterest = cv2.threshold(licensePlateSnip, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                if clearBorder:

                    regionOfInterest = clear_border(regionOfInterest)

                self.debugger.debug_imshow("Region Of Interest", regionOfInterest, waitKey=True)
                break
        
        return (regionOfInterest, licensePlateContour)

    
    def get_tesseract_options(self, psm=7):

        alpha_numeric_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alpha_numeric_chars)
        options += " --psm {}".format(psm)

        return options


    def find_and_extract_text(self, image, psm=7, clearBorder=False):
        
        licensePlateText = None

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.find_license_plate_candidate_regions(grayImage)
        (regionOfInterest, licensePlateContour) = self.locate_license_plate(grayImage, candidates, clearBorder=clearBorder)

        if regionOfInterest is not None:
            options = self.get_tesseract_options(psm=psm)
            licensePlateText = pytesseract.image_to_string(regionOfInterest, config=options)
            self.debugger.debug_imshow("License Plate Region Of Interest", regionOfInterest)

        return (licensePlateText, licensePlateContour)