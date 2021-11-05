import cv2
import imutils
import numpy
import pytesseract

from skimage.segmentation import clear_border



class LicensePlateReader:
    
    
    def __init__(self, minAspectRatio=4, maxAspectRatio=5, debugMode=False):
        
        self.minAspectRatio = minAspectRatio # The minimum aspect ratio used to detect and filter license plates.
        self.maxAspectRatio = maxAspectRatio # The maximum ...                  
        self.debugMode = debugMode           # A flag value used to display intermediate results.
        
        
    def debug_imshow(self, title, image, waitKey=False):
        
        if self.debugMode:
            cv2.imshow(title, image)
            
            if waitKey:
                cv2.waitKey(0)
                
                
    def get_license_plate_candidate_regions(self, grayImage, contoursCnt=5):
        
        rectangleKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackHat = cv2.morphologyEx(grayImage, cv2.MORPH_BLACKHAT, rectangleKernel)
        self.debug_imshow("Black Hat", blackHat)
        
        squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphCloseImage = cv2.morphologyEx(grayImage, cv2.MORPH_CLOSE, squareKernel)
        thresholdImage = cv2.threshold(morphCloseImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Black Hat", thresholdImage)
        
        gradientX = cv2.Sobel(blackHat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradientX = numpy.absolute(gradientX)
        
        minValue = numpy.min(gradientX)
        maxValue = numpy.max(gradientX)
        
        gradientX = 255 * ((gradientX - minValue) / (maxValue - minValue))
        gradientX = gradientX.asType("uint8")
        self.debug_imshow("Scharr", gradientX)
        
        gradientX = cv2.GaussianBlur(gradientX, (5, 5), 0)
        gradientX = cv2.morphologyEx(gradientX, cv2.MORPH_CLOSE, rectangleKernel)
        thresholdGradient = cv2.threshold(gradientX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Gradient Threshold", thresholdGradient)

        thresholdGradient = cv2.erode(thresholdGradient, None, iterations=2)
        thresholdGradient = cv2.dilate(thresholdGradient, None, iterations=2)
        self.debug_imshow("Gradient Erode / Dilate", thresholdGradient)

        thresholdGradient = cv2.bitwise_and(thresholdGradient, thresholdGradient, mask=thresholdImage)
        thresholdGradient = cv2.dilate(thresholdGradient, None, iterations=2)
        thresholdGradient = cv2.erode(thresholdGradient, None, iterations=2)
        self.debug_imshow("Final", thresholdGradient, waitKey=True)

        contours = cv2.findContours(thresholdGradient.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:contoursCnt]

        return contours


    def locate_license_plate(self, grayImage, candidates, clearBorder=False):
        licensePlateContour = None
        regionOfInterest = None

        for candidate in candidates:
            (x, y, w, h) = cv2.boundingRect(candidate)
            aspectRatio = w / float(h)

            if aspectRatio >= self.minAspectRatio and aspectRatio <= self.maxAspectRatio:
                licensePlateContour = candidate
                licensePlate = grayImage[y:y+h, x:x+w]
                regionOfInterest = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            if clearBorder:
                regionOfInterest = clear_border(regionOfInterest)

            self.debug_imshow("License Plate", licensePlate)
            self.debug_imshow("Region Of Interest", regionOfInterest, waitKey=True)
            break;
        
        return (regionOfInterest, licensePlateContour)