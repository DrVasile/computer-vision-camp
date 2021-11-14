import cv2


# The class that contains debug functionality.
class LPRDebugger:


    # The method that implements a debug version for the imshow() method.
    # It is used to debug the image processing pipeline.   
    def debug_imshow(self, title, image, waitKey=False):
        
        if self.debugModeOn:
            cv2.imshow(title, image)
            
            if waitKey:
                cv2.waitKey(0)
