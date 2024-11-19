import cv2
import numpy as np

class Utils:

    @staticmethod
    def select_areas(image):
        im = cv2.imread(image)

        roi1 = cv2.selectROI('Select left area', im, fromCenter=False)
        x, y, w, h = roi1
        area_left = np.s_[y:y + h, x:x + w]

        roi2 = cv2.selectROI('Select right area', im, fromCenter=False)
        x, y, w, h = roi2
        area_right = np.s_[y:y + h, x:x + w]

        cv2.destroyAllWindows()

        return area_left, area_right