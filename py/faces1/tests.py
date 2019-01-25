import unittest
import cv2
import numpy as np

import common

class TestCommon(unittest.TestCase):
   
    def testCutImage(self):
        original_image_name = './images/test/Rubens_Barrichello_0010.jpg'
        test_image_name = './images/test/Rubens_Barrichello_0010_cutted.jpg'
        original_image = cv2.imread(original_image_name)
        test_image = cv2.imread(test_image_name)
        rect = [69, 67, 116, 116]
        res = common.cutImage(original_image, rect)
        self.assertLess(np.median(abs(res - test_image)), 3.0)

if __name__ == '__main__':
    unittest.main()
