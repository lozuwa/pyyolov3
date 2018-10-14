import os
import unittest
import cv2
from pydarknet import *

class pydarknet_test(unittest.TestCase):

	def setUp(self):
		self.path = os.path.join(os.getcwd(), "tests", "people_1.jpg")
		self.image = cv2.imread(self.path)

	def tearDown(self):
		pass

	def test_yolov3(self):
		yolov3 = Yolov3()
		objects = yolov3.findObjects(image=self.image)
		print(objects)

if __name__ == "__main__":
	unittest.main()
