from Classes import AttendanceList
from FaceExtractor import FaceExtractor
from ImageDetector import ImageDetector
import cv2

#extractor = FaceExtractor()
#extractor.extract()

image = cv2.imread("peter.jpg")

img = ImageDetector("Classes")
img.detect(image)

