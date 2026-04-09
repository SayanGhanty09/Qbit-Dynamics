import cv2
import numpy as np

frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame[100:480, 300:340] = (255, 255, 255)
# Make the red brighter (B=0, G=0, R=255) -> grayscale is 76
# OpenCV grayscale coeff: R*0.299 + G*0.587 + B*0.114 => 255*0.299 = 76.245
frame[300:360, 100:160] = (0, 0, 255)
# Add a blue obstacle
frame[300:360, 500:560] = (255, 0, 0) # grayscale 29... this is very dark!

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
val, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu threshold val:", val)
print("Red box max:", np.max(gray[300:360, 100:160]))
print("Blue box max:", np.max(gray[300:360, 500:560]))
