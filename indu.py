import cv2
import numpy as np


image = cv2.imread("underwater.jpg")  
image = cv2.resize(image, (640, 480))


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


lower_bound = np.array([10, 100, 100])
upper_bound = np.array([30, 255, 255])


mask = cv2.inRange(hsv, lower_bound, upper_bound)


result = cv2.bitwise_and(image, image, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500: 
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Detected Object", result)
cv2.waitKey(0)
cv2.destroyAllWindows()