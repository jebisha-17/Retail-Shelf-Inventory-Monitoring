import cv2
import numpy as np

# Load the input image
image = cv2.imread("traffic_light.jpg")

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges for red, yellow, and green lights
red_lower, red_upper = np.array([0, 120, 70]), np.array([10, 255, 255])
yellow_lower, yellow_upper = np.array([20, 100, 100]), np.array([30, 255, 255])
green_lower, green_upper = np.array([40, 50, 50]), np.array([90, 255, 255])

# Create masks for each color
red_mask = cv2.inRange(hsv, red_lower, red_upper)
yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
green_mask = cv2.inRange(hsv, green_lower, green_upper)

# Find contours for each color
red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around detected lights
if red_contours:
    x, y, w, h = cv2.boundingRect(max(red_contours, key=cv2.contourArea))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(image, "Red", (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

if yellow_contours:
    x, y, w, h = cv2.boundingRect(max(yellow_contours, key=cv2.contourArea))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(image, "Yellow", (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

if green_contours:
    x, y, w, h = cv2.boundingRect(max(green_contours, key=cv2.contourArea))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, "Green", (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Save and display result
cv2.imwrite("output_image.jpg", image)
cv2.imshow("Traffic Light Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()