import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("data/stop.jpg")

# print(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img_rgb)

# plt.show()

# Use trained classifer to detect specific objects
# something like Haar Cascade Classifier
stop_cascade = cv2.CascadeClassifier("classifiers/stop_data.xml")

# Detect objects in the image
# Object less than 20*20 will be rejected
found = stop_cascade.detectMultiScale(img_gray, minSize=(20, 20))

# Draw rectangles around detected object
for x, y, w, h in found:
    # `0,255,0` RGB, here it's green color
    # cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)


# cv2.imshow("Detected Stops", img_rgb)
cv2.imshow("Detected Stops", img)

# wait until a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

# countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# for cnt in countours:
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)

#     box = np.int0(box)
#     width, height = rect[1]

#     angle = rect[2]
#     print(f"Object size: {width}x{height} px, angle: {angle}")
