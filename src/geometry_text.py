import cv2
import numpy as np
import easyocr
from matplotlib import pyplot as plt

img = cv2.imread("data/multi-stop-multi-lang.png")
reader = easyocr.Reader(["en", "th"])

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

stop_cascade = cv2.CascadeClassifier("classifiers/stop_data.xml")

found = stop_cascade.detectMultiScale(img_gray, minSize=(20, 20))

print(f"Found: {len(found)} target objects")

# Draw rectangles around detected object
for x, y, w, h in found:
    # Crop the image
    roi = img[y : y + h, x : x + w]

    # OCR on cropped region
    result = reader.readtext(roi)

    for bbox, text, conf in result:
        print(f"Detected text: {text}, Confidence: {conf:.2f}")
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# cv2.imshow("Detected Stops", img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.show()
