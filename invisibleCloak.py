import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(3)
background=0

for i in range(30):
    ret,background = cap.read()
    height_b, width_b, _ = background.shape
    background = cv2.resize(background, (int(width_b * (860 / height_b)), 860))
background = np.flip(background,axis=1)


while(cap.isOpened()):
    ret, img = cap.read()

    # Flip the image
    img = np.flip(img, axis = 1)
    height, width, _ = img.shape

    img = cv2.resize(img, (int(width * (860 / height)), 860))
    # Convert the image to HSV color space.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (35, 35), 0)

    # Defining lower range for red color detection.
    lower = np.array([0,0,0])
    upper = np.array([0,0,0])
    mask1 = cv2.inRange(hsv, lower, upper)

    # Defining upper range for red color detection
    lower_red = np.array([90, 50, 70])
    upper_red = np.array([130, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Addition of the two masks to generate the final mask.
    mask = mask1 + mask2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    # Replacing pixels corresponding to cloak with the background pixels.
    img[np.where(mask == 255)] = background[np.where(mask == 255)]
    cv2.imshow('Display',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
