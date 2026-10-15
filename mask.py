import os
import cv2

import numpy as np

path1= input('face mask image path:')
path2= input('without face mask image path:')
img1 = cv2.imread(path1)  #put face mask image path here
img2 = cv2.imread(path2)  #put without facemask image path here




img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# Ensure the images are aligned first.
diff = cv2.absdiff(img1, img2)

gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

_, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.dilate(mask, kernel, iterations=2)
cv2.imshow('mask',mask)
cv2.imwrite('mask.png',mask)
