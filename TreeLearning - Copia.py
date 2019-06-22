import cv2
import matplotlib.pyplot as plt
import numpy as np

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
len(flags)
flags[40]
nemo = cv2.imread('teste.jpeg')
plt.imshow(nemo)
plt.show()
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
plt.imshow(nemo)
plt.show()
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
light_orange = (234, 240, 226)
dark_orange = (255, 255, 255)
mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
result = cv2.bitwise_and(nemo, nemo, mask=mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
dark_white = (66, 102, 134)
light_white = (135, 155, 191)
mask_white = cv2.inRange(hsv_nemo, light_white, dark_white)
result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)

plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()
final_mask = mask + mask_white

final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)
plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()
blur = cv2.GaussianBlur(final_result, (7, 7), 0)
plt.imshow(blur)
plt.show()
