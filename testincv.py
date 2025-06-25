import cv2
import numpy as np

img = 255 * np.ones((400, 400, 3), dtype=np.uint8)
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
