import cv2
import numpy as np
from core.utils_f2f.flow_viz import flow_to_image

img1 = cv2.imread(f"demo-frames/4.png")
img2 = cv2.imread(f"demo-frames/3.png")

hsv = np.zeros_like(img1)
hsv[..., 1] = 255

img1_ = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_ = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(img2_, img1_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#
# hsv[..., 0] = ang*180/np.pi/2
# hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
# rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

rgb = flow_to_image(flow)

cat = np.concatenate((img1, rgb), 0)
cv2.imwrite("demo-frames/results/result.png", cat)