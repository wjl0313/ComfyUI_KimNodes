import cv2
import numpy as np

def adjust_natural_saturation(image, 自然饱和度):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.addWeighted(s, 自然饱和度, s, 0, 0)
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image
