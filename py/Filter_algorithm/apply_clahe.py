import cv2
import numpy as np

def apply_clahe(image, CLAHE对比度增强限制, clahe_tile_grid_size=(1, 1)):
    if CLAHE对比度增强限制 > 0:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE对比度增强限制, tileGridSize=clahe_tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=CLAHE对比度增强限制, tileGridSize=clahe_tile_grid_size)
        image = clahe.apply(image)
    return image
