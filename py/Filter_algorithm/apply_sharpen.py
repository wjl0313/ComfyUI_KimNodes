import cv2
import numpy as np

def apply_sharpen(image, UM非锐化掩蔽):
    if UM非锐化掩蔽 > 0:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        unsharp_image = cv2.addWeighted(image, 2, blurred, -1, 0)
        alpha = UM非锐化掩蔽
        image = cv2.addWeighted(image, 1 - alpha, unsharp_image, alpha + 0.1, 0)
        brightness_adjust_factor = max(0.96 - 0.16 * (UM非锐化掩蔽 / 4), 0.88)
        image = np.clip(image * brightness_adjust_factor, 0, 255).astype(np.uint8)
    return image
