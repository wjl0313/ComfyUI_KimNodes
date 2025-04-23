import cv2
import numpy as np
import math

def apply_dehaze(image, DCP暗通道先验):
    """去雾处理，接受暗通道先验参数来调整去雾强度。"""
    if DCP暗通道先验 > 0:
        bilateral_radius = 76 # 调整外发光
        im = image.astype('float64') / 255
        dark = dark_channel(im, 15)
        A = atm_light_advanced(im, dark)
        te = transmission_estimate_advanced(im, A, DCP暗通道先验, 15)
        t = transmission_refine_advanced(im, te, bilateral_radius, 0.0001)
        min_transmission = 0.21 - 0.1 * (DCP暗通道先验 * 2)
        print("min_transmission", min_transmission)
        t = np.clip(t, min_transmission, 1)
        image = recover(im, t, A, min_transmission)
        return image
    else:
        return image

def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark_channel = cv2.erode(dc, kernel)
    return dark_channel

def atm_light_advanced(im, dark_channel):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark_channel.reshape(imsz)
    imvec = im.reshape(imsz, 3)
    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]
    atmsum = np.zeros([1, 3])
    for ind in indices:
        atmsum += imvec[ind]
    A = atmsum / numpx
    return A

def transmission_estimate_advanced(im, A, DCP暗通道先验, sz):
    omega = 0.4 * DCP暗通道先验
    im3 = np.empty(im.shape, im.dtype)
    for i in range(3):
        im3[:, :, i] = im[:, :, i] / A[0, i]
    transmission = 1 - omega * dark_channel(im3, sz)
    return transmission

def transmission_refine_advanced(im, et, radius, eps):
    I = im.astype(np.float32) / 255.0
    p = et.astype(np.float32) / 255.0
    ones_array = np.ones(I.shape[:2], I.dtype)
    N = cv2.boxFilter(ones_array, cv2.CV_32F, (radius, radius), borderType=cv2.BORDER_REFLECT)
    mean_a = np.zeros_like(I)
    mean_b = np.zeros_like(I[:, :, 0])
    for i in range(3):
        mean_I = cv2.boxFilter(I[:, :, i], cv2.CV_32F, (radius, radius), borderType=cv2.BORDER_REFLECT) / N
        mean_p = cv2.boxFilter(p, cv2.CV_32F, (radius, radius), borderType=cv2.BORDER_REFLECT) / N
        mean_Ip = cv2.boxFilter(I[:, :, i] * p, cv2.CV_32F, (radius, radius), borderType=cv2.BORDER_REFLECT) / N
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = cv2.boxFilter(I[:, :, i] * I[:, :, i], cv2.CV_32F, (radius, radius), borderType=cv2.BORDER_REFLECT) / N - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a[:, :, i] = cv2.boxFilter(a, cv2.CV_32F, (radius, radius), borderType=cv2.BORDER_REFLECT) / N
        mean_b += cv2.boxFilter(b, cv2.CV_32F, (radius, radius), borderType=cv2.BORDER_REFLECT) / N
    mean_b /= 3
    q = np.sum(mean_a * I, axis=2) + mean_b
    q = np.clip(q * 255.0, 0, 255)
    return q.astype(np.float32) 

def recover(im, t, A, t0):
    t = np.maximum(t, t0)
    adjust_atmos = A * 1  # 调整大气光强度，让暗部稍微亮一些
    recovered = np.empty_like(im)
    for i in range(3): 
        recovered[:, :, i] = (im[:, :, i] - adjust_atmos[0, i]) / t + adjust_atmos[0, i]
    recovered = np.clip(recovered * 255, 0, 255).astype(np.uint8)
    return recovered