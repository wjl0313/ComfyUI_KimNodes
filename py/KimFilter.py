import numpy as np
import cv2
import torch
import math

from PIL import ImageEnhance, Image
from .Filter_algorithm.apply_sharpen import apply_sharpen
from .Filter_algorithm.apply_dehaze import apply_dehaze
from .Filter_algorithm.apply_clahe import apply_clahe
from .Filter_algorithm.adjust_natural_saturation import adjust_natural_saturation
from .Filter_algorithm.adjust_gamma import adjust_gamma

class KimFilter:
    """
    一个图像处理节点，对图像应用锐化、去雾效果、CLAHE、自然饱和度调整及伽马调整。
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "description": "上传您想应用高级图像处理效果的图像。"
                }),
                "UM非锐化掩蔽": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "description": "锐化强度，从0到3。"
                }),
                "DCP暗通道先验": ("FLOAT", {
                    "default": 0.32,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "去雾强度，从0到1。"
                }),
                "CLAHE对比度增强限制": ("FLOAT", {
                    "default": 0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "CLAHE的clip limit，从0到4。"
                }),
                "自然饱和度": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "description": "饱和度强度，从0到2。"
                }),
                "伽马值": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "description": "伽马值，从0.1到3.0。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🍊 Kim-Nodes/🎨Filter | 滤镜"

    def execute(self, image, UM非锐化掩蔽, DCP暗通道先验, CLAHE对比度增强限制, 自然饱和度, 伽马值, clahe_tile_grid_size=1):
        # 确保图像格式正确
        image = self.ensure_image_format(image)
        
        # 多批次处理
        batch_size = image.shape[0]
        processed_images = []

        for i in range(batch_size):
            single_image = image[i]
            processed_image = self.process_single_image(single_image, UM非锐化掩蔽, DCP暗通道先验, CLAHE对比度增强限制, 自然饱和度, 伽马值, clahe_tile_grid_size)
            processed_images.append(processed_image)

        # 合并所有处理后的图像
        processed_images = torch.stack(processed_images)
        return [processed_images]

    def process_single_image(self, image, UM非锐化掩蔽, DCP暗通道先验, CLAHE对比度增强限制, 自然饱和度, 伽马值, clahe_tile_grid_size):
        try:
            processed_image = self.apply_effects(image, UM非锐化掩蔽, DCP暗通道先验, CLAHE对比度增强限制, 自然饱和度, 伽马值, clahe_tile_grid_size)

            # 确保图像格式是 torch.Tensor 并归一化到 [0, 1]
            processed_image = torch.from_numpy(processed_image).float() / 255.0
            return processed_image
        except Exception as e:
            print("在图像处理中发生错误:", str(e))
            black_image = torch.zeros((3, image.shape[1], image.shape[2]), dtype=torch.float32)
            return black_image

    def ensure_image_format(self, image):
        if isinstance(image, torch.Tensor):
            image = image.numpy() * 255
            image = image.astype(np.uint8)
        elif isinstance(image, np.ndarray):
            image = image.astype(np.uint8)
        return image

    def apply_effects(self, image, UM非锐化掩蔽, DCP暗通道先验, CLAHE对比度增强限制, 自然饱和度, 伽马值, clahe_tile_grid_size):
        image = apply_sharpen(image, UM非锐化掩蔽)
        image = apply_dehaze(image, DCP暗通道先验)
        image = apply_clahe(image, CLAHE对比度增强限制, (clahe_tile_grid_size, clahe_tile_grid_size))
        image = adjust_natural_saturation(image, 自然饱和度)
        image = adjust_gamma(image, 伽马值)
        return image
