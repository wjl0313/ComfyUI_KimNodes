from aiohttp import web
import cv2
import numpy as np
import torch

class Whitening_Node:
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义美白节点的输入参数。

        返回:
            dict: 所有输入字段的配置。
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "whitening_strength": ("INT", {
                    "default": 50, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1, 
                    "display": "slider", 
                    "lazy": False
                }),
                "Translucent_skin": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "display": "slider",
                    "lazy": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("whitened_image",)
    FUNCTION = "execute"
    CATEGORY = "🍊 Kim-Nodes/👧🏻美颜"
    DEPRECATED = False
    EXPERIMENTAL = False

    def detect_skin(self, img):
        """
        检测图像中的皮肤区域。
        
        参数:
            img (np.array): RGB格式的图像。
            
        返回:
            np.array: 皮肤区域的掩码（二值图像）。
        """
        # 转换到YCrCb颜色空间
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        
        # 定义皮肤的颜色范围
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([255, 173, 127], np.uint8)
        
        # 创建皮肤掩码
        skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
        
        # 应用形态学操作来改善掩码质量
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        
        # 高斯模糊使边缘更自然
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        return skin_mask

    def execute(self, image, whitening_strength, Translucent_skin):
        """
        对图像的皮肤区域应用美白效果和黄色调节。

        参数:
            image (torch.Tensor): 输入图像张量。
            whitening_strength (int): 美白效果的强度（0-100）。
            Translucent_skin (int): 黄色调节的强度（-100到100）。

        返回:
            tuple: 包含处理后图像的元组。
        """
        if image is None:
            return (None,)

        # 将PyTorch张量转换为NumPy数组进行处理
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_np = img_np[0]
        
        # 检测皮肤区域
        skin_mask = self.detect_skin(img_np)
        skin_mask = skin_mask / 255.0  # 归一化掩码
        
        # 转换到LAB色彩空间
        lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # 创建调整后的LAB图像
        adjusted_lab = lab_img.copy()
        # 只在皮肤区域调整b通道（黄蓝通道）
        adjusted_lab[:, :, 2] = np.clip(
            lab_img[:, :, 2] - Translucent_skin * 0.5, 
            0, 
            255
        )
        
        # 转换回RGB
        adjusted_rgb = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)
        
        # 使用掩码混合原始图像和调整后的图像
        skin_mask = np.expand_dims(skin_mask, axis=2)
        img_np = (img_np * (1 - skin_mask) + adjusted_rgb * skin_mask).astype(np.uint8)
        
        # 应用美白效果
        lookup = self.generate_whitening_lookup(whitening_strength)
        result = img_np.copy()
        for c in range(3):
            result[:, :, c] = lookup[img_np[:, :, c].astype(np.uint8)]

        # 将处理后的图像转换回PyTorch张量
        img_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (img_tensor,)

    def generate_whitening_lookup(self, value):
        """
        根据提供的强度生成美白查找表。

        参数:
            value (int): 美白强度。

        返回:
            np.ndarray: 用于像素值映射的查找表。
        """
        # 中间调增强曲线
        midtones_add = 0.667 * (1 - ((np.arange(256) - 127.0) / 127) ** 2)
        # 创建查找表并将值限制在 0 到 255 之间
        lookup = np.clip(np.arange(256) + (value * midtones_add).astype(np.int16), 0, 255).astype(np.uint8)
        return lookup

