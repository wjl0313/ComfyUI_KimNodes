from aiohttp import web
import cv2
import numpy as np
import torch

class Whitening_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_whitening": ("BOOLEAN", {
                    "default": False,
                    "label": "Auto Whitening"
                }),
                "whitening_strength": ("INT", {
                    "default": 50, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1,
                    "lazy": False
                }),
                "Translucent_skin": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "lazy": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("whitened_image",)
    FUNCTION = "execute"
    CATEGORY = "🍊 Kim-Nodes/👧🏻美颜"

    def __init__(self):
        pass

    def detect_skin(self, img):
        """
        检测图像中的皮肤区域。
        """
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([255, 173, 127], np.uint8)
        skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        return skin_mask

    def generate_whitening_lookup(self, value):
        """
        生成美白查找表
        """
        midtones_add = 0.667 * (1 - ((np.arange(256) - 127.0) / 127) ** 2)
        lookup = np.clip(np.arange(256) + (value * midtones_add).astype(np.int16), 0, 255).astype(np.uint8)
        return lookup

    def analyze_skin_tone(self, img_np):
        """
        智能分析皮肤状况，动态返回建议的美白强度
        """
        skin_mask = self.detect_skin(img_np)
        skin_area = cv2.bitwise_and(img_np, img_np, mask=skin_mask)
        
        # 转换到LAB色彩空间进行分析
        lab_img = cv2.cvtColor(skin_area, cv2.COLOR_RGB2LAB)
        l_channel = lab_img[:, :, 0]  # 亮度通道
        a_channel = lab_img[:, :, 1]  # 红绿通道
        b_channel = lab_img[:, :, 2]  # 黄蓝通道
        
        # 获取有效的皮肤像素
        valid_pixels = (skin_mask > 0)
        if not np.any(valid_pixels):
            return 45  # 如果没有检测到皮肤，返回中等默认值
        
        # 分析亮度分布
        l_values = l_channel[valid_pixels]
        mean_l = np.mean(l_values)
        std_l = np.std(l_values)
        
        # 分析肤色
        a_values = a_channel[valid_pixels]
        b_values = b_channel[valid_pixels]
        mean_a = np.mean(a_values)
        mean_b = np.mean(b_values)
        
        # 重新调整亮度区间的基础强度（降低亮图片的处理强度）
        if mean_l > 200:      # 过度明亮
            base_strength = 5
        elif mean_l > 180:    # 非常亮
            base_strength = 10
        elif mean_l > 160:    # 较亮
            base_strength = 20
        elif mean_l > 140:    # 稍亮
            base_strength = 35
        elif mean_l > 120:    # 中等
            base_strength = 50
        elif mean_l > 100:    # 稍暗
            base_strength = 65
        elif mean_l > 80:     # 较暗
            base_strength = 75
        else:                # 非常暗
            base_strength = 85
        
        # 计算暗区比例（更精确的暗区分析）
        dark_pixels = l_values < 100
        dark_ratio = np.mean(dark_pixels)
        very_dark_pixels = l_values < 80
        very_dark_ratio = np.mean(very_dark_pixels)
        
        # 根据暗区比例调整强度
        if very_dark_ratio > 0.3:  # 大面积深暗区
            base_strength += int(very_dark_ratio * 30)
        elif dark_ratio > 0.4:     # 大面积暗区
            base_strength += int(dark_ratio * 20)
        
        # 光照均匀性分析（更温和的调整）
        if std_l > 45:  # 严重不均匀
            base_strength += int(min(std_l - 45, 15))
        elif std_l > 35:  # 中度不均匀
            base_strength += int(min(std_l - 35, 8))
        elif std_l > 25:  # 轻度不均匀
            base_strength += int(min(std_l - 25, 5))
        
        # 肤色调整（更温和）
        if mean_a > 135:  # 明显偏红
            base_strength += 2
        if mean_b > 135:  # 明显偏黄
            base_strength += 2
        
        # 高光保护：如果高光区域（L>200）占比较大，降低美白强度
        highlight_ratio = np.mean(l_values > 200)
        if highlight_ratio > 0.1:  # 如果高光区域超过10%
            base_strength = max(5, base_strength - int(highlight_ratio * 30))
        
        # 确保最终强度在合理范围内（降低上限）
        final_strength = np.clip(base_strength, 5, 85)
        
        # 打印详细分析结果
        print(f"\nSkin Analysis Results:")
        print(f"Average Brightness: {mean_l:.1f} (L channel)")
        print(f"Brightness Variation: {std_l:.1f}")
        print(f"Dark Area Ratio: {dark_ratio:.2f}")
        print(f"Very Dark Area Ratio: {very_dark_ratio:.2f}")
        print(f"Highlight Area Ratio: {highlight_ratio:.2f}")
        print(f"Red Level: {mean_a:.1f} (a channel)")
        print(f"Yellow Level: {mean_b:.1f} (b channel)")
        print(f"Base Strength: {base_strength}")
        print(f"Final Whitening Strength: {final_strength}")
        
        return final_strength

    def execute(self, image, auto_whitening, whitening_strength, Translucent_skin):
        """
        执行美白处理
        """
        if image is None:
            return (None,)

        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        img_np = img_np[0]
        
        if auto_whitening:
            whitening_strength = self.analyze_skin_tone(img_np)
            print(f"Auto-detected whitening strength: {whitening_strength}")
        
        # 检测皮肤区域
        skin_mask = self.detect_skin(img_np)
        skin_mask = skin_mask / 255.0
        
        # 转换到LAB色彩空间
        lab_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # 调整LAB通道
        adjusted_lab = lab_img.copy()
        adjusted_lab[:, :, 2] = np.clip(lab_img[:, :, 2] - Translucent_skin * 0.5, 0, 255)
        
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
        
        # 转换回PyTorch张量
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_tensor,)

