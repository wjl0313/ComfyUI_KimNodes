import numpy as np
import cv2
import torch
import torchvision.transforms as T

class Bitch_Filter:
    """
    一个图像处理节点，模拟柯达 Gold 200 胶片风格 —— 泛黄、暖调、复古做旧效果。
    滤镜名称为英文“Bitch Filter”，仅供技术实现，实际部署建议更名。
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "description": "输入图像。"
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "description": "滤镜强度，0=无效果，1=完全应用。"
                }),
                "add_grain": ("BOOLEAN", {
                    "default": True,
                    "description": "是否添加胶片颗粒噪点。"
                }),
                "grain_strength": ("FLOAT", {
                    "default": 0.07,
                    "min": 0.0,
                    "max": 0.2,
                    "step": 0.01,
                    "description": "颗粒强度，仅在 add_grain=True 时生效。"
                }),
                "grain_distribution": (["uniform", "gaussian"], {
                    "default": "gaussian",
                    "description": "噪点分布类型：均匀分布或高斯分布。"
                }),
                "warmth": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.8,
                    "max": 1.5,
                    "step": 0.05,
                    "description": "暖色调增强，>1 增加黄色/红色。"
                }),
                "fade": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 0.3,
                    "step": 0.02,
                    "description": "轻微褪色效果，模拟老照片。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🍒 Kim-Nodes/🎨Filter | 滤镜"

    def execute(self, image, intensity, add_grain, grain_strength, grain_distribution, warmth, fade):
        # 将输入转换为正确的格式 [0, 255] uint8 numpy
        image = image.clone().mul(255).clamp(0, 255).byte().cpu().numpy()
        output = []

        # 处理每张图片
        if len(image.shape) == 4:  # 批处理
            for img in image:
                processed = self.apply_kodak_gold_filter(img, intensity, add_grain, grain_strength, grain_distribution, warmth, fade)
                output.append(processed)
        else:  # 单张图片
            processed = self.apply_kodak_gold_filter(image, intensity, add_grain, grain_strength, grain_distribution, warmth, fade)
            output.append(processed)

        # 堆叠并调整维度顺序 -> [B, H, W, C]
        output = torch.stack(output, dim=0).permute([0, 2, 3, 1])
        return (output,)

    def apply_kodak_gold_filter(self, img, intensity, add_grain, grain_strength, grain_distribution, warmth, fade):
        """
        应用柯达 Gold 200 风格滤镜：
        - 暖色调增强（红/黄通道）
        - 轻微褪色（提升亮度，降低对比）
        - 单色灰度噪点（均匀 or 高斯分布）
        - 与原图按强度混合
        """
        img = img.astype(np.float32)

        # Step 1: 暖色调增强 —— 增强红色和黄色
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        r = np.clip(r * (0.95 * warmth), 0, 255)
        g = np.clip(g * (0.9 + 0.2 * warmth), 0, 255)  # 稍微增强绿，避免过黄
        b = np.clip(b * (1.0 - 0.1 * (warmth - 1.0)), 0, 255)  # 蓝色略微降低
        img_warm = np.stack([r, g, b], axis=-1)

        # Step 2: 褪色效果 —— 向白色靠近
        fade_color = np.ones_like(img_warm) * 255
        img_faded = img_warm * (1 - fade) + fade_color * fade

        # Step 3: 添加单色灰度噪点（仅影响亮度）
        if add_grain and grain_strength > 0:
            h, w = img_faded.shape[:2]
            if grain_distribution == "uniform":
                # 均匀分布：[-strength, +strength] * 255
                noise = np.random.uniform(-grain_strength, grain_strength, (h, w)) * 255
            else:  # gaussian
                # 高斯分布：标准差 = strength * 255，均值=0
                noise = np.random.normal(0, grain_strength * 255, (h, w))

            # 将灰度噪点广播到三通道（不改变色相，仅影响亮度）
            noise_3ch = np.stack([noise, noise, noise], axis=-1)
            img_faded = np.clip(img_faded + noise_3ch, 0, 255)

        # Step 4: 与原图混合
        img_filtered = img * (1 - intensity) + img_faded * intensity
        img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

        # 转换为 tensor [C, H, W]
        return T.ToTensor()(img_filtered)