import numpy as np
import cv2
import torch

class KimHDR:
    """
    一个图像处理节点，对图像应用最先进的HDR算法。
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "description": "上传您想应用高级HDR处理效果的图像。"
                }),
                "HDR强度": ("FLOAT", {
                    "default": 1,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.01,
                    "description": "HDR强度，从0到3。"
                }),
                "欠曝光因子": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "欠曝光因子，从0到1。"
                }),
                "过曝光因子": ("FLOAT", {
                    "default": 1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
                    "description": "过曝光因子，从1到2。"
                }),
                "gamma": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "description": "色调映射器的gamma值，从0.1到3.0。"
                }),
                "高光细节": ("FLOAT", {
                    "default": 1/30.0,
                    "min": 1/1000.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "高光细节。"
                }),
                "中间调细节": ("FLOAT", {
                    "default": 0.25,
                    "min": 1/1000.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "中间调细节。"
                }),
                "阴影细节": ("FLOAT", {
                    "default": 2,
                    "min": 1/1000.0,
                    "max": 10.0,
                    "step": 0.1,
                    "description": "阴影细节。"
                }),
                "整体强度": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "处理效果对最终图像的影响程度，从0到1。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🍊 Kim-Nodes/🎨Filter | 滤镜"

    def execute(self, image, HDR强度, 欠曝光因子, 过曝光因子, gamma, 高光细节, 中间调细节, 阴影细节, 整体强度):
        try:

            image = self.ensure_image_format(image)

            processed_image = self.apply_hdr(image, HDR强度, 欠曝光因子, 过曝光因子, gamma, [高光细节, 中间调细节, 阴影细节])

            # 混合原始图像和处理后的图像
            blended_image = cv2.addWeighted(processed_image, 整体强度, image, 1 - 整体强度, 0)

            if isinstance(blended_image, np.ndarray):
                blended_image = np.expand_dims(blended_image, axis=0)

            blended_image = torch.from_numpy(blended_image).float()
            blended_image = blended_image / 255.0
            blended_image = blended_image.to(torch.device('cpu'))

            return [blended_image]
        except Exception as e:
            if image is not None and hasattr(image, 'shape'):
                black_image = torch.zeros((1, 3, image.shape[0], image.shape[1]), dtype=torch.float32)
            else:
                black_image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            return [black_image.to(torch.device('cpu'))]

    def ensure_image_format(self, image):
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.numpy() * 255
            image = image.astype(np.uint8)
        return image

    def apply_hdr(self, image, HDR强度, 欠曝光因子, 过曝光因子, gamma, 曝光时间):
        # 创建HDR合成器
        hdr = cv2.createMergeDebevec()

        # 曝光时间
        times = np.array(曝光时间, dtype=np.float32)

        # 生成不同曝光的图像
        exposure_images = [
            np.clip(image * 欠曝光因子, 0, 255).astype(np.uint8),  # 欠曝光
            image,  # 正常曝光
            np.clip(image * 过曝光因子, 0, 255).astype(np.uint8)   # 过曝光
        ]

        # 合成HDR图像
        hdr_image = hdr.process(exposure_images, times=times.copy())

        # 使用色调映射器
        tonemap = cv2.createTonemapReinhard(gamma=gamma)
        ldr_image = tonemap.process(hdr_image)

        # 调整HDR强度
        ldr_image = ldr_image * HDR强度
        ldr_image = np.clip(ldr_image, 0, 1)

        # 转换为8位图像
        ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

        return ldr_image
