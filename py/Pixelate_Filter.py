import numpy as np
import cv2
import torch
import torchvision.transforms as T

class Pixelate_Filter:
    """
    一个图像处理节点，将图像转换为像素画效果。
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
                "mode": (["lanczos4", "contrast"], {
                    "default": "lanczos4",
                    "description": "像素化模式。lanczos4：高质量缩放；contrast：对比度保留。"
                }),
                "size": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 256,
                    "step": 4,
                    "description": "目标尺寸（较长边将被缩放到这个尺寸）。"
                }),
                "block_size": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 32,
                    "step": 2,
                    "description": "像素块大小（仅在contrast模式下生效）。"
                }),
                "edge_thickness": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "description": "像素边缘厚度（仅在contrast模式下生效）。"
                }),
                "colors": ("INT", {
                    "default": 128,
                    "min": 2,
                    "max": 256,
                    "step": 1,
                    "description": "颜色数量，减少可获得更复古的效果。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🍒 Kim-Nodes/🎨Filter | 滤镜"

    def execute(self, image, mode, size, block_size, edge_thickness, colors):
        # 将输入转换为正确的格式
        image = image.clone().mul(255).clamp(0, 255).byte().cpu().numpy()
        output = []

        # 处理每张图片
        if len(image.shape) == 4:  # 批处理
            for img in image:
                if mode == "contrast":
                    # 使用 pixeloe 的 contrast 模式
                    processed = self.process_contrast(img, size, block_size, edge_thickness, colors)
                else:
                    # 使用自定义的 lanczos4 模式
                    processed = self.process_lanczos4(img, size, colors)
                output.append(processed)
        else:  # 单张图片
            if mode == "contrast":
                processed = self.process_contrast(image, size, block_size, edge_thickness, colors)
            else:
                processed = self.process_lanczos4(image, size, colors)
            output.append(processed)

        # 堆叠并调整维度顺序
        output = torch.stack(output, dim=0).permute([0, 2, 3, 1])
        return (output,)

    def process_contrast(self, image, size, block_size, edge_thickness, colors):
        """使用 pixeloe 的 contrast 模式处理图像"""
        from pixeloe.pixelize import pixelize
        
        # 使用 pixeloe 进行像素化
        img = pixelize(image,
                    mode="contrast",
                    target_size=size,
                    patch_size=block_size,
                    thickness=edge_thickness,
                    contrast=1.0,
                    saturation=1.0,
                    color_matching=True,  # 默认启用颜色匹配
                    no_upscale=False)  # 默认启用放大
        
        # 应用颜色量化（如果需要）
        if colors < 256:
            img = self.quantize_colors(np.array(img), colors)
        
        # 转换为 tensor
        return T.ToTensor()(img)

    def process_lanczos4(self, image, size, colors):
        """使用 lanczos4 模式处理图像"""
        # 获取原始尺寸
        h, w = image.shape[:2]
        
        # 计算目标尺寸，保持宽高比
        if h > w:
            new_h = size
            new_w = int(w * size / h)
        else:
            new_w = size
            new_h = int(h * size / w)
        
        # 使用 LANCZOS4 缩小图像
        small_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 应用颜色量化（如果需要）
        if colors < 256:
            small_image = self.quantize_colors(small_image, colors)
        
        # 放大回原始尺寸（默认启用）
        small_image = cv2.resize(small_image, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 转换为 tensor
        return T.ToTensor()(small_image)

    def quantize_colors(self, image, colors):
        """减少颜色数量以获得更复古的像素画效果"""
        # 计算每个通道的量化因子
        factor = 256 / colors
        
        # 量化颜色
        quantized = np.round(image / factor) * factor
        
        # 确保值在有效范围内
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        return quantized 