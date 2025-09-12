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
        
        try:
            # 尝试 torch 版本
            from pixeloe.torch.pixelize import pixelize
            
            # 将图像转换为正确的格式 [B,C,H,W] range [0..1]
            if len(image.shape) == 3:  # HWC -> CHW
                img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
            else:
                img_tensor = torch.from_numpy(image).float() / 255.0
            
            # 使用 torch 版本的参数
            result = pixelize(
                img_tensor,
                pixel_size=block_size,
                thickness=edge_thickness,
                mode="contrast",
                do_color_match=True,
                do_quant=(colors < 256),
                num_colors=colors if colors < 256 else 32,
                no_post_upscale=False
            )
            # 转换回 numpy 格式
            img = (result.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
        except (ImportError, TypeError):
            try:
                # 如果 torch 版本失败，尝试 legacy 版本
                from pixeloe.legacy.pixelize import pixelize
                img = pixelize(
                    image,
                    mode="contrast", 
                    contrast=1.0,
                    saturation=1.0,
                    colors=colors if colors < 256 else None,
                    color_quant_method='kmeans',
                    no_upscale=False
                )
                
                # legacy 版本可能需要额外的颜色量化
                if colors < 256 and isinstance(img, np.ndarray):
                    img = self.quantize_colors(img, colors)
                    
            except ImportError as e:
                raise ImportError(
                    "无法导入 pixeloe 模块。请确保已安装 pixeloe 包。\n"
                    "您可以通过以下命令安装：\n"
                    "pip install pixeloe\n"
                    f"错误详情: {str(e)}"
                )
        
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