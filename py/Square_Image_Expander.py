import torch
import numpy as np
from PIL import Image
import re

def tensor2pil(image):
    """将tensor转换为PIL图像"""
    if isinstance(image, torch.Tensor):
        # 处理批次维度
        if image.ndim == 4:
            image = image.squeeze(0)
        
        # 判断通道维的位置并转换
        if image.shape[0] <= 4:  # (C, H, W)
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:  # (H, W, C)
            image_np = image.cpu().numpy()
        
        # 缩放到0-255并转换为uint8
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
        
        if image_np.shape[2] == 3:
            return Image.fromarray(image_np, mode='RGB')
        elif image_np.shape[2] == 4:
            return Image.fromarray(image_np, mode='RGBA')
        else:
            raise ValueError(f"不支持的通道数: {image_np.shape[2]}")

def pil2tensor(image):
    """将PIL图像转换为tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def hex_to_rgb(hex_color):
    """将16进制颜色转换为RGB元组"""
    # 移除#号（如果存在）
    hex_color = hex_color.lstrip('#')
    
    # 验证16进制格式
    if not re.match(r'^[0-9a-fA-F]{6}$', hex_color):
        raise ValueError(f"无效的16进制颜色格式: {hex_color}")
    
    # 转换为RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class Image_Square_Pad:
    """
    图片正方形填充器
    
    将图片填充为正方形，以最长边为基准，短边填充透明或指定颜色。
    保持原图比例，居中放置。
    
    功能：
    - 直接支持alpha通道输入
    - 透明填充或色值填充
    - 保持比例不变形
    - 自动居中
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "alpha": ("MASK", {
                    "tooltip": "图片的透明度信息，如果提供将与图片合并"
                }),
                "invert_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否反转alpha值，通常需要开启"
                }),
                "color": ("STRING", {
                    "default": "", 
                    "placeholder": "16进制色值 (如: FF0000 或 #FF0000，留空为透明)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "expand_to_square"
    CATEGORY = "🍒 Kim-Nodes/🏖️图像处理"

    def expand_to_square(self, image, color="", alpha=None, invert_alpha=True):
        """
        将图片扩展为正方形
        
        Args:
            image: 输入图像tensor
            color: 16进制背景色值，留空则为透明
            alpha: 可选的透明度mask tensor
        
        Returns:
            扩展后的图像tensor
        """
        # 转换为PIL图像
        pil_image = tensor2pil(image)
        
        # 如果提供了alpha信息，将其合并到图像中
        if alpha is not None:
            # 将mask tensor转换为PIL图像
            if isinstance(alpha, torch.Tensor):
                # alpha应该是 [batch, height, width] 格式
                if alpha.ndim == 3:
                    alpha_np = alpha.squeeze(0).cpu().numpy()  # 移除batch维度
                elif alpha.ndim == 2:
                    alpha_np = alpha.cpu().numpy()
                else:
                    raise ValueError(f"不支持的alpha维度: {alpha.shape}")
                
                # 根据设置决定是否反转alpha值
                if invert_alpha:
                    alpha_np = 1.0 - alpha_np  # 反转值：mask白色区域变透明，黑色区域变不透明
                
                # 缩放到0-255
                alpha_np = (alpha_np * 255).clip(0, 255).astype(np.uint8)
                alpha_image = Image.fromarray(alpha_np, mode='L')
                
                # 确保alpha图像尺寸与原图匹配
                if alpha_image.size != pil_image.size:
                    alpha_image = alpha_image.resize(pil_image.size, Image.Resampling.LANCZOS)
                
                # 将RGB图像转换为RGBA并应用alpha
                if pil_image.mode != 'RGBA':
                    pil_image = pil_image.convert('RGBA')
                
                # 替换alpha通道
                r, g, b, _ = pil_image.split()
                pil_image = Image.merge('RGBA', (r, g, b, alpha_image))
        
        # 获取原图尺寸
        width, height = pil_image.size
        
        # 计算正方形边长（取最长边）
        max_size = max(width, height)
        
        # 确定填充模式和颜色
        if color.strip():
            # 有背景色，创建RGB图像
            try:
                bg_rgb = hex_to_rgb(color.strip())
                # 创建RGB背景
                square_image = Image.new('RGB', (max_size, max_size), bg_rgb)
                # 确保原图也是RGB模式
                if pil_image.mode == 'RGBA':
                    # 如果原图有透明度，需要合成到背景上
                    temp_bg = Image.new('RGB', pil_image.size, bg_rgb)
                    temp_bg.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
                    pil_image = temp_bg
                elif pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
            except ValueError as e:
                print(f"颜色格式错误: {e}，使用透明背景")
                # 颜色格式错误，回退到透明背景
                square_image = Image.new('RGBA', (max_size, max_size), (0, 0, 0, 0))
                if pil_image.mode != 'RGBA':
                    pil_image = pil_image.convert('RGBA')
        else:
            # 无背景色，创建透明背景
            square_image = Image.new('RGBA', (max_size, max_size), (0, 0, 0, 0))
            if pil_image.mode != 'RGBA':
                pil_image = pil_image.convert('RGBA')
        
        # 计算居中位置
        x_offset = (max_size - width) // 2
        y_offset = (max_size - height) // 2
        
        # 将原图粘贴到中心位置
        if pil_image.mode == 'RGBA' and square_image.mode == 'RGBA':
            square_image.paste(pil_image, (x_offset, y_offset), pil_image)
        else:
            square_image.paste(pil_image, (x_offset, y_offset))
        
        # 如果结果是RGBA但没有透明度，转换为RGB
        if square_image.mode == 'RGBA' and color.strip():
            # 检查是否真的有透明像素
            alpha_channel = square_image.split()[-1]
            if alpha_channel.getextrema()[0] == 255:  # 所有像素都是不透明的
                square_image = square_image.convert('RGB')
        
        # 转换回tensor
        result_tensor = pil2tensor(square_image)
        
        return (result_tensor,)
