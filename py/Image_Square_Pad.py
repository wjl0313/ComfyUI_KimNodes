import torch
import numpy as np
import re

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
    图片正方形填充器 - 画布扩展
    
    类似Photoshop画布大小功能，只扩展画布而不处理原始图像内容。
    完全零损失操作，直接在tensor层面进行画布扩展。
    
    功能：
    - 直接支持alpha通道输入
    - 透明填充或色值填充
    - 保持比例不变形
    - 自动居中
    
    核心特点：
    - 零损失画布扩展（不处理原始图像）
    - 直接tensor操作，避免格式转换
    - 类似PS画布大小的逻辑
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
    FUNCTION = "pad_to_square"
    CATEGORY = "🍒 Kim-Nodes/🏖️图像处理"

    def pad_to_square(self, image, color="", alpha=None, invert_alpha=True):
        """
        画布扩展为正方形 - 类似PS画布大小功能
        
        直接在tensor层面操作，零损失扩展画布，不处理原始图像内容
        
        Args:
            image: 输入图像tensor [B, H, W, C]
            color: 16进制背景色值，留空则为透明
            alpha: 可选的透明度mask tensor [B, H, W]
            invert_alpha: 是否反转alpha值
        
        Returns:
            扩展后的图像tensor
        """
        # 获取原图尺寸
        batch_size, orig_height, orig_width, channels = image.shape
        
        # 处理alpha通道
        if alpha is not None:
            # 确保alpha维度正确，支持多种格式
            if alpha.ndim == 4:
                # [B, C, H, W] 格式，取第一个batch和第一个通道
                alpha_data = alpha.squeeze(0).squeeze(0)  # [H, W]
            elif alpha.ndim == 3 and alpha.shape[0] == batch_size:
                # [B, H, W] 格式
                alpha_data = alpha.squeeze(0)  # [H, W]
            elif alpha.ndim == 3 and alpha.shape[0] == 1:
                # [1, H, W] 格式
                alpha_data = alpha.squeeze(0)  # [H, W]
            elif alpha.ndim == 2:
                # [H, W] 格式
                alpha_data = alpha  # [H, W]
            else:
                raise ValueError(f"不支持的alpha维度: {alpha.shape}，支持的格式: [H,W], [1,H,W], [B,H,W], [B,C,H,W]")
            
            # 确保alpha尺寸与图像匹配
            if alpha_data.shape != (orig_height, orig_width):
                raise ValueError(f"Alpha尺寸 {alpha_data.shape} 与图像尺寸 ({orig_height}, {orig_width}) 不匹配")
            
            # 根据设置决定是否反转alpha值
            if invert_alpha:
                alpha_data = 1.0 - alpha_data
            
            # 将alpha添加到图像中，创建RGBA
            if channels == 3:
                # 扩展为RGBA
                image_with_alpha = torch.cat([image, alpha_data.unsqueeze(0).unsqueeze(-1)], dim=-1)
                channels = 4
            else:
                # 替换现有alpha通道
                image_with_alpha = torch.cat([image[..., :3], alpha_data.unsqueeze(0).unsqueeze(-1)], dim=-1)
                channels = 4
            image = image_with_alpha
        
        # 计算正方形边长（取最长边）
        max_size = max(orig_height, orig_width)
        
        # 如果已经是正方形，直接返回
        if orig_height == orig_width == max_size:
            return (image,)
        
        # 确定填充值和通道数
        if color.strip():
            # 有背景色
            try:
                bg_rgb = hex_to_rgb(color.strip())
                if channels == 4:
                    # RGBA模式，背景色+不透明
                    fill_value = [bg_rgb[0]/255.0, bg_rgb[1]/255.0, bg_rgb[2]/255.0, 1.0]
                else:
                    # RGB模式
                    fill_value = [bg_rgb[0]/255.0, bg_rgb[1]/255.0, bg_rgb[2]/255.0]
            except ValueError as e:
                print(f"颜色格式错误: {e}，使用透明背景")
                # 错误时使用透明背景
                if channels == 3:
                    # 需要扩展为RGBA以支持透明
                    alpha_channel = torch.ones((batch_size, orig_height, orig_width, 1), 
                                             dtype=image.dtype, device=image.device)
                    image = torch.cat([image, alpha_channel], dim=-1)
                    channels = 4
                fill_value = [0.0, 0.0, 0.0, 0.0]
        else:
            # 无背景色，透明填充
            if channels == 3:
                # 需要扩展为RGBA以支持透明
                alpha_channel = torch.ones((batch_size, orig_height, orig_width, 1), 
                                         dtype=image.dtype, device=image.device)
                image = torch.cat([image, alpha_channel], dim=-1)
                channels = 4
            fill_value = [0.0, 0.0, 0.0, 0.0]
        
        # 创建新的正方形tensor
        square_tensor = torch.zeros((batch_size, max_size, max_size, channels), 
                                   dtype=image.dtype, device=image.device)
        
        # 填充背景色
        if channels == 3:
            square_tensor[:, :, :, 0] = fill_value[0]  # R
            square_tensor[:, :, :, 1] = fill_value[1]  # G
            square_tensor[:, :, :, 2] = fill_value[2]  # B
        elif channels == 4:
            square_tensor[:, :, :, 0] = fill_value[0]  # R
            square_tensor[:, :, :, 1] = fill_value[1]  # G
            square_tensor[:, :, :, 2] = fill_value[2]  # B
            square_tensor[:, :, :, 3] = fill_value[3]  # A
        
        # 计算居中位置
        y_offset = (max_size - orig_height) // 2
        x_offset = (max_size - orig_width) // 2
        
        # 直接复制原图到中心位置 - 零损失操作
        square_tensor[:, y_offset:y_offset+orig_height, x_offset:x_offset+orig_width, :] = image
        
        return (square_tensor,)