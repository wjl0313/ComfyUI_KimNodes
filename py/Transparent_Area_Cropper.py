import torch
import numpy as np
from PIL import Image

class Transparent_Area_Cropper:
    """
    自动裁剪图像中的透明区域，只保留非透明部分。
    支持边缘按百分比扩展，当扩展超出原图时显示透明背景。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "百分比扩展": ("INT", {
                    "default": 0,
                    "min": -50,
                    "max": 200,
                    "step": 1,
                    "display": "number"
                }),
                "最小扩展像素": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("裁剪后图片", "透明蒙版",)
    FUNCTION = "crop_transparent_area"
    CATEGORY = "🍒 Kim-Nodes/✂ Crop | 裁剪工具"

    def tensor2pil(self, image):
        # 如果输入是(batch, height, width, channels)格式，取第一个样本
        if len(image.shape) == 4:
            image = image[0]
        return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

    def pil2tensor(self, image):
        # 转换为numpy数组并归一化
        img_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
        # 添加batch维度
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def crop_transparent_area(self, 图片, 百分比扩展=0.0, 最小扩展像素=0):
        print(f"输入图片维度: {图片.shape}")
        
        # 检查图片是否有透明通道
        if 图片.shape[-1] != 4:
            print("输入图片没有透明通道，返回原图")
            # 创建一个全不透明的蒙版
            alpha_tensor = torch.ones((1, 图片.shape[1], 图片.shape[2])).float()
            return (图片, alpha_tensor,)
        
        # 处理输入图片
        if isinstance(图片, torch.Tensor):
            if len(图片.shape) == 4:
                img_tensor = 图片[0]
            else:
                img_tensor = 图片
                
            # 转换为PIL图像以便处理透明通道
            pil_img = self.tensor2pil(img_tensor)
        else:
            pil_img = 图片  # 如果已经是PIL图像，直接使用
        
        # 确保图像有透明通道
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        
        # 提取透明通道
        _, _, _, alpha = pil_img.split()
        
        # 获取非透明区域的边界框
        bbox = alpha.getbbox()  # 返回(left, upper, right, lower)
        
        if not bbox:
            print("未检测到非透明区域，返回原图")
            alpha_tensor = torch.zeros((1, pil_img.height, pil_img.width)).float()
            return (图片, alpha_tensor,)
        
        # 解包边界框
        left, top, right, bottom = bbox
        
        # 计算原始尺寸
        orig_width = right - left
        orig_height = bottom - top
        
        # 计算百分比扩展像素（至少满足最小扩展像素要求）
        width_expand = max(int(orig_width * (百分比扩展 / 100.0)), 最小扩展像素 if 百分比扩展 > 0 else 0)
        height_expand = max(int(orig_height * (百分比扩展 / 100.0)), 最小扩展像素 if 百分比扩展 > 0 else 0)
        
        # 如果百分比为负，则收缩而不是扩展
        if 百分比扩展 < 0:
            width_expand = int(orig_width * (百分比扩展 / 100.0))
            height_expand = int(orig_height * (百分比扩展 / 100.0))
        
        # 计算扩展后的边界
        new_left = max(0, left - width_expand)
        new_top = max(0, top - height_expand)
        new_right = min(pil_img.width, right + width_expand)
        new_bottom = min(pil_img.height, bottom + height_expand)
        
        # 计算扩展后的尺寸
        new_width = new_right - new_left
        new_height = new_bottom - new_top
        
        # 确保新尺寸不会小于1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        print(f"边界点: 左({left}), 右({right}), 上({top}), 下({bottom})")
        print(f"原始尺寸: {orig_width}x{orig_height}")
        print(f"扩展像素: 宽={width_expand}, 高={height_expand} (百分比={百分比扩展}%)")
        print(f"扩展尺寸: {new_width}x{new_height}")
        print(f"扩展边界: left={new_left}, right={new_right}, top={new_top}, bottom={new_bottom}")
        
        # 裁剪图像
        cropped_img = pil_img.crop((new_left, new_top, new_right, new_bottom))
        
        # 转换回tensor
        cropped_tensor = self.pil2tensor(cropped_img)
        
        # 创建透明蒙版
        alpha_mask = np.array(cropped_img.split()[-1])  # 获取alpha通道
        alpha_tensor = torch.from_numpy(alpha_mask).float() / 255.0
        alpha_tensor = alpha_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        return (cropped_tensor, alpha_tensor,) 