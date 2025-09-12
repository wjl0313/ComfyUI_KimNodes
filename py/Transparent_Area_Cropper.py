import torch
import numpy as np
from PIL import Image

class Transparent_Area_Cropper:
    """
    根据mask元素位置裁剪画布，类似PS的画布大小调整功能。
    支持白底黑元素的mask格式。
    正方形模式：根据裁剪区域的最大维度创建正方形画布。
    矩形模式：根据mask边界框精确裁剪画布。
    注意：只裁剪画布大小，不修改图像内容本身。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "蒙版": ("MASK",),
                "正方形输出": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
                "百分比扩展": ("INT", {
                    "default": 0,
                    "min": 0,
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("合并Alpha图片", "裁剪后图片", "透明蒙版",)
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
    
    def create_masked_alpha_image(self, image, mask):
        """
        使用mask创建透明底图像
        mask中黑色区域(0)保留图像内容，白色区域(255)变为透明
        """
        # 确保图像是RGBA格式
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 获取图像和mask的数组
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # 创建结果数组
        result_array = img_array.copy()
        
        # mask中白色区域(255)对应透明，黑色区域(0)保留
        # 所以我们需要反转mask：白色->透明，黑色->不透明
        alpha_mask = 255 - mask_array  # 反转mask
        
        # 将反转后的mask应用到alpha通道
        result_array[:, :, 3] = np.minimum(result_array[:, :, 3], alpha_mask)
        
        return Image.fromarray(result_array, 'RGBA')

    def crop_transparent_area(self, 图片, 蒙版, 正方形输出=True, 百分比扩展=0.0, 最小扩展像素=0):
        print(f"输入图片维度: {图片.shape}")
        print(f"输入蒙版维度: {蒙版.shape}")
        
        # 处理输入图片
        if isinstance(图片, torch.Tensor):
            if len(图片.shape) == 4:
                img_tensor = 图片[0]
            else:
                img_tensor = 图片
            # 转换为PIL图像
            pil_img = self.tensor2pil(img_tensor)
        else:
            pil_img = 图片
            
        # 确保图像有透明通道
        if pil_img.mode != 'RGBA':
            if pil_img.mode == 'RGB':
                pil_img = pil_img.convert('RGBA')
            else:
                pil_img = pil_img.convert('RGBA')
        
        # 处理蒙版
        if isinstance(蒙版, torch.Tensor):
            # 处理不同维度的蒙版
            if len(蒙版.shape) == 4:  # (batch, channels, height, width)
                mask_np = 蒙版[0, 0].cpu().numpy()
            elif len(蒙版.shape) == 3:  # (batch, height, width) 或 (channels, height, width)
                if 蒙版.shape[0] == 1:
                    mask_np = 蒙版[0].cpu().numpy()
                else:
                    mask_np = 蒙版[0].cpu().numpy()
            elif len(蒙版.shape) == 2:  # (height, width)
                mask_np = 蒙版.cpu().numpy()
            else:
                raise ValueError(f"不支持的蒙版维度: {蒙版.shape}")
            
            # 将蒙版转换为二值图像（0或255）
            mask_np = (mask_np * 255).astype(np.uint8)
        else:
            mask_np = np.array(蒙版)
        
        # 对于白底黑元素的mask，需要反转：黑色区域(0)是有效区域，白色区域(255)是背景
        # 反转mask：黑色变白色，白色变黑色
        inverted_mask_np = 255 - mask_np
        
        print(f"原始mask统计: 最小值={mask_np.min()}, 最大值={mask_np.max()}")
        print(f"反转mask统计: 最小值={inverted_mask_np.min()}, 最大值={inverted_mask_np.max()}")
        
        # 从反转后的蒙版获取非零区域的边界框（现在黑色元素变成了白色）
        mask_pil = Image.fromarray(mask_np, mode='L')  # 保持原始mask用于后续处理
        inverted_mask_pil = Image.fromarray(inverted_mask_np, mode='L')  # 反转mask用于获取边界框
        bbox = inverted_mask_pil.getbbox()  # 返回(left, upper, right, lower)
        
        if not bbox:
            print("蒙版中未检测到非零区域，返回空图")
            # 创建1x1的透明图片
            empty_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            empty_tensor = self.pil2tensor(empty_img)
            empty_mask = torch.zeros((1, 1, 1)).float()
            # 创建1x1的透明图片作为alpha合成结果
            empty_composite = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            empty_composite_tensor = self.pil2tensor(empty_composite)
            return (empty_composite_tensor, empty_tensor, empty_mask,)
        
        # 解包边界框 - 这就是mask的有效区域
        left, top, right, bottom = bbox
        
        # 计算原始mask区域尺寸
        orig_width = right - left
        orig_height = bottom - top
        
        # 应用扩展参数（如果有的话）
        if 百分比扩展 != 0 or 最小扩展像素 != 0:
            # 计算百分比扩展像素
            width_expand = max(int(orig_width * (百分比扩展 / 100.0)), 最小扩展像素 if 百分比扩展 > 0 else 0)
            height_expand = max(int(orig_height * (百分比扩展 / 100.0)), 最小扩展像素 if 百分比扩展 > 0 else 0)
            
            # 如果百分比为负，则收缩而不是扩展
            if 百分比扩展 < 0:
                width_expand = int(orig_width * (百分比扩展 / 100.0))
                height_expand = int(orig_height * (百分比扩展 / 100.0))
            
            # 计算扩展后的边界
            crop_left = left - width_expand
            crop_top = top - height_expand
            crop_right = right + width_expand
            crop_bottom = bottom + height_expand
        else:
            # 如果没有扩展，直接使用mask的边界框
            crop_left = left
            crop_top = top
            crop_right = right
            crop_bottom = bottom
        
        # 限制裁剪区域在原图范围内
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(pil_img.width, crop_right)
        crop_bottom = min(pil_img.height, crop_bottom)
        
        # 计算实际裁剪尺寸
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top
        
        # 确保尺寸不会小于1
        crop_width = max(1, crop_width)
        crop_height = max(1, crop_height)
        
        print(f"检测到mask元素边界: 左({left}), 右({right}), 上({top}), 下({bottom})")
        print(f"mask元素尺寸: {orig_width}x{orig_height}")
        if 百分比扩展 != 0 or 最小扩展像素 != 0:
            print(f"画布扩展参数: 百分比={百分比扩展}%, 最小像素={最小扩展像素}")
        print(f"画布裁剪区域: ({crop_left}, {crop_top}) 到 ({crop_right}, {crop_bottom})")
        print(f"画布裁剪尺寸: {crop_width}x{crop_height}")
        
        # 确保裁剪区域有效
        if crop_left >= crop_right or crop_top >= crop_bottom:
            print("无有效裁剪区域，返回空图")
            empty_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            empty_tensor = self.pil2tensor(empty_img)
            empty_mask = torch.zeros((1, 1, 1)).float()
            # 创建1x1的透明图片作为alpha合成结果
            empty_composite = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            empty_composite_tensor = self.pil2tensor(empty_composite)
            return (empty_composite_tensor, empty_tensor, empty_mask,)
        
        # 裁剪源图像和蒙版
        cropped_src = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        # 使用反转后的mask进行裁剪，这样黑色元素区域变成白色（有效区域）
        cropped_mask = inverted_mask_pil.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        # 确保图像和蒙版尺寸一致
        if cropped_src.size != cropped_mask.size:
            min_width = min(cropped_src.width, cropped_mask.width)
            min_height = min(cropped_src.height, cropped_mask.height)
            cropped_src = cropped_src.crop((0, 0, min_width, min_height))
            cropped_mask = cropped_mask.crop((0, 0, min_width, min_height))
        
        # 画布裁剪模式：直接使用裁剪后的图像，不应用mask处理
        # 确保图像是RGBA格式以保持透明度
        if cropped_src.mode != 'RGBA':
            if cropped_src.mode == 'RGB':
                cropped_src = cropped_src.convert('RGBA')
            else:
                cropped_src = cropped_src.convert('RGBA')
        
        # 直接使用裁剪后的图像，保持原始内容和透明度
        canvas_cropped_img = cropped_src
        
        # 为了输出mask，我们需要将反转的mask再反转回来，保持与输入相同的格式（白底黑元素）
        cropped_original_mask = mask_pil.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        # 根据输出模式处理最终结果
        if 正方形输出:
            # 创建正方形输出 - 使用裁剪后的最大维度
            max_dim = max(canvas_cropped_img.width, canvas_cropped_img.height)
            output_img = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
            # 计算居中位置
            center_x = (max_dim - canvas_cropped_img.width) // 2
            center_y = (max_dim - canvas_cropped_img.height) // 2
            output_img.paste(canvas_cropped_img, (center_x, center_y), canvas_cropped_img)
            
            # 创建正方形蒙版 - 使用原始格式（白底黑元素）
            output_mask_img = Image.new('L', (max_dim, max_dim), 255)  # 白色背景
            output_mask_img.paste(cropped_original_mask, (center_x, center_y))
            
            # 创建透明底的alpha合成图像 - 使用mask来控制透明度
            alpha_composite_img = self.create_masked_alpha_image(output_img, output_mask_img)
            
            print(f"正方形画布输出尺寸: {max_dim}x{max_dim}")
        else:
            # 直接使用裁剪后的结果
            output_img = canvas_cropped_img
            output_mask_img = cropped_original_mask  # 使用原始格式的mask
            
            # 创建透明底的alpha合成图像 - 使用mask来控制透明度
            alpha_composite_img = self.create_masked_alpha_image(output_img, output_mask_img)
            
            print(f"矩形画布输出尺寸: {canvas_cropped_img.width}x{canvas_cropped_img.height}")
        
        # 转换回tensor
        output_tensor = self.pil2tensor(output_img)
        
        # 转换输出蒙版为tensor
        mask_array = np.array(output_mask_img).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array)
        mask_tensor = mask_tensor.unsqueeze(0)  # 添加batch维度
        
        # 转换alpha合成图像为tensor
        alpha_composite_tensor = self.pil2tensor(alpha_composite_img)
        
        return (alpha_composite_tensor, output_tensor, mask_tensor,) 