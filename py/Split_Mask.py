import numpy as np
import torch
import cv2
from PIL import Image
import torch.nn.functional as F

def images_process(images):
    """处理图像，将其转换为PIL图像"""
    if isinstance(images, torch.Tensor):
        # 如果是张量，转换为PIL图像
        images = images.cpu().numpy()
        if len(images.shape) == 4:
            # 批处理图像
            processed_images = []
            for img in images:
                # 将值范围从[0,1]转换为[0,255]
                img = (img * 255).astype(np.uint8)
                # 如果图像是RGB或RGBA格式
                if img.shape[2] == 3:
                    img = Image.fromarray(img, 'RGB')
                elif img.shape[2] == 4:
                    img = Image.fromarray(img, 'RGBA')
                elif img.shape[2] == 1:
                    img = Image.fromarray(img.squeeze(), 'L')
                processed_images.append(img)
            return processed_images
        else:
            # 单张图像
            img = (images * 255).astype(np.uint8)
            if images.shape[2] == 3:
                return [Image.fromarray(img, 'RGB')]
            elif images.shape[2] == 4:
                return [Image.fromarray(img, 'RGBA')]
            elif images.shape[2] == 1:
                return [Image.fromarray(img.squeeze(), 'L')]
    elif isinstance(images, list):
        # 如果已经是列表，检查每个元素
        processed_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
                img = (img * 255).astype(np.uint8)
                if img.shape[2] == 3:
                    img = Image.fromarray(img, 'RGB')
                elif img.shape[2] == 4:
                    img = Image.fromarray(img, 'RGBA')
                elif img.shape[2] == 1:
                    img = Image.fromarray(img.squeeze(), 'L')
            processed_images.append(img)
        return processed_images
    elif isinstance(images, Image.Image):
        # 如果是单个PIL图像
        return [images]
    else:
        raise ValueError("Unsupported image type")

def _split_mask(image, mask, padding=10, filtration_area=0.0025):
    """根据mask分割图像元素"""
    # 将PIL图像转换为numpy数组
    image_np = np.array(image)
    
    # 确保mask是单通道uint8类型图像
    if isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        mask_np = np.array(mask)
    
    # 确保mask是单通道图像
    if len(mask_np.shape) > 2:
        mask_np = mask_np[:, :, 0] if mask_np.shape[-1] > 1 else mask_np.squeeze()
    
    # 确保mask是uint8类型
    if mask_np.dtype != np.uint8:
        mask_np = (mask_np * 255).astype(np.uint8)
    
    # 二值化处理
    _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 排除轮廓面积占比小于filtration_area的轮廓
    contours = [contour for contour in contours if cv2.contourArea(contour) > filtration_area * image_np.shape[0] * image_np.shape[1]]
    result_images = []
    
    for contour in contours:
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 添加padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image_np.shape[1] - x, w + 2 * padding)
        h = min(image_np.shape[0] - y, h + 2 * padding)
        
        # 创建透明背景
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 复制RGB通道和原始alpha通道
        if image_np.shape[-1] == 4:  # 如果原图有alpha通道
            rgba[:, :, :4] = image_np[y:y+h, x:x+w, :4]
        else:  # 如果原图只有RGB通道
            rgba[:, :, :3] = image_np[y:y+h, x:x+w, :3]
        
        # 创建当前轮廓的mask
        contour_mask = np.zeros_like(mask_binary)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # 只取当前轮廓内的区域作为alpha通道
        alpha = contour_mask[y:y+h, x:x+w]
        
        # 如果原图有alpha通道，需要将原始alpha和轮廓mask结合
        if image_np.shape[-1] == 4:
            original_alpha = image_np[y:y+h, x:x+w, 3]
            rgba[:, :, 3] = cv2.bitwise_and(original_alpha, alpha)
        else:
            rgba[:, :, 3] = alpha
        
        # 将numpy数组转换为PIL图像
        result_image = Image.fromarray(rgba)
        result_images.append(result_image)
    
    return result_images

def process_mask(mask):
    if isinstance(mask, torch.Tensor):
        # 处理[1, 1, H, W]格式
        if len(mask.shape) == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
            mask = mask[0, 0]
        # 处理[1, H, W]格式
        elif len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask[0]
            
        # 转换为numpy数组并调整值范围
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        return mask_np  # 直接返回uint8类型的numpy数组
    
    return mask

class Split_Mask:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
                "filtration_area": ("FLOAT", {"default": 0.0025, "min": 0, "max": 1, "step": 0.0001}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split_mask"
    CATEGORY = "🍒 Kim-Nodes/🔲Mask_Tools | 蒙板工具"

    def split_mask(self, image, mask, padding=10, filtration_area=0.0025):
        # 处理输入图像
        image = images_process(image)[0]
        
        # 处理mask
        mask = process_mask(mask)
        
        # 分割mask元素
        result_images = _split_mask(image, mask, padding, filtration_area)
        
        # 处理结果图像
        processed_images = []
        for result_image in result_images:
            # 将PIL图像转换为numpy数组并归一化
            result_np = np.array(result_image).astype(np.float32) / 255
            # 添加批次维度
            result_np = np.expand_dims(result_np, axis=0)
            # 转换为张量
            img = torch.from_numpy(result_np)
            processed_images.append(img)
            
        return (processed_images,)