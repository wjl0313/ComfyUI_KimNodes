import torch
import numpy as np

class Mask_Noise_Cleaner:
    """
    去除mask中主体白色区域以外的小白色区域。
    保留最大的白色连通区域，去除其他小的白色区域。
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "仅保留最大连通区域": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox"
                }),
                "保留面积阈值": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("清理后的遮罩", "遮罩图像")
    FUNCTION = "remove_outliers"
    CATEGORY = "🍒 Kim-Nodes/🔲Mask_Tools | 蒙板工具"

    def remove_outliers(self, mask, 仅保留最大连通区域, 保留面积阈值):
        # 处理列表输入
        if isinstance(mask, list):
            mask = mask[0]
            
        # 确保mask是2D张量
        if len(mask.shape) > 2:
            mask = mask.squeeze()
            
        # 转换为numpy数组进行处理
        mask_np = mask.cpu().numpy()
        
        # 创建新的mask用于结果
        cleaned_mask = np.zeros_like(mask_np)
        
        # 转换为8位图像
        import cv2
        mask_8bit = (mask_np * 255).astype(np.uint8)
        
        # 找到所有轮廓
        contours, hierarchy = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # 如果没有轮廓，返回空mask
            return (torch.from_numpy(cleaned_mask).to(mask.device), 
                    torch.from_numpy(np.stack([cleaned_mask, cleaned_mask, cleaned_mask], axis=2)).unsqueeze(0).to(mask.device))
        
        # 按面积大小排序轮廓
        contours_with_area = [(contour, cv2.contourArea(contour)) for contour in contours]
        contours_with_area.sort(key=lambda x: x[1], reverse=True)
        
        if 仅保留最大连通区域:
            # 只保留最大的轮廓
            cv2.fillPoly(cleaned_mask, [contours_with_area[0][0]], 1)
        else:
            # 保留所有大于阈值的轮廓
            for contour, area in contours_with_area:
                if area >= 保留面积阈值:
                    cv2.fillPoly(cleaned_mask, [contour], 1)
        
        # 转换cleaned_mask回torch张量
        cleaned_mask_tensor = torch.from_numpy(cleaned_mask).to(mask.device)
        
        # 创建图像输出
        # 将mask转换为3通道图像
        mask_image = np.stack([cleaned_mask, cleaned_mask, cleaned_mask], axis=2)
        mask_image_tensor = torch.from_numpy(mask_image).unsqueeze(0).to(mask.device)
        
        return (cleaned_mask_tensor, mask_image_tensor)