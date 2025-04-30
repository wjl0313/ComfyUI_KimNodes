import numpy as np
import torch

class Mask_White_Area_Ratio:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "area_threshold": ("FLOAT", {
                    "default": 50.0, 
                    "min": 0.0, 
                    "max": 100.0, 
                    "step": 1,
                    "display": "number"
                }),
            },
            "hidden": {
                "custom_white_level": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "BOOLEAN",)
    RETURN_NAMES = ("white_area_ratio", "boolean",)
    FUNCTION = "mask_white_area_ratio"
    CATEGORY = "🍒 Kim-Nodes/🔲Mask_Tools | 蒙板工具"
    DESCRIPTION = "计算mask中白色区域占比。area_threshold用于判断白色区域占比是否超过设定值。白色判断标准默认为>0.5。"
    
    def mask_white_area_ratio(self, mask, area_threshold, custom_white_level=0.5):
        """计算mask中白色区域占比，并判断是否超过指定阈值"""
        # 获取批次
        batch_size = mask.shape[0]
        # 创建存储结果的列表
        white_area_ratios = []
        is_above_threshold_list = []
        
        # 计算白色区域占比
        for i in range(batch_size):
            # 将mask转换为numpy数组
            mask_np = mask[i].cpu().numpy()
            # 将浮点数转换为整数，使用默认或用户指定的白色判断标准
            mask_np = (mask_np > custom_white_level).astype(np.uint8)
            # 计算白色区域占比
            white_area_ratio = np.sum(mask_np) / (mask_np.shape[0] * mask_np.shape[1])
            # 计算占比是否超过阈值
            ratio_percent = white_area_ratio * 100.0
            # 使用Python原生布尔类型，而不是NumPy布尔类型
            is_above = bool(ratio_percent > area_threshold)
            
            # 将结果添加到列表中
            white_area_ratios.append(float(white_area_ratio))  # 确保返回Python float而不是numpy.float
            is_above_threshold_list.append(is_above)  # 已经是Python bool类型
            
        # 返回白色区域占比和是否超过阈值的布尔值
        return (white_area_ratios, is_above_threshold_list,)