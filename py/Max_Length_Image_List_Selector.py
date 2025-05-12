import torch
import numpy as np

class MaxLength_ImageListSelector:
    """
    从多组图片列表中选择长度最长的那组图片列表输出。
    适用于需要从多个处理分支中选择包含最多图片的分支结果。
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片列表1": ("IMAGE",),
                "图片列表2": ("IMAGE",),
            },
            "optional": {
                "图片列表3": ("IMAGE",),
                "图片列表4": ("IMAGE",),
                "图片列表5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("最长图片列表",)
    FUNCTION = "select_max_length_list"
    CATEGORY = "🍒 Kim-Nodes/✔️Selector |选择器"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def select_max_length_list(self, 图片列表1, 图片列表2, 图片列表3=None, 图片列表4=None, 图片列表5=None):
        """
        从多个输入的图片列表中选择长度最长的列表
        
        参数:
            图片列表1-5: 输入的图片列表
            
        返回:
            最长的图片列表
        """
        # 收集所有非空的图片列表
        all_lists = [图片列表1, 图片列表2]
        
        # 添加可选的列表（如果提供）
        if 图片列表3 is not None:
            all_lists.append(图片列表3)
        if 图片列表4 is not None:
            all_lists.append(图片列表4)
        if 图片列表5 is not None:
            all_lists.append(图片列表5)
            
        # 确保所有输入都是列表
        for i in range(len(all_lists)):
            if not isinstance(all_lists[i], list):
                all_lists[i] = [all_lists[i]]
        
        # 查找最长的列表
        max_length = 0
        max_length_list = []
        
        for img_list in all_lists:
            if len(img_list) > max_length:
                max_length = len(img_list)
                max_length_list = img_list
        
        # 如果所有列表都是空的，返回一个空列表
        if max_length == 0:
            empty_img = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return ([empty_img],)
            
        # 返回最长的图片列表
        return (max_length_list,) 