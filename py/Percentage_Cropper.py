import torch
import numpy as np

class Percentage_Cropper:    
    """
    按照图片宽高的百分比向内裁切图片的节点。
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "horizontal_percent": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "vertical_percent": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "🍒 Kim-Nodes/✂ Crop | 裁剪工具"

    def crop_image(self, image, horizontal_percent, vertical_percent):
        """
        按照给定的百分比裁切图片
        
        参数:
            image: 输入图片张量 (B,H,W,C)
            horizontal_percent: 水平方向裁切的百分比
            vertical_percent: 垂直方向裁切的百分比
        """
        
        # 确保输入图片是正确的格式
        if len(image.shape) != 4:
            image = image.unsqueeze(0)
            
        B, H, W, C = image.shape
        
        # 计算需要裁切的像素数
        h_crop = int(W * (horizontal_percent / 100.0) / 2)
        v_crop = int(H * (vertical_percent / 100.0) / 2)
        
        # 进行裁切
        cropped = image[:, v_crop:H-v_crop, h_crop:W-h_crop, :]
        
        return (cropped,) 