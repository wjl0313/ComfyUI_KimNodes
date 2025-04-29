import torch
import numpy as np

class BoundingBox_Cropper:
    """
    根据边界框坐标裁切图片的节点。
    当bbox_index为-1时，输出所有检测到的边界框对应的裁剪图片。
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOXES", {"forceInput": True}),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 100,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "🍒 Kim-Nodes/✂ Crop | 裁剪工具"
    OUTPUT_IS_LIST = (True,)

    def crop_image(self, image, bboxes, bbox_index=0):
        """
        根据边界框坐标裁切图片
        
        参数:
            image: 输入图片张量 (B,H,W,C)
            bboxes: 边界框坐标列表
            bbox_index: 要使用的边界框索引，-1表示输出所有边界框
        """
        
        # 确保输入图片是正确的格式
        if len(image.shape) != 4:
            image = image.unsqueeze(0)
        
        result_images = []
        
        for i in range(len(image)):
            # 获取当前图像对应的边界框
            current_bboxes = bboxes[min(i, len(bboxes)-1)]
            
            # 如果没有边界框，返回原始图像
            if len(current_bboxes) == 0:
                result_images.append(image[i].unsqueeze(0))
                continue
            
            # 特殊情况：如果bbox_index为-1，裁切所有边界框
            if bbox_index == -1:
                for bbox in current_bboxes:
                    # 提取边界框坐标
                    x1, y1, x2, y2 = bbox
                    
                    # 确保坐标在有效范围内
                    B, H, W, C = image.shape
                    x1 = max(0, min(x1, W-1))
                    y1 = max(0, min(y1, H-1))
                    x2 = max(x1+1, min(x2, W))
                    y2 = max(y1+1, min(y2, H))
                    
                    # 进行裁切
                    cropped = image[i:i+1, y1:y2, x1:x2, :]
                    result_images.append(cropped)
            else:
                # 获取指定索引的边界框，如果索引超出范围，使用第一个边界框
                box_idx = min(bbox_index, len(current_bboxes)-1)
                bbox = current_bboxes[box_idx]
                
                # 提取边界框坐标
                x1, y1, x2, y2 = bbox
                
                # 确保坐标在有效范围内
                B, H, W, C = image.shape
                x1 = max(0, min(x1, W-1))
                y1 = max(0, min(y1, H-1))
                x2 = max(x1+1, min(x2, W))
                y2 = max(y1+1, min(y2, H))
                
                # 进行裁切
                cropped = image[i:i+1, y1:y2, x1:x2, :]
                result_images.append(cropped)
        
        # 返回图片列表
        return (result_images,) 