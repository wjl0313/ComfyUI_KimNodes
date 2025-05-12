import torch
import numpy as np
from PIL import Image

class Image_PixelFilter:
    """
    过滤图像列表，将最大边长小于指定阈值的图像过滤掉，并同时返回原始图像和被过滤的小尺寸图像
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "像素阈值": ("INT", {
                    "default": 512,  # 默认边长512
                    "min": 1,
                    "max": 4096, 
                    "step": 1,
                    "description": "过滤掉最大边长小于此阈值的图片"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("过滤后图像", "原始图像", "被过滤小图像",)
    OUTPUT_IS_LIST = (True, True, True,)
    FUNCTION = "filter_images"
    CATEGORY = "🍒 Kim-Nodes/✔️Selector |选择器"

    def filter_images(self, images, 像素阈值):
        """将最大边长小于阈值的图片过滤掉，同时返回原始图像列表和被过滤掉的小图像"""
        filtered_images = []      # 存储通过过滤的图像
        original_images = []      # 存储所有原始图像
        filtered_out_images = []  # 存储被过滤掉的小图像
        filtered_count = 0
        total_count = 0
        
        # 如果输入是单个图像张量，将其转换为列表
        if not isinstance(images, list):
            images = [images]
            
        # 处理空输入的情况
        if len(images) == 0:
            print("警告：没有输入图像")
            # 创建一个1x1的黑色图像作为占位符
            empty_image = torch.zeros((1, 1, 1, 3))
            return ([empty_image], [empty_image], [empty_image])
        
        # 遍历所有图像
        for img in images:
            total_count += 1
            # 确保图像是4D张量 [B, H, W, C]
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            # 保存原始图像
            original_images.append(img)
            
            # 计算图像的最大边长
            height = img.shape[1]
            width = img.shape[2]
            max_edge_length = max(height, width)
            
            # 如果最大边长小于阈值，过滤掉该图像
            if max_edge_length < 像素阈值:
                filtered_count += 1
                filtered_out_images.append(img)  # 保存被过滤掉的小图像
                continue
                
            filtered_images.append(img)
        
        print(f"过滤前图片数量: {total_count}")
        print(f"被过滤掉的图片数量: {filtered_count}")
        print(f"剩余图片数量: {len(filtered_images)}")
        
        # 如果过滤后没有图片剩余，使用第一张原始图像作为占位符
        if len(filtered_images) == 0:
            print("所有图像都被过滤掉了，将使用第一张原始图像作为占位符")
            filtered_images = [original_images[0]]
        
        # 如果没有被过滤掉的图像，也打印一条消息
        if len(filtered_out_images) == 0:
            print("没有图像被过滤掉")
            filtered_out_images = [original_images[0]]
        
        return (filtered_images, original_images, filtered_out_images) 