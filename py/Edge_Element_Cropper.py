import torch
import numpy as np
from PIL import Image
import cv2

class Edge_Element_Cropper:
    """
    检测图片中四边最接近边缘的元素顶点，并据此裁剪图片。
    支持边缘按百分比扩展，当扩展超出原图时显示透明背景。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "原始图片": ("IMAGE",),
                "蒙版图片": ("IMAGE",),
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("裁剪后图片", "区域测试", "透明蒙版",)
    FUNCTION = "crop_edges"
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

    def crop_edges(self, 原始图片, 蒙版图片, 百分比扩展=0.0, 最小扩展像素=0):
        print(f"输入图片维度: {原始图片.shape}")
        
        # 处理输入图片
        if isinstance(原始图片, torch.Tensor):
            if len(原始图片.shape) == 4:
                原始图片 = 原始图片[0]
            img_np = (原始图片.cpu().numpy() * 255).astype(np.uint8)
            
        if isinstance(蒙版图片, torch.Tensor):
            if len(蒙版图片.shape) == 4:
                蒙版图片 = 蒙版图片[0]
            mask_np = (蒙版图片.cpu().numpy() * 255).astype(np.uint8)
        
        # 转换蒙版为灰度图
        if len(mask_np.shape) == 3:
            mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask_np
            
        # 使用形态学操作来优化区域
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(mask_gray, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # 找到轮廓
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("未检测到元素")
            empty_result = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
            empty_mask = torch.zeros((1, img_np.shape[0], img_np.shape[1])).float()
            return (empty_result, empty_result, empty_mask,)
            
        # 创建可视化图像
        vis_img = img_np.copy()
        
        # 在原图上用红线画出所有轮廓
        cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)  # 红色线条(0,0,255)
        
        # 合并所有轮廓点以找到整体边界
        all_points = np.concatenate(contours)
        leftmost = tuple(all_points[all_points[:,:,0].argmin()][0])
        rightmost = tuple(all_points[all_points[:,:,0].argmax()][0])
        topmost = tuple(all_points[all_points[:,:,1].argmin()][0])
        bottommost = tuple(all_points[all_points[:,:,1].argmax()][0])
        
        # 计算中心点
        center_x = (leftmost[0] + rightmost[0]) // 2
        center_y = (topmost[1] + bottommost[1]) // 2
        
        # 计算原始尺寸（加1是因为边界点之差需要加1才是实际尺寸）
        orig_width = rightmost[0] - leftmost[0] + 1
        orig_height = bottommost[1] - topmost[1] + 1
        
        # 计算百分比扩展像素（至少满足最小扩展像素要求）
        width_expand = max(int(orig_width * (百分比扩展 / 100.0)), 最小扩展像素 if 百分比扩展 > 0 else 0)
        height_expand = max(int(orig_height * (百分比扩展 / 100.0)), 最小扩展像素 if 百分比扩展 > 0 else 0)
        
        # 如果百分比为负，则收缩而不是扩展
        if 百分比扩展 < 0:
            width_expand = int(orig_width * (百分比扩展 / 100.0))
            height_expand = int(orig_height * (百分比扩展 / 100.0))
        
        # 计算扩展后的尺寸
        new_width = orig_width + (width_expand * 2)  # 左右各扩展一次
        new_height = orig_height + (height_expand * 2)  # 上下各扩展一次
        
        # 确保新尺寸不会小于1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # 计算新的边界（从边缘扩展）
        min_x = leftmost[0] - width_expand
        max_x = rightmost[0] + width_expand
        min_y = topmost[1] - height_expand
        max_y = bottommost[1] + height_expand
        
        print(f"中心点: ({center_x}, {center_y})")
        print(f"边界点: 左({leftmost[0]}), 右({rightmost[0]}), 上({topmost[1]}), 下({bottommost[1]})")
        print(f"原始尺寸: {orig_width}x{orig_height}")
        print(f"扩展像素: 宽={width_expand}, 高={height_expand} (百分比={百分比扩展}%)")
        print(f"扩展尺寸: {new_width}x{new_height}")
        print(f"扩展边界: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
        
        # 用蓝色线条画出边界框
        vis_min_x = max(0, min_x)
        vis_max_x = min(vis_img.shape[1], max_x)
        vis_min_y = max(0, min_y)
        vis_max_y = min(vis_img.shape[0], max_y)
        cv2.rectangle(vis_img, (vis_min_x, vis_min_y), (vis_max_x, vis_max_y), (255, 0, 0), 2)
        
        # 添加文字标注
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 255, 255)  # 白色
        font_thickness = 2
        
        cv2.putText(vis_img, 'L', (leftmost[0]-20, leftmost[1]), 
                    font, font_scale, font_color, font_thickness)
        cv2.putText(vis_img, 'R', (rightmost[0]+10, rightmost[1]), 
                    font, font_scale, font_color, font_thickness)
        cv2.putText(vis_img, 'T', (topmost[0], topmost[1]-10), 
                    font, font_scale, font_color, font_thickness)
        cv2.putText(vis_img, 'B', (bottommost[0], bottommost[1]+30), 
                    font, font_scale, font_color, font_thickness)
        
        # 转换原始图像为PIL Image
        orig_pil = Image.fromarray(img_np)
        
        # 创建新的透明图像
        new_im = Image.new("RGBA", (new_width, new_height), (0,0,0,0))
        
        # 计算粘贴位置（中心对齐）
        paste_x = (new_width - orig_width) // 2
        paste_y = (new_height - orig_height) // 2
        
        # 创建原始内容的蒙版
        mask_pil = Image.fromarray(mask_gray)
        mask_pil = mask_pil.crop((leftmost[0], topmost[1], rightmost[0], bottommost[1]))
        
        # 裁剪原始图像
        cropped_orig = orig_pil.crop((leftmost[0], topmost[1], rightmost[0], bottommost[1]))
        
        # 将原始内容粘贴到新图像的中心
        new_im.paste(cropped_orig, (paste_x, paste_y), mask=mask_pil)
        
        # 转换回tensor
        new_im_tensor = self.pil2tensor(new_im)
        
        # 创建透明蒙版
        alpha_mask = np.array(new_im.split()[-1])  # 获取alpha通道
        alpha_tensor = torch.from_numpy(alpha_mask).float() / 255.0
        alpha_tensor = alpha_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
        
        # 转换可视化结果
        vis_result = torch.from_numpy(vis_img).float() / 255.0
        vis_result = vis_result.unsqueeze(0)
        
        return (new_im_tensor, vis_result, alpha_tensor,) 