import torch
import numpy as np
from PIL import Image, ImageDraw
import random
import base64
import io
import json
from .tiling_templates import template_manager

class SeamlessTilingGenerator:
    """
    无缝四方连续拼图生成器
    
    输入image list，将它组合成指定尺寸的无缝四方连续拼图：
    - 四个角是同一张image的四等分（仅一张）
    - 上边与下边是同一张图的对等分
    - 左边与右边是同一张图的对等分
    - 中间空余部分使用image list中不同的图片进行贴图
    """

    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        # 获取可用模板列表
        available_templates = template_manager.get_available_templates()
        template_choices = list(available_templates.keys())
        
        return {
            "required": {
                "image_list": ("IMAGE", ),
                "拼图模板": (template_choices, {
                    "default": "经典四方连续" if "经典四方连续" in template_choices else template_choices[0]
                }),
                "输出宽度": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 32,
                    "display": "number"
                }),
                "输出高度": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 32,
                    "display": "number"
                }),
                "基础图片尺寸": ("INT", {
                    "default": 84,
                    "min": 32,
                    "max": 512,
                    "step": 4,
                    "display": "number",
                    "description": "控制四个角的尺寸（中心图片会是其4倍大小）"
                }),
                "四边图片尺寸": ("INT", {
                    "default": 168,
                    "min": 32,
                    "max": 512,
                    "step": 4,
                    "display": "number",
                    "description": "单独控制四条边的图片尺寸"
                }),
                "填充中间区域": ("BOOLEAN", {
                    "default": True,
                    "description": "是否在中间区域放置图片"
                }),
                "随机种子": ("INT", {
                    "default": 0,
                     "min": 0,
                    "max": 4294967295,
                    "step": 1,
                    "display": "number"
                }),
                "启用随机": ("BOOLEAN", {
                    "default": True
                }),
                "背景颜色": ("STRING", {
                    "default": "#FFFFFF",
                    "description": "十六进制背景颜色值，例如：#FFFFFF"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "output_json")
    FUNCTION = "generate_seamless_tiling"
    CATEGORY = "🍒 Kim-Nodes/🧩Icon Processing | 图标处理"
    INPUT_IS_LIST = True

    def preprocess_images(self, image_tensor):
        """将张量转换为PIL Image对象列表"""
        image_list = []
        
        # 由于 INPUT_IS_LIST = True，image_tensor 是一个列表
        for tensor in image_tensor:
            if isinstance(tensor, torch.Tensor):
                if tensor.ndim == 4:  # (B, C, H, W)
                    if tensor.shape[1] in (3, 4):
                        tensor = tensor.permute(0, 2, 3, 1)  # -> (B, H, W, C)
                    img_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
                elif tensor.ndim == 3:  # (C, H, W)
                    if tensor.shape[0] in (3, 4):
                        tensor = tensor.permute(1, 2, 0)  # -> (H, W, C)
                    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
                else:
                    raise ValueError(f"不支持的张量维度: {tensor.shape}")
                
                img_pil = Image.fromarray(img_np)
                image_list.append(img_pil)
            else:
                raise ValueError("输入的每个元素必须是torch.Tensor类型")
        
        return image_list

    def image_to_base64(self, image, format='PNG'):
        """将PIL图像转换为base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"

    def generate_seamless_tiling(self, image_list, 拼图模板="经典四方连续", 输出宽度=1024, 输出高度=1024, 基础图片尺寸=128, 
                               四边图片尺寸=128, 填充中间区域=True, 随机种子=0, 启用随机=True, 背景颜色="#FFFFFF"):
        """使用模板系统生成无缝拼图"""
        
        # 由于INPUT_IS_LIST=True，所有参数都是列表，需要取第一个元素
        拼图模板 = 拼图模板[0] if isinstance(拼图模板, list) else 拼图模板
        输出宽度 = 输出宽度[0] if isinstance(输出宽度, list) else 输出宽度
        输出高度 = 输出高度[0] if isinstance(输出高度, list) else 输出高度
        基础图片尺寸 = 基础图片尺寸[0] if isinstance(基础图片尺寸, list) else 基础图片尺寸
        四边图片尺寸 = 四边图片尺寸[0] if isinstance(四边图片尺寸, list) else 四边图片尺寸
        填充中间区域 = 填充中间区域[0] if isinstance(填充中间区域, list) else 填充中间区域
        随机种子 = 随机种子[0] if isinstance(随机种子, list) else 随机种子
        启用随机 = 启用随机[0] if isinstance(启用随机, list) else 启用随机
        背景颜色 = 背景颜色[0] if isinstance(背景颜色, list) else 背景颜色

        print(f"🎨 使用模板: {拼图模板}")
        print(f"📏 基础图片尺寸: {基础图片尺寸} (控制四个角)")
        print(f"📐 四边图片尺寸: {四边图片尺寸} (控制四条边的图片)")
        print(f"🎯 中间图片尺寸: 将被模板设置为 {基础图片尺寸 * 4} (4倍基础尺寸)")

        # 预处理图片
        images = self.preprocess_images(image_list)
        print(f"📷 预处理完成，共获得 {len(images)} 张图片")
        
        if len(images) < 1:
            raise ValueError("至少需要1张图片")
        
        # 获取模板实例
        try:
            template = template_manager.get_template(拼图模板)
            template_info = template.get_template_info()
            print(f"📋 模板信息: {template_info['name']} - {template_info['description']}")
        except ValueError as e:
            print(f"错误: {e}")
            # 回退到默认模板
            template = template_manager.get_template("经典四方连续")
            print("已回退到经典四方连续模板")
        
        # 准备模板参数 - 分别控制不同区域的尺寸
        params = {
            "边界宽度": 四边图片尺寸,      # 四条边的图片尺寸
            "角落大小": 基础图片尺寸,      # 四个角的尺寸
            "中间图片大小": 基础图片尺寸,    # 中心图片的尺寸（注：大部分模板会内部乘以2）
            "基础图片尺寸": 基础图片尺寸,    # 提供基础尺寸给模板参考
            "四边图片尺寸": 四边图片尺寸,    # 提供四边尺寸给模板参考
            "填充中间区域": 填充中间区域,
            "随机种子": 随机种子,
            "启用随机": 启用随机,
            "背景颜色": 背景颜色
        }
        
        # 使用模板生成拼图
        canvas_size = (输出宽度, 输出高度)
        result_tuple = template.generate_tiling(images, canvas_size, params)
        
        # 兼容旧版模板（只返回canvas和mask_canvas）和新版模板（返回positions）
        if len(result_tuple) == 2:
            canvas, mask_canvas = result_tuple
            positions = []  # 旧版模板没有位置信息
            print("⚠️  使用的模板不支持位置信息输出，将返回空的JSON数据")
        elif len(result_tuple) == 3:
            canvas, mask_canvas, positions = result_tuple
        else:
            raise ValueError(f"模板返回了不支持的结果数量: {len(result_tuple)}")
        
        # 转换图像为张量
        canvas_rgb = canvas.convert('RGB')
        result = np.array(canvas_rgb, dtype=np.float32) / 255.0
        result = np.expand_dims(result, axis=0)
        result_tensor = torch.from_numpy(result)
        
        # 转换遮罩为张量
        mask_array = np.array(mask_canvas, dtype=np.float32) / 255.0
        mask_array = np.expand_dims(mask_array, axis=0)
        mask_tensor = torch.from_numpy(mask_array)
        
        # 生成JSON输出
        masks_data = []
        for i, pos in enumerate(positions):
            # 裁剪对应区域的图像
            bbox = pos["bbox"]
            cropped_img = canvas.crop(bbox)
            
            # 转换为base64
            mask_base64 = self.image_to_base64(cropped_img, 'PNG')
            
            # 格式化边界框坐标为字符串列表
            bbox_str = [
                f"{bbox[0]},{bbox[1]}",  # 左上角
                f"{bbox[2]},{bbox[1]}",  # 右上角  
                f"{bbox[2]},{bbox[3]}",  # 右下角
                f"{bbox[0]},{bbox[3]}"   # 左下角
            ]
            
            masks_data.append({
                "mask": mask_base64,
                "bbox": bbox_str,
                "type": pos.get("type", "unknown"),
                "position": pos.get("position", "unknown"),
                "image_index": pos.get("image_index", -1)
            })
        
        # 构建最终JSON结构
        output_json = {
            "masks": masks_data
        }
        
        json_string = json.dumps(output_json, ensure_ascii=False, indent=2)
        
        print(f"✅ 拼图生成完成！")
        print(f"图像张量形状: {result_tensor.shape}")
        print(f"遮罩张量形状: {mask_tensor.shape}")
        print(f"🎯 最终输出: 1张 {输出宽度}x{输出高度} 的无缝拼图、遮罩及 {len(positions)} 个图像区域的JSON信息")
        
        return (result_tensor, mask_tensor, json_string) 