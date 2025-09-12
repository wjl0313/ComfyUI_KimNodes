import torch
import numpy as np
from PIL import Image
import json
import base64
import io

class JSONImageCompositor:
    """
    JSON图像合成器
    
    将 SeamlessTilingGenerator 输出的JSON数据直接合成为完整图像：
    - 解析JSON中的base64图像数据和坐标信息
    - 根据bbox坐标将图像精确放置到对应位置
    - 自动计算或使用自定义画布尺寸
    - 支持背景颜色自定义
    """

    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {
                    "multiline": True,
                    "description": "来自SeamlessTilingGenerator的JSON输出"
                }),
                "背景颜色": ("STRING", {
                    "default": "#FFFFFF",
                    "description": "合成图背景颜色，例如：#FFFFFF"
                }),
            },
            "optional": {
                "自定义画布尺寸": ("BOOLEAN", {
                    "default": False,
                    "description": "是否使用自定义画布尺寸（否则自动计算）"
                }),
                "画布宽度": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 32,
                    "display": "number"
                }),
                "画布高度": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 32,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("合成预览图", "统计信息")
    FUNCTION = "compose_json_images"
    CATEGORY = "🍒 Kim-Nodes/🧩Icon Processing | 图标处理"

    def hex_to_rgb(self, hex_color):
        """将十六进制颜色转换为RGB元组"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            hex_color = "FFFFFF"  # 默认白色
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return (255, 255, 255)  # 默认白色

    def base64_to_image(self, base64_str):
        """将base64字符串转换为PIL Image"""
        try:
            # 移除data:image/png;base64,前缀
            if base64_str.startswith('data:image/'):
                base64_str = base64_str.split(',', 1)[1]
            
            # 解码base64
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            print(f"❌ 解码base64图像失败: {e}")
            # 返回一个空白图像作为占位符
            return Image.new('RGB', (64, 64), color=(200, 200, 200))


    def parse_bbox_coordinates(self, bbox):
        """解析边界框坐标字符串列表为数值"""
        try:
            # bbox格式: ["x1,y1", "x2,y2", "x3,y3", "x4,y4"]
            # 我们只需要左上角(x1,y1)和右下角(x3,y3)
            x1, y1 = map(int, bbox[0].split(','))
            x2, y2 = map(int, bbox[2].split(','))  # 右下角坐标
            return (x1, y1, x2, y2)
        except (ValueError, IndexError) as e:
            print(f"❌ 解析坐标失败: {bbox}, 错误: {e}")
            return (0, 0, 64, 64)  # 默认坐标

    def compose_json_images(self, json_data, 背景颜色="#FFFFFF", 自定义画布尺寸=False, 
                           画布宽度=1024, 画布高度=1024):
        """直接根据JSON坐标信息合成图像"""
        
        print(f"🎨 开始根据JSON坐标合成图像")
        
        try:
            # 解析JSON数据
            data = json.loads(json_data)
            masks = data.get("masks", [])
            
            if not masks:
                print("⚠️  JSON中没有找到图像数据")
                # 创建空白图像
                empty_img = Image.new('RGB', (512, 512), color=self.hex_to_rgb(背景颜色))
                result = np.array(empty_img, dtype=np.float32) / 255.0
                result = np.expand_dims(result, axis=0)
                return (torch.from_numpy(result), "没有找到图像数据")
            
            print(f"📷 找到 {len(masks)} 个图像区域")
            
            # 计算画布尺寸
            if 自定义画布尺寸:
                canvas_width = 画布宽度
                canvas_height = 画布高度
                print(f"📐 使用自定义画布尺寸: {canvas_width}x{canvas_height}")
            else:
                # 自动计算画布尺寸：找到所有坐标的最大值
                max_x, max_y = 0, 0
                for mask_data in masks:
                    try:
                        bbox = mask_data.get("bbox", [])
                        if bbox:
                            x1, y1, x2, y2 = self.parse_bbox_coordinates(bbox)
                            max_x = max(max_x, x2)
                            max_y = max(max_y, y2)
                    except Exception as e:
                        print(f"⚠️  跳过无效坐标: {e}")
                        continue
                
                # 添加一些边距
                canvas_width = max_x + 50
                canvas_height = max_y + 50
                print(f"📐 自动计算画布尺寸: {canvas_width}x{canvas_height}")
            
            # 创建画布
            canvas = Image.new('RGB', (canvas_width, canvas_height), 
                             color=self.hex_to_rgb(背景颜色))
            
            # 统计信息
            type_counts = {}
            position_counts = {}
            successful_placements = 0
            
            # 处理每个图像
            for i, mask_data in enumerate(masks):
                try:
                    # 解析坐标
                    bbox = mask_data.get("bbox", [])
                    if not bbox:
                        print(f"⚠️  图像 {i+1} 缺少坐标信息")
                        continue
                    
                    x1, y1, x2, y2 = self.parse_bbox_coordinates(bbox)
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 检查坐标是否在画布范围内
                    if x1 < 0 or y1 < 0 or x2 > canvas_width or y2 > canvas_height:
                        print(f"⚠️  图像 {i+1} 坐标超出画布范围: ({x1},{y1},{x2},{y2})")
                        # 如果不是自定义尺寸，调整画布大小
                        if not 自定义画布尺寸:
                            new_width = max(canvas_width, x2 + 50)
                            new_height = max(canvas_height, y2 + 50)
                            if new_width != canvas_width or new_height != canvas_height:
                                print(f"📐 扩展画布尺寸至: {new_width}x{new_height}")
                                # 创建新的更大画布
                                new_canvas = Image.new('RGB', (new_width, new_height), 
                                                     color=self.hex_to_rgb(背景颜色))
                                new_canvas.paste(canvas, (0, 0))
                                canvas = new_canvas
                                canvas_width, canvas_height = new_width, new_height
                    
                    # 解码base64图像
                    img = self.base64_to_image(mask_data["mask"])
                    
                    # 调整图像大小以匹配bbox尺寸
                    if width > 0 and height > 0:
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    # 粘贴到指定位置
                    canvas.paste(img, (x1, y1))
                    successful_placements += 1
                    
                    # 收集统计信息
                    img_type = mask_data.get("type", "unknown")
                    img_position = mask_data.get("position", "unknown")
                    
                    type_counts[img_type] = type_counts.get(img_type, 0) + 1
                    position_counts[img_position] = position_counts.get(img_position, 0) + 1
                    
                    print(f"✅ 图像 {i+1}/{len(masks)}: {img_type} at {img_position} -> ({x1},{y1},{x2},{y2})")
                    
                except Exception as e:
                    print(f"❌ 处理图像 {i+1} 时出错: {e}")
                    continue
            
            # 转换为张量
            result = np.array(canvas, dtype=np.float32) / 255.0
            result = np.expand_dims(result, axis=0)
            result_tensor = torch.from_numpy(result)
            
            # 生成统计信息
            stats = {
                "总图像数": len(masks),
                "成功放置": successful_placements,
                "画布尺寸": f"{canvas_width}x{canvas_height}",
                "类型统计": type_counts,
                "位置统计": position_counts
            }
            
            stats_text = json.dumps(stats, ensure_ascii=False, indent=2)
            
            print(f"✅ 合成完成！")
            print(f"📊 统计信息: {stats}")
            
            return (result_tensor, stats_text)
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            # 创建错误图像
            error_img = Image.new('RGB', (512, 256), color=(255, 100, 100))
            
            result = np.array(error_img, dtype=np.float32) / 255.0
            result = np.expand_dims(result, axis=0)
            return (torch.from_numpy(result), f"JSON解析错误: {e}")
            
        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")
            # 创建通用错误图像
            error_img = Image.new('RGB', (512, 256), color=(255, 100, 100))
            
            result = np.array(error_img, dtype=np.float32) / 255.0
            result = np.expand_dims(result, axis=0)
            return (torch.from_numpy(result), f"处理错误: {e}")
