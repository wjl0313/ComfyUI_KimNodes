"""
随机偏移拼图模板

基于经典四方连续模板，重新设计图片分配逻辑：
- 四个角固定使用图1（四等分）
- 中心点固定使用图1（如果启用填充中间区域）
- 上下左右边界归入"填充中间区域"控制
- 当图片>2张时：上下边使用图2，左右边使用图3（不重复）
- 当图片=2张时：上下边使用图2，左右边使用图2
- 当图片=1张时：全部使用图1
- 中间元素位置添加以中心点向外半径64-256的随机偏移
- 边界保持无缝拼接特性（不添加偏移）
- 修正版：移除边缘偏移，确保四方连续无缝拼接
"""

import random
import math
from PIL import Image
from .base_template import TilingTemplateBase


class RandomOffsetTemplate(TilingTemplateBase):
    """随机偏移拼图模板"""
    
    def __init__(self):
        super().__init__()
        self.template_name = "随机偏移拼图"
        self.template_description = "角落固定图1，边界不重复使用图2/图3，中心随机偏移，保持无缝连续"
    
    def get_template_info(self):
        """返回模板信息"""
        return {
            "name": self.template_name,
            "description": self.template_description
        }
    
    def validate_params(self, params):
        """验证参数有效性"""
        # 检查基础尺寸参数（优先）或传统分离参数
        has_basic_size = "基础图片尺寸" in params
        has_separate_sizes = all(param in params for param in ["边界宽度", "角落大小", "中间图片大小"])
        
        if not (has_basic_size or has_separate_sizes):
            print("警告: 缺少尺寸参数（基础图片尺寸 或 边界宽度/角落大小/中间图片大小）")
            return False
        
        # 检查其他必需参数
        other_required = ["填充中间区域"]
        for param in other_required:
            if param not in params:
                print(f"警告: 缺少必需参数 {param}")
                return False
        
        return True
    
    def create_corner_pieces(self, corner_image, corner_size):
        """从一张图片创建四个角，特殊排列以实现无缝效果"""
        # 计算缩放比例，保持原始比例
        w, h = corner_image.size
        scale = min(corner_size * 2 / w, corner_size * 2 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图片，保持原始比例
        corner_img = corner_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 创建一个临时画布来居中放置图片
        temp_canvas = Image.new('RGBA', (corner_size * 2, corner_size * 2), (0, 0, 0, 0))
        paste_x = (corner_size * 2 - new_w) // 2
        paste_y = (corner_size * 2 - new_h) // 2
        temp_canvas.paste(corner_img, (paste_x, paste_y), corner_img)
        
        # 分割成四等分
        half_w = corner_size
        half_h = corner_size
        
        # 创建四个角的画布
        top_left = Image.new('RGBA', (corner_size, corner_size), (0, 0, 0, 0))
        top_right = Image.new('RGBA', (corner_size, corner_size), (0, 0, 0, 0))
        bottom_left = Image.new('RGBA', (corner_size, corner_size), (0, 0, 0, 0))
        bottom_right = Image.new('RGBA', (corner_size, corner_size), (0, 0, 0, 0))
        
        # 从临时画布中裁剪并粘贴到对应位置
        # 右下角 -> 左上角
        br_piece = temp_canvas.crop((half_w, half_h, half_w * 2, half_h * 2))
        top_left.paste(br_piece, (0, 0), br_piece)
        
        # 左下角 -> 右上角
        bl_piece = temp_canvas.crop((0, half_h, half_w, half_h * 2))
        top_right.paste(bl_piece, (0, 0), bl_piece)
        
        # 左上角 -> 右下角
        tl_piece = temp_canvas.crop((0, 0, half_w, half_h))
        bottom_right.paste(tl_piece, (0, 0), tl_piece)
        
        # 右上角 -> 左下角
        tr_piece = temp_canvas.crop((half_w, 0, half_w * 2, half_h))
        bottom_left.paste(tr_piece, (0, 0), tr_piece)
        
        return top_left, top_right, bottom_left, bottom_right
    
    def apply_random_offset(self, center_x, center_y, random_seed, min_radius=64, max_radius=256):
        """为中心位置添加随机偏移"""
        # 使用独立的随机状态，避免影响其他随机操作
        rng = random.Random(random_seed)
        
        # 生成随机半径（64-256之间）
        radius = rng.uniform(min_radius, max_radius)
        
        # 生成随机角度（0-2π）
        angle = rng.uniform(0, 2 * math.pi)
        
        # 计算偏移量
        offset_x = int(radius * math.cos(angle))
        offset_y = int(radius * math.sin(angle))
        
        # 计算新位置
        new_x = center_x + offset_x
        new_y = center_y + offset_y
        
        print(f"🎲 随机偏移: 半径={radius:.1f}, 角度={math.degrees(angle):.1f}°")
        print(f"📍 原始中心: ({center_x}, {center_y}) → 偏移后: ({new_x}, {new_y})")
        print(f"📏 偏移距离: ({offset_x:+d}, {offset_y:+d})")
        
        return new_x, new_y
    
    def create_edge_pair(self, edge_image, edge_type, target_size):
        """创建边界对，保持无缝拼接特性
        Args:
            edge_image: 边界图片
            edge_type: 边界类型 'horizontal' 或 'vertical'
            target_size: 目标尺寸 (width, height)
        """
        if edge_type == 'horizontal':
            # 水平边界：上下对等分（无偏移，保持无缝）
            target_width, target_height = target_size
            scale = min(target_width / edge_image.size[0], target_height / (edge_image.size[1] / 2))
            
            new_width = int(edge_image.size[0] * scale)
            new_height = int(edge_image.size[1] * scale)
            resized_img = edge_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 分割成上下两部分
            split_point = new_height // 2
            top_half = resized_img.crop((0, 0, new_width, split_point))
            bottom_half = resized_img.crop((0, split_point, new_width, new_height))
            
            # 创建目标尺寸的画布
            top_canvas = Image.new('RGBA', target_size, (0, 0, 0, 0))
            bottom_canvas = Image.new('RGBA', target_size, (0, 0, 0, 0))
            
            # 居中粘贴（保持无缝对称）
            paste_x = (target_width - new_width) // 2
            top_canvas.paste(bottom_half, (paste_x, 0), bottom_half)  # 上边界使用下半部分
            bottom_canvas.paste(top_half, (paste_x, 0), top_half)     # 下边界使用上半部分
            
            return top_canvas, bottom_canvas
            
        else:  # vertical
            # 垂直边界：左右对等分（无偏移，保持无缝）
            target_width, target_height = target_size
            scale = min(target_height / edge_image.size[1], target_width / (edge_image.size[0] / 2))
            
            new_width = int(edge_image.size[0] * scale)
            new_height = int(edge_image.size[1] * scale)
            resized_img = edge_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 分割成左右两部分
            split_point = new_width // 2
            left_half = resized_img.crop((0, 0, split_point, new_height))
            right_half = resized_img.crop((split_point, 0, new_width, new_height))
            
            # 创建目标尺寸的画布
            left_canvas = Image.new('RGBA', target_size, (0, 0, 0, 0))
            right_canvas = Image.new('RGBA', target_size, (0, 0, 0, 0))
            
            # 居中粘贴（保持无缝对称）
            paste_y = (target_height - new_height) // 2
            left_canvas.paste(right_half, (0, paste_y), right_half)   # 左边界使用右半部分
            right_canvas.paste(left_half, (0, paste_y), left_half)    # 右边界使用左半部分
            
            return left_canvas, right_canvas
    
    def fill_center_area(self, canvas, mask_canvas, center_image, start_x, start_y, end_x, end_y, tile_size, random_seed):
        """填充中心区域，使用固定图片并添加随机偏移"""
        if not center_image:
            return []
        
        center_positions = []
            
        print(f"🎯 随机偏移模板中心填充：固定图1+随机偏移")
        print(f"📐 填充区域: ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
        print(f"📏 设定的图片大小: {tile_size}")
        
        # 计算可用区域大小
        available_width = end_x - start_x
        available_height = end_y - start_y
        
        # 确保图片大小不超过可用空间
        max_size = min(available_width, available_height)
        if tile_size > max_size:
            tile_size = max_size
            print(f"⚠️  图片尺寸调整为: {tile_size} (受可用空间限制)")
        
        # 计算原始居中位置
        original_center_x = start_x + (available_width - tile_size) // 2
        original_center_y = start_y + (available_height - tile_size) // 2
        
        # 应用随机偏移
        offset_x, offset_y = self.apply_random_offset(
            original_center_x + tile_size // 2,  # 转换为图片中心点
            original_center_y + tile_size // 2,
            random_seed + 12345,  # 使用不同的种子避免冲突
            min_radius=64,
            max_radius=256
        )
        
        # 转换回左上角坐标
        x = offset_x - tile_size // 2
        y = offset_y - tile_size // 2
        
        # 确保图片不会完全超出可用区域
        min_x = start_x - tile_size + 32
        max_x = end_x - 32
        min_y = start_y - tile_size + 32
        max_y = end_y - 32
        
        x = max(min_x, min(x, max_x))
        y = max(min_y, min(y, max_y))
        
        print(f"🎯 最终图片放置位置: ({x}, {y})")
        
        # 缩放图片
        tile_img = self.resize_image_keep_ratio(center_image, (tile_size, tile_size), force_size=True)
        
        # 粘贴到画布和遮罩
        canvas.paste(tile_img, (x, y), tile_img)
        if tile_img.mode == 'RGBA':
            mask_canvas.paste(0, (x, y), tile_img)
        
        # 记录中心位置信息
        center_positions.append({
            "type": "center",
            "position": "center_offset",
            "bbox": [x, y, x + tile_size, y + tile_size],
            "image_index": 0  # 固定使用第一张图片
        })
        
        print(f"✅ 中心区域填充完成，使用图1（带随机偏移）")
        
        return center_positions
    
    def generate_tiling(self, images, canvas_size, params):
        """生成随机偏移无缝拼图"""
        
        if not self.validate_params(params):
            raise ValueError("参数验证失败")
        
        # 初始化位置信息列表
        positions = []
        
        if len(images) < 1:
            raise ValueError("至少需要1张图片")
        
        # 获取参数
        输出宽度, 输出高度 = canvas_size
        基础图片尺寸 = params.get("基础图片尺寸", 128)
        边界宽度 = params.get("边界宽度", 基础图片尺寸)
        角落大小 = params.get("角落大小", 基础图片尺寸)
        中间图片大小 = params.get("中间图片大小", 基础图片尺寸)
        填充中间区域 = params.get("填充中间区域", True)
        随机种子 = params.get("随机种子", 0)
        启用随机 = params.get("启用随机", True)
        背景颜色 = params.get("背景颜色", "#FFFFFF")
        
        # 创建画布和遮罩
        bg_color = tuple(int(背景颜色.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        canvas = Image.new('RGBA', (输出宽度, 输出高度), bg_color)
        mask_canvas = Image.new('L', (输出宽度, 输出高度), 255)
        print(f"🎨 创建随机偏移模板画布，尺寸: {输出宽度} x {输出高度}，背景颜色: {背景颜色}")
        
        # 设置随机种子
        if 启用随机:
            random.seed(随机种子)
        
        total_images = len(images)
        print(f"🎯 随机偏移模板图片分配：输入图片数量 = {total_images}")
        
        # 图片分配策略
        corner_image = images[0]  # 角落固定使用图1
        center_image = images[0]  # 中心固定使用图1
        
        # 边界图片分配：确保上下边与左右边不重复
        if total_images >= 3:
            h_edge_image = images[1]  # 上下边使用图2
            v_edge_image = images[2]  # 左右边使用图3
            print(f"📋 图片分配（无重复）：")
            print(f"  • 角落位置：图1（四等分）")
            print(f"  • 中心位置：图1（随机偏移）")
            print(f"  • 上下边界：图2（对等分，无缝拼接）")
            print(f"  • 左右边界：图3（对等分，无缝拼接）")
        elif total_images >= 2:
            h_edge_image = images[1]  # 上下边使用图2
            v_edge_image = images[1]  # 左右边也使用图2
            print(f"📋 图片分配（部分重复）：")
            print(f"  • 角落位置：图1（四等分）")
            print(f"  • 中心位置：图1（随机偏移）")
            print(f"  • 上下边界：图2（对等分，无缝拼接）")
            print(f"  • 左右边界：图2（对等分，无缝拼接）")
        else:
            h_edge_image = images[0]  # 全部使用图1
            v_edge_image = images[0]
            print(f"📋 图片分配（全部重复）：")
            print(f"  • 角落位置：图1（四等分）")
            print(f"  • 中心位置：图1（随机偏移）")
            print(f"  • 上下边界：图1（对等分，无缝拼接）")
            print(f"  • 左右边界：图1（对等分，无缝拼接）")
        
        # 创建并粘贴四个角（总是显示）
        print(f"🔲 创建四个角落（固定使用图1）...")
        tl_corner, tr_corner, bl_corner, br_corner = self.create_corner_pieces(corner_image, 角落大小)
        
        canvas.paste(tl_corner, (0, 0), tl_corner)
        canvas.paste(tr_corner, (输出宽度 - 角落大小, 0), tr_corner)
        canvas.paste(bl_corner, (0, 输出高度 - 角落大小), bl_corner)
        canvas.paste(br_corner, (输出宽度 - 角落大小, 输出高度 - 角落大小), br_corner)
        
        # 记录四个角的位置信息
        positions.extend([
            {
                "type": "corner",
                "position": "top_left",
                "bbox": [0, 0, 角落大小, 角落大小],
                "image_index": images.index(corner_image)
            },
            {
                "type": "corner", 
                "position": "top_right",
                "bbox": [输出宽度 - 角落大小, 0, 输出宽度, 角落大小],
                "image_index": images.index(corner_image)
            },
            {
                "type": "corner",
                "position": "bottom_left", 
                "bbox": [0, 输出高度 - 角落大小, 角落大小, 输出高度],
                "image_index": images.index(corner_image)
            },
            {
                "type": "corner",
                "position": "bottom_right",
                "bbox": [输出宽度 - 角落大小, 输出高度 - 角落大小, 输出宽度, 输出高度],
                "image_index": images.index(corner_image)
            }
        ])
        
        # 遮罩
        if tl_corner.mode == 'RGBA':
            mask_canvas.paste(0, (0, 0), tl_corner)
            mask_canvas.paste(0, (输出宽度 - 角落大小, 0), tr_corner)
            mask_canvas.paste(0, (0, 输出高度 - 角落大小), bl_corner)
            mask_canvas.paste(0, (输出宽度 - 角落大小, 输出高度 - 角落大小), br_corner)
        
        # 填充中间区域（包括边界和中心）
        if 填充中间区域:
            print(f"🎯 开始填充中间区域（边界+中心）...")
            
            # 创建水平边界（上下）
            h_edge_length = 输出宽度 - 2 * 角落大小
            if h_edge_length > 0:
                print(f"📏 创建水平边界（上下，保持无缝）...")
                top_edge, bottom_edge = self.create_edge_pair(
                    h_edge_image, 'horizontal', (h_edge_length, 边界宽度)
                )
                
                canvas.paste(top_edge, (角落大小, 0), top_edge)
                canvas.paste(bottom_edge, (角落大小, 输出高度 - 边界宽度), bottom_edge)
                
                # 记录水平边界位置信息
                positions.extend([
                    {
                        "type": "edge",
                        "position": "top",
                        "bbox": [角落大小, 0, 角落大小 + h_edge_length, 边界宽度],
                        "image_index": images.index(h_edge_image)
                    },
                    {
                        "type": "edge",
                        "position": "bottom", 
                        "bbox": [角落大小, 输出高度 - 边界宽度, 角落大小 + h_edge_length, 输出高度],
                        "image_index": images.index(h_edge_image)
                    }
                ])
                
                if top_edge.mode == 'RGBA':
                    mask_canvas.paste(0, (角落大小, 0), top_edge)
                    mask_canvas.paste(0, (角落大小, 输出高度 - 边界宽度), bottom_edge)
            
            # 创建垂直边界（左右）
            v_edge_length = 输出高度 - 2 * 角落大小
            if v_edge_length > 0:
                print(f"📏 创建垂直边界（左右，保持无缝）...")
                left_edge, right_edge = self.create_edge_pair(
                    v_edge_image, 'vertical', (边界宽度, v_edge_length)
                )
                
                canvas.paste(left_edge, (0, 角落大小), left_edge)
                canvas.paste(right_edge, (输出宽度 - 边界宽度, 角落大小), right_edge)
                
                # 记录垂直边界位置信息
                positions.extend([
                    {
                        "type": "edge",
                        "position": "left",
                        "bbox": [0, 角落大小, 边界宽度, 角落大小 + v_edge_length],
                        "image_index": images.index(v_edge_image)
                    },
                    {
                        "type": "edge",
                        "position": "right",
                        "bbox": [输出宽度 - 边界宽度, 角落大小, 输出宽度, 角落大小 + v_edge_length],
                        "image_index": images.index(v_edge_image)
                    }
                ])
                
                if left_edge.mode == 'RGBA':
                    mask_canvas.paste(0, (0, 角落大小), left_edge)
                    mask_canvas.paste(0, (输出宽度 - 边界宽度, 角落大小), right_edge)
            
            # 填充中心区域
            center_start_x = max(角落大小, 边界宽度)
            center_start_y = max(角落大小, 边界宽度)
            center_end_x = 输出宽度 - max(角落大小, 边界宽度)
            center_end_y = 输出高度 - max(角落大小, 边界宽度)
            
            if center_end_x > center_start_x and center_end_y > center_start_y:
                中间图片实际尺寸 = 基础图片尺寸 * 2
                print(f"🎯 填充中心位置（图1+随机偏移）...")
                center_positions = self.fill_center_area(canvas, mask_canvas, center_image, center_start_x, center_start_y, 
                                    center_end_x, center_end_y, 中间图片实际尺寸, 随机种子)
                positions.extend(center_positions)
            
        else:
            print("🚫 已禁用中间区域填充（不显示边界和中心）")
        
        print(f"✅ 随机偏移拼图模板生成完成")
        print(f"📊 模板特征:")
        print(f"   • 角落位置: 固定使用图1（四等分）")
        print(f"   • 中心位置: 固定使用图1（随机偏移64-256px）")
        print(f"   • 边界特性: 保持无缝四方连续拼接")
        print(f"   • 图片分配: >2张时上下左右不重复，≤2张时智能重复")
        print(f"   • 图片需求: 最少1张，推荐3张（实现无重复）")
        print(f"   • 填充控制: {'✅ 启用边界+中心' if 填充中间区域 else '❌ 只显示角落'}")
        
        return canvas, mask_canvas, positions