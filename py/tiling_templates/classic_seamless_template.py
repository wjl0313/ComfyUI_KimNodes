"""
经典四方连续无缝拼图模板

实现传统的四方连续拼图效果，从图片列表中随机选择4张不同图片：
- 四个角：随机选择一张图片的四等分，实现无缝连接
- 上边与下边：随机选择一张图片的对等分
- 左边与右边：随机选择一张图片的对等分  
- 中间区域：随机选择一张图片进行填充

每次生成时随机分配4张不同图片到四个位置，确保无重复且具有随机性。
"""

import random
from PIL import Image
from .base_template import TilingTemplateBase


class ClassicSeamlessTemplate(TilingTemplateBase):
    """经典四方连续无缝拼图模板"""
    
    def __init__(self):
        super().__init__()
        self.template_name = "经典四方连续"
        self.template_description = "传统的四方连续拼图，边界和角落实现无缝对接。从图片列表中随机选择4张不同图片分配到角落、上下边、左右边、中心位置"
    
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
        """从一张图片创建四个角，特殊排列以实现无缝效果：
        - 左上角 = 原图右下角
        - 右上角 = 原图左下角
        - 右下角 = 原图左上角
        - 左下角 = 原图右上角
        """
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
    
    def fill_center_area(self, canvas, mask_canvas, images, start_x, start_y, end_x, end_y, tile_size, random_seed):
        """使用不同图片填充中间区域，自动调整布局"""
        if not images:
            return []
        
        center_positions = []
            
        print(f"🧩 经典模板中间填充策略：单个图片居中放置")
        print(f"📷 可用图片数量: {len(images)}")
        print(f"📐 填充区域: ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
        print(f"📏 设定的图片大小: {tile_size}")
        
        # 计算可用区域大小
        available_width = end_x - start_x
        available_height = end_y - start_y
        
        # 确保图片大小不超过可用空间，但尽量保持设定尺寸
        max_size = min(available_width, available_height)
        if tile_size > max_size:
            tile_size = max_size
            print(f"⚠️  图片尺寸调整为: {tile_size} (受可用空间限制)")
        
        # 经典模板策略：默认只在中心填充一个图片
        num_cols = 1
        num_rows = 1
            
        print(f"使用设定的图片大小: {tile_size}")
        
        # 计算实际的图片大小（保持设定大小）
        actual_tile_width = tile_size
        actual_tile_height = tile_size
        
        # 计算图片之间的间距
        x_spacing = (available_width - num_cols * tile_size) / max(1, num_cols - 1) if num_cols > 1 else 0
        y_spacing = (available_height - num_rows * tile_size) / max(1, num_rows - 1) if num_rows > 1 else 0
        
        print(f"📋 布局策略: {num_rows} 行 x {num_cols} 列 (单个图片居中)")
        print(f"📐 实际图片大小: {actual_tile_width} x {actual_tile_height}")
        
        # 设置随机种子并选择图片
        random.seed(random_seed)
        selected_img = random.choice(images)
        
        # 计算居中位置
        x = start_x + (available_width - tile_size) // 2
        y = start_y + (available_height - tile_size) // 2
        
        print(f"🎯 图片放置位置: ({x}, {y})")
        
        # 缩放图片到实际大小，保持原始比例
        tile_img = self.resize_image_keep_ratio(selected_img, (tile_size, tile_size), force_size=True)
        
        # 粘贴到画布和遮罩
        canvas.paste(tile_img, (x, y), tile_img)
        # 在遮罩上使用图片的alpha通道
        if tile_img.mode == 'RGBA':
            mask_canvas.paste(0, (x, y), tile_img)
        
        # 记录中心图片位置信息
        center_positions.append({
            "type": "center",
            "position": "center",
            "bbox": [x, y, x + tile_size, y + tile_size],
            "image_index": images.index(selected_img)
        })
        
        print(f"✅ 经典模板中间区域填充完成，放置了 1 个图片")
        
        # 重置随机种子
        random.seed()
        
        return center_positions
    
    def generate_tiling(self, images, canvas_size, params):
        """生成经典四方连续无缝拼图"""
        
        if not self.validate_params(params):
            raise ValueError("参数验证失败")
        
        if len(images) < 4:
            raise ValueError("经典四方连续模板至少需要4张不同的图片（角落、上下边、左右边、中心各1张）")
        
        # 初始化位置信息列表
        positions = []
        
        # 获取参数 - 优先使用基础图片尺寸确保一致性
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
        # 将十六进制颜色转换为RGBA
        bg_color = tuple(int(背景颜色.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        canvas = Image.new('RGBA', (输出宽度, 输出高度), bg_color)
        mask_canvas = Image.new('L', (输出宽度, 输出高度), 255)  # 创建白色背景的遮罩
        print(f"🎨 创建画布和遮罩，尺寸: {输出宽度} x {输出高度}，背景颜色: {背景颜色}")
        
        # 设置随机种子
        if 启用随机:
            random.seed(随机种子)
        
        # 从图片列表中随机选择4张不同的图片用于四个位置
        if len(images) >= 4:
            # 随机选择4张不重复的图片
            selected_images = random.sample(images, 4)
            corner_image = selected_images[0]    # 角落图片
            h_edge_image = selected_images[1]    # 水平边界图片（上下边）
            v_edge_image = selected_images[2]    # 垂直边界图片（左右边）
            center_image = selected_images[3]    # 中心图片
        else:
            # 如果图片不足4张，按顺序分配（保持原有逻辑作为后备）
            corner_image = images[0]
            h_edge_image = images[1] if len(images) >= 2 else images[0]
            v_edge_image = images[2] if len(images) >= 3 else images[0]
            center_image = images[3] if len(images) >= 4 else images[0]
        
        # 中间区域填充只使用专门的中心图片
        fill_images = [center_image]
            
        print(f"🎲 随机选择的图片分配:")
        print(f"🔲 角落图片: {images.index(corner_image) + 1}号图片")
        print(f"🔄 水平边界（上下边）: {images.index(h_edge_image) + 1}号图片") 
        print(f"↕️ 垂直边界（左右边）: {images.index(v_edge_image) + 1}号图片")
        print(f"🎯 中心图片: {images.index(center_image) + 1}号图片")
        print(f"🧩 四个位置使用4张不同图片，随机分配，无重复")
        
        # 创建四个角
        tl_corner, tr_corner, bl_corner, br_corner = self.create_corner_pieces(corner_image, 角落大小)
        
        # 粘贴四个角到画布和遮罩
        # 画布
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
        
        # 遮罩（使用图片的alpha通道）
        if tl_corner.mode == 'RGBA':
            mask_canvas.paste(0, (0, 0), tl_corner)
            mask_canvas.paste(0, (输出宽度 - 角落大小, 0), tr_corner)
            mask_canvas.paste(0, (0, 输出高度 - 角落大小), bl_corner)
            mask_canvas.paste(0, (输出宽度 - 角落大小, 输出高度 - 角落大小), br_corner)
        
        # 创建水平边界（上下）- 同一张图片的对等分
        h_edge_length = 输出宽度 - 2 * 角落大小
        if h_edge_length > 0:
            # 计算缩放比例
            scale = min(h_edge_length / h_edge_image.size[0], 边界宽度 / (h_edge_image.size[1] / 2))
            
            # 使用相同的缩放比例处理图片
            new_width = int(h_edge_image.size[0] * scale)
            new_height = int(h_edge_image.size[1] * scale)
            h_edge_img = h_edge_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 计算上下分割点
            split_point = new_height // 2
            
            # 分割成上下两部分
            top_half = h_edge_img.crop((0, 0, new_width, split_point))
            bottom_half = h_edge_img.crop((0, split_point, new_width, new_height))
            
            # 创建目标尺寸的画布并对齐到边缘
            top_canvas = Image.new('RGBA', (h_edge_length, 边界宽度), (0, 0, 0, 0))
            bottom_canvas = Image.new('RGBA', (h_edge_length, 边界宽度), (0, 0, 0, 0))
            
            # 水平居中，垂直对齐到边缘
            paste_x = (h_edge_length - new_width) // 2
            top_canvas.paste(bottom_half, (paste_x, 0), bottom_half)  # 上边界使用下半部分，对齐到顶部
            bottom_canvas.paste(top_half, (paste_x, 0), top_half)     # 下边界使用上半部分，对齐到顶部
            
            # 粘贴到主画布和遮罩
            canvas.paste(top_canvas, (角落大小, 0), top_canvas)
            canvas.paste(bottom_canvas, (角落大小, 输出高度 - 边界宽度), bottom_canvas)
            
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
            
            # 在遮罩上使用图片的alpha通道
            if top_canvas.mode == 'RGBA':
                mask_canvas.paste(0, (角落大小, 0), top_canvas)
                mask_canvas.paste(0, (角落大小, 输出高度 - 边界宽度), bottom_canvas)
        
        # 创建垂直边界（左右）- 同一张图片的对等分
        v_edge_length = 输出高度 - 2 * 角落大小
        if v_edge_length > 0:
            # 计算缩放比例
            scale = min(v_edge_length / v_edge_image.size[1], 边界宽度 / (v_edge_image.size[0] / 2))
            
            # 使用相同的缩放比例处理图片
            new_width = int(v_edge_image.size[0] * scale)
            new_height = int(v_edge_image.size[1] * scale)
            v_edge_img = v_edge_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 计算左右分割点
            split_point = new_width // 2
            
            # 分割成左右两部分
            left_half = v_edge_img.crop((0, 0, split_point, new_height))
            right_half = v_edge_img.crop((split_point, 0, new_width, new_height))
            
            # 创建目标尺寸的画布并对齐到边缘
            left_canvas = Image.new('RGBA', (边界宽度, v_edge_length), (0, 0, 0, 0))
            right_canvas = Image.new('RGBA', (边界宽度, v_edge_length), (0, 0, 0, 0))
            
            # 垂直居中，水平对齐到边缘
            paste_y = (v_edge_length - new_height) // 2
            left_canvas.paste(right_half, (0, paste_y), right_half)    # 左边界使用右半部分，对齐到左边
            right_canvas.paste(left_half, (0, paste_y), left_half)     # 右边界使用左半部分，对齐到左边
            
            # 粘贴到主画布和遮罩
            canvas.paste(left_canvas, (0, 角落大小), left_canvas)
            canvas.paste(right_canvas, (输出宽度 - 边界宽度, 角落大小), right_canvas)
            
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
            
            # 在遮罩上使用图片的alpha通道
            if left_canvas.mode == 'RGBA':
                mask_canvas.paste(0, (0, 角落大小), left_canvas)
                mask_canvas.paste(0, (输出宽度 - 边界宽度, 角落大小), right_canvas)
        
        # 填充中间区域
        if 填充中间区域:
            print("🎨 开始经典模板中间区域填充...")
            center_start_x = max(角落大小, 边界宽度)
            center_start_y = max(角落大小, 边界宽度)
            center_end_x = 输出宽度 - max(角落大小, 边界宽度)
            center_end_y = 输出高度 - max(角落大小, 边界宽度)
            
            print(f"📐 中间区域范围: ({center_start_x}, {center_start_y}) 到 ({center_end_x}, {center_end_y})")
            
            # 中间图片需要放大一倍，因为边缘使用的是裁切后的图片片段
            # 而中间使用完整图片，为了视觉一致性，需要放大
            中间图片实际尺寸 = 基础图片尺寸 * 2
            print(f"📏 基础图片尺寸: {基础图片尺寸}")
            print(f"🔍 中间图片实际尺寸: {中间图片实际尺寸} (放大一倍以匹配边缘裁切效果)")
            
            if center_end_x > center_start_x and center_end_y > center_start_y:
                center_positions = self.fill_center_area(canvas, mask_canvas, fill_images, center_start_x, center_start_y, 
                                    center_end_x, center_end_y, 中间图片实际尺寸, 随机种子)
                positions.extend(center_positions)
            else:
                print("⚠️  中间区域空间不足，跳过填充")
        else:
            print("🚫 已禁用中间区域填充")
        
        return canvas, mask_canvas, positions
