"""
增强经典无缝拼图模板

基于经典模板，在四个象限的几何中心增加4个图片位置，
形成1个中心 + 4个象限中心的5图片系统，
共同受"填充中间区域"开关控制
"""

import random
from PIL import Image
from .classic_seamless_template import ClassicSeamlessTemplate


class EnhancedClassicTemplate(ClassicSeamlessTemplate):
    """增强经典无缝拼图模板 - 在经典模板基础上增加4个象限中心图片"""
    
    def __init__(self):
        super().__init__()
        self.template_name = "增强经典拼图"
        self.template_description = "基于经典模板增加4个象限中心图片，形成5图片中心系统"
    
    def get_template_info(self):
        """返回模板信息"""
        return {
            "name": self.template_name,
            "description": self.template_description
        }
    
    def allocate_images_for_enhanced_template(self, images, random_enabled, random_seed):
        """为增强模板分配图片 - 边缘尽量不重复，中心使用剩余图片"""
        if random_enabled:
            random.seed(random_seed)
            shuffled_images = images.copy()
            random.shuffle(shuffled_images)
        else:
            shuffled_images = images.copy()
        
        total_images = len(shuffled_images)
        print(f"🎯 图片分配策略：输入图片数量 = {total_images}")
        
        # 边缘位置分配（8个位置：4个角落 + 4个边缘）
        edge_images = {}
        used_for_edges = []
        
        if total_images >= 3:
            # 足够的图片，按经典模板的正确分配（3张图片：角落+水平边界+垂直边界）
            edge_images["角落"] = shuffled_images[0]           # 角落图片
            edge_images["水平边界"] = shuffled_images[1]        # 上下边缘共用
            edge_images["垂直边界"] = shuffled_images[2]        # 左右边缘共用
            used_for_edges = shuffled_images[:3]
            print(f"📋 边缘分配：3个图片，符合经典模板标准（角落+水平边界+垂直边界）")
            
        elif total_images >= 2:
            # 2个图片，角落1张，水平垂直边界共用1张
            edge_images["角落"] = shuffled_images[0]           # 角落图片
            edge_images["水平边界"] = shuffled_images[1]        # 水平边界
            edge_images["垂直边界"] = shuffled_images[1]        # 垂直边界重复使用
            used_for_edges = shuffled_images[:2]
            print(f"📋 边缘分配：2个图片，角落独立，水平垂直边界共用")
            
        else:
            # 只有1个图片，全部位置重复使用
            edge_images["角落"] = shuffled_images[0]
            edge_images["水平边界"] = shuffled_images[0]
            edge_images["垂直边界"] = shuffled_images[0]
            used_for_edges = shuffled_images
            print(f"📋 边缘分配：1个图片重复使用，覆盖所有边缘位置")
        
        # 中心位置分配（使用剩余图片）
        if total_images >= 3:
            # 有足够图片时，中心使用未被边缘使用的图片
            center_images = shuffled_images[3:]
            if not center_images:  # 如果正好3个图片
                center_images = shuffled_images  # 使用所有图片，允许重复
            print(f"🎯 中心分配：{len(center_images)}个剩余图片，与边缘{'不重复' if len(center_images) > 0 and total_images > 3 else '可能重复'}")
        else:
            # 图片不足时，中心可以使用所有图片
            center_images = shuffled_images
            print(f"🎯 中心分配：{len(center_images)}个图片（与边缘可能重复，因为图片总数 < 3）")
        
        print(f"📊 分配结果：")
        print(f"   • 边缘使用：{len(used_for_edges)}个不同图片")
        print(f"   • 中心使用：{len(center_images)}个图片")
        print(f"   • 是否重复：{'❌ 无重复' if total_images >= 3 else '✅ 允许重复'}")
        
        return edge_images, center_images
    
    def calculate_enhanced_quadrant_positions(self, canvas_size, tile_size):
        """计算增强象限位置 - 真正的象限中心，不受边缘图片大小影响"""
        输出宽度, 输出高度 = canvas_size
        
        # 计算真正的象限中心位置（基于整个画布，不受边缘图片影响）
        # 象限划分：将画布分为4个相等的象限
        quadrant_width = 输出宽度 // 2
        quadrant_height = 输出高度 // 2
        
        print(f"📐 画布尺寸：{输出宽度} x {输出高度}")
        print(f"📐 象限尺寸：{quadrant_width} x {quadrant_height}")
        
        positions = {}
        
        # 计算画布的整体中心
        canvas_center_x = 输出宽度 // 2
        canvas_center_y = 输出高度 // 2
        
        # 整体中心位置
        positions["整体中心"] = (
            canvas_center_x - tile_size // 2,
            canvas_center_y - tile_size // 2
        )
        
        # 真正的象限中心位置（不受边缘图片大小影响）
        # 4个象限中心位置（图片左上角坐标）
        positions["左上象限中心"] = (
            quadrant_width // 2 - tile_size // 2,
            quadrant_height // 2 - tile_size // 2
        )
        
        positions["右上象限中心"] = (
            quadrant_width + quadrant_width // 2 - tile_size // 2,
            quadrant_height // 2 - tile_size // 2
        )
        
        positions["左下象限中心"] = (
            quadrant_width // 2 - tile_size // 2,
            quadrant_height + quadrant_height // 2 - tile_size // 2
        )
        
        positions["右下象限中心"] = (
            quadrant_width + quadrant_width // 2 - tile_size // 2,
            quadrant_height + quadrant_height // 2 - tile_size // 2
        )
        
        print(f"🎯 真正的象限中心位置（固定比例位置）：")
        for name, (x, y) in positions.items():
            print(f"   {name}: ({x}, {y})")
        
        # 计算象限中心间的距离验证
        center_pos = positions["整体中心"]
        center_x = center_pos[0] + tile_size // 2
        center_y = center_pos[1] + tile_size // 2
        
        print(f"🔗 象限中心距离验证：")
        distances = []
        for name, (pos_x, pos_y) in positions.items():
            if name != "整体中心":
                pos_center_x = pos_x + tile_size // 2
                pos_center_y = pos_y + tile_size // 2
                distance = ((pos_center_x - center_x) ** 2 + (pos_center_y - center_y) ** 2) ** 0.5
                distances.append(distance)
                print(f"   整体中心 → {name}: {distance:.1f}像素")
        
        # 检查是否等距
        if len(set([round(d) for d in distances])) == 1:
            print(f"   ✅ 所有象限中心等距离分布")
        else:
            print(f"   ⚠️  象限中心距离不完全相等")
        
        return positions
    
    def fill_center_area_enhanced(self, canvas, mask_canvas, center_images, canvas_size, tile_size, random_enabled, random_seed):
        """增强的中心区域填充 - 填充5个象限位置，使用剩余图片"""
        
        if not center_images:
            print("⚠️  没有中心图片可填充")
            return []
        
        center_positions = []
        
        print(f"🎯 增强经典模板：填充5个象限位置（使用{len(center_images)}个剩余图片）")
        
        # 计算所有5个象限位置（按照原始需求）
        positions = self.calculate_enhanced_quadrant_positions(canvas_size, tile_size)
        
        # 位置顺序 - 先中心，再四个象限（按照你原始需求的红点位置）
        fill_order = ["整体中心", "左上象限中心", "右上象限中心", "左下象限中心", "右下象限中心"]
        
        # 设置随机种子用于图片选择
        if random_enabled:
            random.seed(random_seed)
        
        # 创建图片索引，确保有足够的变化
        img_indices = []
        available_indices = list(range(len(center_images)))
        
        for i in range(5):  # 5个位置
            if available_indices:
                chosen_idx = random.choice(available_indices)
                img_indices.append(chosen_idx)
                # 如果图片足够多，移除已选择的图片避免重复
                if len(available_indices) > 1:
                    available_indices.remove(chosen_idx)
            else:
                # 重新从所有中心图片中选择
                available_indices = list(range(len(center_images)))
                chosen_idx = random.choice(available_indices)
                img_indices.append(chosen_idx)
                available_indices.remove(chosen_idx)
        
        # 填充所有5个位置
        for i, pos_name in enumerate(fill_order):
            x, y = positions[pos_name]
            
            # 确保位置在有效范围内
            输出宽度, 输出高度 = canvas_size
            x = max(0, min(x, 输出宽度 - tile_size))
            y = max(0, min(y, 输出高度 - tile_size))
            
            # 选择图片
            img = center_images[img_indices[i]]
            
            # 缩放图片，保持原始比例
            tile_img = self.resize_image_keep_ratio(img, (tile_size, tile_size), force_size=True)
            
            # 粘贴到画布和遮罩
            canvas.paste(tile_img, (x, y), tile_img)
            if tile_img.mode == 'RGBA':
                mask_canvas.paste(0, (x, y), tile_img)
            
            # 记录位置信息
            center_positions.append({
                "type": "center",
                "position": pos_name,
                "bbox": [x, y, x + tile_size, y + tile_size],
                "image_index": center_images.index(img)
            })
            
            print(f"🎯 {pos_name}({i+1}/5) 位置: ({x}, {y})")
        
        print(f"✅ 增强中心区域填充完成，放置了5个象限图片")
        print(f"📏 所有图片尺寸: {tile_size}x{tile_size}")
        print(f"🔗 形成真正的象限中心布局（红点位置），不受边缘图片大小影响")
        
        # 重置随机种子
        if random_enabled:
            random.seed()
        
        return center_positions
    
    def generate_tiling(self, images, canvas_size, params):
        """生成增强经典无缝拼图"""
        
        if not self.validate_params(params):
            raise ValueError("参数验证失败")
        
        if len(images) < 1:
            raise ValueError("至少需要1张图片")
        
        # 初始化位置信息列表
        positions = []
        
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
        print(f"🎨 创建增强经典模板画布，尺寸: {输出宽度} x {输出高度}，背景颜色: {背景颜色}")
        
        # 增强经典模板使用与经典模板相同的尺寸计算方式
        象限图片尺寸 = int(基础图片尺寸 * 2)  # 与经典模板的中心图片保持相同尺寸
        print(f"📏 基础图片尺寸: {基础图片尺寸}")
        print(f"🎯 象限图片尺寸: {象限图片尺寸} (与经典模板中心图片尺寸一致)")
        
        # 智能图片分配：边缘尽量不重复，中心使用剩余图片
        edge_images, center_images = self.allocate_images_for_enhanced_template(images, 启用随机, 随机种子)
        
        # 1. 创建四个角落（总是显示，不受开关控制）- 参考经典模板的正确处理
        # 角落必须来自同一张图片的四等分，确保无缝拼图效果
        corner_image = edge_images["角落"]  # 角落图片
        h_edge_image = edge_images["水平边界"]  # 水平边界图片（上下共用）
        v_edge_image = edge_images["垂直边界"]  # 垂直边界图片（左右共用）
        
        print(f"🔲 角落图片: {type(corner_image).__name__}, 🔄 水平边界: {type(h_edge_image).__name__}, ↕️ 垂直边界: {type(v_edge_image).__name__}")
        
        tl_corner, tr_corner, bl_corner, br_corner = self.create_corner_pieces(corner_image, 角落大小)
        
        # 粘贴四个角到画布和遮罩 - 与经典模板相同的处理方式
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
        
        print(f"🔲 角落处理：使用1张图片创建4个角落片段（无缝拼图标准）")
        
        # 2. 创建边缘（总是显示，不受开关控制）- 参考经典模板的正确处理
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
        
        print(f"🔄 边缘处理：水平边界（上下共用）+ 垂直边界（左右共用），符合无缝拼图标准")
        
        # 3. 填充增强的中心区域（5个图片，受开关控制）
        if 填充中间区域:
            print(f"🎯 开始填充增强中心区域（5个象限位置，使用剩余图片）")
            center_positions = self.fill_center_area_enhanced(
                canvas, mask_canvas, center_images, 
                (输出宽度, 输出高度), 象限图片尺寸,
                启用随机, 随机种子
            )
            positions.extend(center_positions)
        else:
            print("⏸️  中间区域填充已禁用，跳过5个中心位置的填充")
        
        print(f"✅ 增强经典模板生成完成")
        print(f"📊 模板特征:")
        print(f"   • 边缘图片: 3个位置（1个角落+1个水平边界+1个垂直边界），符合经典模板标准")
        print(f"   • 角落处理: 1张图片的四等分（符合无缝拼图标准）") 
        print(f"   • 边缘处理: 水平边界（上下共用）+ 垂直边界（左右共用）")
        print(f"   • 中心图片: 5个（受开关控制 - 1个整体中心 + 4个象限中心）")
        print(f"   • 图片分配: 边缘优先，中心使用剩余，{'无重复' if len(images) >= 3 else '允许重复'}")
        print(f"   • 布局方式: 真正的象限中心布局（红点位置），固定几何位置")
        print(f"   • 开关状态: {'✅ 启用' if 填充中间区域 else '❌ 禁用'}")
        
        # 重置随机种子
        random.seed()
        
        return canvas, mask_canvas, positions