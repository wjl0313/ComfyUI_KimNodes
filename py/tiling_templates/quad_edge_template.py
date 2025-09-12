"""
四边界拼图模板

改进的边界布局策略：
- 四个角固定使用图1（四等分）
- 中心点固定使用图1（2倍尺寸，居中无偏移）
- 上下边界：4对图片，每个图片独立上下偏移±30px
- 左右边界：4对图片，每个图片独立左右偏移±30px
- 防重叠设计：偏移范围控制，确保不与角落图片重叠
- 对称美学：对边图片反向偏移，保持视觉平衡
- 视觉丰富：每个边界图片位置随机化，避免单调排列
"""

import random
import math
from PIL import Image
from .base_template import TilingTemplateBase


class QuadEdgeTemplate(TilingTemplateBase):
    """四边界拼图模板"""
    
    def __init__(self):
        super().__init__()
        self.template_name = "四边界拼图"
        self.template_description = "角落固定图1，每边4对边界图片独立随机偏移，上下边界上下偏移，左右边界左右偏移，避免重叠确保视觉丰富性"
    
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
    
    def fill_center_area(self, canvas, mask_canvas, center_image, start_x, start_y, end_x, end_y, tile_size):
        """填充中心区域，使用固定图片居中放置（不添加随机偏移）"""
        if not center_image:
            return []
        
        center_positions = []
            
        print(f"🎯 四边界模板中心填充：固定图1（2倍尺寸）+居中放置")
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
        
        # 计算居中位置（不添加随机偏移）
        x = start_x + (available_width - tile_size) // 2
        y = start_y + (available_height - tile_size) // 2
        
        print(f"🎯 最终图片放置位置: ({x}, {y}) - 居中无偏移")
        
        # 缩放图片
        tile_img = self.resize_image_keep_ratio(center_image, (tile_size, tile_size), force_size=True)
        
        # 粘贴到画布和遮罩
        canvas.paste(tile_img, (x, y), tile_img)
        if tile_img.mode == 'RGBA':
            mask_canvas.paste(0, (x, y), tile_img)
        
        # 记录中心位置信息
        center_positions.append({
            "type": "center",
            "position": "center",
            "bbox": [x, y, x + tile_size, y + tile_size],
            "image_index": 0  # 固定使用第一张图片
        })
        
        print(f"✅ 中心区域填充完成，使用图1（2倍尺寸，居中无偏移）")
        
        return center_positions
    
    def fill_multiple_horizontal_edges(self, canvas, mask_canvas, h_edge_images, start_x, end_x, 
                                     top_y, bottom_y, edge_width, random_seed):
        """填充多个水平边界（上下四对，每个图片独立随机上下偏移）"""
        h_edge_length = end_x - start_x
        if h_edge_length <= 0:
            return
            
        print(f"📏 创建多个水平边界（上下四对，每个图片独立上下偏移）...")
        
        # 计算四对边界的位置
        segment_width = h_edge_length // 4
        
        # 设置随机种子确保可重现
        random.seed(random_seed)
        
        for i in range(4):
            # 计算当前段的位置
            segment_start_x = start_x + i * segment_width
            segment_end_x = start_x + (i + 1) * segment_width
            if i == 3:  # 最后一段使用剩余空间
                segment_end_x = end_x
            
            current_width = segment_end_x - segment_start_x
            
            # 获取对应的图片（每对使用同一张图）
            edge_image = h_edge_images[i] if i < len(h_edge_images) else h_edge_images[0]
            
            # 缩放图片（适当放大）
            scale = min(current_width / edge_image.size[0], edge_width / edge_image.size[1]) * 1.5
            new_width = int(edge_image.size[0] * scale)
            new_height = int(edge_image.size[1] * scale)
            resized_img = edge_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 为每个图片生成独立的上下偏移（范围缩小避免重叠）
            individual_y_offset = random.randint(-30, 30)  # 减小偏移范围
            
            # 水平中心位置（不偏移，保持在段内居中）
            center_x = (segment_start_x + segment_end_x) // 2
            
            # 计算基础位置（避免与角落重叠）
            # 上边界：基础位置在边界区域内
            top_base_y = top_y + edge_width // 2
            # 下边界：基础位置在边界区域内  
            bottom_base_y = bottom_y + edge_width // 2
            
            # 应用独立偏移
            top_center_y = top_base_y + individual_y_offset
            bottom_center_y = bottom_base_y - individual_y_offset  # 下边界反向偏移保持对称
            
            # 基于中心点计算图片放置位置
            paste_x = center_x - (new_width // 2)
            top_paste_y = top_center_y - (new_height // 2)
            bottom_paste_y = bottom_center_y - (new_height // 2)
            
            # 直接在主画布上粘贴
            canvas.paste(resized_img, (paste_x, top_paste_y), resized_img)
            canvas.paste(resized_img, (paste_x, bottom_paste_y), resized_img)
            
            # 更新遮罩
            if resized_img.mode == 'RGBA':
                mask_canvas.paste(0, (paste_x, top_paste_y), resized_img)
                mask_canvas.paste(0, (paste_x, bottom_paste_y), resized_img)
            
            print(f"  完成第{i+1}对水平边界: 上边中心({center_x}, {top_center_y}) 下边中心({center_x}, {bottom_center_y}) 偏移:{individual_y_offset}px")
    
    def fill_multiple_vertical_edges(self, canvas, mask_canvas, v_edge_images, start_y, end_y,
                                   left_x, right_x, edge_width, random_seed):
        """填充多个垂直边界（左右四对，每个图片独立随机左右偏移）"""
        v_edge_length = end_y - start_y
        if v_edge_length <= 0:
            return
            
        print(f"📏 创建多个垂直边界（左右四对，每个图片独立左右偏移）...")
        
        # 计算四对边界的位置
        segment_height = v_edge_length // 4
        
        # 使用不同的随机种子避免与水平边界偏移相同
        random.seed(random_seed + 1000)
        
        for i in range(4):
            # 计算当前段的位置
            segment_start_y = start_y + i * segment_height
            segment_end_y = start_y + (i + 1) * segment_height
            if i == 3:  # 最后一段使用剩余空间
                segment_end_y = end_y
            
            current_height = segment_end_y - segment_start_y
            
            # 获取对应的图片（每对使用同一张图）
            edge_image = v_edge_images[i] if i < len(v_edge_images) else v_edge_images[0]
            
            # 缩放图片（适当放大）
            scale = min(edge_width / edge_image.size[0], current_height / edge_image.size[1]) * 1.5
            new_width = int(edge_image.size[0] * scale)
            new_height = int(edge_image.size[1] * scale)
            resized_img = edge_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 为每个图片生成独立的左右偏移（范围缩小避免重叠）
            individual_x_offset = random.randint(-30, 30)  # 减小偏移范围
            
            # 垂直中心位置（不偏移，保持在段内居中）
            center_y = (segment_start_y + segment_end_y) // 2
            
            # 计算基础位置（避免与角落重叠）
            # 左边界：基础位置在边界区域内
            left_base_x = left_x + edge_width // 2
            # 右边界：基础位置在边界区域内
            right_base_x = right_x + edge_width // 2
            
            # 应用独立偏移
            left_center_x = left_base_x + individual_x_offset
            right_center_x = right_base_x - individual_x_offset  # 右边界反向偏移保持对称
            
            # 基于中心点计算图片放置位置
            left_paste_x = left_center_x - (new_width // 2)
            right_paste_x = right_center_x - (new_width // 2)
            paste_y = center_y - (new_height // 2)
            
            # 直接在主画布上粘贴
            canvas.paste(resized_img, (left_paste_x, paste_y), resized_img)
            canvas.paste(resized_img, (right_paste_x, paste_y), resized_img)
            
            # 更新遮罩
            if resized_img.mode == 'RGBA':
                mask_canvas.paste(0, (left_paste_x, paste_y), resized_img)
                mask_canvas.paste(0, (right_paste_x, paste_y), resized_img)
            
            print(f"  完成第{i+1}对垂直边界: 左边中心({left_center_x}, {center_y}) 右边中心({right_center_x}, {center_y}) 偏移:{individual_x_offset}px")
    
    def generate_tiling(self, images, canvas_size, params):
        """生成四边界无缝拼图"""
        
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
        print(f"🎨 创建四边界模板画布，尺寸: {输出宽度} x {输出高度}，背景颜色: {背景颜色}")
        
        # 设置随机种子
        if 启用随机:
            random.seed(随机种子)
        
        # 移除全局偏移，改为每个图片独立偏移
        print(f"🎯 使用独立随机偏移策略：每个边界图片独立偏移，避免与角落重叠")
        print(f"📏 偏移范围：±30px（减小范围确保不重叠）")
        
        total_images = len(images)
        print(f"🎯 四边界模板图片分配：输入图片数量 = {total_images}")
        print(f"📊 图片需求：角落+中心(图1) + 上下边界(4对) + 左右边界(4对) = 共需9张不同图片")
        
        # 图片分配策略
        corner_image = images[0]  # 角落固定使用图1
        center_image = images[0]  # 中心固定使用图1
        
        # 上下边界分配：4对，每对使用同一张图（图2、图3、图4、图5）
        h_edge_images = []
        for i in range(4):  # 需要4张图片用于4对上下边界
            img_index = (i + 1) % total_images  # 从图2开始循环使用
            h_edge_images.append(images[img_index])
        
        # 左右边界分配：4对，每对使用同一张图（图6、图7、图8、图9）
        v_edge_images = []
        for i in range(4):  # 需要4张图片用于4对左右边界
            img_index = (i + 5) % total_images  # 从图6开始，如果不够则循环
            v_edge_images.append(images[img_index])
        
        print(f"📋 图片分配策略：")
        print(f"  • 角落位置：图1（四等分）")
        print(f"  • 中心位置：图1（2倍尺寸，居中无偏移）")
        print(f"  • 上下边界：图{[((i+1)%total_images)+1 for i in range(4)]}（4对，每对内保持无缝对称）")
        print(f"  • 左右边界：图{[((i+5)%total_images)+1 for i in range(4)]}（4对，每对内保持无缝对称）")
        
        if total_images >= 9:
            print(f"✅ 图片充足，可实现完全无重复分配")
        else:
            print(f"⚠️  图片数量{total_images}张，会有循环重复")
        
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
            print(f"🎯 开始填充中间区域（四边界+中心）...")
            
            # 创建多个水平边界（上下四对）
            self.fill_multiple_horizontal_edges(
                canvas, mask_canvas, h_edge_images,
                角落大小, 输出宽度 - 角落大小,
                0, 输出高度 - 边界宽度, 边界宽度,
                随机种子
            )
            
            # 创建多个垂直边界（左右四对）
            self.fill_multiple_vertical_edges(
                canvas, mask_canvas, v_edge_images,
                角落大小, 输出高度 - 角落大小,
                0, 输出宽度 - 边界宽度, 边界宽度,
                随机种子
            )
            
            # 填充中心区域
            center_start_x = max(角落大小, 边界宽度)
            center_start_y = max(角落大小, 边界宽度)
            center_end_x = 输出宽度 - max(角落大小, 边界宽度)
            center_end_y = 输出高度 - max(角落大小, 边界宽度)
            
            if center_end_x > center_start_x and center_end_y > center_start_y:
                中间图片实际尺寸 = 基础图片尺寸 * 2  # 固定为2倍尺寸
                print(f"🎯 填充中心位置（图1，2倍尺寸，居中无偏移）...")
                center_positions = self.fill_center_area(canvas, mask_canvas, center_image, center_start_x, center_start_y, 
                                    center_end_x, center_end_y, 中间图片实际尺寸)
                positions.extend(center_positions)
            
        else:
            print("🚫 已禁用中间区域填充（不显示边界和中心）")
        
        print(f"✅ 四边界拼图模板生成完成")
        print(f"📊 模板特征:")
        print(f"   • 角落位置: 固定使用图1（四等分）")
        print(f"   • 中心位置: 固定使用图1（2倍尺寸，居中无偏移）")
        print(f"   • 上下边界: 4对边界（每个图片独立上下偏移±30px）")
        print(f"   • 左右边界: 4对边界（每个图片独立左右偏移±30px）")
        print(f"   • 偏移策略: 每个边界图片独立随机偏移，避免与角落重叠")
        print(f"   • 对称特性: 对边图片反向偏移，保持视觉平衡")
        print(f"   • 图片放置: 适度放大(1.5倍)，基于边界区域中心定位")
        print(f"   • 防重叠: 偏移范围控制在±30px，确保不与角落冲突")
        print(f"   • 图片分配: 自动循环分配，确保对间不重复")
        print(f"   • 图片需求: 最少1张，推荐9张（实现完全无重复）")
        print(f"   • 填充控制: {'✅ 启用边界+中心' if 填充中间区域 else '❌ 只显示角落'}")
        
        return canvas, mask_canvas, positions
