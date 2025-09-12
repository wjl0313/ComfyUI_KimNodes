"""
象限边缘无缝拼图模板

基于classic_seamless_template，实现新的布局：
- 去除四个角的图片  
- 边缘从原来的4个位置增加到8个位置（上下左右各2个）
- 每个象限边缘位置使用完整图片，中心对齐黑点坐标
- 图片允许超出画布边界，只要中心对齐即可
- 总共需要5张图片：从图片列表顺序使用，不重复（除非图片数<5）
- 实现无缝四方连续：按SVG颜色分组
  * 绿色组：垂直线X=256（上左、下左）→ 图片#1
  * 黄色组：垂直线X=768（上右、下右）→ 图片#2
  * 青色组：水平线Y=256（左上、右上）→ 图片#3
  * 品红组：水平线Y=768（左下、右下）→ 图片#4
- 中心区域只有1个图片，位于画布正中央 → 图片#5
"""

import random
from PIL import Image
from .base_template import TilingTemplateBase


class QuadrantEdgeTemplate(TilingTemplateBase):
    """象限边缘无缝拼图模板"""
    
    def __init__(self):
        super().__init__()
        self.template_name = "象限边缘拼图"
        self.template_description = "去除角落，边缘8个位置+中心1个位置，需要5张图片，按SVG颜色分组实现无缝四方连续"
    
    def get_template_info(self):
        """返回模板信息"""
        return {
            "name": self.template_name,
            "description": self.template_description
        }
    
    def validate_params(self, params):
        """验证参数有效性"""
        has_basic_size = "基础图片尺寸" in params
        has_separate_sizes = all(param in params for param in ["边界图片大小", "中间图片大小"])
        
        if not (has_basic_size or has_separate_sizes):
            print("警告: 缺少尺寸参数（基础图片尺寸 或 边界图片大小/中间图片大小）")
            return False
        
        other_required = ["填充中间区域"]
        for param in other_required:
            if param not in params:
                print(f"警告: 缺少必需参数 {param}")
                return False
        
        return True
    
    def calculate_quadrant_edge_positions(self, canvas_size, edge_size):
        """计算8个象限边缘位置，基于SVG坐标参考"""
        输出宽度, 输出高度 = canvas_size
        
        # 基于SVG坐标系统（1024x1024参考）
        # 黑点坐标表示完整图片的中心点位置：
        # 上边缘：(256,0), (768,0) - 图片中心点
        # 下边缘：(256,1024), (768,1024) - 图片中心点
        # 左边缘：(0,256), (0,768) - 图片中心点  
        # 右边缘：(1024,256), (1024,768) - 图片中心点
        
        scale_x = 输出宽度 / 1024.0
        scale_y = 输出高度 / 1024.0
        
        positions = {}
        
        # 图片中心对齐到黑点位置，允许超出画布边界
        # 计算图片左上角位置 = 黑点坐标 - 图片尺寸的一半
        
        # 上边缘：黑点(256,0), (768,0) - 图片中心对齐到这些点
        positions["上左"] = (int(256 * scale_x) - edge_size // 2, int(0 * scale_y) - edge_size // 2)
        positions["上右"] = (int(768 * scale_x) - edge_size // 2, int(0 * scale_y) - edge_size // 2)
        
        # 下边缘：黑点(256,1024), (768,1024) - 图片中心对齐到这些点
        positions["下左"] = (int(256 * scale_x) - edge_size // 2, int(1024 * scale_y) - edge_size // 2)
        positions["下右"] = (int(768 * scale_x) - edge_size // 2, int(1024 * scale_y) - edge_size // 2)
        
        # 左边缘：黑点(0,256), (0,768) - 图片中心对齐到这些点
        positions["左上"] = (int(0 * scale_x) - edge_size // 2, int(256 * scale_y) - edge_size // 2)
        positions["左下"] = (int(0 * scale_x) - edge_size // 2, int(768 * scale_y) - edge_size // 2)
        
        # 右边缘：黑点(1024,256), (1024,768) - 图片中心对齐到这些点
        positions["右上"] = (int(1024 * scale_x) - edge_size // 2, int(256 * scale_y) - edge_size // 2)
        positions["右下"] = (int(1024 * scale_x) - edge_size // 2, int(768 * scale_y) - edge_size // 2)
        
        print(f"📐 象限边缘位置（8个黑点，基于SVG坐标）：")
        for name, (x, y) in positions.items():
            print(f"   {name}: ({x}, {y})")
        
        print(f"📍 完整图片的中心对齐定位：")
        print(f"   上边缘黑点：({int(256 * scale_x)}, {int(0 * scale_y)}), ({int(768 * scale_x)}, {int(0 * scale_y)})")
        print(f"   下边缘黑点：({int(256 * scale_x)}, {int(1024 * scale_y)}), ({int(768 * scale_x)}, {int(1024 * scale_y)})")
        print(f"   左边缘黑点：({int(0 * scale_x)}, {int(256 * scale_y)}), ({int(0 * scale_x)}, {int(768 * scale_y)})")
        print(f"   右边缘黑点：({int(1024 * scale_x)}, {int(256 * scale_y)}), ({int(1024 * scale_x)}, {int(768 * scale_y)})")
        print(f"   黑点含义：完整图片的中心点，允许超出画布边界")
        
        return positions
    
    def calculate_center_cross_positions(self, canvas_size, tile_size, edge_size):
        """计算中心5个位置的十字摆放，基于新的SVG坐标参考"""
        输出宽度, 输出高度 = canvas_size
        
        # 基于SVG坐标比例计算（SVG参考: 1024x1024）
        # SVG中心红点坐标：
        # 中心：(512, 512)
        # 上：(512, 256), 下：(512, 768)
        # 左：(256, 512), 右：(768, 512)
        
        scale_x = 输出宽度 / 1024.0
        scale_y = 输出高度 / 1024.0
        
        positions = {}
        
        # 基于新SVG坐标计算中心红点位置
        positions["中心"] = (int(512 * scale_x) - tile_size // 2, int(512 * scale_y) - tile_size // 2)
        positions["上"] = (int(512 * scale_x) - tile_size // 2, int(256 * scale_y) - tile_size // 2)
        positions["下"] = (int(512 * scale_x) - tile_size // 2, int(768 * scale_y) - tile_size // 2)
        positions["左"] = (int(256 * scale_x) - tile_size // 2, int(512 * scale_y) - tile_size // 2)
        positions["右"] = (int(768 * scale_x) - tile_size // 2, int(512 * scale_y) - tile_size // 2)
        
        print(f"🎯 中心十字位置（5个红点，基于SVG坐标）：")
        for name, (x, y) in positions.items():
            print(f"   {name}: ({x}, {y})")
        
        print(f"📍 SVG坐标映射：")
        print(f"   中心: ({int(512 * scale_x)}, {int(512 * scale_y)}) ← SVG(512, 512)")
        print(f"   上下红点X: {int(512 * scale_x)} ← SVG 512")
        print(f"   左右红点X: {int(256 * scale_x)}, {int(768 * scale_x)} ← SVG 256, 768")
        print(f"   上下红点Y: {int(256 * scale_y)}, {int(768 * scale_y)} ← SVG 256, 768")
        print(f"   左右红点Y: {int(512 * scale_y)} ← SVG 512")
        
        return positions
    

    
    def fill_edge_positions(self, canvas, mask_canvas, edge_images, canvas_size, edge_size):
        """填充8个边缘位置，实现无缝四方连续效果"""
        positions = self.calculate_quadrant_edge_positions(canvas_size, edge_size)
        输出宽度, 输出高度 = canvas_size
        
        edge_positions = []
        
        print(f"🔲 开始填充8个边缘位置（无缝四方连续效果）")
        
        # 获取可用图片列表
        available_images = list(edge_images.values())
        
        # 为4个颜色组分配图片（确保无缝连续）
        if len(available_images) >= 4:
            color_group_assignment = {
                "绿色": available_images[0],    # 绿色组（上左、下左）
                "黄色": available_images[1],    # 黄色组（上右、下右）
                "青色": available_images[2],    # 青色组（左上、右上）
                "品红": available_images[3]     # 品红组（左下、右下）
            }
        else:
            # 图片不足时循环使用
            color_group_assignment = {
                "绿色": available_images[0 % len(available_images)],
                "黄色": available_images[1 % len(available_images)],
                "青色": available_images[2 % len(available_images)],
                "品红": available_images[3 % len(available_images)]
            }
        
        # 边缘位置映射到颜色组（参考SVG颜色分组）
        position_to_color_group = {
            "上左": "绿色", "下左": "绿色",     # 左边垂直线：相同X坐标(256)
            "上右": "黄色", "下右": "黄色",     # 右边垂直线：相同X坐标(768)
            "左上": "青色", "右上": "青色",     # 上边水平线：相同Y坐标(256)
            "左下": "品红", "右下": "品红"      # 下边水平线：相同Y坐标(768)
        }
        
        print(f"📋 无缝连续分配（按SVG颜色分组）：")
        print(f"  • 绿色组 → 上左、下左（垂直线X=256）")
        print(f"  • 黄色组 → 上右、下右（垂直线X=768）")  
        print(f"  • 青色组 → 左上、右上（水平线Y=256）")
        print(f"  • 品红组 → 左下、右下（水平线Y=768）")
        
        edge_names = ["上左", "上右", "下左", "下右", "左上", "左下", "右上", "右下"]
        
        for pos_name in edge_names:
            x, y = positions[pos_name]
            
            # 根据位置选择对应颜色组的图片
            color_group = position_to_color_group[pos_name]
            img = color_group_assignment[color_group]
            
            # 缩放图片到目标尺寸，保持比例
            edge_img = self.resize_image_keep_ratio(img, (edge_size, edge_size), force_size=True)
            
            # 粘贴到画布 - 允许超出边界的精确定位
            canvas.paste(edge_img, (x, y), edge_img)
            if edge_img.mode == 'RGBA':
                mask_canvas.paste(0, (x, y), edge_img)
            
            # 记录位置信息
            edge_positions.append({
                "type": "edge",
                "position": pos_name,
                "bbox": [x, y, x + edge_size, y + edge_size],
                "image_index": available_images.index(img),
                "color_group": color_group
            })
            
            print(f"  • {pos_name}({color_group}): ({x}, {y}) - 图片中心对齐黑点")
        
        print(f"✅ 边缘处理完成：8个位置，4个颜色组，实现无缝四方连续效果")
        print(f"📍 定位说明：按SVG颜色分组，确保垂直/水平线上的位置共用图片")
        
        return edge_positions
    
    def fill_center_cross(self, canvas, mask_canvas, center_images, canvas_size, tile_size, edge_size, random_enabled, random_seed):
        """填充中心位置（只有1个图片），与象限边缘黑点对齐"""
        if not center_images:
            print("⚠️  没有中心图片可填充")
            return []
        
        center_positions = []
        
        print(f"🎯 填充中心位置（1个红点，位于画布正中央）")
        
        positions = self.calculate_center_cross_positions(canvas_size, tile_size, edge_size)
        
        # 设置随机种子
        if random_enabled:
            random.seed(random_seed)
        
        # 只填充中心位置
        pos_name = "中心"
        x, y = positions[pos_name]
        
        # 确保位置在有效范围内
        输出宽度, 输出高度 = canvas_size
        x = max(0, min(x, 输出宽度 - tile_size))
        y = max(0, min(y, 输出高度 - tile_size))
        
        # 选择图片（随机选择或使用第一张）
        if random_enabled:
            img = random.choice(center_images)
        else:
            img = center_images[0]
        
        # 缩放图片
        tile_img = self.resize_image_keep_ratio(img, (tile_size, tile_size), force_size=True)
        
        # 粘贴到画布
        canvas.paste(tile_img, (x, y), tile_img)
        if tile_img.mode == 'RGBA':
            mask_canvas.paste(0, (x, y), tile_img)
        
        # 记录中心位置信息
        center_positions.append({
            "type": "center",
            "position": "center",
            "bbox": [x, y, x + tile_size, y + tile_size],
            "image_index": center_images.index(img)
        })
        
        print(f"🎯 中心位置: ({x}, {y})")
        
        print(f"✅ 中心区域填充完成，放置了1个红点，位于画布正中央")
        
        if random_enabled:
            random.seed()
        
        return center_positions
    
    def allocate_images_for_quadrant_template(self, images, random_enabled, random_seed):
        """为象限边缘模板分配图片 - 总共需要5张图片（4个边缘组+1个中心）"""
        if random_enabled:
            random.seed(random_seed)
            shuffled_images = images.copy()
            random.shuffle(shuffled_images)
        else:
            shuffled_images = images.copy()
        
        total_images = len(shuffled_images)
        print(f"🎯 象限边缘模板图片分配：输入图片数量 = {total_images}")
        
        # 总共需要5张图片：4个边缘颜色组 + 1个中心
        required_images = 5
        
        if total_images >= required_images:
            # 图片充足，顺序分配不重复
            edge_images = {
                "边缘0": shuffled_images[0],  # 绿色组
                "边缘1": shuffled_images[1],  # 黄色组
                "边缘2": shuffled_images[2],  # 青色组
                "边缘3": shuffled_images[3]   # 品红组
            }
            center_images = [shuffled_images[4]]  # 中心使用第5张图片
            
            print(f"📋 图片充足模式：使用前5张图片不重复")
            print(f"  • 绿色组(上左、下左)：图片 #{1}")
            print(f"  • 黄色组(上右、下右)：图片 #{2}")
            print(f"  • 青色组(左上、右上)：图片 #{3}")
            print(f"  • 品红组(左下、右下)：图片 #{4}")
            print(f"  • 中心位置：图片 #{5}")
            
        else:
            # 图片不足，循环使用
            edge_images = {}
            for i in range(4):  # 4个边缘颜色组
                edge_images[f"边缘{i}"] = shuffled_images[i % total_images]
            
            # 中心图片使用所有可用图片
            center_images = shuffled_images
            
            print(f"📋 图片不足模式：{total_images}张图片循环使用")
            print(f"  • 绿色组(上左、下左)：图片 #{(0 % total_images) + 1}")
            print(f"  • 黄色组(上右、下右)：图片 #{(1 % total_images) + 1}")
            print(f"  • 青色组(左上、右上)：图片 #{(2 % total_images) + 1}")
            print(f"  • 品红组(左下、右下)：图片 #{(3 % total_images) + 1}")
            print(f"  • 中心位置：从{total_images}张图片中选择")
        
        print(f"🎯 分配总结：边缘4个颜色组 + 中心1个位置 = 5个图片需求")
        
        return edge_images, center_images
    
    def generate_tiling(self, images, canvas_size, params):
        """生成象限边缘无缝拼图"""
        
        if not self.validate_params(params):
            raise ValueError("参数验证失败")
        
        if len(images) < 1:
            raise ValueError("至少需要1张图片")
        
        # 初始化位置信息列表
        positions = []
        
        # 获取参数
        输出宽度, 输出高度 = canvas_size
        基础图片尺寸 = params.get("基础图片尺寸", 128)
        # 边界图片默认使用基础图片尺寸的2倍，确保有足够的覆盖范围
        边界图片大小 = params.get("边界图片大小", 基础图片尺寸 * 2)
        中间图片大小 = params.get("中间图片大小", 基础图片尺寸)
        填充中间区域 = params.get("填充中间区域", True)
        随机种子 = params.get("随机种子", 0)
        启用随机 = params.get("启用随机", True)
        背景颜色 = params.get("背景颜色", "#FFFFFF")
        
        # 创建画布和遮罩
        bg_color = tuple(int(背景颜色.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        canvas = Image.new('RGBA', (输出宽度, 输出高度), bg_color)
        mask_canvas = Image.new('L', (输出宽度, 输出高度), 255)
        print(f"🎨 创建象限边缘模板画布，尺寸: {输出宽度} x {输出高度}，背景颜色: {背景颜色}")
        print(f"⚙️ 模板参数：")
        print(f"   • 基础图片尺寸: {基础图片尺寸}px")
        print(f"   • 边界图片大小: {边界图片大小}px")
        print(f"   • 中间图片大小: {中间图片大小}px")
        print(f"   • 填充中间区域: {'✅' if 填充中间区域 else '❌'}")
        
        # 分配图片
        edge_images, center_images = self.allocate_images_for_quadrant_template(images, 启用随机, 随机种子)
        
        # 1. 填充8个边缘位置（总是显示） - 使用完整图片中心对齐黑点
        print(f"🔲 开始填充8个象限边缘位置（完整图片中心对齐黑点）...")
        edge_positions = self.fill_edge_positions(canvas, mask_canvas, edge_images, (输出宽度, 输出高度), 边界图片大小)
        positions.extend(edge_positions)
        
        # 2. 填充中心位置（受开关控制）
        if 填充中间区域:
            # 中间图片大小动态计算，确保与边界图片协调
            中间图片大小 = 基础图片尺寸 * 2
            print(f"🎯 开始填充中心位置（1个红点，画布正中央）")
            center_positions = self.fill_center_cross(
                canvas, mask_canvas, center_images,
                (输出宽度, 输出高度), 中间图片大小, 边界图片大小,
                启用随机, 随机种子
            )
            positions.extend(center_positions)
        else:
            print("⏸️  中间区域填充已禁用，跳过十字位置填充")
        
        print(f"✅ 象限边缘模板生成完成")
        print(f"📊 模板特征:")
        print(f"   • 边缘图片: 8个位置，4个颜色组，实现无缝四方连续")
        print(f"   • 中心图片: 1个（受开关控制 - 画布正中央）")
        print(f"   • 图片需求: 总共5张图片（4个边缘组+1个中心），顺序使用不重复")
        print(f"   • 无缝连续: 按SVG颜色分组（垂直线共用，水平线共用）")
        print(f"   • 定位方式: 图片中心对齐到SVG黑点坐标，允许超出画布边界")
        print(f"   • 坐标定位: 黑点为图片中心点（非裁切边缘）")
        print(f"   • 边缘坐标: 上下边缘(256,768)，左右边缘(256,768)")
        print(f"   • 边界图片: 默认大小为基础图片尺寸×2，确保充分覆盖")
        print(f"   • 图片处理: 不裁切，使用完整图片，缩放到目标尺寸")
        print(f"   • 开关状态: {'✅ 启用' if 填充中间区域 else '❌ 禁用'}")
        
        return canvas, mask_canvas, positions