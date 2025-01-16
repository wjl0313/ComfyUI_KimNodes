import torch
import numpy as np
from PIL import Image, ImageDraw
import random

class SeamlessIconGenerator:
    """
    将图标按照类似 Distribute_icons_in_grid.py 的方式进行排列，
    保持每个图标的宽高比不变，先从左至右纵向排列，超过指定数量后换列继续排列，
    最终叠加到底图 scene_image 上。
    并在网格拼贴时画出边缘，并确保若 icon 太大时会被等比缩小到不会超出格子范围。
    """

    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "icons_1": ("IMAGE",),
                "icons_2": ("IMAGE",),
                "scene_image": ("IMAGE",),
                "icon_size": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 1600,
                    "step": 5,
                    "display": "number"
                }),
                "num_rows": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
                "max_repeats": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "display": "number",
                    "label": "列重复次数"
                }),
                "total_height": ("INT", {
                    "default": 1536,
                    "min": 100,
                    "max": 4096,
                    "step": 8,
                    "display": "number",
                    "label": "总高度"
                }),
                "column_spacing": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "label": "列间距"
                }),
                "random_order": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启",
                    "label_off": "关闭"
                }),
                "seed": ("INT", {
                    "default": 6666,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
                "column_offset": ("INT", {
                    "default": 320,
                    "min": -1024,
                    "max": 1024,
                    "step": 4,
                    "display": "number",
                    "label": "列错落值"
                }),
                "rotation": ("FLOAT", {
                    "default": 35.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "display": "number",
                    "label": "图标旋转角度"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = True
    FUNCTION = "generate_seamless_icon"
    CATEGORY = "🍊 Kim-Nodes/🛑Icon Processing | 图标处理"

    def preprocess_icons(self, icons):
        """将批次或列表张量类型图片转换为PIL Image 对象列表"""
        icon_list = []
        
        # 由于 INPUT_IS_LIST = True，icons 现在是一个列表
        for icon_tensor in icons:
            if isinstance(icon_tensor, torch.Tensor):
                if icon_tensor.shape[1] in (3, 4):
                    icon_tensor = icon_tensor.permute(0, 2, 3, 1)
                icon_np = (icon_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                icon_pil = Image.fromarray(icon_np)
                icon_list.append(icon_pil)
            else:
                raise ValueError("输入的图标必须是张量类型。")

        return icon_list

    def create_grid_layout(self, icons_1, icons_2, icon_size, num_rows, total_height, scene_height, scene_width, column_spacing, column_offset, rotation, max_repeats):
        if not icons_1 or not icons_2:
            raise ValueError("没有输入任何图标。")

        # 限制两组图标数量为num_rows
        transformed_icons_1 = [icon for icon in icons_1[:num_rows]]
        transformed_icons_2 = [icon for icon in icons_2[:num_rows]]
        
        total_icons = len(transformed_icons_1) + len(transformed_icons_2)
        num_columns = 1  # 因为现在只取num_rows个图标，所以只需要一列

        # 存储基础列组的图标信息
        base_columns = [transformed_icons_1 + transformed_icons_2]  # 直接将所有图标放在一列中

        # 计算列的最大宽度（使用原始图标尺寸）
        col_widths = [max(icon.size[0] for icon in transformed_icons_1 + transformed_icons_2)]

        # 计算基础列组的总宽度
        base_group_width = sum(col_widths) + column_spacing * (num_columns - 1) if num_columns > 0 else 0
        
        # 计算需要重复的列组数量
        repeat_columns = (scene_width + base_group_width - 1) // base_group_width

        # 创建画布
        canvas_width = scene_width
        canvas_height = scene_height
        collage = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(collage)

        # 对每个水平重复的列组进行处理
        for repeat_x in range(repeat_columns):
            x_offset = repeat_x * (base_group_width + column_spacing)
            y_offset = (repeat_x % 2) * column_offset
            
            # 根据列的奇偶选择使用哪组图标
            current_icons = transformed_icons_2 if repeat_x % 2 == 1 else transformed_icons_1
            
            for col_idx, col_icons in enumerate(base_columns):
                current_x = x_offset + sum(col_widths[:col_idx]) + column_spacing * col_idx
                col_width = col_widths[col_idx]
                
                # 计算每组图标的实际高度（不包含间距）
                icons_heights = [icon.size[1] for icon in current_icons]
                
                # 计算每个重复组的区域高度
                section_height = total_height / max_repeats
                
                for repeat_y in range(max_repeats):
                    # 计算当前重复组的起始y坐标
                    section_start = y_offset + repeat_y * section_height
                    
                    # 计算当前组内图标的间距
                    if len(current_icons) > 1:
                        total_icons_height = sum(icons_heights)
                        available_space = section_height - total_icons_height
                        spacing_between = available_space / (len(current_icons) - 1)
                    else:
                        spacing_between = 0
                    
                    # 在当前区域内均匀分布图标
                    current_y = section_start
                    for idx, icon in enumerate(current_icons):
                        w, h = icon.size
                        x_centered = current_x + (col_width - w) // 2
                        
                        # 将图标垂直居中放置在其分配的空间内
                        y_centered = current_y + h/2
                        
                        if 0 <= y_centered < scene_height and x_centered + w <= scene_width:
                            # 创建旋转画布
                            diagonal = int(((w ** 2 + h ** 2) ** 0.5))
                            rotated_canvas = Image.new('RGBA', (diagonal, diagonal), (0, 0, 0, 0))
                            rotated_draw = ImageDraw.Draw(rotated_canvas)
                            
                            # 将图标粘贴到画布中心
                            paste_x = (diagonal - w) // 2
                            paste_y = (diagonal - h) // 2
                            rotated_canvas.paste(icon, (paste_x, paste_y), icon)
                            
                            # 绘制红框
                            rotated_draw.rectangle(
                                [paste_x, paste_y, paste_x + w, paste_y + h],
                                outline=(255, 0, 0, 255),
                                width=2
                            )
                            
                            # 旋转画布
                            rotated_image = rotated_canvas.rotate(rotation, expand=True, resample=Image.BICUBIC)
                            
                            # 计算旋转后的位置
                            new_w, new_h = rotated_image.size
                            paste_x = x_centered - (new_w - w) // 2
                            paste_y = int(y_centered - h/2) - (new_h - h) // 2
                            
                            # 粘贴旋转后的图像
                            collage.paste(rotated_image, (paste_x, paste_y), rotated_image)
                        
                        # 更新下一个图标的y坐标
                        current_y += h + spacing_between

        return collage

    def create_flow_layout(self, icons, spacing=10, max_width=1024):
        """
        简单的水平流式布局示例，先从左到右依次摆放图标，
        超过 max_width 就换行，把下一张图标贴到新行。
        红框大小随图标实际尺寸变化。
        """
        if not icons:
            raise ValueError("没有输入任何图标。")

        transformed_icons = [icon for icon in icons]
        
        # 记录贴好的 (x, y) 坐标及每行占用的最大高度
        current_x, current_y = 0, 0
        line_height = 0

        # 为了统计画布的总宽高，逐个计算
        positions = []
        max_canvas_width = 0
        total_canvas_height = 0

        for icon in transformed_icons:
            w, h = icon.size
            # 若放不下，将 x 重置为 0，并 y += 当前行最大高度+spacing
            # 同时更新下一行的 line_height
            if current_x + w > max_width:
                # 换行
                current_x = 0
                current_y += line_height + spacing
                line_height = 0
            
            # 记录位置
            positions.append((current_x, current_y, w, h))
            # 更新下一次贴图的 x
            current_x += w + spacing
            # 更新当前行最大高
            line_height = max(line_height, h)
            # 更新整幅画布的宽度
            max_canvas_width = max(max_canvas_width, current_x)

        # 全部贴完后，加上最后一行高度
        total_canvas_height = current_y + line_height

        # 创建画布
        collage = Image.new('RGBA', (max_canvas_width, total_canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(collage)

        # 根据 positions 将图标贴到对应位置，并绘制红框
        for icon, (x, y, w, h) in zip(transformed_icons, positions):
            collage.paste(icon, (x, y), icon)
            # 删除绘制红框的代码
            # draw.rectangle(
            #     [x, y, x + w, y + h],
            #     outline=(255, 0, 0, 255),
            #     width=0
            # )

        return collage

    def generate_seamless_icon(self, icons_1, icons_2, scene_image, icon_size=50, num_rows=5, total_height=1536, column_spacing=0, column_offset=0, rotation=0.0, random_order=False, seed=0, max_repeats=10):
        """
        处理输入参数，确保它们是正确的类型
        """
        # 处理参数
        if isinstance(total_height, list):
            total_height = total_height[0]
        if isinstance(icon_size, list):
            icon_size = icon_size[0]
        if isinstance(num_rows, list):
            num_rows = num_rows[0]
        if isinstance(random_order, list):
            random_order = random_order[0]
        if isinstance(seed, list):
            seed = seed[0]
        if isinstance(column_spacing, list):
            column_spacing = column_spacing[0]
        if isinstance(column_offset, list):
            column_offset = column_offset[0]
        if isinstance(rotation, list):
            rotation = rotation[0]
        if isinstance(max_repeats, list):
            max_repeats = max_repeats[0]

        # 预处理两组图标
        icon_list_1 = self.preprocess_icons(icons_1)
        icon_list_2 = self.preprocess_icons(icons_2)
        
        # 随机顺序处理（如果启用）
        if random_order:
            if seed != 0:
                random.seed(seed)
            else:
                random.seed(None)
            
            random.shuffle(icon_list_1)
            random.shuffle(icon_list_2)
            
            random.seed()

        # 处理背景图 scene_image
        if isinstance(scene_image, list):
            scene_image = scene_image[0]

        if isinstance(scene_image, torch.Tensor):
            if scene_image.shape[0] != 1:
                raise ValueError("scene_image 只支持 batch_size=1, 当前 batch_size={}".format(scene_image.shape[0]))
            if scene_image.shape[1] in (3, 4):
                scene_image = scene_image.permute(0, 2, 3, 1)
            scene_np = (scene_image[0].cpu().numpy() * 255).astype(np.uint8)
            scene_pil = Image.fromarray(scene_np)
        elif isinstance(scene_image, np.ndarray):
            if scene_image.ndim == 4 and scene_image.shape[0] == 1:
                scene_image = scene_image[0]
            if scene_image.ndim == 3 and scene_image.shape[0] in (3, 4):
                scene_image = np.transpose(scene_image, (1, 2, 0))
            scene_np = (scene_image * 255).astype(np.uint8)
            scene_pil = Image.fromarray(scene_np)
        else:
            raise TypeError("scene_image 必须是 torch.Tensor 或 numpy.ndarray。")

        # 获取场景图片的高度和宽度
        scene_width = scene_pil.size[0]
        scene_height = scene_pil.size[1]

        # 在独立画布上按网格排列图标，并绘制边框，传入场景高度和宽度
        grid_collage = self.create_grid_layout(icon_list_1, icon_list_2, icon_size, num_rows, 
                                             total_height, scene_height, scene_width, 
                                             column_spacing, column_offset, rotation, max_repeats)

        # 将网格贴到场景图上 (从左上角开始贴)
        scene_pil.paste(grid_collage, (0, 0), grid_collage)

        # 转为 (1, H, W, 3/4) 的张量返回
        result = np.array(scene_pil, dtype=np.float32) / 255.0
        if result.shape[-1] == 4:
            result = result[..., :3]  # 去掉 alpha 通道
        result = np.expand_dims(result, axis=0)
        return torch.from_numpy(result), 