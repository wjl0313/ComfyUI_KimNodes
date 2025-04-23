import os
import random
import numpy as np
from PIL import Image
import cv2
import torch

class IconDistributeByGrid:
    """
    基于蒙版在场景图上分布图标的节点。
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_image": ("IMAGE",),
                "mask_image": ("MASK",),
                "icons": ("IMAGE",),
                "icon_size": ("INT", {
                    "default": 50,
                    "min": 10,  
                    "max": 1600,
                    "step": 5,
                    "display": "number"
                }),
                "min_distance": ("INT", {
                    "default": 0,
                    "min": 0,
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
                "num_cols": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
                "vertical_offset": ("INT", {
                    "default": 0,
                    "min": -1000,  # 可根据实际需求调整最小值
                    "max": 1000,   # 可根据实际需求调整最大值
                    "step": 5,
                    "display": "number"
                })
            },
            "hidden": {  
                "max_scale": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "distribute_icons_in_grid" 
    CATEGORY = "🍊 Kim-Nodes/🛑Icon Processing | 图标处理"

    def distribute_icons_in_grid(self, scene_image, mask_image, icons, icon_size,
                    min_distance, num_rows=5, num_cols=10, max_scale=1.0, vertical_offset=0):

        def icons_preprocess(icons):
            """将批次或列表张量类型图片转换为PIL Image 对象列表"""

                        # 如果输入的是四维张量，先将其转为列表
            if isinstance(icons, torch.Tensor):
                # 将传入的张量转置为(B, H, W, C)
                if icons.shape[1] == 3 or icons.shape[1] == 4:
                    icons = icons.permute(0, 2, 3, 1) # (B, H, W, C)

                icons_tensor = [icons[i:i+1] for i in range(icons.shape[0])]  # 将每个批次作为独立的图片
                print("传入的是批次图片，图片张量的形状, ", icons_tensor[0].shape)
                # 图片列表容器
                icon_list = []
                # 遍历每张贴纸
                for icon_tensor in icons_tensor:
                    # 假设输入张量是 (1, H, W, C) 格式，取第一维的1（即去掉批次维度）
                    icon_tensor = icon_tensor.squeeze(0)  # 去掉批次维度，得到(C, H, W)格式
                    # 将张量转换为PIL图像
                    icon_np = icon_tensor.cpu().numpy()  # 将张量转换为numpy数组
                    icon_np = (icon_np * 255).astype(np.uint8)
                    icon = Image.fromarray(icon_np)
                    # 加入列表容器
                    icon_list.append(icon)
            elif isinstance(icons, list):
                print("传入的是列表图片, 图片的张量形状，", icons[0].shape)
                icon_list = []
                for icon_tensor in icons:
                    # 将图片张量专职为(1, H, W, C)
                    if icon_tensor.shape[1] == 3 or icon_tensor.shape[1] == 4:
                        icon_tensor = icon_tensor.permute(0, 2, 3, 1)
                    icon_np = icon_tensor.cpu().numpy()  # 将张量转换为numpy数组
                    icon_np = (icon_np * 255).astype(np.uint8)
                    icon = Image.fromarray(icon_np)
                    # 加入列表容器
                    icon_list.append(icon)
            else:
                raise ValueError("输入的贴纸必须是四维张量的图片或元素为张量（四维张量且批次维度为1）的列表")    

            return icon_list

        def preprocess_mask_image(mask_image):
            """预处理蒙版，确保维度和类型正确"""
            if isinstance(mask_image, torch.Tensor):
                mask_image_np = mask_image.cpu().numpy()
            elif isinstance(mask_image, np.ndarray):
                mask_image_np = mask_image
            else:
                raise TypeError("mask_image 应该是一个 torch.Tensor 或 np.ndarray，但得到了 {}".format(type(mask_image)))

            if len(mask_image_np.shape) == 3:
                if mask_image_np.shape[0] == 1:
                    mask_image_np = mask_image_np[0]
                else:
                    mask_image_np = np.mean(mask_image_np, axis=0)
            elif len(mask_image_np.shape) != 2:
                raise ValueError(f"Unexpected mask dimensions: {mask_image_np.shape}")

            return (mask_image_np * 255).clip(0, 255).astype(np.uint8)

        def get_white_area(mask_np):
            """获取白色区域的轮廓范围"""
            _, binary_mask = cv2.threshold(mask_np, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours, binary_mask

        def transform_icon(icon, target_size):
            """根据目标尺寸调整图标"""
            # 调整到目标尺寸
            icon = icon.resize((target_size, target_size), Image.LANCZOS)
            return icon

        def get_grid_positions(binary_mask, icon_width, icon_height, num_rows, num_cols, icons):
            """根据蒙版获取按格子排列的所有可用位置"""
            mask_height, mask_width = binary_mask.shape
            positions = []

            # 计算蒙版有效区域的边界
            min_x = np.min(np.where(binary_mask == 255)[1])  # 蒙版的最小X坐标
            max_x = np.max(np.where(binary_mask == 255)[1])  # 蒙版的最大X坐标
            min_y = np.min(np.where(binary_mask == 255)[0])  # 蒙版的最小Y坐标
            max_y = np.max(np.where(binary_mask == 255)[0])  # 蒙版的最大Y坐标

            # 计算有效区域的宽高
            valid_width = max_x - min_x
            valid_height = max_y - min_y

            # 根据有效区域和网格大小，动态计算每个格子的宽度和高度
            grid_width = max(icon_width, valid_width // num_cols)
            grid_height = max(icon_height, valid_height // num_rows)

            count = 0
            icon_count = len(icons)
            # 如果行数为7，则第7行列数为8
            for row in range(num_rows):
                count += 1
                if count == 7 and num_cols == 7 and icon_count == 50:
                    num_cols = 8
                for col in range(num_cols):
                    if num_cols == 8:
                        grid_width = valid_width // num_cols
                    x = min_x + col * grid_width
                    y = min_y + row * grid_height

                    # 确保网格完全在蒙版白色区域内
                    if np.all(binary_mask[y:y + grid_height, x:x + grid_width] == 255):
                        positions.append((x, y))

            return positions

        def align_positions_to_mask_center(positions, scene_width, scene_height, binary_mask, icon_width, icon_height, vertical_offset):
            """将网格放置位置按蒙板区域中心进行对齐，并应用垂直偏移"""
            mask_height, mask_width = binary_mask.shape

            # 获取所有网格位置的边界
            min_x = min(positions, key=lambda p: p[0])[0]
            max_x = max(positions, key=lambda p: p[0])[0]
            min_y = min(positions, key=lambda p: p[1])[1]
            max_y = max(positions, key=lambda p: p[1])[1]

            # 计算总区域的中心点
            total_width = max_x - min_x + icon_width
            total_height = max_y - min_y + icon_height
            center_x = min_x + total_width // 2
            center_y = min_y + total_height // 2

            # 计算蒙板的中心点
            mask_center_x = mask_width // 2
            mask_center_y = mask_height // 2

            # 计算偏移量
            offset_x = mask_center_x - center_x
            offset_y = mask_center_y - center_y + vertical_offset  # 应用垂直偏移

            # 根据偏移量调整格子位置
            aligned_positions = [(x + offset_x, y + offset_y) for (x, y) in positions]

            return aligned_positions

        def place_icons_on_scene(positions, scene_image_pil, icons, icon_size):
            """
            在场景上按顺序放置图标，只放置可用的图标数量
            """
            placed_positions = []
            
            # 只使用可用的图标数量，不循环使用
            available_positions = positions[:len(icons)]
            
            # 按顺序放置图标
            for position, icon in zip(available_positions, icons):
                transformed_icon = transform_icon(icon, icon_size)
                x, y = position
                scene_image_pil.paste(transformed_icon, (x, y), transformed_icon)
                placed_positions.append((x, y))

            return scene_image_pil, placed_positions

        # 开始处理
        # 处理 scene_image
        if isinstance(scene_image, torch.Tensor):
            scene_image_np = scene_image.cpu().numpy()
            if scene_image_np.ndim == 4:
                if scene_image_np.shape[0] == 1:
                    scene_image_np = scene_image_np[0]
                else:
                    raise ValueError(f"批次大小大于 1 不受支持：{scene_image_np.shape[0]}")
            if scene_image_np.ndim == 3:
                if scene_image_np.shape[0] == 3 or scene_image_np.shape[0] == 4:
                    scene_image_np = np.transpose(scene_image_np, (1, 2, 0))
            scene_image_np = (scene_image_np * 255).astype(np.uint8)
        elif isinstance(scene_image, np.ndarray):
            scene_image_np = scene_image
            if scene_image_np.ndim == 4 and scene_image_np.shape[0] == 1:
                scene_image_np = scene_image_np[0]
            if scene_image_np.ndim == 3:
                if scene_image_np.shape[2] == 3 or scene_image_np.shape[2] == 4:
                    pass
                elif scene_image_np.shape[0] == 3 or scene_image_np.shape[0] == 4:
                    scene_image_np = np.transpose(scene_image_np, (1, 2, 0))
            scene_image_np = (scene_image_np * 255).astype(np.uint8)
        else:
            raise TypeError(f"scene_image 类型错误：{type(scene_image)}")

        # 处理蒙版
        mask_np = preprocess_mask_image(mask_image)
        contours, binary_mask = get_white_area(mask_np)
        icons = icons_preprocess(icons)
        positions = get_grid_positions(binary_mask, icon_size, icon_size, num_rows, num_cols, icons)

        # 对齐网格到蒙版中心并应用垂直偏移
        aligned_positions = align_positions_to_mask_center(positions, scene_image_np.shape[1], scene_image_np.shape[0], binary_mask, icon_size, icon_size, vertical_offset)

        # 创建场景图并放置图标
        scene_image_pil = Image.fromarray(scene_image_np)
        scene_image_pil, placed_positions = place_icons_on_scene(aligned_positions, scene_image_pil, icons, icon_size)

        # 将结果转换为模型所需的格式返回
        result_image = np.array(scene_image_pil).astype(np.float32) / 255.0

        # 如果结果是 RGBA (H, W, 4)，需要转换回 RGB (H, W, 3)
        if result_image.shape[-1] == 4:
            # 使用 alpha 通道进行混合
            alpha = result_image[..., 3:4]
            rgb = result_image[..., :3]
            result_image = rgb

        # 添加批次维度
        result_image = np.expand_dims(result_image, axis=0)

        # 转换为张量
        result_tensor = torch.tensor(result_image, dtype=torch.float32)

        return result_tensor,
