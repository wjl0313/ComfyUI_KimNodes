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
                "icon_folder": ("STRING", {
                    "multiline": False,
                    "default": "F:/龙品/all",
                    "lazy": True
                }),
                "icon_size": ("INT", {
                    "default": 50,
                    "min": 10,  
                    "max": 512,
                    "step": 5,
                    "display": "number"
                }),
                "min_distance": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 512, 
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
    CATEGORY = "🍊 Kim-Nodes"

    def distribute_icons_in_grid(self, scene_image, mask_image, icon_folder, icon_size,
                    min_distance, num_rows=5, num_cols=10, max_scale=1.0, vertical_offset=0):

        def load_icons(icon_folder):
            """加载文件夹内所有图标"""
            icons = []
            for file in os.listdir(icon_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                    icon_path = os.path.join(icon_folder, file)
                    icon = Image.open(icon_path).convert("RGBA")
                    icons.append(icon)
            return icons

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

        def get_grid_positions(binary_mask, icon_width, icon_height, num_rows, num_cols):
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

            for row in range(num_rows):
                for col in range(num_cols):
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
            placed_positions = []
            for position in positions:
                icon = random.choice(icons)
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
        icons = load_icons(icon_folder)
        positions = get_grid_positions(binary_mask, icon_size, icon_size, num_rows, num_cols)

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
