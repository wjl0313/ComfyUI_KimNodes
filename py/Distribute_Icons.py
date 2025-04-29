import os
import random
import numpy as np
from PIL import Image
import cv2
import torch

class Distribute_Icons:
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
                    "default": "./icons",
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
                "min_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number" 
                }),
                "rotation_angle": ("INT", {
                    "default": 90,
                    "min": 0,
                    "max": 180,
                    "step": 1,
                    "display": "number"
                }),
            },
            "hidden": {  # 添加hidden部分
                "max_scale": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "distribute_icons" 
    CATEGORY = "🍒 Kim-Nodes/🧩Icon Processing | 图标处理"

    def distribute_icons(self, scene_image, mask_image, icon_folder, icon_size,
                    min_distance, min_scale, rotation_angle, max_scale=1.0):

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

        def transform_icon(icon, target_size, min_scale, max_scale, rotation_angle):
            """根据目标尺寸调整图标，并进行旋转和缩放"""
            # 调整到目标尺寸
            icon = icon.resize((target_size, target_size), Image.LANCZOS)

            # 随机缩放比例
            scale_percent = random.uniform(min_scale, max_scale)
            new_size = int(icon.width * scale_percent)
            icon = icon.resize((new_size, new_size), Image.LANCZOS)

            # 如果设置了旋转角度
            if rotation_angle > 0:
                angle = random.uniform(-rotation_angle, rotation_angle)
                icon = icon.rotate(angle, expand=True)

            return icon

        def is_fully_within_white_area(icon, x_offset, y_offset, binary_mask):
            """检查图标是否完全位于白色区域内"""
            icon_array = np.array(icon.split()[3])
            icon_height, icon_width = icon_array.shape

            for i in range(icon_height):
                for j in range(icon_width):
                    if icon_array[i, j] > 0:
                        xi, yi = x_offset + j, y_offset + i
                        if xi < 0 or yi < 0 or xi >= binary_mask.shape[1] or yi >= binary_mask.shape[0]:
                            return False
                        if binary_mask[yi, xi] == 0:
                            return False
            return True

        def check_minimum_distance(x_offset, y_offset, icon_width, icon_height, placed_positions, min_distance):
            """检查新位置是否与已放置的图标保持足够的距离"""
            for placed_x, placed_y, placed_width, placed_height in placed_positions:
                dx = abs(x_offset - placed_x)
                dy = abs(y_offset - placed_y)
                if dx < ((icon_width + placed_width) // 2 + min_distance) and dy < ((icon_height + placed_height) // 2 + min_distance):
                    return False
            return True

        def find_valid_position_with_distance(icon, binary_mask, placed_positions, min_distance, max_attempts=500):
            """寻找一个有效的位置，使图标完全位于白色区域内且与其他图标保持一定距离"""
            icon_width, icon_height = icon.size
            mask_height, mask_width = binary_mask.shape

            for attempt in range(max_attempts):
                x_offset = random.randint(0, mask_width - icon_width)
                y_offset = random.randint(0, mask_height - icon_height)

                if not is_fully_within_white_area(icon, x_offset, y_offset, binary_mask):
                    continue

                if not check_minimum_distance(x_offset, y_offset, icon_width, icon_height, placed_positions, min_distance):
                    continue

                return x_offset, y_offset

            print(f"警告：尝试 {max_attempts} 次后未找到有效位置，图标尺寸({icon_width}, {icon_height})")
            return None

        # 开始处理
        # 预处理蒙版和场景图
        # 处理 scene_image
        if isinstance(scene_image, torch.Tensor):
            print(f"[DEBUG] 输入 scene_image (torch.Tensor) 的原始维度: {scene_image.shape}")
            scene_image_np = scene_image.cpu().numpy()
            # 如果存在批次维度，移除它
            if scene_image_np.ndim == 4:
                # 检查批次大小是否为 1
                if scene_image_np.shape[0] == 1:
                    scene_image_np = scene_image_np[0]
                else:
                    raise ValueError(f"批次大小大于 1 不受支持：{scene_image_np.shape[0]}")
            # 现在，scene_image_np 应该是 3D 数组 (C, H, W) 或 (H, W, C)
            if scene_image_np.ndim == 3:
                # 检查通道是否在第一个维度或最后一个维度
                if scene_image_np.shape[0] == 3 or scene_image_np.shape[0] == 4:
                    # 通道在第一个维度 (C, H, W) -> 转置为 (H, W, C)
                    scene_image_np = np.transpose(scene_image_np, (1, 2, 0))
                elif scene_image_np.shape[2] == 3 or scene_image_np.shape[2] == 4:
                    # 通道在最后一个维度 (H, W, C)，无需转置
                    pass
                else:
                    raise ValueError(f"无法识别的 scene_image 格式，形状为：{scene_image_np.shape}")
            else:
                raise ValueError(f"在移除批次维度后，scene_image 具有意外的维度：{scene_image_np.shape}")
            scene_image_np = (scene_image_np * 255).astype(np.uint8)
        elif isinstance(scene_image, np.ndarray):
            scene_image_np = scene_image
            if scene_image_np.ndim == 4 and scene_image_np.shape[0] == 1:
                scene_image_np = scene_image_np[0]
            if scene_image_np.ndim == 3:
                if scene_image_np.shape[2] == 3 or scene_image_np.shape[2] == 4:
                    pass  # 形状已经是 (H, W, C)
                elif scene_image_np.shape[0] == 3 or scene_image_np.shape[0] == 4:
                    # 从 (C, H, W) 转置为 (H, W, C)
                    scene_image_np = np.transpose(scene_image_np, (1, 2, 0))
                else:
                    raise ValueError(f"无法识别的 scene_image 格式，形状为：{scene_image_np.shape}")
            else:
                raise ValueError(f"scene_image 具有意外的维度：{scene_image_np.shape}")
            scene_image_np = (scene_image_np * 255).astype(np.uint8)
        else:
            raise TypeError("scene_image 应该是 torch.Tensor 或 np.ndarray，但得到的是 {}".format(type(scene_image)))

        # 转换为 PIL 图像
        scene_image_pil = Image.fromarray(scene_image_np).convert("RGBA")
        print(f"[DEBUG] 转换为 PIL 图像后的尺寸: {scene_image_pil.size}")

        # 处理 mask_image
        mask_image_np = preprocess_mask_image(mask_image)

        contours, binary_mask = get_white_area(mask_image_np)

        # 加载图标
        icons = load_icons(icon_folder)
        if not icons:
            raise FileNotFoundError(f"图标文件夹内没有有效的图像：{icon_folder}")

        placed_positions = []  # 用于记录已放置图标的位置和尺寸

        for index, icon in enumerate(icons):
            # 修改后的代码
            transformed_icon = transform_icon(icon, icon_size, min_scale, max_scale, rotation_angle)

            # 直接找到一个完全适合白色区域且保持距离的位置
            position = find_valid_position_with_distance(transformed_icon, binary_mask, placed_positions, min_distance)
            if position:
                x_offset, y_offset = position
                scene_image_pil.paste(transformed_icon, (x_offset, y_offset), transformed_icon)
                placed_positions.append((x_offset, y_offset, transformed_icon.width, transformed_icon.height))
                print(f"图标 {index + 1} 成功放置在位置 ({x_offset}, {y_offset})")
            else:
                print(f"图标 {index + 1} 放置失败：无法找到合适的位置")
        

        # 在 distribute_icons 方法的末尾，替换原来的返回部分：

        # 将结果转换为模型所需的格式返回
        result_image = np.array(scene_image_pil).astype(np.float32) / 255.0
        print(f"[DEBUG] 输出结果 (result_image) 的维度: {result_image.shape}")

        # 如果结果是 RGBA (H, W, 4)，需要转换回 RGB (H, W, 3)
        if result_image.shape[-1] == 4:
            # 使用 alpha 通道进行混合
            alpha = result_image[..., 3:4]
            rgb = result_image[..., :3]
            result_image = rgb

        # 添加批次维度
        result_image = np.expand_dims(result_image, axis=0)
        print(f"[DEBUG] 添加批次维度后的 result_image 维度: {result_image.shape}")

        # 转换为张量
        result_tensor = torch.from_numpy(result_image)
        print(f"[DEBUG] 输出 result_tensor 的维度: {result_tensor.shape}")

        return (result_tensor,)