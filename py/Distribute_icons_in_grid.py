import os
import random
import numpy as np
from PIL import Image
import cv2
import torch
import base64
import io
import json

class IconDistributeByGrid:
    """
    在蒙版区域内随机分布图标的节点。
    将所有输入的图标随机分布在第一张场景图的蒙版区域内。
    - 只处理第一张场景图和蒙版
    - 收集所有输入的图标
    - 随机分布在同一张图上
    - 确保图标不重叠
    """

    def __init__(self):
        pass

    def get_random_icon(self, icon_list, used_icons, total_needed):
        """
        从图标列表中随机选择一个未使用的图标。
        如果所有图标都已使用，则重新开始一个新的随机序列。
        """
        # 如果已使用的图标数量达到列表长度，重置使用记录
        if len(used_icons) >= len(icon_list):
            used_icons.clear()
        
        # 获取所有未使用的图标索引
        available_indices = [i for i in range(len(icon_list)) if i not in used_icons]
        
        # 随机选择一个未使用的图标
        chosen_index = random.choice(available_indices)
        used_icons.add(chosen_index)
        
        return icon_list[chosen_index]

    def image_to_base64(self, image, format='PNG'):
        """将PIL图像转换为base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/{format.lower()};base64,{img_str}"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_image": ("IMAGE", {
                    "description": "场景背景图像，图标将放置在此图像上"
                }),
                "mask_image": ("MASK", {
                    "description": "蒙版图像，白色区域表示可以放置图标的区域"
                }),
                "icons": ("IMAGE", {
                    "description": "要分布的图标图像，支持批量输入多个图标"
                }),
                "icon_size": ("INT", {
                    "default": 256,
                    "min": 64,  
                    "max": 512,
                    "step": 4,
                    "display": "number",
                    "description": "图标的最大尺寸（像素），算法会从此尺寸开始尝试放置"
                }),
                "min_icon_size": ("INT", {
                    "default": 128,
                    "min": 64,
                    "max": 256,
                    "step": 4,
                    "display": "number",
                    "description": "图标的最小尺寸，防止图标缩放得过小"
                }),
                "icon_count": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "description": "要放置的图标数量"
                }),
                "min_distance": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 200, 
                    "step": 5,
                    "display": "number",
                    "description": "图标之间的最小间距（像素）"
                }),
                "edge_padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "display": "number",
                    "description": "与mask边缘的最小距离"
                }),
                "spacing_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "number",
                    "description": "图标间距的缩放因子，1.0表示标准间距，大于1.0增加间距，小于1.0减少间距"
                }),
                "random_seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number",
                    "description": "随机种子，-1表示使用随机种子，其他值用于生成可重复的结果"
                }),
                "enable_rotation": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用旋转",
                    "label_off": "禁用旋转",
                    "description": "是否启用图标的随机旋转"
                }),
            },
            "optional": {
                "input_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "description": "输入的JSON数据，将在此基础上追加新的图标信息"
                }),
                "rotation_range": ("INT", {
                    "default": 360,
                    "min": 0,
                    "max": 360,
                    "step": 15,
                    "display": "number",
                    "description": "随机旋转的角度范围（度），0表示不旋转，360表示全范围旋转"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("processed_image", "updated_mask", "output_json")
    FUNCTION = "distribute_icons_randomly" 
    CATEGORY = "🍒 Kim-Nodes/🧩Icon Processing | 图标处理"
    OUTPUT_IS_LIST = (False, False, False)  # 明确指定输出不是列表

    def distribute_icons_randomly(self, scene_image, mask_image, icons, icon_size, icon_count, min_distance, edge_padding, spacing_factor, random_seed=-1, min_icon_size=128, enable_rotation=False, rotation_range=360, input_json=""):
        """
        将所有输入图标随机分布在第一张场景图的蒙版区域内。
        无论输入多少张图或图标，都只输出一张结果图和一张更新后的蒙版。
        
        参数:
        - edge_padding: 与mask边缘的最小距离
        - spacing_factor: 图标间距的缩放因子
        - min_icon_size: 图标的最小尺寸，防止图标缩放得过小
        - enable_rotation: 是否启用图标的随机旋转
        - rotation_range: 随机旋转的角度范围（度），0表示不旋转，360表示全范围旋转
        
        返回:
        - processed_image: 贴入图标后的场景图像
        - updated_mask: 只显示本次贴入图标位置的蒙版（白色背景，黑色=图标位置）
        """
        

        
        # 处理输入的JSON数据
        icon_positions = []
        try:
            if input_json and input_json.strip():
                input_data = json.loads(input_json)
                if "masks" in input_data:
                    icon_positions = input_data["masks"].copy()
                    print(f"📄 读取到输入JSON数据，包含 {len(icon_positions)} 个已有图标信息")
                else:
                    print("📄 输入JSON数据格式不完整，将创建新的数据结构")
            else:
                print("📄 没有输入JSON数据，将创建新的数据结构")
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析错误: {e}，将创建新的数据结构")
        except Exception as e:
            print(f"⚠️ 处理输入JSON时出错: {e}，将创建新的数据结构")
        
        # 设置随机种子
        if random_seed != -1:
            seed = max(0, min(random_seed, 2**32 - 1))
            random.seed(seed)
            np.random.seed(seed)
        else:
            print("使用随机种子")
        
        # 1. 提取第一张场景图
        if isinstance(scene_image, torch.Tensor):

            if scene_image.ndim == 4:
                first_scene = scene_image[0]  # 只取第一张
                print(f"- 提取第一张后 shape: {first_scene.shape}")
            else:
                first_scene = scene_image
                print(f"- 保持原样 shape: {first_scene.shape}")
            scene_np = first_scene.cpu().numpy()
            if scene_np.ndim == 3 and scene_np.shape[0] in [3, 4]:
                scene_np = np.transpose(scene_np, (1, 2, 0))
            scene_np = (scene_np * 255).astype(np.uint8)
            print(f"- 最终scene_np shape: {scene_np.shape}")
        else:
            raise TypeError("scene_image必须是torch.Tensor")
        
        # 2. 提取第一张蒙版
        if isinstance(mask_image, torch.Tensor):
            if mask_image.ndim == 4:
                first_mask = mask_image[0]  # 只取第一张
            elif mask_image.ndim == 3:
                first_mask = mask_image[0]
            else:
                first_mask = mask_image
            mask_np = first_mask.cpu().numpy()
            if mask_np.ndim > 2:
                mask_np = mask_np[0]
            mask_np = (mask_np * 255).clip(0, 255).astype(np.uint8)
        else:
            raise TypeError("mask_image必须是torch.Tensor")
        
        # 3. 收集所有图标
        icon_list = []
        if isinstance(icons, torch.Tensor):
            if icons.ndim == 3:
                # 单个图标
                icon = icons
                if icon.shape[0] in [3, 4]:
                    icon = icon.permute(1, 2, 0)
                icon_np = icon.cpu().numpy()
                icon_np = (icon_np * 255).clip(0, 255).astype(np.uint8)
                # 根据通道数确定正确的模式
                mode = 'RGBA' if icon_np.shape[-1] == 4 else 'RGB'
                icon_list.append(Image.fromarray(icon_np, mode=mode))
            elif icons.ndim == 4:
                # 批次图标 - 收集所有
                for i in range(icons.shape[0]):
                    icon = icons[i]
                    if icon.shape[0] in [3, 4]:
                        icon = icon.permute(1, 2, 0)
                    icon_np = icon.cpu().numpy()
                    icon_np = (icon_np * 255).clip(0, 255).astype(np.uint8)
                    # 根据通道数确定正确的模式
                    mode = 'RGBA' if icon_np.shape[-1] == 4 else 'RGB'
                    icon_list.append(Image.fromarray(icon_np, mode=mode))
        else:
            raise TypeError("icons必须是torch.Tensor")
        
        print(f"处理结果：场景图 {scene_np.shape}, 蒙版 {mask_np.shape}, 图标数量 {len(icon_list)}")
        
        # 4. 获取白色区域和边缘
        threshold = 127 if mask_np.max() > 1 else 0.5
        _, binary_mask = cv2.threshold(mask_np, threshold, 255, cv2.THRESH_BINARY)
        
        # 获取mask边缘
        edges = cv2.Canny(binary_mask, 100, 200)
        # 膨胀边缘，确保与边缘保持距离
        kernel = np.ones((edge_padding, edge_padding), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 创建有效区域mask（白色区域减去边缘缓冲区）
        valid_area = np.logical_and(binary_mask == 255, dilated_edges == 0)
        valid_pixels = np.where(valid_area)
        
        if len(valid_pixels[0]) == 0:
            print("警告：没有足够的有效区域放置图标")
            result = scene_np.astype(np.float32) / 255.0
            if result.shape[-1] == 4:
                result = result[..., :3]
            # 返回原始图像和全白mask（没有放置任何图标）
            empty_mask = np.ones_like(binary_mask).astype(np.float32)
            
            # 生成只包含输入数据的JSON
            output_json = {
                "masks": icon_positions
            }
            json_string = json.dumps(output_json, ensure_ascii=False, indent=2)
            
            return (torch.tensor(result).unsqueeze(0), torch.tensor(empty_mask).unsqueeze(0), json_string)

        # 5. 计算可用空间和布局
        min_x, max_x = np.min(valid_pixels[1]), np.max(valid_pixels[1])
        min_y, max_y = np.min(valid_pixels[0]), np.max(valid_pixels[0])
        
        # 初始化结果列表，存储所有成功的位置和对应的缩放尺寸
        final_positions = []
        remaining_icons = icon_count
        current_icon_size = icon_size
        # 使用用户自定义的最小图标尺寸，但确保不会大于初始尺寸
        actual_min_icon_size = min(min_icon_size, icon_size)
        
        while remaining_icons > 0 and current_icon_size >= actual_min_icon_size:
            # 计算当前尺寸下的有效区域
            current_valid_area = valid_area.copy()
            positions_this_round = []
            
            # 计算当前尺寸下可以放置的图标数量
            valid_area_size = np.sum(current_valid_area)
            # 应用间距缩放因子
            adjusted_min_distance = int(min_distance * spacing_factor)
            current_icon_area = (current_icon_size + adjusted_min_distance) ** 2
            max_possible_icons = int(valid_area_size / current_icon_area)
            
            if max_possible_icons == 0:
                # 当前尺寸放不下，缩小尺寸继续尝试
                current_icon_size = int(current_icon_size * 0.9)  # 缩小到80%
                continue
            
            # 尝试放置图标
            icons_to_try = min(remaining_icons, max_possible_icons)
            
            for i in range(icons_to_try):
                best_position = None
                max_min_distance = 0
                
                # 在有效区域内采样多个位置
                sample_points = 50
                for _ in range(sample_points):
                    valid_indices = np.where(current_valid_area)
                    if len(valid_indices[0]) == 0:
                        continue
                        
                    idx = random.randint(0, len(valid_indices[0]) - 1)
                    y = valid_indices[0][idx]
                    x = valid_indices[1][idx]
                    
                    # 检查是否有足够空间放置图标
                    if y + current_icon_size >= binary_mask.shape[0] or x + current_icon_size >= binary_mask.shape[1]:
                        continue
                        
                    # 检查图标区域是否完全在有效区域内
                    icon_region = current_valid_area[y:y + current_icon_size, x:x + current_icon_size]
                    if not np.all(icon_region):
                        continue
                    
                    # 计算与已放置图标的最小距离（包括本轮和之前轮次的）
                    min_dist = float('inf')
                    
                    # 检查与本轮已放置的图标的距离
                    for px, py in positions_this_round:
                        dist = np.sqrt((x - px)**2 + (y - py)**2)
                        min_dist = min(min_dist, dist)
                    
                    # 检查与之前轮次放置的图标的距离
                    for pos, size in final_positions:
                        px, py = pos
                        # 计算中心点距离
                        center_dist = np.sqrt((x + current_icon_size/2 - (px + size/2))**2 + 
                                            (y + current_icon_size/2 - (py + size/2))**2)
                        # 考虑不同大小图标的间距，应用间距缩放因子
                        required_dist = ((current_icon_size + size)/2 + adjusted_min_distance)
                        if center_dist < required_dist:
                            min_dist = 0  # 表示位置无效
                            break
                    
                    # 如果是第一个图标，或者这个位置比之前找到的更好
                    if min_dist == float('inf') or (min_dist > adjusted_min_distance + current_icon_size and min_dist > max_min_distance):
                        max_min_distance = min_dist
                        best_position = (x, y)
                
                if best_position is None:
                    break
                
                # 更新有效区域
                x, y = best_position
                margin = int(adjusted_min_distance / 2)
                y1 = max(0, y - margin)
                y2 = min(current_valid_area.shape[0], y + current_icon_size + margin)
                x1 = max(0, x - margin)
                x2 = min(current_valid_area.shape[1], x + current_icon_size + margin)
                current_valid_area[y1:y2, x1:x2] = False
                
                positions_this_round.append(best_position)
            
            # 更新剩余图标数量和保存本轮结果
            for pos in positions_this_round:
                final_positions.append((pos, current_icon_size))
            remaining_icons -= len(positions_this_round)
            
            # 如果这一轮没有放置任何图标，减小尺寸继续尝试
            if not positions_this_round:
                current_icon_size = int(current_icon_size * 0.8)  # 缩小到80%
            
            print(f"当前轮次：尺寸={current_icon_size}, 放置数量={len(positions_this_round)}, 剩余数量={remaining_icons}")
        
        if remaining_icons > 0:
            print(f"警告：仍有 {remaining_icons} 个图标无法放置")
        
        # 7. 放置图标
        scene_pil = Image.fromarray(scene_np)
        
        # 创建新的mask，初始化为全白（底色为白色）
        updated_mask = np.ones_like(binary_mask) * 255  # 全白背景
        print(f"初始化全白mask，尺寸: {updated_mask.shape}")
        
        # 跟踪已使用的图标
        used_icons = set()
        
        # 根据不同尺寸放置图标
        for i, (pos, size) in enumerate(final_positions):
            x, y = pos
            # 随机选择未使用的图标
            icon = self.get_random_icon(icon_list, used_icons, len(final_positions))
            # 使用高质量的缩放方法
            if hasattr(Image, 'Resampling'):
                resized = icon.resize((size, size), Image.Resampling.LANCZOS)
            else:
                resized = icon.resize((size, size), Image.LANCZOS)
            
            # 应用随机旋转（如果启用）
            rotation_angle = 0
            if enable_rotation and rotation_range > 0:
                # 生成随机旋转角度
                rotation_angle = random.uniform(-rotation_range/2, rotation_range/2)
                # 旋转图标，使用expand=True避免裁剪，fillcolor透明
                if resized.mode == 'RGBA':
                    resized = resized.rotate(rotation_angle, expand=True, fillcolor=(0, 0, 0, 0))
                else:
                    # 对于RGB图像，先转为RGBA再旋转，避免边缘问题
                    resized = resized.convert('RGBA')
                    resized = resized.rotate(rotation_angle, expand=True, fillcolor=(0, 0, 0, 0))
                
                # 调整放置位置，使旋转后的图标居中对齐原始位置
                rotated_w, rotated_h = resized.size
                offset_x = (rotated_w - size) // 2
                offset_y = (rotated_h - size) // 2
                x = max(0, x - offset_x)
                y = max(0, y - offset_y)
                
                print(f"  图标 {i+1} 旋转角度: {rotation_angle:.1f}°, 调整位置: ({offset_x}, {offset_y})")
            
            # 确保图标转换为RGBA格式（旋转后或原始RGBA）
            if resized.mode != 'RGBA':
                resized = resized.convert('RGBA')
            
            # 使用alpha通道作为mask进行粘贴
            icon_mask = resized.split()[-1]
            scene_pil.paste(resized, (x, y), icon_mask)
            
            # 记录图标位置信息
            actual_w, actual_h = resized.size
            icon_bbox = [x, y, x + actual_w, y + actual_h]
            
            # 裁剪图标区域并转换为base64
            icon_region = scene_pil.crop(icon_bbox)
            icon_base64 = self.image_to_base64(icon_region, 'PNG')
            
            # 格式化边界框坐标为字符串列表
            bbox_str = [
                f"{icon_bbox[0]},{icon_bbox[1]}",  # 左上角
                f"{icon_bbox[2]},{icon_bbox[1]}",  # 右上角  
                f"{icon_bbox[2]},{icon_bbox[3]}",  # 右下角
                f"{icon_bbox[0]},{icon_bbox[3]}"   # 左下角
            ]
            
            # 添加到位置列表
            icon_positions.append({
                "mask": icon_base64,
                "bbox": bbox_str,
                "type": "icon",
                "position": f"icon_{i+1}",
                "size": size,
                "rotation": rotation_angle if enable_rotation else 0,
                "original_size": [size, size],
                "actual_size": [actual_w, actual_h]
            })
            
            print(f"  📍 记录图标 {i+1} 位置信息: ({x}, {y}) 尺寸: {actual_w}x{actual_h}")
            
            # 更新mask：精确标记图标的实际形状
            icon_mask_np = np.array(icon_mask)
            rotated_w, rotated_h = resized.size
            
            # 计算在场景中的实际边界
            y_end = min(y + rotated_h, updated_mask.shape[0])
            x_end = min(x + rotated_w, updated_mask.shape[1])
            
            # 确保不超出边界
            if x >= 0 and y >= 0 and x < updated_mask.shape[1] and y < updated_mask.shape[0]:
                # 获取实际的图标mask尺寸（处理边界情况）
                actual_h = y_end - y
                actual_w = x_end - x
                
                # 裁剪图标mask到实际可用区域
                icon_mask_cropped = icon_mask_np[:actual_h, :actual_w]
                
                # 将图标的不透明区域（alpha > 127）标记为黑色（显示图标位置）
                icon_opaque = icon_mask_cropped > 127
                updated_mask[y:y_end, x:x_end][icon_opaque] = 0  # 黑色显示图标位置
                
                # 统计新增的像素
                new_pixels = np.sum(icon_opaque)
                rotation_info = f", 旋转{rotation_angle:.1f}°" if rotation_angle != 0 else ""
                print(f"  图标 {i+1}: 位置({x},{y}), 原始尺寸{size}, 实际尺寸{rotated_w}x{rotated_h}, 标记像素{new_pixels}{rotation_info}")
        
        print(f"最终放置了 {len(final_positions)} 个图标，使用了 {len(set(size for _, size in final_positions))} 种不同尺寸")
        
        # 统计mask像素
        total_marked_pixels = np.sum(updated_mask == 0)
        print(f"输出Mask统计: 总共标记了{total_marked_pixels}个黑色像素显示图标位置")
        
        # 8. 转换回张量 - 确保只返回一张图和一张mask
        result_np = np.array(scene_pil).astype(np.float32) / 255.0
        if result_np.shape[-1] == 4:
            result_np = result_np[..., :3]
        
        # 转换图像张量
        result_tensor = torch.tensor(result_np).unsqueeze(0)
        
        # 转换mask张量（归一化到0-1范围）
        mask_normalized = updated_mask.astype(np.float32) / 255.0
        mask_tensor = torch.tensor(mask_normalized).unsqueeze(0)
        
        # 生成输出JSON
        output_json = {
            "masks": icon_positions
        }
        
        json_string = json.dumps(output_json, ensure_ascii=False, indent=2)
        
        print(f"✅ 图标分布完成！")
        print(f"📊 总计处理: {len(final_positions)} 个图标")
        print(f"📄 输出JSON包含: {len(icon_positions)} 个图标信息（包含输入的 {len(icon_positions) - len(final_positions)} 个已有图标）")
        
        return (result_tensor, mask_tensor, json_string)