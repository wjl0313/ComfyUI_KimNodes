import os
import logging
from PIL import Image, PngImagePlugin
from typing import Tuple
import torch
import numpy as np
from datetime import datetime

"""
Add_ImageMetadata

一个用于添加元数据到图片并保存的节点。
"""

def tensor2pil(image_tensor):
    """
    将图像张量转换为 PIL.Image 对象。
    """
    if isinstance(image_tensor, torch.Tensor):
        print(f"Original tensor shape: {image_tensor.shape}")
        logging.info(f"Original tensor shape: {image_tensor.shape}")

        # 如果张量有 4 个维度，我们需要处理批次维度
        if image_tensor.ndim == 4:
            # 检查批次维度是否为 1
            if image_tensor.shape[0] == 1:
                image_tensor = image_tensor.squeeze(0)
                print(f"After squeezing batch dimension: {image_tensor.shape}")
                logging.info(f"After squeezing batch dimension: {image_tensor.shape}")
            else:
                # 如果批次维度大于 1，我们只处理第一个样本
                image_tensor = image_tensor[0]
                print(f"Selected first sample from batch: {image_tensor.shape}")
                logging.info(f"Selected first sample from batch: {image_tensor.shape}")

        # 现在，image_tensor 应该是 3 维的
        if image_tensor.ndim == 3:
            print(f"Processing 3D tensor with shape: {image_tensor.shape}")
            logging.info(f"Processing 3D tensor with shape: {image_tensor.shape}")

            # 判断通道维的位置
            if image_tensor.shape[0] <= 4:
                # 通道在第一个维度 [C, H, W]
                image_numpy = image_tensor.permute(1, 2, 0).cpu().numpy()
            elif image_tensor.shape[2] <= 4:
                # 通道在最后一个维度 [H, W, C]
                image_numpy = image_tensor.cpu().numpy()
            else:
                raise ValueError(f"无法解释张量形状: {image_tensor.shape}")

            print(f"image_numpy.shape: {image_numpy.shape}")
            logging.info(f"image_numpy.shape: {image_numpy.shape}")

            # 缩放到 0-255 并转换为 uint8
            image_numpy = (image_numpy * 255).clip(0, 255).astype(np.uint8)

            # 处理不同的通道数
            if image_numpy.shape[2] == 1:
                image_numpy = image_numpy.squeeze(2)
                return Image.fromarray(image_numpy, mode='L')
            elif image_numpy.shape[2] == 3:
                return Image.fromarray(image_numpy, mode='RGB')
            elif image_numpy.shape[2] == 4:
                return Image.fromarray(image_numpy, mode='RGBA')
            else:
                raise ValueError(f"不支持的通道数: {image_numpy.shape[2]}")

        elif image_tensor.ndim == 2:
            # 灰度图像
            image_numpy = image_tensor.cpu().numpy()
            image_numpy = (image_numpy * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(image_numpy, mode='L')

        else:
            raise ValueError(f"无法处理张量维度: {image_tensor.ndim}")

    else:
        raise TypeError("输入必须是 torch.Tensor 类型的图像张量。")

class Add_ImageMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入被覆写图片": ("IMAGE",),  # 需要添加元数据的图片
                "输入图片元数据": ("DICT",),  # 接收元数据
                "output_dir": ("STRING", {  # 保存图片的目录
                    "multiline": False,
                    "default": "output",
                    "placeholder": "保存图片的目录（例如：output/ 或 /custom/path/）",
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "add_metadata"
    CATEGORY = "🍊 Kim-Nodes/🔢Metadata | 元数据处理"
    OUTPUT_NODE = True  # 标记为输出节点

    def __init__(self):
        pass

    def add_metadata(self, 输入图片元数据, 输入被覆写图片, output_dir="output") -> Tuple:
        try:
            print("Add_ImageMetadata 节点开始执行。")
            logging.info("Add_ImageMetadata 节点开始执行。")

            # 设置保存路径到用户指定的目录，默认为 'output'
            if output_dir:
                save_dir = os.path.abspath(output_dir)
            else:
                project_root = os.getcwd()  # 获取当前工作目录作为项目根目录
                save_dir = os.path.join(project_root, "output")

            # 确保保存目录存在
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"已创建目录: {save_dir}")
                    logging.info(f"已创建目录: {save_dir}")
                except Exception as e:
                    logging.error(f"创建目录失败: {e}")
                    print(f"创建目录失败: {e}")
                    return ()

            # 生成文件名：日期加序号.png
            date_str = datetime.now().strftime("%Y%m%d")
            serial = 1
            while True:
                filename = f"{date_str}_{serial}.png"
                new_image_path = os.path.join(save_dir, filename)
                if not os.path.exists(new_image_path):
                    break
                serial += 1

            print(f"生成的文件名: {filename}")
            logging.info(f"生成的文件名: {filename}")

            # 处理 输入被覆写图片
            try:
                # 从输入被覆写图片字典中提取图像张量
                if isinstance(输入被覆写图片, dict) and 'samples' in 输入被覆写图片:
                    image_tensor_output = 输入被覆写图片['samples'][0]
                elif isinstance(输入被覆写图片, torch.Tensor):
                    image_tensor_output = 输入被覆写图片
                else:
                    raise TypeError("输入被覆写图片不是有效的图像数据。")

                输入被覆写图片_pil = tensor2pil(image_tensor_output)
                print("输入被覆写图片已转换为 PIL.Image 对象。")
                logging.info("输入被覆写图片已转换为 PIL.Image 对象。")
            except Exception as e:
                logging.error(f"转换输入被覆写图片失败: {e}")
                print(f"转换输入被覆写图片失败: {e}")
                return ()

            # 使用 输入图片元数据
            input_info = 输入图片元数据  # 直接使用输入的元数据
            print(f"获取到输入的元数据: {list(input_info.keys())}")
            logging.info(f"获取到输入的元数据: {list(input_info.keys())}")

            # 创建一个新的 PngInfo 对象，用于存储元数据
            png_info = PngImagePlugin.PngInfo()

            # 将输入的元数据添加到 PngInfo 对象中
            for key, value in input_info.items():
                # 处理可能的编码问题，确保所有字符串都是 UTF-8 编码
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='replace')
                elif not isinstance(value, str):
                    value = str(value)

                # 使用与原始图片相同的字段名称和块类型
                if key.lower() == 'parameters':
                    metadata_key = 'parameters'  # 确保字段名称一致
                    # 使用 add_text 方法，并通过 lang 参数确保使用 iTXt 块
                    png_info.add_text(metadata_key, value, zip=False)
                else:
                    # 添加其他元数据字段，使用 tEXt 块
                    png_info.add_text(key, value, zip=False)

                print(f"添加元数据: {key} = {value}")
                logging.info(f"添加元数据: {key} = {value}")

            # 保存新图片并附加元数据
            try:
                输入被覆写图片_pil.save(new_image_path, "PNG", pnginfo=png_info)
                logging.info(f"已将元数据添加到新图片并保存到 '{new_image_path}'")
                print(f"已将元数据添加到新图片并保存到 '{new_image_path}'")
            except Exception as e:
                logging.error(f"保存图片失败: {e}")
                print(f"保存图片失败: {e}")
                return ()

            return ()

        except Exception as e:
            logging.error(f"发生异常: {e}")
            print(f"发生异常: {e}")
            return ()
