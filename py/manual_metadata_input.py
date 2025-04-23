import os  
import logging
from PIL import Image, PngImagePlugin
from typing import Tuple, Optional
import torch
import numpy as np

def tensor2pil(image_tensor: torch.Tensor) -> Image.Image:
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

class Manual_MetadataInput:
    """
    Manual_MetadataInput

    一个用于手动填写元数据参数并输出元数据字典的节点，同时支持自动获取图片尺寸。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "placeholder": "输入提示词（prompt）",
                }),
                "图像": ("IMAGE", {  # 将图片输入设置为必填项
                    "description": "输入图片，用于自动获取尺寸",
                }),
            },
            "optional": {
                "steps": ("INT", {
                    "default": 10,  # 从样本中获取的默认值
                    "min": 1,
                    "max": 1000,
                }),
                "sampler": ("STRING", {
                    "default": "Euler",  # 从样本中获取的默认值
                    "placeholder": "输入采样器名称",
                }),
                "schedule_type": ("STRING", {
                    "default": "Simple",  # 从样本中获取的默认值
                    "placeholder": "输入调度类型（Schedule type）",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.0,  # 从样本中获取的默认值
                    "min": 0.0,
                    "max": 100.0,
                }),
                "distilled_cfg_scale": ("FLOAT", {
                    "default": 3.5,  # 从样本中获取的默认值
                    "min": 0.0,
                    "max": 100.0,
                }),
                "seed": ("INT", {
                    "default": 1173957321,  # 从样本中获取的默认值
                    "min": 0,
                    "max": 4294967295,
                }),
                "model_hash": ("STRING", {
                    "default": "9965eb995e",  # 从样本中获取的默认值
                    "placeholder": "输入模型哈希值",
                }),
                "model": ("STRING", {
                    "default": "kimVixen_fp8_e4m3fn",  # 从样本中获取的默认值
                    "placeholder": "输入模型名称",
                }),
                "lora_hashes": ("STRING", {
                    "default": '"flux_loraName: e3b0c44298fc"',  # 从样本中获取的默认值
                    "placeholder": "输入 Lora 哈希值",
                }),
                "version": ("STRING", {
                    "default": "",  # 从样本中获取的默认值
                    "placeholder": "输入版本信息",
                }),
                "module_1": ("STRING", {
                    "default": "ae",  # 从样本中获取的默认值
                    "placeholder": "输入模块 1 信息",
                }),
                "module_2": ("STRING", {
                    "default": "clip_l",  # 从样本中获取的默认值
                    "placeholder": "输入模块 2 信息",
                }),
                "module_3": ("STRING", {
                    "default": "t5xxl_fp16",  # 从样本中获取的默认值
                    "placeholder": "输入模块 3 信息",
                }),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("输出元数据",)
    FUNCTION = "generate_metadata"
    CATEGORY = "🍊 Kim-Nodes/🔢Metadata | 元数据处理"

    def __init__(self):
        pass

    def generate_metadata(self, prompt: str, 图像: torch.Tensor,
                         steps: int = 10, sampler: str = "Euler", schedule_type: str = "Simple",
                         cfg_scale: float = 1.0, distilled_cfg_scale: float = 3.5, seed: int = 1173957321,
                         model_hash: str = "9965eb995e",
                         model: str = "kimVixen_fp8_e4m3fn",
                         lora_hashes: str = '"flux_loraName: e3b0c44298fc"',
                         version: str = "",
                         module_1: str = "ae", module_2: str = "clip_l",
                         module_3: str = "t5xxl_fp16") -> Tuple[dict]:
        """
        生成元数据字典。
        """
        try:
            print("Manual_MetadataInput 节点开始执行。")
            logging.info("Manual_MetadataInput 节点开始执行。")

            # 自动获取图片尺寸
            try:
                pil_image = tensor2pil(图像)
                width, height = pil_image.size
                size = f"{width}x{height}"
                print(f"自动获取图片尺寸: {size}")
                logging.info(f"自动获取图片尺寸: {size}")
            except Exception as e:
                logging.error(f"自动获取图片尺寸失败: {e}")
                print(f"自动获取图片尺寸失败: {e}")
                # 如果获取尺寸失败，设置一个默认值或抛出异常
                size = "1024x1024"  # 或者根据需求处理
                print(f"使用默认尺寸: {size}")
                logging.info(f"使用默认尺寸: {size}")

            # 构建参数列表
            parameters_list = []

            # 添加参数到列表
            parameters_list.append(f"Steps: {steps}")
            parameters_list.append(f"Sampler: {sampler}")
            parameters_list.append(f"Schedule type: {schedule_type}")
            parameters_list.append(f"CFG scale: {cfg_scale}")
            parameters_list.append(f"Distilled CFG Scale: {distilled_cfg_scale}")
            parameters_list.append(f"Seed: {seed}")
            parameters_list.append(f"Size: {size}")
            if model_hash:
                parameters_list.append(f"Model hash: {model_hash}")
            if model:
                parameters_list.append(f"Model: {model}")
            if lora_hashes:
                parameters_list.append(f"Lora hashes: {lora_hashes}")
            if version:
                parameters_list.append(f"Version: {version}")
            if module_1:
                parameters_list.append(f"Module 1: {module_1}")
            if module_2:
                parameters_list.append(f"Module 2: {module_2}")
            if module_3:
                parameters_list.append(f"Module 3: {module_3}")

            # 将参数列表组合成字符串
            parameters_string = ', '.join(parameters_list)

            # 组合提示词和参数
            full_parameters = f"{prompt}\n{parameters_string}"

            # 创建元数据字典，使用 'Parameters' 作为键
            metadata = {'Parameters': full_parameters}

            print("元数据生成成功。")
            logging.info("元数据生成成功。")

            return (metadata,)

        except Exception as e:
            logging.error(f"发生异常: {e}")
            print(f"发生异常: {e}")
            return ()
