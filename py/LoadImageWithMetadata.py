import os
import logging
from PIL import Image
import numpy as np

"""
LoadImageWithMetadata

一个用于从文件路径加载图像，并输出图像元数据的节点。
"""

class LoadImage_Metadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "输入图片的文件路径",
                }),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("输出元数据",)
    FUNCTION = "load_image"
    CATEGORY = "🍒 Kim-Nodes/🔢Metadata | 元数据处理"

    def __init__(self):
        pass

    def load_image(self, image_path):
        try:
            print(f"正在加载图像: {image_path}")
            logging.info(f"正在加载图像: {image_path}")

            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"文件不存在: {image_path}")

            # 打开图像并获取元数据
            image = Image.open(image_path)
            metadata = image.info  # 获取元数据

            print(f"图像元数据: {list(metadata.keys())}")
            logging.info(f"图像元数据: {list(metadata.keys())}")

            return (metadata,)

        except Exception as e:
            logging.error(f"加载图像失败: {e}")
            print(f"加载图像失败: {e}")
            return (None,)
