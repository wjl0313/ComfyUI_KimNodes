import os
import torch
import numpy as np
from typing import List
from ultralytics import YOLO  # 确保已安装 ultralytics 库

# 设置模型目录为当前路径下的 /models/yolo-world
yolo_world_model_path = os.path.abspath(os.path.join(os.getcwd(), "models", "yolo-world"))

def get_files(directory, extensions):
    """
    获取指定目录下指定扩展名的文件。

    :param directory: 要搜索的目录路径
    :param extensions: 文件扩展名列表，例如 [".pt"]
    :return: 字典，键为文件名，值为完整路径
    """
    files = {}
    if not os.path.exists(directory):
        return files

    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in extensions):
            files[filename] = os.path.abspath(os.path.join(directory, filename))
    return files

def tensor2np(tensor):
    """
    将 PyTorch Tensor 转换为 NumPy 数组。

    :param tensor: torch.Tensor，输入张量。
    :return: np.ndarray，转换后的 NumPy 数组。
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()  # 如果张量在 GPU 上，移动到 CPU
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # 假设形状为 [1, H, W, C]
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # 移除单通道
    return tensor.numpy()

class YOLOWorld_Match:
    def __init__(self):
        self.NODE_NAME = 'YOLOWorld Object Matcher'
        os.environ['MODEL_CACHE_DIR'] = yolo_world_model_path  # 设置模型缓存目录

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义输入类型，包括图像、模型选择、置信度阈值和类别。
        """
        model_ext = [".pt"]
        FILES_DICT = get_files(yolo_world_model_path, model_ext)
        FILE_LIST = list(FILES_DICT.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "yolo_world_model": (FILE_LIST,),
                "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0, "max": 1, "step": 0.01}),  # 调低默认阈值
                "category": ("STRING", {"default": "person"}),  # 更改默认类别为常见类别
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = 'object_matcher'
    CATEGORY = "\U0001F34A Kim-Nodes/\U0001F50DYOLOWorld_Match | 特征匹配"

    def object_matcher(self, image, yolo_world_model, confidence_threshold, category):
        """
        使用 YOLO 模型进行对象检测，返回是否匹配的结果。

        :param image: 输入图像（torch.Tensor）。
        :param yolo_world_model: 模型文件名称。
        :param confidence_threshold: 置信度阈值。
        :param category: 检测类别。
        :return: 检测结果的字符串。
        """
        if isinstance(image, torch.Tensor):
            image = [image]

        model = self.load_yolo_world_model(yolo_world_model)
        result_strings = []

        model_classes = model.names
        target_category = category.strip().lower()

        if target_category not in [name.lower() for name in model_classes.values()]:
            return ("false",)

        for img_tensor in image:
            img_np = tensor2np(img_tensor)

            if img_np.size == 0:
                result_strings.append("false")
                continue

            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            elif img_np.shape[-1] == 4:
                img_np = img_np[:, :, :3]

            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)

            try:
                results = model.predict(source=img_np, conf=confidence_threshold, verbose=False)
            except Exception as e:
                result_strings.append("false")
                continue

            if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                detections = results[0].boxes
                detected = False
                for box in detections:
                    cls_id = int(box.cls[0].item())
                    cls_name = model_classes.get(cls_id, "").lower()
                    confidence = box.conf[0].item()
                    if cls_name == target_category and confidence >= confidence_threshold:
                        detected = True
                        break

                if detected:
                    result_strings.append("true")
                else:
                    result_strings.append("false")
            else:
                result_strings.append("false")

        if len(result_strings) == 1:
            return (result_strings[0],)
        else:
            return (", ".join(result_strings),)

    def load_yolo_world_model(self, model_id: str):
        """
        加载 YOLO 模型。

        :param model_id: 模型文件名称。
        :return: 加载后的 YOLO 模型。
        """
        full_model_path = os.path.abspath(os.path.join(yolo_world_model_path, model_id))
        try:
            model = YOLO(full_model_path)
        except Exception as e:
            raise
        return model
