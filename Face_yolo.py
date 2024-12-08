from PIL import Image, ImageOps
import numpy as np
from ultralytics import YOLO
import torch
import logging
import os
import os.path

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义模型目录为相对路径：根目录下的 /models/yolo
base_dir = os.getcwd()  # 获取当前工作目录
model_path = os.path.join(base_dir, "models", "yolo")


# 定义 get_files 函数
def get_files(directory, extensions):
    """
    获取指定目录下指定扩展名的文件。

    :param directory: 要搜索的目录路径
    :param extensions: 文件扩展名列表，例如 [".pt"]
    :return: 字典，键为文件名，值为完整路径
    """
    files = {}
    if not os.path.exists(directory):
        logging.warning(f"目录不存在: {directory}")
        return files

    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in extensions):
            files[filename] = os.path.join(directory, filename)
    return files

class FaceDetectionNode:
    """
    Face Detection Node for processing images and detecting faces.
    """
    # 模型缓存，避免重复加载
    model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        model_ext = [".pt"]
        FILES_DICT = get_files(model_path, model_ext)
        FILE_LIST = list(FILES_DICT.keys())
        # logger.info(f"Model path: {model_path}")
        # logger.info(f"Available models: {FILE_LIST}")
        return {
            "required": {
                "image": ("IMAGE", ),
                "yolo_model": (FILE_LIST,),
                "confidence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "expand_percent": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "slider",
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "Face_yolo"
    CATEGORY = "🍊 Kim-Nodes"

    def __init__(self):
        pass

    def Face_yolo(self, image, confidence, expand_percent, yolo_model):
        logger.info(f"Executing face detection with confidence={confidence}, expand_percent={expand_percent}")

        # 检查模型是否已加载
        if yolo_model in self.model_cache:
            model = self.model_cache[yolo_model]
            logger.info(f"Using cached model: {yolo_model}")
        else:
            # 加载模型
            full_model_path = os.path.join(model_path, yolo_model)
            logger.info(f"Loading YOLO model from {full_model_path}")
            try:
                model = YOLO(full_model_path)
                self.model_cache[yolo_model] = model  # 缓存模型
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise

        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # 检查并调整输入图像形状
        logger.debug(f"输入的 image 类型: {type(image)}")
        if isinstance(image, torch.Tensor):
            logger.debug(f"输入的 image 张量形状: {image.shape}")
            # 调整输入图像形状为 (batch_size, channels, height, width)
            if len(image.shape) == 4 and image.shape[-1] in [1, 3, 4]:
                # 将最后一维的通道移动到第二维
                image = image.permute(0, 3, 1, 2)  # 形状从 (batch_size, height, width, channels) 转为 (batch_size, channels, height, width)
            elif len(image.shape) != 4 or image.shape[1] not in [1, 3, 4]:
                raise ValueError(f"输入的 image 形状不正确，应为 (batch_size, channels, height, width)，但得到 {image.shape}")

            # 将 PyTorch 张量转换为 NumPy 数组
            image_np = image.cpu().numpy()

            # 去除批次维度
            image_np = image_np[0]  # 形状: (channels, height, width)

            # 调整形状为 (height, width, channels)
            image_np = np.transpose(image_np, (1, 2, 0))
            logger.debug(f"处理后的图像形状: {image_np.shape}")

        else:
            raise TypeError("输入的 image 不是 torch.Tensor 类型")


        # 确保输入是标准格式 (height, width, channels)
        channels = image_np.shape[2]
        if channels == 1:
            # 灰度图像，转换为 RGB
            logger.debug("将灰度图像转换为 RGB 格式。")
            image_np = np.repeat(image_np, 3, axis=2)
            channels = 3
        elif channels == 4:
            # RGBA 图像，转换为 RGB
            logger.debug("将 RGBA 图像转换为 RGB 格式。")
            image_np = image_np[:, :, :3]
            channels = 3

        if channels != 3:
            raise ValueError(f"期望图像有 3 个通道，但得到 {channels} 个通道。")

        logger.debug(f"最终图像尺寸 - 高度: {image_np.shape[0]}, 宽度: {image_np.shape[1]}, 通道数: {channels}")

        # 将像素值从 [0,1] 转换为 [0,255]
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)

        # 将 NumPy 数组转换为 PIL 图像
        image_pil = Image.fromarray(image_np)

        # 使用 YOLO 模型检测脸部
        try:
            results = model.predict(source=image_np, conf=confidence, device=device)
        except AttributeError as e:
            logger.error(f"Error during prediction: {e}")
            raise

        bboxes = []
        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()
                cls = box.cls.cpu().numpy()

                if cls == 0 and conf >= confidence:  # 确保检测到脸部
                    bboxes.append({
                        "xmin": xmin / image_np.shape[1],
                        "ymin": ymin / image_np.shape[0],
                        "width": (xmax - xmin) / image_np.shape[1],
                        "height": (ymax - ymin) / image_np.shape[0],
                    })

        if not bboxes:
            logger.warning("No faces detected.")
            # 处理输出并返回原始图像
            result_tensor = self.process_output(image_pil)
            return (result_tensor,)  # 返回处理后的原图

        cropped_faces = []
        for bbox in bboxes:
            cropped_face = self.crop_face(image_pil, bbox, expand_percent)
            cropped_faces.append(cropped_face)

        # 这里只处理第一个裁剪的脸部图像
        result_tensor = self.process_output(cropped_faces[0])
        return (result_tensor,)


    def crop_face(self, image, bbox, expand_percent):
        """
        Crop a face from the image based on bounding box.
        """
        width, height = image.size
        left = int(bbox["xmin"] * width)
        top = int(bbox["ymin"] * height)
        right = left + int(bbox["width"] * width)
        bottom = top + int(bbox["height"] * height)

        # 扩展边界框
        expand_x = int((right - left) * expand_percent / 100)
        expand_y = int((bottom - top) * expand_percent / 100)
        left = max(0, left - expand_x)
        top = max(0, top - expand_y)
        right = min(width, right + expand_x)
        bottom = min(height, bottom + expand_y)

        # 裁剪并返回
        cropped_image = image.crop((left, top, right, bottom))

        # 确保返回的是 RGB 图像
        if cropped_image.mode != "RGB":
            cropped_image = cropped_image.convert("RGB")

        return cropped_image


    def process_output(self, scene_image_pil):
        """
        将结果转换为模型所需的格式返回。
        """
        # 将 PIL 图像转换为 NumPy 数组并归一化
        result_image = np.array(scene_image_pil).astype(np.float32) / 255.0
        print(f"[DEBUG] 输出结果 (result_image) 的维度: {result_image.shape}")

        # 如果结果是 RGBA (H, W, 4)，需要转换回 RGB (H, W, 3)
        if result_image.shape[-1] == 4:
            # 使用 alpha 通道进行混合
            alpha = result_image[..., 3:4]
            rgb = result_image[..., :3]
            result_image = rgb
            print(f"[DEBUG] 转换后的 RGB 图像维度: {result_image.shape}")

        # 如果结果是灰度图像 (H, W, 1)，需要转换为 RGB
        if result_image.shape[-1] == 1:
            result_image = np.repeat(result_image, 3, axis=-1)
            print(f"[DEBUG] 灰度图像转换为 RGB 后的维度: {result_image.shape}")

        # 添加批次维度
        result_image = np.expand_dims(result_image, axis=0)
        print(f"[DEBUG] 添加批次维度后的 result_image 维度: {result_image.shape}")

        # 转换为张量
        result_tensor = torch.from_numpy(result_image)
        print(f"[DEBUG] 输出 result_tensor 的维度: {result_tensor.shape}")

        return result_tensor


