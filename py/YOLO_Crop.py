from PIL import Image, ImageOps
import numpy as np
from ultralytics import YOLO
import torch
import logging
import os

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置模型目录为当前路径下的 /models/yolo
model_path = os.path.abspath(os.path.join(os.getcwd(), "models", "yolo"))


# 定义 get_files 函数
def get_files(directory, extensions):
    """
    获取指定目录下指定扩展名的文件。

    :param directory: 要搜索的目录路径
    :param extensions: 文件扩展名列表，例如 [".pt"]
    :return: 字典，键为文件名，值为完整路径
    """
    files = {}
    if not os.path.exists(directory):  # 检查目录是否存在
        logging.warning(f"目录不存在: {directory}")
        return files

    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in extensions):  # 匹配扩展名
            files[filename] = os.path.join(directory, filename)
    return files


class YOLO_Crop:
    """
    Face Detection Node for processing images and detecting faces.
    """
    model_cache = {}  # 模型缓存，避免重复加载

    @classmethod
    def INPUT_TYPES(cls):
        model_ext = [".pt"]
        FILES_DICT = get_files(model_path, model_ext)  # 获取模型文件列表
        FILE_LIST = list(FILES_DICT.keys())  # 提取文件名列表
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "yolo_model": (FILE_LIST,),  # 可选模型文件
                "confidence": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.01
                }),  # 置信度阈值
                "square_size": ("FLOAT", {
                    "default": 100.0, "min": 10.0, "max": 200.0, "step": 1.0
                }),  # 现在使用图像尺寸的百分比
                "vertical_offset": ("FLOAT", {
                    "default": 0.0, "min": -512.0, "max": 512.0, "step": 1.0
                }),  # 上下偏移百分比
                "horizontal_offset": ("FLOAT", {
                    "default": 0.0, "min": -512.0, "max": 512.0, "step": 1.0
                })  # 左右偏移百分比
            },
        }

    RETURN_TYPES = ("IMAGE", "DATA")  # 增加输出项 DATA
    FUNCTION = "Face_yolo"
    CATEGORY = "🍒 Kim-Nodes/✂ Crop | 裁剪工具"

    def __init__(self):
        pass

    def Face_yolo(self, image, confidence, square_size, yolo_model, vertical_offset, horizontal_offset):
        # 检查并加载模型
        if yolo_model in self.model_cache:
            model = self.model_cache[yolo_model]  # 使用缓存模型
        else:
            full_model_path = os.path.join(model_path, yolo_model)  # 获取模型路径
            try:
                model = YOLO(full_model_path)  # 加载模型
                self.model_cache[yolo_model] = model  # 缓存模型
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise

        device = "cuda" if torch.cuda.is_available() else "cpu"  # 确定设备

        # 处理输入图像
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4 and image.shape[-1] in [1, 3, 4]:  # 检查形状
                image = image.permute(0, 3, 1, 2)  # 调整形状为 (batch_size, channels, height, width)
            image_np = image.cpu().numpy()[0]  # 去除批次维度
            image_np = np.transpose(image_np, (1, 2, 0))  # 转为 (height, width, channels)
        else:
            raise TypeError("输入的 image 不是 torch.Tensor 类型")

        # 确保输入为 RGB 格式
        channels = image_np.shape[2]
        if channels == 1:
            image_np = np.repeat(image_np, 3, axis=2)  # 灰度图转 RGB
        elif channels == 4:
            image_np = image_np[:, :, :3]  # 去掉 alpha 通道

        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)  # 转换为 8 位像素值
        image_pil = Image.fromarray(image_np)  # 转换为 PIL 图像

        # 使用 YOLO 模型检测
        results = model.predict(source=image_np, conf=confidence, device=device)

        bboxes = []  # 存储检测到的边界框
        width, height = image_np.shape[1], image_np.shape[0]  # 获取图像的宽和高

        for result in results:
            for box in result.boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()
                cls = box.cls.cpu().numpy()

                if cls == 0 and conf >= confidence:
                    # 计算检测到的人脸框的宽度和高度
                    face_width = xmax - xmin
                    face_height = ymax - ymin
                    face_size = max(face_width, face_height)  # 使用较大的边作为基准
                    
                    # 计算图像的最小边长
                    min_side = min(width, height)
                    
                    # 根据人脸尺寸计算合适的方块大小
                    # 添加额外的边距（这里使用1.5作为系数，您可以根据需要调整）
                    face_margin = 1.5
                    actual_square_size = face_size * face_margin
                    
                    # 应用用户指定的百分比调整
                    actual_square_size = actual_square_size * (square_size / 100.0)
                    
                    # 计算边界框的中心
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2

                    # 使用实际的方块大小
                    half_size = actual_square_size / 2

                    # 计算偏移后的坐标
                    vertical_offset_px = (actual_square_size) * vertical_offset / 100
                    horizontal_offset_px = (actual_square_size) * horizontal_offset / 100

                    # 计算新的边界框坐标，保持正方形大小
                    xmin_new = max(0, int(center_x - half_size + horizontal_offset_px))
                    xmax_new = min(width, int(center_x + half_size + horizontal_offset_px))
                    ymin_new = max(0, int(center_y - half_size + vertical_offset_px))
                    ymax_new = min(height, int(center_y + half_size + vertical_offset_px))

                    # 存储归一化后的边界框坐标和像素坐标
                    bboxes.append({
                        "xmin": xmin_new / width,
                        "ymin": ymin_new / height,
                        "xmax": xmax_new / width,
                        "ymax": ymax_new / height,
                        "xmin_pixel": xmin_new,
                        "ymin_pixel": ymin_new,
                        "xmax_pixel": xmax_new,
                        "ymax_pixel": ymax_new
                    })

        if not bboxes:
            return (self.process_output(image_pil), {"pixels": image_np.tolist(), "bboxes": [], "square_size": square_size})  # 无检测结果

        # 裁剪并更新边界框
        cropped_faces = [self.crop_face(image_pil, bbox) for bbox in bboxes]  # 裁剪脸部
        return (
            self.process_output(cropped_faces[0]),  # 返回裁剪后的第一张图像
            {
                "pixels": image_np.tolist(),  # 返回图像像素数据
                "bboxes": bboxes  # 返回检测到的边界框
            }
        )

    def crop_face(self, image, bbox):
        """
        Crop a face from the image based on bounding box.
        """
        width, height = image.size  # 获取图像的宽度和高度
        left = bbox["xmin_pixel"]  # 使用传入的像素坐标
        top = bbox["ymin_pixel"]
        right = bbox["xmax_pixel"]
        bottom = bbox["ymax_pixel"]

        # 使用新的坐标裁剪图像
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image.convert("RGB") if cropped_image.mode != "RGB" else cropped_image


    def process_output(self, scene_image_pil):
        """
        将结果转换为模型所需的格式返回。
        """
        result_image = np.array(scene_image_pil).astype(np.float32) / 255.0
        if result_image.shape[-1] == 4:
            result_image = result_image[..., :3]  # 去掉 alpha 通道
        elif result_image.shape[-1] == 1:
            result_image = np.repeat(result_image, 3, axis=-1)  # 灰度图转 RGB

        result_image = np.expand_dims(result_image, axis=0)  # 添加批次维度
        # logger.info(f"faceyolo输出的维度 {result_image.shape}")
        return torch.from_numpy(result_image)  # 转换为张量
