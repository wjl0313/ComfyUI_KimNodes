from PIL import Image, ImageOps
import numpy as np
from ultralytics import YOLO
import torch
import logging
import os

# 将日志级别设置为ERROR，只保留错误信息
logging.basicConfig(level=logging.ERROR)
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


class YOLO_Multi_Crop:
    """
    多人物检测和裁剪节点，可以检测图像中的多个人物并分别裁剪。
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
                }),  # 使用图像尺寸的百分比
                "max_detections": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1
                }),  # 最大检测数量
            },
        }

    RETURN_TYPES = ("IMAGE", "DATA")
    FUNCTION = "multi_crop"
    OUTPUT_IS_LIST = (True, False)  # 图像输出为列表，DATA输出不是列表
    CATEGORY = "🍒 Kim-Nodes/✂ Crop | 裁剪工具"

    def __init__(self):
        pass

    def multi_crop(self, image, confidence, square_size, yolo_model, max_detections):
        # 检查并加载模型
        if yolo_model in self.model_cache:
            model = self.model_cache[yolo_model]  # 使用缓存模型
        else:
            full_model_path = os.path.join(model_path, yolo_model)  # 获取模型路径
            try:
                model = YOLO(full_model_path)  # 加载模型
                self.model_cache[yolo_model] = model  # 缓存模型
            except Exception as e:
                logger.error(f"加载YOLO模型失败: {e}")
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

                if cls == 0 and conf >= confidence:  # 假设类别0是人
                    # 计算检测到的人物框的宽度和高度
                    person_width = xmax - xmin
                    person_height = ymax - ymin
                    person_size = max(person_width, person_height)  # 使用较大的边作为基准
                    
                    # 计算图像的最小边长
                    min_side = min(width, height)
                    
                    # 根据人物尺寸计算合适的方块大小
                    # 添加额外的边距
                    person_margin = 1.5
                    actual_square_size = person_size * person_margin
                    
                    # 应用用户指定的百分比调整
                    actual_square_size = actual_square_size * (square_size / 100.0)
                    
                    # 计算边界框的中心
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2

                    # 使用实际的方块大小
                    half_size = actual_square_size / 2

                    # 计算新的边界框坐标，保持正方形大小
                    xmin_new = max(0, int(center_x - half_size))
                    xmax_new = min(width, int(center_x + half_size))
                    ymin_new = max(0, int(center_y - half_size))
                    ymax_new = min(height, int(center_y + half_size))

                    # 存储归一化后的边界框坐标和像素坐标
                    bboxes.append({
                        "xmin": xmin_new / width,
                        "ymin": ymin_new / height,
                        "xmax": xmax_new / width,
                        "ymax": ymax_new / height,
                        "xmin_pixel": xmin_new,
                        "ymin_pixel": ymin_new,
                        "xmax_pixel": xmax_new,
                        "ymax_pixel": ymax_new,
                        "confidence": float(conf),
                        "center_x": center_x,  # 添加中心点坐标用于排序
                        "center_y": center_y
                    })

        if not bboxes:
            # 如果没有检测到人物，返回原图和空数据
            return ([self.process_single_image(image_pil)], {"pixels": image_np.tolist(), "bboxes": []})

        # 按照从左到右、从上到下的顺序排序边界框
        # 首先按行分组（使用一个阈值来确定是否在同一行）
        y_threshold = height * 0.1  # 10%的图像高度作为阈值
        
        # 按y坐标（从上到下）粗略排序
        bboxes.sort(key=lambda x: x["center_y"])
        
        # 分组到不同的行
        rows = []
        current_row = [bboxes[0]]
        
        for i in range(1, len(bboxes)):
            # 如果当前边界框与当前行的第一个边界框在y方向上的距离小于阈值，则认为它们在同一行
            if abs(bboxes[i]["center_y"] - current_row[0]["center_y"]) < y_threshold:
                current_row.append(bboxes[i])
            else:
                # 否则开始一个新行
                rows.append(current_row)
                current_row = [bboxes[i]]
        
        # 添加最后一行
        if current_row:
            rows.append(current_row)
        
        # 对每一行内的边界框按x坐标（从左到右）排序
        for row in rows:
            row.sort(key=lambda x: x["center_x"])
        
        # 将排序后的边界框重新展平为一个列表
        sorted_bboxes = []
        for row in rows:
            sorted_bboxes.extend(row)
        
        # 更新边界框列表
        bboxes = sorted_bboxes[:max_detections]
        
        # 裁剪所有检测到的人物
        cropped_persons = []
        face_data = []  # 用于存储每个脸部的详细信息

        for i, bbox in enumerate(bboxes):
            cropped_img = self.crop_person(image_pil, bbox)
            img_width, img_height = cropped_img.size
            
            # 记录每个脸部的详细信息
            face_info = {
                "index": i,
                "center_x": float(bbox["center_x"]),
                "center_y": float(bbox["center_y"]),
                "xmin_pixel": int(bbox["xmin_pixel"]),
                "ymin_pixel": int(bbox["ymin_pixel"]),
                "xmax_pixel": int(bbox["xmax_pixel"]),
                "ymax_pixel": int(bbox["ymax_pixel"]),
                "width": img_width,
                "height": img_height,
                "confidence": float(bbox["confidence"])
            }
            face_data.append(face_info)
            
            cropped_persons.append(cropped_img)

        # 处理所有裁剪后的图像为单独的张量列表
        processed_images = []
        for i, img in enumerate(cropped_persons):
            processed = self.process_single_image(img)
            processed_images.append(processed)
        
        return (
            processed_images,  # 返回所有裁剪后的图像列表
            {
                # 移除像素数据，避免数据过大
                "bboxes": bboxes,  # 返回检测到的所有边界框
                "face_data": face_data,  # 添加结构化的脸部数据
                "count": len(bboxes),  # 返回检测到的人物数量
                "image_size": {"width": width, "height": height}  # 添加图像尺寸信息
            }
        )

    def crop_person(self, image, bbox):
        """
        根据边界框裁剪人物图像
        """
        width, height = image.size  # 获取图像的宽度和高度
        left = bbox["xmin_pixel"]  # 使用传入的像素坐标
        top = bbox["ymin_pixel"]
        right = bbox["xmax_pixel"]
        bottom = bbox["ymax_pixel"]

        # 使用坐标裁剪图像
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image.convert("RGB") if cropped_image.mode != "RGB" else cropped_image

    def process_single_image(self, pil_image):
        """
        将单个PIL图像转换为模型所需的格式返回。
        """
        # 转换为numpy数组并归一化
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # 确保是RGB格式
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]  # 去掉alpha通道
        elif len(img_array.shape) == 2 or img_array.shape[-1] == 1:
            # 处理灰度图像
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.repeat(img_array, 3, axis=-1)
            
        # 添加批次维度
        img_array = np.expand_dims(img_array, axis=0)
        return torch.from_numpy(img_array)  # 转换为张量 