from PIL import Image 
import numpy as np
import logging
import os

# 导入 torch
try:
    import torch
except ImportError:
    torch = None  # 如果未安装 torch，则处理为 None

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Crop_Paste:
    """
    Node for merging a single cropped image back into the original image
    based on the bounding box coordinates.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入的原始图片
                "crop_image": ("IMAGE",),    # 置信度最高的裁剪图像
                "data": ("DATA",),              # 包含边界框信息的字典
            },
        }

    RETURN_TYPES = ("IMAGE",)  # 返回修改后的图片
    FUNCTION = "crop_paste"
    CATEGORY = "🍊 Kim-Nodes/✂ Crop | 裁剪处理"

    def __init__(self):
        pass

    def crop_paste(self, data, image, crop_image):
        """
        Merge the single cropped image back into the original image.
        """
        # 打印输入图像的类型和维度信息
        logger.info(f"Original input type: {type(image)}")
        if isinstance(image, torch.Tensor):
            logger.info(f"Original input shape (torch.Tensor): {image.shape}")
        elif isinstance(image, np.ndarray):
            logger.info(f"Original input shape (NumPy array): {image.shape}")
        elif isinstance(image, Image.Image):
            logger.info(f"Original input size (PIL.Image): {image.size}")
        else:
            logger.warning(f"Unknown image type: {type(image)}")

        # 确保输入图像是 PIL.Image
        image = self._ensure_pil_image(image)
        crop_image = self._ensure_pil_image(crop_image)

        # 再次打印转换后的 PIL.Image 的尺寸
        logger.info(f"Converted input image size (PIL.Image): {image.size}")

        # 确保原始图像和裁剪图像都是 'RGB' 模式
        if image.mode != 'RGB':
            logger.info(f"Converting original image from {image.mode} to RGB")
            image = image.convert('RGB')

        if crop_image.mode != 'RGB':
            logger.info(f"Converting cropped image from {crop_image.mode} to RGB")
            crop_image = crop_image.convert('RGB')

        # 提取图像尺寸和边界框
        width, height = image.size  # width 和 height 对应于 W 和 H
        bboxes = data.get("bboxes", [])
        logger.info(f"Original image dimensions: {width}x{height}")
        logger.info(f"Number of bounding boxes: {len(bboxes)}")

        if not bboxes:
            logger.warning("No bounding boxes detected. Returning the original image.")
            return self.process_output(image)  # 返回原始图片

        # 假设置信度最高的边界框是第一个
        bbox = bboxes[0]
        logger.info(f"Selected bounding box: {bbox}")

        # 计算实际裁切的像素坐标
        left = max(0, int(bbox["xmin"] * width))
        top = max(0, int(bbox["ymin"] * height))
        right = min(width, int(bbox["xmax"] * width))
        bottom = min(height, int(bbox["ymax"] * height))

        logger.info(f"Bounding box coordinates: left={left}, top={top}, right={right}, bottom={bottom}")

        if left >= right or top >= bottom:
            logger.warning(f"Invalid bounding box: {left}, {top}, {right}, {bottom}")
            return self.process_output(image)

        # 调整裁剪图像的尺寸以匹配边界框的尺寸
        bbox_width = right - left
        bbox_height = bottom - top
        logger.info(f"BBox width: {bbox_width}, BBox height: {bbox_height}")
        crop_image_resized = crop_image.resize((bbox_width, bbox_height))

        # 确保裁剪的图像与原始图像模式一致
        if crop_image_resized.mode != image.mode:
            logger.info(f"Converting resized cropped image from {crop_image_resized.mode} to {image.mode}")
            crop_image_resized = crop_image_resized.convert(image.mode)

        # 将裁剪图像粘贴回原始图像
        image_paste = image.copy()
        image_paste.paste(crop_image_resized, (left, top))

        logger.info(f"Result image size: {image_paste.size}")
        return self.process_output(image_paste)

    def _ensure_pil_image(self, image):
        """
        Ensure that the input is a PIL.Image. Convert if necessary.
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return self._convert_to_image(image)
        elif torch and isinstance(image, torch.Tensor):
            logger.info(f"Input image tensor shape before squeeze: {image.shape}")
            image = image.squeeze()  # 去掉多余的维度
            logger.info(f"Image shape after squeeze (torch.Tensor): {image.shape}")

            if image.ndim == 3:
                if image.shape[0] in [1, 3, 4]:  # (C, H, W)
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                elif image.shape[2] in [1, 3, 4]:  # (H, W, C)
                    image_np = image.cpu().numpy()
                else:
                    raise ValueError(f"Unsupported tensor shape: {image.shape}")
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")

            return self._convert_to_image(image_np)
        else:
            raise ValueError("输入的图片必须是 PIL.Image、NumPy 或 torch.Tensor 类型。")

    def _convert_to_image(self, array):
        """
        Convert a NumPy array to a PIL.Image.
        """
        # 如果数组是多维，尝试去掉无用的维度
        if array.ndim > 3:
            array = array.squeeze()
        logger.info(f"Image shape after squeeze in _convert_to_image: {array.shape}")

        # 检查数据范围并转换为 [0, 255]
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
            logger.info("Normalized array to uint8 with range [0, 255]")
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)

        # 创建 PIL.Image
        if array.ndim == 2:  # 单通道灰度图
            image = Image.fromarray(array, mode='L')
        elif array.ndim == 3:
            if array.shape[-1] == 1:
                array = array.squeeze(-1)
                image = Image.fromarray(array, mode='L')
            elif array.shape[-1] == 3:
                image = Image.fromarray(array, mode='RGB')
            elif array.shape[-1] == 4:
                array = array[..., :3]
                image = Image.fromarray(array, mode='RGB')
            else:
                raise ValueError(f"Unsupported number of channels: {array.shape[-1]}")
        else:
            raise ValueError(f"无法将输入数据转换为图像，形状: {array.shape}")

        return image

    def process_output(self, image):
        """
        Process the final image to match the output format.
        Converts to a torch.Tensor of shape (1, H, W, 3).
        """
        # 将 PIL.Image 转换为 NumPy 数组，形状为 (H, W, C)
        result_image = np.array(image).astype(np.float32) / 255.0
        logger.info(f"Result image shape: {result_image.shape}")

        # 添加批次维度，形状为 (1, H, W, C)
        result_image = np.expand_dims(result_image, axis=0)
        logger.info(f"Final image shape with batch dimension: {result_image.shape}")

        # 转换为 torch.Tensor，形状为 (1, H, W, C)
        result_tensor = torch.from_numpy(result_image)
        logger.info(f"Final PyTorch tensor shape: {result_tensor.shape}")

        # 返回张量，形状为 (1, H, W, 3)
        return (result_tensor,)
