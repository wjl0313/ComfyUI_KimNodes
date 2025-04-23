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
                "crop_images": ("IMAGE",),    # 裁剪的图像列表
                "data": ("DATA",),              # 包含边界框信息的字典
                "feather_amount": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01
                }),  # 边缘渐变程度控制
            },
        }

    RETURN_TYPES = ("IMAGE",)  # 返回修改后的图片
    FUNCTION = "crop_paste"
    CATEGORY = "🍊 Kim-Nodes/✂ Crop | 裁剪处理"

    def __init__(self):
        pass

    def crop_paste(self, data, image, crop_images, feather_amount):
        """
        将多个裁剪的图像粘贴回原始图像的对应位置
        """
        print("\n===== Crop_Paste 节点输入信息 =====")
        print(f"输入图像类型: {type(image)}")
        
        # 打印裁剪图像信息
        print(f"裁剪图像类型: {type(crop_images)}")
        if isinstance(crop_images, torch.Tensor):
            print(f"裁剪图像形状: {crop_images.shape}")
            if len(crop_images.shape) == 4:
                print(f"裁剪图像批次数量: {crop_images.shape[0]}")
        
        # 处理原始图像
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 4:
                image = image[0]
            image_np = image.cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            print(f"原始图像尺寸: {image_pil.size}")
        
        # 获取边界框
        bboxes = data.get("bboxes", [])
        print(f"边界框数量: {len(bboxes)}")
        
        # 创建一个新的图像用于粘贴
        image_paste = image_pil.copy()
        width, height = image_pil.size
        
        if image_paste.mode != 'RGB':
            image_paste = image_paste.convert('RGB')
        
        # 处理裁剪图像
        print("\n处理裁剪图像...")
        if isinstance(crop_images, torch.Tensor) and len(crop_images.shape) == 4:
            batch_size = crop_images.shape[0]
            print(f"检测到批次张量，包含 {batch_size} 个图像")
            
            # 遍历所有裁剪图像和对应的边界框
            for i in range(min(batch_size, len(bboxes))):
                print(f"\n处理第 {i+1}/{batch_size} 个裁剪图像")
                
                try:
                    # 安全获取图像
                    crop_img = crop_images[i]
                    if crop_img.shape[0] == 1 and crop_img.shape[1] == 1:
                        print(f"警告: 图像 {i} 尺寸过小 ({crop_img.shape})，跳过")
                        continue
                    
                    # 打印形状和数值范围
                    print(f"裁剪图像形状: {crop_img.shape}")
                    
                    # 裁剪图像预处理
                    crop_np = crop_img.cpu().numpy()
                    
                    # 规范化数值范围
                    if crop_np.max() > 1.0 or crop_np.min() < 0:
                        print(f"警告: 图像 {i} 数值范围异常 ({crop_np.min()} 到 {crop_np.max()})，进行规范化")
                        crop_np = np.clip(crop_np, 0, 1)
                    
                    # 转换为8位格式
                    crop_np = (crop_np * 255).astype(np.uint8)
                    
                    # 计算边界框坐标
                    bbox = bboxes[i]
                    left = max(0, int(bbox["xmin"] * width))
                    top = max(0, int(bbox["ymin"] * height))
                    right = min(width, int(bbox["xmax"] * width))
                    bottom = min(height, int(bbox["ymax"] * height))
                    
                    print(f"边界框坐标: ({left}, {top}) -> ({right}, {bottom})")
                    
                    if left >= right or top >= bottom:
                        print(f"警告: 无效的边界框坐标，跳过")
                        continue
                    
                    # 创建PIL图像并调整尺寸
                    crop_pil = Image.fromarray(crop_np)
                    bbox_width = right - left
                    bbox_height = bottom - top
                    crop_resized = crop_pil.resize((bbox_width, bbox_height))
                    
                    # 粘贴图像
                    image_paste.paste(crop_resized, (left, top))
                    print(f"图像 {i} 粘贴成功")
                    
                except Exception as e:
                    print(f"处理图像 {i} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
        
        print("\n所有裁剪图像处理完成")
        
        # 转换为张量并返回
        result_np = np.array(image_paste).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)
        
        return (result_tensor,)

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
