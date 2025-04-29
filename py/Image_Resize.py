from PIL import Image
import numpy as np
import torch

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Image_Resize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "supersample": (["true", "false"],),
                "resampling": (["lanczos", "nearest", "bilinear", "bicubic"],),
                "target_size": ("INT", {"default": 1024, "min": 256, "max": 15360, "step": 4}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_resize"
    CATEGORY = "🍒 Kim-Nodes/🏖️图像处理"

    def image_resize(self, image, supersample, resampling, target_size):
        # 直接处理单张图片，避免循环
        img = image[0] if len(image.shape) == 4 else image
        
        # 使用内存优化的方式进行转换
        with torch.no_grad():  # 减少内存使用
            pil_image = tensor2pil(img)
            resized_image = self.apply_resize_image(pil_image, supersample, target_size, resampling)
            result = pil2tensor(resized_image)
        
        return (result, )

    def apply_resize_image(self, image: Image.Image, supersample='true', target_size: int = 1024, resample='bicubic'):
        # 预先计算尺寸，避免重复计算
        current_width, current_height = image.size
        
        # 使用更高效的尺寸计算方法
        if current_width > current_height:
            new_height = target_size + (-target_size % 8)  # 更快的 8 的倍数计算
            scale = new_height / current_height
            new_width = int(current_width * scale + 7) & -8  # 位运算获取 8 的倍数
        else:
            new_width = target_size + (-target_size % 8)
            scale = new_width / current_width
            new_height = int(current_height * scale + 7) & -8

        # 缓存重采样过滤器
        resample_filters = {
            'nearest': Image.Resampling.NEAREST,  # 直接使用 PIL 常量
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'lanczos': Image.Resampling.LANCZOS
        }
        
        current_filter = resample_filters[resample]
        
        # 如果需要超采样，先放大
        if supersample == 'true':
            image = image.resize((new_width * 8, new_height * 8), resample=current_filter)
        
        # 最终缩放
        resized_image = image.resize((new_width, new_height), resample=current_filter)
        
        return resized_image
