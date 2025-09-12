import torch
import random

class Image_List_Splitter:
    """
    将输入的图片列表分割成两组：
    - 第一组：指定数量的图片
    - 第二组：剩余的图片
    支持固定顺序或随机选择
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 输入图片列表
                "split_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 99999,
                    "step": 1,
                    "display": "number"
                }),  # 第一组要选取的图片数量
                "enable_random": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用随机",
                    "label_off": "固定顺序"
                }),  # 是否启用随机选择
                "random_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "display": "number"
                }),  # 随机种子
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("selected_images", "remaining_images",)
    FUNCTION = "split_images"
    CATEGORY = "🍒 Kim-Nodes/🏖️图像处理"
    
    # 指定输入和输出是否为列表
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)  # 两个输出都是列表
    
    def split_images(self, images, split_count, enable_random, random_seed):
        """
        分割图片列表
        
        Args:
            images: 输入图片tensor列表
            split_count: 第一组要选取的图片数量（列表）
            enable_random: 是否启用随机选择（列表）
            random_seed: 随机种子（列表）
            
        Returns:
            tuple: (selected_images_list, remaining_images_list)
        """
        # 确保images是列表
        if not isinstance(images, list):
            images = [images]
            
        # 获取参数的第一个值（因为INPUT_IS_LIST=True，所有输入都是列表）
        split_count = split_count[0] if isinstance(split_count, list) else split_count
        enable_random = enable_random[0] if isinstance(enable_random, list) else enable_random
        random_seed = random_seed[0] if isinstance(random_seed, list) else random_seed
        
        # 获取图片总数
        total_images = len(images)
        
        # 确保split_count不超过总图片数
        actual_split_count = min(split_count, total_images)
        
        if enable_random:
            # 使用随机种子
            random.seed(random_seed)
            # 复制图片列表，避免修改原列表
            images_copy = images.copy()
            # 随机打乱
            random.shuffle(images_copy)
            # 选择指定数量
            selected_images = images_copy[:actual_split_count]
            remaining_images = images_copy[actual_split_count:]
        else:
            # 从开头选取固定数量
            selected_images = images[:actual_split_count]
            remaining_images = images[actual_split_count:]
        
        # 如果remaining_images为空，创建一个黑色图片
        if not remaining_images:
            # 使用第一张图片的尺寸创建黑色图片
            sample_image = images[0]
            black_image = torch.zeros_like(sample_image)
            remaining_images = [black_image]
        
        return (selected_images, remaining_images)

