import torch
import numpy as np

class Transparent_Image_Filter:
    """
    过滤图像列表，将完全透明的图像过滤掉
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("过滤后有效图像",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "filter_images"
    CATEGORY = "🍒 Kim-Nodes/✔️Selector |选择器"

    def check_image(self, img_tensor):
        """检查图像是否完全透明"""
        # 将张量转换为NumPy数组
        img_np = img_tensor.cpu().numpy()
        
        # 获取图像尺寸
        if len(img_np.shape) == 4:
            img_np = img_np[0]  # 处理批处理图像
        
        # 确保图像是3D张量 [height, width, channels]
        if len(img_np.shape) != 3:
            raise ValueError(f"图像张量形状不正确: {img_np.shape}")
        
        # 检查是否为RGBA图像
        if img_np.shape[2] == 4:
            # 只检查alpha通道
            alpha = img_np[:, :, 3]
            # 如果所有像素的alpha值都接近0，则图像完全透明
            return not np.all(alpha < 0.01)
        else:
            # 如果不是RGBA图像，则认为是有效图像
            return True

    def filter_images(self, images):
        """将完全透明的图片过滤掉，只返回有效图像"""
        filtered_images = []
        filtered_count = 0
        total_count = 0
        
        # 处理单个图像输入
        if not isinstance(images, list):
            images = [images]
            
        # 处理空输入
        if len(images) == 0:
            print("警告：没有输入图像")
            return ([],)
        
        # 遍历所有图像
        for img in images:
            total_count += 1
            try:
                if self.check_image(img):
                    filtered_images.append(img)
                    print(f"图像 #{total_count} 有效，保留")
                else:
                    filtered_count += 1
                    print(f"图像 #{total_count} 被过滤，完全透明")
            except Exception as e:
                print(f"处理图像 #{total_count} 时出错: {e}")
                continue
        
        print(f"过滤前图片数量: {total_count}")
        print(f"被过滤掉的透明图片数量: {filtered_count}")
        print(f"剩余有效图片数量: {len(filtered_images)}")
        
        return (filtered_images,) 