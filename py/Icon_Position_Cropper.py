import torch
import numpy as np
from PIL import Image

class IconPositionCropper:
    """
    根据指定的四个坐标点来裁切图片
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片": ("IMAGE",),
                "位置数据": ("DATA",),
                "起始列号": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
                "终止列号": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
                "行号": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("裁切后图像",)
    FUNCTION = "crop_by_positions"
    CATEGORY = "🍊 Kim-Nodes/🛑Icon Processing | 图标处理"

    def crop_by_positions(self, 图片, 位置数据, 起始列号, 终止列号, 行号):
        # 添加详细的调试信息
        print("\n=== 输入参数 ===")
        print(f"起始列号: {起始列号}")
        print(f"终止列号: {终止列号}")
        print(f"结束行号: {行号}")
        
        # 处理输入图片
        if isinstance(图片, torch.Tensor):
            if 图片.shape[0] != 1:
                raise ValueError("图片只支持 batch_size=1")
            if 图片.shape[1] in (3, 4):
                图片 = 图片.permute(0, 2, 3, 1)
            image_np = (图片[0].cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
        else:
            raise ValueError("图片必须是 torch.Tensor 类型")

        # 找到四个角落的中心点
        左上中心点 = None
        右上中心点 = None
        左下中心点 = None
        右下中心点 = None
        
        # 遍历所有位置找到四个角落
        for pos in 位置数据:
            if pos["列号"] == 起始列号 and pos["重复组号"] == 0:
                center_x = pos["x"] + pos["宽"]/2
                center_y = pos["y"] + pos["高"]/2
                左上中心点 = (center_x, center_y)
            
            if pos["列号"] == 终止列号 and pos["重复组号"] == 0:
                center_x = pos["x"] + pos["宽"]/2
                center_y = pos["y"] + pos["高"]/2
                右上中心点 = (center_x, center_y)
                
            if pos["列号"] == 起始列号 and pos["重复组号"] == 行号:
                center_x = pos["x"] + pos["宽"]/2
                center_y = pos["y"] + pos["高"]/2
                左下中心点 = (center_x, center_y)
                
            if pos["列号"] == 终止列号 and pos["重复组号"] == 行号:
                center_x = pos["x"] + pos["宽"]/2
                center_y = pos["y"] + pos["高"]/2
                右下中心点 = (center_x, center_y)
        
        if not all([左上中心点, 右上中心点, 左下中心点, 右下中心点]):
            raise ValueError(f"未找到所需的四个角落点，请检查参数是否正确。\n"
                          f"需要的范围：第{起始列号}列到第{终止列号}列，第0行到第{行号}行")

        print(f"\n=== 四个角落的中心点位置 ===")
        print(f"左上中心点: {左上中心点}")
        print(f"右上中心点: {右上中心点}")
        print(f"左下中心点: {左下中心点}")
        print(f"右下中心点: {右下中心点}")

        # 计算裁切区域
        left = int(min(左上中心点[0], 左下中心点[0]))
        right = int(max(右上中心点[0], 右下中心点[0]))
        top = int(min(左上中心点[1], 右上中心点[1]))
        bottom = int(max(左下中心点[1], 右下中心点[1]))

        print(f"\n=== 最终裁切区域 ===")
        print(f"左上角: ({left}, {top})")
        print(f"右下角: ({right}, {bottom})")
        print(f"宽度: {right - left}, 高度: {bottom - top}")

        # 裁切图片
        cropped_image = image_pil.crop((left, top, right, bottom))

        # 转换回 tensor
        result = np.array(cropped_image, dtype=np.float32) / 255.0
        if result.shape[-1] == 4:
            result = result[..., :3]  # 去掉 alpha 通道
        result = np.expand_dims(result, axis=0)
        
        return (torch.from_numpy(result),) 