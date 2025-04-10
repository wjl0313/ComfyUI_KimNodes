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
        print(f"输入图片批次大小: {图片.shape[0]}")
        
        # 调试输出位置数据格式
        if 位置数据 and len(位置数据) > 0:
            print("\n=== 位置数据结构 ===")
            print(f"位置数据类型: {type(位置数据)}")
            print(f"位置数据长度: {len(位置数据)}")
            print(f"第一个元素键: {list(位置数据[0].keys()) if isinstance(位置数据[0], dict) else '不是字典'}")
            
            # 检查是否有嵌套结构
            if '图标位置' in 位置数据[0]:
                print(f"找到图标位置字段，包含 {len(位置数据[0]['图标位置'])} 个位置项")
                if len(位置数据[0]['图标位置']) > 0:
                    实际位置数据 = 位置数据[0]['图标位置']
                    print(f"图标位置项示例: {实际位置数据[0]}")
                    print(f"图标位置项键: {list(实际位置数据[0].keys()) if isinstance(实际位置数据[0], dict) else '不是字典'}")
            else:
                print("未找到图标位置字段")
                实际位置数据 = 位置数据
        else:
            print("警告: 位置数据为空")
            raise ValueError("位置数据为空，无法进行裁切")
        
        # 处理输入图片
        if not isinstance(图片, torch.Tensor):
            raise ValueError("图片必须是 torch.Tensor 类型")

        # 找到四个角落的中心点
        左上中心点 = None
        右上中心点 = None
        左下中心点 = None
        右下中心点 = None
        
        # 如果数据是嵌套的，使用图标位置数据
        if '图标位置' in 位置数据[0]:
            实际位置数据 = 位置数据[0]['图标位置']
        else:
            实际位置数据 = 位置数据
            
        print("\n=== 查找四个角落点 ===")
        print(f"使用数据长度: {len(实际位置数据)}")
        print(f"查找范围: 第{起始列号}列到第{终止列号}列，第0行到第{行号}行")
        
        # 遍历所有位置找到四个角落
        for pos in 实际位置数据:
            try:
                if pos["列号"] == 起始列号 and pos["重复组号"] == 0:
                    center_x = pos["x"] + pos["宽"]/2
                    center_y = pos["y"] + pos["高"]/2
                    左上中心点 = (center_x, center_y)
                    print(f"找到左上中心点: {左上中心点}")
                
                if pos["列号"] == 终止列号 and pos["重复组号"] == 0:
                    center_x = pos["x"] + pos["宽"]/2
                    center_y = pos["y"] + pos["高"]/2
                    右上中心点 = (center_x, center_y)
                    print(f"找到右上中心点: {右上中心点}")
                    
                if pos["列号"] == 起始列号 and pos["重复组号"] == 行号:
                    center_x = pos["x"] + pos["宽"]/2
                    center_y = pos["y"] + pos["高"]/2
                    左下中心点 = (center_x, center_y)
                    print(f"找到左下中心点: {左下中心点}")
                    
                if pos["列号"] == 终止列号 and pos["重复组号"] == 行号:
                    center_x = pos["x"] + pos["宽"]/2
                    center_y = pos["y"] + pos["高"]/2
                    右下中心点 = (center_x, center_y)
                    print(f"找到右下中心点: {右下中心点}")
            except KeyError as e:
                print(f"警告：处理位置时出错 - 键{e}不存在")
                continue
        
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

        # 批量处理所有图片
        结果列表 = []
        批次大小 = 图片.shape[0]
        print(f"开始批量处理 {批次大小} 张图片")
        
        for i in range(批次大小):
            处理图片 = 图片[i:i+1]  # 获取当前图片
            
            if 处理图片.shape[1] in (3, 4):
                处理图片 = 处理图片.permute(0, 2, 3, 1)
            
            image_np = (处理图片[0].cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            
            # 裁切图片
            cropped_image = image_pil.crop((left, top, right, bottom))
            
            # 转换回 tensor
            result = np.array(cropped_image, dtype=np.float32) / 255.0
            if result.shape[-1] == 4:
                result = result[..., :3]  # 去掉 alpha 通道
            
            # 调整维度顺序以符合 IMAGE 格式 [H, W, C]
            结果列表.append(torch.from_numpy(result))
        
        print(f"已完成 {len(结果列表)} 张图片的裁切处理")
        # 将列表转换成批次张量 [B, H, W, C]
        return (torch.stack(结果列表),) 