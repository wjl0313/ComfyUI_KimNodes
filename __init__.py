import logging
# 图标类
from .py.Distribute_Icons import Distribute_Icons
from .py.Distribute_icons_in_grid import IconDistributeByGrid
from .py.Seamless_Icon_Generator import SeamlessIconGenerator
from .py.Icon_Position_Cropper import IconPositionCropper
# 裁切贴回
from .py.YOLO_Crop import YOLO_Crop
from .py.YOLO_Multi_Crop import YOLO_Multi_Crop
from .py.Crop_Paste import Crop_Paste
# 裁剪工具
from .py.Edge_Element_Cropper import Edge_Element_Cropper
from .py.Transparent_Area_Cropper import Transparent_Area_Cropper
from .py.Percentage_Cropper import Percentage_Cropper
from .py.BoundingBox_Cropper import BoundingBox_Cropper
# 滤镜
from .py.KimFilter import KimFilter
from .py.KimHDR import KimHDR
from .py.Whitening import Whitening_Node
# 文本工具
from .py.Prompt_Text import Prompt_Text
from .py.Text_Match import Text_Match
from .py.Text_Processor import Text_Processor
from .py.Image_Classification import Image_Classification
from .py.Save_Image import Save_Image
# 元数据相关
from .py.Kim_image_metadata import Add_ImageMetadata
from .py.LoadImageWithMetadata import LoadImage_Metadata
from .py.manual_metadata_input import Manual_MetadataInput
# 数据匹配
from .py.YOLOWorld_Match import YOLOWorld_Match
# 图像处理
from .py.Image_Resize import Image_Resize
# 蒙板处理
from .py.Split_Mask import Split_Mask
# from .py.Lora_Difference_extraction import ExtractDifferenceLora

# 插件的节点类映射
NODE_CLASS_MAPPINGS = {
    # 图标类
    "Distribute_Icons": Distribute_Icons,
    "IconDistributeByGrid": IconDistributeByGrid,
    "Seamless_Icon_Generator": SeamlessIconGenerator,
    "Icon_Position_Cropper": IconPositionCropper,
    # 裁切贴回
    "YOLO_Crop": YOLO_Crop,
    "YOLO_Multi_Crop": YOLO_Multi_Crop,
    "Crop_Paste": Crop_Paste,
    # 裁剪工具
    "Edge_Element_Cropper": Edge_Element_Cropper,
    "Transparent_Area_Cropper": Transparent_Area_Cropper,
    "Percentage_Cropper": Percentage_Cropper,
    "BoundingBox_Cropper": BoundingBox_Cropper,
    # 滤镜
    "KimFilter": KimFilter,
    "KimHDR": KimHDR,
    "Whitening_Node": Whitening_Node,
    # 文本工具
    "Prompt_Text": Prompt_Text,
    "Text_Match": Text_Match,
    "Text_Processor": Text_Processor,
    "Image_Classification": Image_Classification,
    "Save_Image": Save_Image,
    # 元数据相关
    "Add_ImageMetadata": Add_ImageMetadata,
    "LoadImage_Metadata": LoadImage_Metadata,
    "Manual_MetadataInput": Manual_MetadataInput,
    # 数据匹配
    "YOLOWorld_Match": YOLOWorld_Match,
    # 图像处理
    "Image_Resize": Image_Resize,
    # 蒙板处理
    "Split_Mask": Split_Mask,
    # "ExtractDifferenceLora": ExtractDifferenceLora
}

# 节点的显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
#图标类
    "Distribute_Icons": "🍒istribute_Icons",
    "IconDistributeByGrid": "🍒IconDistributeByGrid",
    "Seamless_Icon_Generator": "🍒Seamless_Icon_Generator",  
    "Icon_Position_Cropper": "🍒Icon_Position_Cropper",
#裁切贴回
    "YOLO_Crop": "🍒YOLO_Crop / YOLO裁切",
    "YOLO_Multi_Crop": "🍒YOLO_Multi_Crop / 多人物裁切",
    "Crop_Paste": "🍒Crop_Paste / 裁切粘贴",
#裁剪工具
    "Edge_Element_Cropper": "🍒边缘元素裁剪器",
    "Transparent_Area_Cropper": "🍒透明区域裁剪器",
    "Percentage_Cropper": "🍒百分比裁剪器",
    "BoundingBox_Cropper": "🍒边界框裁剪器",
#滤镜
    "KimFilter": "🍒Filter🎨滤镜",
    "KimHDR": "🍒HDR🌈",
    "Whitening_Node": "🍒Whitening_👧🏻牛奶肌",
#文本工具
    "Prompt_Text": "🍒Prompt_Text / 文本输出",
    "Text_Match": "🍒Text_Match / 文本匹配",
    "Text_Processor": "🍒Text_Processor / 文本数字提取",
    "Image_Classification": "🍒Image_Classification / 图像分类",
    "Save_Image": "🍒Save_Image / 判断路径保存",

    "Add_ImageMetadata": "🍒Add_ImageMetadata / 合并保存图像元数据",
    "LoadImage_Metadata": "🍒LoadImage_Metadata / 加载workflow图片",
    "Manual_MetadataInput": "🍒Manual_MetadataInput / 填写元数据",
#数据匹配
    "YOLOWorld_Match": "🍒YOLOWorld_Match🔍特征匹配",
#图像处理
    "Image_Resize": "🍒Image_Resize📐图像尺寸缩放",
#蒙板处理
    "Split_Mask": "🍒Split_Mask🔍蒙版元素分割器",
}

# 插件初始化
def setup_plugin():
    print("设置插件环境...")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("comfyui_plugin.log"),
            logging.StreamHandler()
        ]
    )
    # 这里可以添加更多的设置代码，例如初始化资源等

# 调用 setup_plugin
setup_plugin()