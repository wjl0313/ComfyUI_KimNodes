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
from .py.Pixelate_Filter import Pixelate_Filter
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
from .py.LoRA_Metadata_Reader import LoRA_Metadata_Reader
# 数据匹配
from .py.YOLOWorld_Match import YOLOWorld_Match
# 图像处理
from .py.Image_Resize import Image_Resize
# 蒙板处理
from .py.Split_Mask import Split_Mask
from .py.MaskWhiteAreaRatio import Mask_White_Area_Ratio
from .py.Mask_Noise_Cleaner import Mask_Noise_Cleaner
# from .py.Lora_Difference_extraction import ExtractDifferenceLora
from .py.Max_Length_Image_List_Selector import MaxLength_ImageListSelector
from .py.Transparent_Image_Filter import Transparent_Image_Filter
from .py.Image_Pixel_Filter import Image_PixelFilter

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
    "Pixelate_Filter": Pixelate_Filter,
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
    "LoRA_Metadata_Reader": LoRA_Metadata_Reader,
    # 数据匹配
    "YOLOWorld_Match": YOLOWorld_Match,
    # 图像处理
    "Image_Resize": Image_Resize,
    # 蒙板处理
    "Split_Mask": Split_Mask,
    "Mask_White_Area_Ratio": Mask_White_Area_Ratio,
    "Mask_Noise_Cleaner": Mask_Noise_Cleaner,
    # "ExtractDifferenceLora": ExtractDifferenceLora,
    "MaxLength_ImageListSelector": MaxLength_ImageListSelector,
    "Transparent_Image_Filter": Transparent_Image_Filter,
    "Image_PixelFilter": Image_PixelFilter,
}

# 节点的显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
#图标类
    "Distribute_Icons": "🍒Distribute_Icons / 分发图标",
    "IconDistributeByGrid": "🍒IconDistributeByGrid / 区域分发图标",
    "Seamless_Icon_Generator": "🍒Seamless_Icon_Generator / 无缝图标生成",  
    "Icon_Position_Cropper": "🍒Icon_Position_Cropper / 图标位置裁剪",
#裁剪工具
    "YOLO_Crop": "🍒YOLO_Crop✀YOLO裁切",
    "YOLO_Multi_Crop": "🍒YOLO_Multi_Crop✀多人物裁切",
    "Crop_Paste": "🍒Crop_Paste✀裁切粘贴",
    "Edge_Element_Cropper": "🍒Edge_Element_Cropper✀边缘元素裁剪",
    "Transparent_Area_Cropper": "🍒Transparent_Area_Cropper✀透明区域裁剪",
    "Percentage_Cropper": "🍒Percentage_Cropper✀百分比裁剪",
    "BoundingBox_Cropper": "🍒BoundingBox_Cropper✀边界框裁剪",
#滤镜
    "KimFilter": "🍒Filter🎨滤镜",
    "KimHDR": "🍒HDR🌈",
    "Whitening_Node": "🍒Whitening_👧🏻牛奶肌",
    "Pixelate_Filter": "🍒Pixelate_Filter🎮像素化滤镜",
#文本工具
    "Prompt_Text": "🍒Prompt_Text / 文本输出",
    "Text_Match": "🍒Text_Match / 文本匹配",
    "Text_Processor": "🍒Text_Processor / 文本数字提取",
    "Image_Classification": "🍒Image_Classification / 图像分类",
    "Save_Image": "🍒Save_Image / 判断路径保存",
#数据处理
    "Add_ImageMetadata": "🍒Add_ImageMetadata / 合并保存图像元数据",
    "LoadImage_Metadata": "🍒LoadImage_Metadata / 加载workflow图片",
    "Manual_MetadataInput": "🍒Manual_MetadataInput / 填写元数据",
    "LoRA_Metadata_Reader": "🍒LoRA_Metadata_Reader📋LoRA元数据读取器",
#特征匹配
    "YOLOWorld_Match": "🍒YOLOWorld_Match🔍特征匹配",
#图像处理
    "Image_Resize": "🍒Image_Resize📐图像尺寸缩放",
#蒙板处理
    "Split_Mask": "🍒Split_Mask🔪蒙版元素分割",
    "Mask_White_Area_Ratio": "🍒Mask_White_Area_Ratio📊蒙版白色区域占比",
    "Mask_Noise_Cleaner": "🍒Mask_Noise_Cleaner🧹蒙版噪点清理器",
#选择器
    "MaxLength_ImageListSelector": "🍒MaxLength_ImageListSelector✔️最长图片列表选择",
    "Transparent_Image_Filter": "🍒Transparent_ImageFilter✔️无色图像过滤",
    "Image_PixelFilter": "🍒Image_PixelFilter✔️图像像素过滤",
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