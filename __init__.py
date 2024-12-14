import logging
from .py.Distribute_Icons import Distribute_Icons
from .py.Distribute_icons_in_grid import IconDistributeByGrid
from .py.YOLO_Crop import YOLO_Crop
from .py.Crop_Paste import Crop_Paste
from .py.KimFilter import KimFilter
from .py.Prompt_Text import Prompt_Text
from .py.Text_Match import Text_Match
from .py.Image_Classification import Image_Classification
from .py.Save_Image import Save_Image
from .py.KimHDR import KimHDR
from .py.Kim_image_metadata import Add_ImageMetadata
from .py.LoadImageWithMetadata import LoadImage_Metadata
from .py.manual_metadata_input import Manual_MetadataInput
from .py.YOLOWorld_Match import YOLOWorld_Match

# 插件的节点类映射
NODE_CLASS_MAPPINGS = {
    "Distribute_Icons": Distribute_Icons,
    "IconDistributeByGrid": IconDistributeByGrid,
    "YOLO_Crop": YOLO_Crop,
    "Crop_Paste": Crop_Paste,
    "KimFilter": KimFilter,
    "Text_Match": Text_Match,
    "Prompt_Text": Prompt_Text,
    "Image_Classification": Image_Classification,
    "Save_Image": Save_Image,
    "KimHDR": KimHDR,
    "Add_ImageMetadata": Add_ImageMetadata,
    "LoadImage_Metadata": LoadImage_Metadata,
    "Manual_MetadataInput": Manual_MetadataInput,
    "YOLOWorld_Match": YOLOWorld_Match,
}

# 节点的显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "Distribute_Icons": "Kim_🍊istribute_Icons🛑",
    "IconDistributeByGrid": "Kim_🍊IconDistributeByGrid🛑",
    "YOLO_Crop": "Kim_🍊YOLO_Crop✂YOLO裁切",
    "Crop_Paste": "Kim_🍊Crop_Paste✂裁切粘贴",
    "KimFilter": "Kim_🍊Filter🎨滤镜",
    "Prompt_Text": "Kim_🍊Prompt_Text",
    "Text_Match": "Kim_🍊Text_Match🔍文本匹配",
    "Image_Classification": "Kim_🍊Image_Classification🔍图像分类",
    "Save_Image": "Kim_🍊Save_Image🔍判断路径保存",
    "Manual_MetadataInput": "Kim_🍊Manual_MetadataInput📝填写元数据",
    "KimHDR": "Kim_🍊KimHDR",
    "Add_ImageMetadata": "Kim_🍊Add_ImageMetadata📝合并保存图像元数据",
    "LoadImage_Metadata": "Kim_🍊LoadImage_Metadata📝加载workflow图片",
    "YOLOWorld_Match": "Kim_🍊YOLOWorld_Match🔍特征匹配"
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