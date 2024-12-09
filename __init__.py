import logging
from .Distribute_Icons import Distribute_Icons
from .Distribute_icons_in_grid import IconDistributeByGrid
from .YOLO_Crop import YOLO_Crop
from .Crop_Paste import Crop_Paste
from .KimFilter import KimFilter
from .Text_Match import Text_Match

# 插件的节点类映射
NODE_CLASS_MAPPINGS = {
    "Distribute_Icons": Distribute_Icons,
    "IconDistributeByGrid": IconDistributeByGrid,
    "YOLO_Crop": YOLO_Crop,
    "Crop_Paste": Crop_Paste,
    "KimFilter": KimFilter,
    "Text_Match": Text_Match
}

# 节点的显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "Distribute_Icons": "Kim_🍊istribute_Icons 🛑",
    "IconDistributeByGrid": "Kim_🍊IconDistributeByGrid 🛑",
    "YOLO_Crop": "Kim_🍊YOLO_Crop ✂",
    "Crop_Paste": "Kim_🍊Crop_Paste ✂",
    "KimFilter": "Kim_🍊Filter🎨",
    "Text_Match": "Kim_🍊Text_Match🔍"
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