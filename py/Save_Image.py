import os
import glob  # 导入 glob 模块
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo  # 修改导入方式
import string
import random
import folder_paths

# 定义生成随机字符串的函数
def generate_random_string(length=8):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for i in range(length))

# 定义 args 类或根据实际情况调整
class Args:
    disable_metadata = False  # 根据需要设置

args = Args()

class Save_Image:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "file_path": ("STRING", {"multiline": False, "placeholder": "输入图片文件所在的目录路径"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True
    CATEGORY = "🍒 Kim-Nodes/🔍Text_Tools | 文本工具"

    def save_images(self, images, file_path, prompt=None, extra_pnginfo=None):
        filename_prefix = os.path.basename(file_path)
        if file_path == '':
            filename_prefix = "ComfyUI"
        
        filename_prefix, _ = os.path.splitext(filename_prefix)

        _, extension = os.path.splitext(file_path)

        if extension:
            # 如果有扩展名，处理文件路径
            file_path = os.path.dirname(file_path)

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )

        if not os.path.exists(file_path):
            # 创建目录
            os.makedirs(file_path)
            print("目录已创建")
        else:
            print("目录已存在")

        # 获取目录下的所有文件
        if file_path == "":
            files = glob.glob(os.path.join(full_output_folder, '*'))
        else:
            files = glob.glob(os.path.join(file_path, '*'))
        
        # 统计文件数量
        file_count = len(files)
        counter += file_count
        print('统计文件数量:', file_count, '计数器:', counter)

        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))

            file = f"{filename}_{counter:05}_.png"
            
            if file_path == "":
                fp = os.path.join(full_output_folder, file)
                if os.path.exists(fp):
                    file = f"{filename}_{counter:05}_{generate_random_string(8)}.png"
                    fp = os.path.join(full_output_folder, file)
                img.save(fp, pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
            else:
                fp = os.path.join(file_path, file)
                if os.path.exists(fp):
                    file = f"{filename}_{counter:05}_{generate_random_string(8)}.png"
                    fp = os.path.join(file_path, file)

                img.save(fp, pnginfo=metadata, compress_level=self.compress_level)
                results.append({
                    "filename": file,
                    "subfolder": file_path,
                    "type": self.type
                })
            counter += 1

        return ()
