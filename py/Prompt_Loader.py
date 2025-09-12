import os
import re
import logging
import hashlib

# 使用字典存储每个节点实例的计数器
counter_map = {}

def get_counter_key(prompt_dir, unique_id):
    # 使用目录路径和unique_id的组合作为键
    return f"{prompt_dir}_{unique_id}"

def natural_sort_key(s):
    # 将字符串中的数字部分转换为整数进行比较
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class Prompt_Loader:
    def __init__(self):
        self.current_seed = 0

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt_dir": ("STRING", {"multiline": False, "placeholder": "输入提示文件所在的目录路径"}),
                    "sequential_mode": ("BOOLEAN", {"default": False, "label_on": "开启顺序选择", "label_off": "关闭顺序选择"}),
                    "reset_to_first": ("BOOLEAN", {"default": False, "label_on": "重置到起始位置", "label_off": "继续当前位置"}),
                    "start_index": ("INT", {"default": 0, "min": -1, "step": 1, "max": 0xffffffffffffffff, "tooltip": "设置从第几个提示词开始加载:\n-1: 从最后一个开始\n0或更大: 从指定序号开始"}),
                    "load_cap": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "设置每次加载的提示词数量:\n0: 加载全部\n1或更大: 加载指定数量"}),
                    "seed": ("INT", {"default": 666, "min": 0, "max": 0xffffffffffffffff}),
                    },
                "optional": {
                    "reload": ("BOOLEAN", { "default": False, "label_on": "if file changed", "label_off": "if value changed"}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("STRING", "STRING", "INT",)
    RETURN_NAMES = ("prompt_list", "file_names", "current_index",)
    OUTPUT_IS_LIST = (True, True, False,)

    FUNCTION = "doit"

    CATEGORY = "🍒 Kim-Nodes/🔍Text_Tools | 文本工具"

    @staticmethod
    def IS_CHANGED(prompt_dir, sequential_mode=False, reset_to_first=False, start_index=-1, load_cap=0, seed=0, reload=False, unique_id=None):
        if reload:
            if not os.path.exists(prompt_dir):
                return float('NaN')

            prompt_files = []
            for root, dirs, files in os.walk(prompt_dir):
                for file in files:
                    if file.endswith(".txt"):
                        prompt_files.append(os.path.join(root, file))

            # 使用自然排序
            prompt_files.sort(key=natural_sort_key)

            md5 = hashlib.md5()
            for file_name in prompt_files:
                md5.update(file_name.encode('utf-8'))
                with open(file_name, 'rb') as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        md5.update(chunk)

            return md5.hexdigest()

        return prompt_dir

    def doit(self, prompt_dir, sequential_mode=False, reset_to_first=False, start_index=-1, load_cap=0, seed=0, reload=False, unique_id=None):
        if not os.path.exists(prompt_dir):
            logging.warning(f"[KimNodes] Prompt_Loader: 目录不存在 '{prompt_dir}'")
            return ([], [], 0)

        prompt_files = []
        for root, dirs, files in os.walk(prompt_dir):
            for file in files:
                if file.endswith(".txt"):
                    prompt_files.append(os.path.join(root, file))

        if not prompt_files:
            return ([], [], 0)

        # 使用自然排序
        prompt_files.sort(key=natural_sort_key)

        # 处理start_index为负数的情况
        if start_index < 0:
            start_index = len(prompt_files) + start_index

        # 确保start_index在有效范围内
        start_index = max(0, min(start_index, len(prompt_files) - 1))

        prompts = []
        file_names = []
        
        # 如果是顺序模式，只处理一个文件
        if sequential_mode and unique_id is not None:
            counter_key = get_counter_key(prompt_dir, unique_id)
            
            # 如果重置开关打开，从start_index开始
            if reset_to_first:
                counter_map[counter_key] = start_index
            # 否则使用seed来控制索引
            elif seed != self.current_seed:
                if counter_key not in counter_map:
                    counter_map[counter_key] = start_index
                else:
                    counter_map[counter_key] = (counter_map[counter_key] + 1) % len(prompt_files)
                self.current_seed = seed
            
            current_index = counter_map[counter_key]
            prompt_files = [prompt_files[current_index]]
            
            # 记录日志
            logging.info(f"[KimNodes] Prompt_Loader: 节点 {unique_id}, 目录 {prompt_dir}, 当前索引 {current_index}, 文件 {os.path.basename(prompt_files[0])}, seed {seed}, 重置 {reset_to_first}, 起始位置 {start_index}")
        else:
            current_index = start_index
            
        for file_name in prompt_files:
            try:
                with open(file_name, "r", encoding="utf-8") as file:
                    text = file.read().strip()
                    if text:  # 只有当文本不为空时才添加
                        prompts.append(text)
                        # 获取不带路径和扩展名的文件名
                        base_name = os.path.splitext(os.path.basename(file_name))[0]
                        file_names.append(base_name)
            except Exception as e:
                logging.error(f"[KimNodes] Prompt_Loader: 处理文件时发生错误 '{file_name}': {str(e)}\n注意：仅支持UTF-8编码的文件。")

        # 如果不是顺序模式，才应用load_cap
        if not sequential_mode:
            prompts = prompts[start_index:]
            file_names = file_names[start_index:]
            if load_cap > 0:
                prompts = prompts[:load_cap]
                file_names = file_names[:load_cap]

        return (prompts, file_names, current_index)
