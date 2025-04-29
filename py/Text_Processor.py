import torch

class Text_Processor:
    """
    文本处理器：接受文本输入，处理后输出字符串和整数。
    可以用于提取文本中的数值或处理文本内容。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入文本": ("STRING", {"multiline": True, "default": "请输入文本内容"}),
                "提取数字": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("文本", "数字提取",)
    FUNCTION = "process_text"
    CATEGORY = "🍒 Kim-Nodes/🔍Text_Tools | 文本工具"

    def process_text(self, 输入文本, 提取数字=True):
        # 处理文本内容，可以根据需要修改处理逻辑
        processed_text = 输入文本.strip()
        
        # 提取文本中的数字（如果有）
        extracted_number = 0
        if 提取数字:
            # 查找文本中的数字
            import re
            numbers = re.findall(r'\d+', 输入文本)
            if numbers:
                # 取第一个找到的数字
                extracted_number = int(numbers[0])
        
        return (processed_text, extracted_number) 