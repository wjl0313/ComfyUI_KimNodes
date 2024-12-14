import re
import os, sys
import folder_paths
import importlib.util
import comfy.utils

global _available
_available = True

class Prompt_Text:
    global _available
    available = _available

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "execute"
    CATEGORY = "🍊 Kim-Nodes/🔍Text_Match | 文本匹配"
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def execute(self, text):
        # 简单的文本输入输出节点，将输入原样输出
        # 如果需要可以在这里对文本进行简单处理
        pbar = comfy.utils.ProgressBar(len(text))
        prompt_result = []
        for t in text:
            prompt_result.append(t)
            pbar.update(1)

        return {
            "ui": {
                "prompt": prompt_result
            },
            "result": (prompt_result,)
        }
