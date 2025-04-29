class Image_Classification:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_tag": ("STRING", {"forceInput": True}),
                "true_path": ("STRING", {"forceInput": True, "default": "/path/to/true_output"}),
                "false_path": ("STRING", {"forceInput": True, "default": "/path/to/false_output"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "execute"
    CATEGORY = "🍒 Kim-Nodes/🔍Text_Tools | 文本工具"

    def execute(self, text_tag, true_path, false_path):
        """
        根据text_tag的值，输出对应的路径字符串。
        如果text_tag为"true"，输出true_path；否则，输出false_path。
        
        Returns:
            tuple: 包含一个字符串路径，根据text_tag的值选择true_path或false_path。
        """
        if text_tag.lower() == "true":
            return (true_path,)
        else:
            return (false_path,)
