class Text_Match:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_match": ("STRING", {"forceInput":True}),
                "text_tag": ("STRING", {"forceInput":True}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN",)
    RETURN_NAMES = ("文本", "布尔值",)
    FUNCTION = "execute"
    CATEGORY = "🍊 Kim-Nodes/🔍Text_Match | 文本匹配"

    def execute(self, text_match, text_tag):
        matched = text_match in text_tag
        return (str(matched), matched,)