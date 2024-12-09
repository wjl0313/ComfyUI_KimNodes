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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "execute"
    CATEGORY = "🍊 Kim-Nodes"

    def execute(self, text_match, text_tag):
        matched = text_match in text_tag
        return (str(matched),)