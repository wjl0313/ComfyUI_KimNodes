import torch


class Mask_Add_Switch:
    """
    蒙版合并节点，支持不同格式的mask合并
    - 白底模式(True)：合并白底黑色mask的黑色区域
    - 黑底模式(False)：合并黑底白色mask的白色区域
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks_a": ("MASK",),
                "masks_b": ("MASK",),
                "invert_switch": ("BOOLEAN", {"default": False, "label_on": "白底", "label_off": "黑底"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "add_masks_with_switch"
    CATEGORY = "🍒 Kim-Nodes/🔲Mask_Tools | 蒙板工具"
    
    def add_masks_with_switch(self, masks_a, masks_b, invert_switch):
        """
        合并两个蒙版，并根据开关决定输出格式
        
        Args:
            masks_a: 第一个蒙版（白底黑色mask）
            masks_b: 第二个蒙版（白底黑色mask）  
            invert_switch: 格式开关，True时输出白底黑色mask，False时输出黑底白色mask
        
        Returns:
            处理后的蒙版
        """
        # 确保输入mask是正确的格式（压缩到2D）
        # ComfyUI的MASK类型应该是2D张量
        if masks_a.ndim > 2:
            masks_a = masks_a.squeeze()
        if masks_b.ndim > 2:
            masks_b = masks_b.squeeze()
            
        # 合并白底黑色mask的黑色区域（使用最小值）
        # 白色=1.0，黑色=0.0，min操作可以合并黑色区域
        merged_masks = torch.min(masks_a, masks_b)
        
        # 根据开关决定输出格式
        if invert_switch:
            # True时：输出白底黑色mask（保持原样）
            result_masks = merged_masks
        else:
            # False时：输出黑底白色mask（反转结果）
            result_masks = 1.0 - merged_masks
            
        # 确保值在[0,1]范围内
        result_masks = torch.clamp(result_masks, 0, 1)
        
        # 确保输出是2D张量（符合ComfyUI MASK类型的标准）
        if result_masks.ndim > 2:
            result_masks = result_masks.squeeze()
        
        return (result_masks,)
