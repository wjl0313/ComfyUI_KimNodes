"""
无缝拼图模板基类
定义了模板系统的基础接口，所有具体模板都应该继承这个基类
"""

from abc import ABC, abstractmethod
from PIL import Image
import torch
import numpy as np


class TilingTemplateBase(ABC):
    """无缝拼图模板基类"""
    
    def __init__(self):
        self.template_name = "Base Template"
        self.template_description = "基础模板，需要被具体模板继承"
    
    @abstractmethod
    def get_template_info(self):
        """返回模板信息
        Returns:
            dict: 包含模板名称和描述的字典
        """
        return {
            "name": self.template_name,
            "description": self.template_description
        }
    
    @abstractmethod
    def generate_tiling(self, images, canvas_size, params):
        """生成无缝拼图
        Args:
            images (list): 输入图片列表
            canvas_size (tuple): 画布尺寸 (width, height)
            params (dict): 模板参数
            
        Returns:
            tuple: (canvas, mask_canvas, positions) 
                   canvas: PIL Image对象
                   mask_canvas: PIL Image对象  
                   positions: list of dict，包含每个图片的位置信息
        """
        pass
    
    def validate_params(self, params):
        """验证参数有效性
        Args:
            params (dict): 模板参数
            
        Returns:
            bool: 参数是否有效
        """
        return True
    
    def resize_image_keep_ratio(self, image, target_size, force_size=False, align='center'):
        """保持宽高比缩放图片
        Args:
            image: 输入图片
            target_size: 目标尺寸 (width, height)
            force_size: 是否强制输出为目标尺寸
            align: 对齐方式，可选 'center', 'top', 'bottom', 'left', 'right'
        """
        w, h = image.size
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图片
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        if force_size:
            # 创建目标大小的画布
            result = Image.new('RGBA', target_size, (0, 0, 0, 0))
            
            # 根据对齐方式计算粘贴位置
            if align == 'center':
                paste_x = (target_w - new_w) // 2
                paste_y = (target_h - new_h) // 2
            elif align == 'top':
                paste_x = (target_w - new_w) // 2
                paste_y = 0
            elif align == 'bottom':
                paste_x = (target_w - new_w) // 2
                paste_y = target_h - new_h
            elif align == 'left':
                paste_x = 0
                paste_y = (target_h - new_h) // 2
            elif align == 'right':
                paste_x = target_w - new_w
                paste_y = (target_h - new_h) // 2
            
            result.paste(resized, (paste_x, paste_y), resized)
            return result
        else:
            # 直接返回保持比例的图片
            return resized


class TemplateManager:
    """模板管理器"""
    
    def __init__(self):
        self.templates = {}
        # 英文ID到中文名称的映射
        self.chinese_names = {
            "classic_seamless": "经典四方连续",
            "enhanced_classic": "增强经典拼图", 
            "quadrant_edge": "象限边缘拼图",
            "random_offset": "随机偏移拼图",
            "multi_edge": "多边界拼图"
        }
        # 中文名称到英文ID的映射
        self.english_ids = {v: k for k, v in self.chinese_names.items()}
        self.register_default_templates()
    
    def register_template(self, template_id, template_class, chinese_name=None):
        """注册模板
        Args:
            template_id (str): 模板ID
            template_class (class): 模板类
            chinese_name (str): 中文名称（可选）
        """
        self.templates[template_id] = template_class
        
        # 如果提供了中文名称，更新映射
        if chinese_name:
            self.chinese_names[template_id] = chinese_name
            self.english_ids[chinese_name] = template_id
    
    def get_template(self, template_id):
        """获取模板实例
        Args:
            template_id (str): 模板ID（支持中文名称或英文ID）
            
        Returns:
            TilingTemplateBase: 模板实例
        """
        # 如果是中文名称，转换为英文ID
        if template_id in self.english_ids:
            template_id = self.english_ids[template_id]
        
        if template_id not in self.templates:
            raise ValueError(f"未找到模板: {template_id}")
        
        return self.templates[template_id]()
    
    def get_available_templates(self):
        """获取所有可用模板的信息
        Returns:
            dict: 中文模板名称 -> 模板信息的字典
        """
        result = {}
        for template_id, template_class in self.templates.items():
            instance = template_class()
            # 使用中文名称作为键
            chinese_name = self.chinese_names.get(template_id, template_id)
            template_info = instance.get_template_info()
            template_info['english_id'] = template_id  # 保留英文ID用于内部使用
            result[chinese_name] = template_info
        return result
    
    def register_default_templates(self):
        """注册默认模板"""
        # 导入并注册默认模板
        try:
            from .classic_seamless_template import ClassicSeamlessTemplate
            self.register_template("classic_seamless", ClassicSeamlessTemplate, "经典四方连续")
            print("✅ 已注册经典四方连续模板")
        except ImportError:
            print("警告: 无法导入经典四方连续模板")
        
        try:
            from .enhanced_classic_template import EnhancedClassicTemplate
            self.register_template("enhanced_classic", EnhancedClassicTemplate, "增强经典拼图")
            print("✅ 已注册增强经典拼图模板")
        except ImportError:
            print("警告: 无法导入增强经典拼图模板")
        
        try:
            from .quadrant_edge_template import QuadrantEdgeTemplate
            self.register_template("quadrant_edge", QuadrantEdgeTemplate, "象限边缘拼图")
            print("✅ 已注册象限边缘拼图模板")
        except ImportError:
            print("警告: 无法导入象限边缘拼图模板")
        
        try:
            from .random_offset_template import RandomOffsetTemplate
            self.register_template("random_offset", RandomOffsetTemplate, "随机偏移拼图")
            print("✅ 已注册随机偏移拼图模板")
        except ImportError:
            print("警告: 无法导入随机偏移拼图模板")
        
        try:
            from .multi_edge_template import MultiEdgeTemplate
            self.register_template("multi_edge", MultiEdgeTemplate, "多边界拼图")
            print("✅ 已注册多边界拼图模板")
        except ImportError:
            print("警告: 无法导入多边界拼图模板")
        
        try:
            from .quad_edge_template import QuadEdgeTemplate
            self.register_template("quad_edge", QuadEdgeTemplate, "四边界拼图")
            print("✅ 已注册四边界拼图模板")
        except ImportError:
            print("警告: 无法导入四边界拼图模板")


# 全局模板管理器实例
template_manager = TemplateManager()
