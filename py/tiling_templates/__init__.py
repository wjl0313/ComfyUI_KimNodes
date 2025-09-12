"""
无缝拼图模板包

这个包包含了各种无缝拼图模板的实现。
"""

from .base_template import TilingTemplateBase, TemplateManager, template_manager
from .classic_seamless_template import ClassicSeamlessTemplate
from .enhanced_classic_template import EnhancedClassicTemplate
from .quadrant_edge_template import QuadrantEdgeTemplate
from .random_offset_template import RandomOffsetTemplate
from .multi_edge_template import MultiEdgeTemplate
from .quad_edge_template import QuadEdgeTemplate

__all__ = [
    'TilingTemplateBase',
    'TemplateManager', 
    'template_manager',
    'ClassicSeamlessTemplate',
    'EnhancedClassicTemplate',
    'QuadrantEdgeTemplate',
    'RandomOffsetTemplate',
    'MultiEdgeTemplate',
    'QuadEdgeTemplate'
]
