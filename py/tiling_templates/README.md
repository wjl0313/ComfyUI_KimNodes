# 无缝拼图模板系统

## 设计原则

1. **参数统一管理**: 所有参数都由主节点 `SeamlessTilingGenerator` 统一定义和管理
2. **模板职责单一**: 模板只负责实现具体的拼图算法逻辑
3. **接口标准化**: 所有模板都继承相同的基类，保证接口一致性
4. **通用性保证**: 不同模板使用相同的参数接口，便于切换和扩展
5. **填充策略独立**: 每个模板可以实现不同的中间填充策略和数量

## 架构说明

### 基础组件

- `base_template.py`: 模板基类，定义标准接口
- `classic_seamless_template.py`: 经典四方连续模板实现（单个图片居中填充）
- `grid_seamless_template.py`: 网格无缝拼图模板实现（2x2网格填充）
- `enhanced_classic_template.py`: 增强经典拼图模板实现（基于经典模板增加4个象限中心图片）
- `__init__.py`: 模板包初始化和注册

### 模板基类接口

```python
class TilingTemplateBase(ABC):
    def get_template_info(self):
        """返回模板基本信息（名称、描述）"""
        
    def validate_params(self, params):
        """验证参数有效性"""
        
    def generate_tiling(self, images, canvas_size, params):
        """生成无缝拼图的核心算法"""
```

### 标准参数

所有模板都接收以下标准参数：

- `基础图片尺寸`: 统一的尺寸基准，用于边界宽度、角落大小、中间图片大小
- `边界宽度`: 边界区域的宽度（默认使用基础图片尺寸）
- `角落大小`: 四个角落的大小（默认使用基础图片尺寸）
- `中间图片大小`: 中间填充图片的大小（默认使用基础图片尺寸）
- `填充中间区域`: 是否在中间区域放置图片
- `随机种子`: 随机数种子
- `启用随机`: 是否启用随机模式
- `背景颜色`: 背景颜色值

**注意**: 现在主要使用 `基础图片尺寸` 作为统一的尺寸控制，其他三个尺寸参数会自动使用该基础尺寸。模板可以选择使用基础尺寸或分别使用具体的尺寸参数。

## 中间填充策略

不同模板可以实现不同的中间区域填充策略：

### 经典四方连续模板 (classic_seamless)
- **填充数量**: 1个图片
- **布局方式**: 居中放置
- **尺寸调整**: 中间图片放大2倍（匹配边缘裁切效果）
- **适用场景**: 传统四方连续效果，简洁优雅

### 网格无缝拼图模板 (grid_seamless)  
- **填充数量**: 最多4个图片 (2x2网格)
- **布局方式**: 网格排列，均匀分布
- **尺寸调整**: 网格图片放大1.5倍（适应多图布局）
- **适用场景**: 丰富的视觉效果，适合多图片展示

### 增强经典拼图模板 (enhanced_classic)
- **填充数量**: 13个图片 (8个固定 + 5个可选)
- **布局方式**: 基于经典模板，在四个象限的几何中心增加4个图片
- **固定图片**: 4个边缘 + 4个角落（总是显示，不受开关控制）
- **可选图片**: 1个整体中心 + 4个象限中心（受"填充中间区域"开关控制）
- **图片分配策略**: 
  - 输入图片 ≥ 3张：边缘3个位置（1个角落+1个水平边界+1个垂直边界）使用不重复图片，中心使用剩余图片
  - 输入图片 < 3张：边缘智能重复分配，中心可与边缘重复
  - 角落处理：始终使用1张图片的四等分，符合无缝拼图标准
  - 边缘处理：水平边界（上下共用）+ 垂直边界（左右共用），符合经典模板标准
- **象限分布**: 真正的象限几何中心位置，不受边缘图片大小影响
- **位置算法**: 基于画布的固定比例位置 (1/4, 1/4), (3/4, 1/4), (1/4, 3/4), (3/4, 3/4)
- **尺寸调整**: 象限图片放大2倍（与经典模板中心图片尺寸一致）
- **适用场景**: 增强的经典效果，智能图片分配，保持正确的象限中心位置

## 尺寸一致性原理

为了确保中间图片与边缘图片在视觉上保持一致的大小，模板系统会自动调整中间图片的尺寸：

### 问题分析
- **边缘区域**: 使用原图的裁切片段（如上半部分、下半部分）
- **角落区域**: 使用原图的四等分片段
- **中间区域**: 使用完整的图片

### 解决方案
- **经典模板**: 中间图片放大2倍，因为边缘使用原图的一半
- **网格模板**: 网格图片放大1.5倍，适应多图片布局的视觉平衡
- **增强经典模板**: 象限图片放大2倍，与经典模板中心图片保持相同尺寸标准
- **自定义模板**: 可根据边缘处理方式调整相应的放大倍数

### 自定义模板填充策略
开发者可以根据需要实现：
- 单行/单列布局
- 不规则排列
- 重叠效果
- 渐变过渡
- 等等...

## 添加新模板

1. 继承 `TilingTemplateBase` 基类
2. 实现所有抽象方法
3. 在 `base_template.py` 的 `TemplateManager.register_default_templates()` 中注册
4. 模板只实现算法逻辑，不定义特殊参数

## 示例：创建新模板

```python
from .base_template import TilingTemplateBase

class MyCustomTemplate(TilingTemplateBase):
    def __init__(self):
        super().__init__()
        self.template_name = "我的自定义模板"
        self.template_description = "自定义拼图算法描述"
    
    def get_template_info(self):
        return {
            "name": self.template_name,
            "description": self.template_description
        }
    
    def validate_params(self, params):
        # 验证必需参数 - 支持新的统一尺寸参数
        has_basic_size = "基础图片尺寸" in params
        has_separate_sizes = all(param in params for param in ["边界宽度", "角落大小", "中间图片大小"])
        
        if not (has_basic_size or has_separate_sizes):
            return False
            
        # 检查其他必需参数
        if "填充中间区域" not in params:
            return False
            
        return True
    
    def generate_tiling(self, images, canvas_size, params):
        # 实现自定义拼图算法
        # 返回 (canvas, mask_canvas)
        pass
```

然后在 `base_template.py` 中注册：

```python
def register_default_templates(self):
    try:
        from .classic_seamless_template import ClassicSeamlessTemplate
        from .my_custom_template import MyCustomTemplate
        
        self.register_template("classic_seamless", ClassicSeamlessTemplate)
        self.register_template("my_custom", MyCustomTemplate)
    except ImportError as e:
        print(f"警告: 无法导入模板 - {e}")
```
