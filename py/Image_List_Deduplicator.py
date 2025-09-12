import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class Image_List_Deduplicator:
    """
    对输入的图片列表进行去重，支持过滤相似图片。
    使用多种哈希算法的组合来检测相似图片：
    - Average Hash (aHash)：基于平均灰度值
    - Perceptual Hash (pHash)：基于DCT变换
    - Difference Hash (dHash)：基于相邻像素差异
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "hash_size": ("INT", {
                    "default": 16,
                    "min": 8,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                "keep_best_quality": ("BOOLEAN", {
                    "default": True,
                    "label_on": "保留最高质量",
                    "label_off": "保留第一张"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("unique_images", "removed_count")
    FUNCTION = "deduplicate_images"
    CATEGORY = "🍒 Kim-Nodes/🏖️图像处理"
    
    # 输入输出为列表
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, False)
    
    def tensor_to_pil(self, tensor_image: torch.Tensor) -> Image.Image:
        """将张量转换为PIL图像"""
        # 确保张量在CPU上
        tensor_image = tensor_image.cpu()
        # 转换为numpy数组 (H, W, C)
        numpy_image = (tensor_image * 255).clamp(0, 255).numpy().astype(np.uint8)
        # 如果是RGB图像
        if numpy_image.shape[-1] == 3:
            return Image.fromarray(numpy_image, mode='RGB')
        # 如果是RGBA图像
        elif numpy_image.shape[-1] == 4:
            return Image.fromarray(numpy_image, mode='RGBA')
        # 如果是灰度图像
        elif len(numpy_image.shape) == 2:
            return Image.fromarray(numpy_image, mode='L')
        else:
            raise ValueError(f"Unsupported image shape: {numpy_image.shape}")
    
    def average_hash(self, image: Image.Image, hash_size: int = 16) -> str:
        """计算平均哈希"""
        # 调整图像大小并转换为灰度
        img = image.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        
        # 计算平均值
        avg = pixels.mean()
        
        # 生成哈希
        hash_bits = pixels > avg
        return ''.join(str(int(bit)) for bit in hash_bits.flatten())
    
    def difference_hash(self, image: Image.Image, hash_size: int = 16) -> str:
        """计算差异哈希"""
        # 调整图像大小（宽度多一个像素用于计算差异）
        img = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        
        # 计算相邻像素的差异
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        # 生成哈希
        return ''.join(str(int(bit)) for bit in diff.flatten())
    
    def perceptual_hash(self, image: Image.Image, hash_size: int = 16) -> str:
        """计算感知哈希（改进版）"""
        # 对于较大的hash_size，使用更大的初始图像
        img_size = max(64, hash_size * 4)
        img = image.convert('L').resize((img_size, img_size), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float32)
        
        # 简单的平滑处理（使用均值滤波代替高斯滤波）
        kernel_size = 3
        pad = kernel_size // 2
        padded = np.pad(pixels, pad, mode='edge')
        smoothed = np.zeros_like(pixels)
        
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                smoothed[i, j] = padded[i:i+kernel_size, j:j+kernel_size].mean()
        
        pixels = smoothed
        
        # 简化的DCT实现：使用分块并计算梯度
        # 将图像分成hash_size x hash_size的块
        block_size = img_size // hash_size
        dct_like = np.zeros((hash_size, hash_size))
        
        for i in range(hash_size):
            for j in range(hash_size):
                # 获取块
                block = pixels[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                # 计算块的特征（使用标准差来捕捉纹理信息）
                mean_val = block.mean()
                std_val = block.std()
                # 结合均值和标准差
                dct_like[i, j] = mean_val + std_val * 0.5
        
        # 计算中位数而不是平均值（更鲁棒）
        median_val = np.median(dct_like)
        
        # 生成哈希
        hash_bits = dct_like > median_val
        return ''.join(str(int(bit)) for bit in hash_bits.flatten())
    
    def compute_hash_tuple(self, image: Image.Image, hash_size: int) -> Tuple[str, str, str]:
        """
        计算图像的多种哈希值组合
        返回: (average_hash, perceptual_hash, difference_hash)
        """
        # 计算三种哈希，每种使用原始图像以保留更多细节
        ahash = self.average_hash(image, hash_size)
        phash = self.perceptual_hash(image, hash_size)
        dhash = self.difference_hash(image, hash_size)
        
        return (ahash, phash, dhash)
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """计算两个哈希字符串之间的汉明距离"""
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2))
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def combined_hamming_distance(self, hash_tuple1: Tuple[str, str, str], 
                                 hash_tuple2: Tuple[str, str, str]) -> int:
        """计算组合哈希的总汉明距离"""
        total_distance = 0
        for h1, h2 in zip(hash_tuple1, hash_tuple2):
            total_distance += self.hamming_distance(h1, h2)
        return total_distance
    
    def are_similar(self, hash_tuple1: Tuple[str, str, str], 
                   hash_tuple2: Tuple[str, str, str], 
                   threshold: float, hash_size: int) -> bool:
        """
        判断两个图像是否相似
        threshold: 0.0-1.0 的相似度阈值，越接近1.0要求越相似
        """
        # 分别计算每种哈希的相似度
        similarities = []
        hash_names = ['average', 'perceptual', 'difference']
        
        for idx, (h1, h2) in enumerate(zip(hash_tuple1, hash_tuple2)):
            if len(h1) != len(h2):
                logger.warning(f"{hash_names[idx]} hash长度不匹配: {len(h1)} vs {len(h2)}")
                return False
            
            # 计算汉明距离
            distance = self.hamming_distance(h1, h2)
            # 转换为相似度（0-1范围）
            max_distance = len(h1)
            similarity = 1.0 - (distance / max_distance)
            similarities.append(similarity)
        
        # 输出调试信息（仅在相似度较高时）
        weighted_similarity = sum(s * w for s, w in zip(similarities, [0.25, 0.5, 0.25]))
        if weighted_similarity > 0.8:
            logger.info(f"高相似度对比 - A:{similarities[0]:.3f}, P:{similarities[1]:.3f}, D:{similarities[2]:.3f}, 加权:{weighted_similarity:.3f}, 阈值:{threshold:.2f}")
        
        # 更严格的判断逻辑
        if threshold >= 0.99:
            # 极高阈值：要求所有哈希都非常相似
            return all(s >= 0.98 for s in similarities)
        elif threshold >= 0.95:
            # 高阈值：至少两种哈希非常相似，且没有哈希相似度过低
            high_similarity_count = sum(1 for s in similarities if s >= threshold)
            min_similarity = min(similarities)
            return high_similarity_count >= 2 and min_similarity >= (threshold - 0.15)
        elif threshold >= 0.90:
            # 中高阈值：感知哈希必须相似，且加权平均达标
            return similarities[1] >= threshold and weighted_similarity >= threshold
        elif threshold >= 0.85:
            # 中阈值：至少有一种哈希非常相似，且加权平均达标
            max_similarity = max(similarities)
            return max_similarity >= (threshold + 0.1) and weighted_similarity >= threshold
        else:
            # 低阈值：使用加权平均，但要求至少有一种哈希达到基准
            return weighted_similarity >= threshold and max(similarities) >= threshold
    
    def get_image_quality_score(self, tensor_image: torch.Tensor) -> Tuple[int, int]:
        """
        计算图像质量分数
        返回: (分辨率, 数据大小估算)
        """
        height, width = tensor_image.shape[0], tensor_image.shape[1]
        resolution = height * width
        # 估算数据大小（基于张量元素数量）
        data_size = tensor_image.numel()
        return (resolution, data_size)
    
    def deduplicate_images(self, images, similarity_threshold, hash_size, keep_best_quality):
        """
        对图片列表进行去重
        
        Args:
            images: 输入图片tensor列表
            similarity_threshold: 相似度阈值（0.0-1.0）
            hash_size: 哈希大小
            keep_best_quality: 是否保留最高质量的图片
            
        Returns:
            tuple: (去重后的图片列表, 移除的图片数量)
        """
        # 确保输入是列表
        if not isinstance(images, list):
            images = [images]
        
        # 获取参数值（处理列表输入）
        similarity_threshold = similarity_threshold[0] if isinstance(similarity_threshold, list) else similarity_threshold
        hash_size = hash_size[0] if isinstance(hash_size, list) else hash_size
        keep_best_quality = keep_best_quality[0] if isinstance(keep_best_quality, list) else keep_best_quality
        
        # 如果没有图片或只有一张图片，直接返回
        if len(images) <= 1:
            return (images, 0)
        
        # 计算所有图像的哈希值和质量分数
        image_data = []
        for i, img_tensor in enumerate(images):
            try:
                pil_image = self.tensor_to_pil(img_tensor)
                hash_tuple = self.compute_hash_tuple(pil_image, hash_size)
                quality_score = self.get_image_quality_score(img_tensor)
                image_data.append({
                    'index': i,
                    'tensor': img_tensor,
                    'hash': hash_tuple,
                    'quality': quality_score,
                    'is_duplicate': False
                })
            except Exception as e:
                logger.error(f"处理图片 {i} 时出错: {str(e)}")
                # 出错的图片保留
                image_data.append({
                    'index': i,
                    'tensor': img_tensor,
                    'hash': None,
                    'quality': (0, 0),
                    'is_duplicate': False
                })
        
        # 查找重复组
        duplicate_groups = []
        processed = set()
        
        for i in range(len(image_data)):
            if i in processed or image_data[i]['hash'] is None:
                continue
                
            # 创建新的重复组，只包含真正相似的图片
            group = [i]
            processed.add(i)
            
            # 只与当前图片比较，不传递相似性
            for j in range(i + 1, len(image_data)):
                if j in processed or image_data[j]['hash'] is None:
                    continue
                    
                # 只有与第一张图片相似的才加入组
                if self.are_similar(image_data[i]['hash'], image_data[j]['hash'], 
                                  similarity_threshold, hash_size):
                    group.append(j)
                    processed.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                
        # 输出调试信息
        logger.info(f"发现 {len(duplicate_groups)} 个重复组")
        for idx, group in enumerate(duplicate_groups):
            logger.info(f"组 {idx + 1}: 包含 {len(group)} 张图片")
        
        # 处理重复组，标记要删除的图片
        removed_count = 0
        for group in duplicate_groups:
            if keep_best_quality:
                # 按质量排序，保留最好的
                group_data = [(idx, image_data[idx]['quality']) for idx in group]
                # 按分辨率和数据大小排序，最后按索引排序（保持稳定性）
                group_data.sort(key=lambda x: (x[1][0], x[1][1], x[0]))
                # 标记除最后一个（最高质量）外的所有图片为重复
                for idx, _ in group_data[:-1]:
                    image_data[idx]['is_duplicate'] = True
                    removed_count += 1
            else:
                # 保留第一张，删除其他
                for idx in group[1:]:
                    image_data[idx]['is_duplicate'] = True
                    removed_count += 1
        
        # 收集未标记为重复的图片
        unique_images = []
        for data in image_data:
            if not data['is_duplicate']:
                unique_images.append(data['tensor'])
        
        # 如果所有图片都被标记为重复（不应该发生），至少保留第一张
        if not unique_images and images:
            unique_images = [images[0]]
            removed_count = len(images) - 1
        
        logger.info(f"去重完成：原始图片 {len(images)} 张，保留 {len(unique_images)} 张，移除 {removed_count} 张")
        
        return (unique_images, removed_count) 