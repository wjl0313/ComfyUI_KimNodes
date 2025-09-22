import os
import json
import logging
from typing import Dict, Any, Optional
import folder_paths

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class LoRA_Metadata_Reader:
    """
    LoRA元数据读取节点
    
    用于读取LoRA模型文件的元数据信息，包括训练参数、标签频率、数据集信息等。
    支持.safetensors格式的LoRA模型文件。
    """

    def __init__(self):
        self.lora_dir = folder_paths.get_folder_paths("loras")[0] if folder_paths.get_folder_paths("loras") else ""

    @classmethod
    def INPUT_TYPES(cls):
        # 获取LoRA模型文件列表
        lora_files = []
        if folder_paths.get_folder_paths("loras"):
            lora_dir = folder_paths.get_folder_paths("loras")[0]
            if os.path.exists(lora_dir):
                for file in os.listdir(lora_dir):
                    if file.endswith('.safetensors'):
                        lora_files.append(file)
        
        if not lora_files:
            lora_files = ["None"]

        return {
            "required": {
                "lora_file": (lora_files, {
                    "default": lora_files[0] if lora_files else "None"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("元数据",)
    FUNCTION = "read_lora_metadata"
    CATEGORY = "🍒 Kim-Nodes/🔢Metadata | 元数据处理"

    def read_safetensors_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """读取safetensors文件的元数据"""
        if not SAFETENSORS_AVAILABLE:
            logging.error("safetensors库未安装，无法读取元数据")
            return None
        
        try:
            # 直接读取safetensors文件的header来获取元数据
            import struct
            with open(file_path, 'rb') as f:
                # 读取header长度 (前8字节)
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    return None
                header_size = struct.unpack('<Q', header_size_bytes)[0]
                
                # 读取header内容
                header_bytes = f.read(header_size)
                if len(header_bytes) < header_size:
                    return None
                
                header = json.loads(header_bytes.decode('utf-8'))
                metadata = header.get('__metadata__', {})
                return metadata
        except Exception as e:
            logging.error(f"读取safetensors元数据失败: {e}")
            # 备用方法：使用safe_open
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
                    return metadata
            except Exception as e2:
                logging.error(f"备用方法也失败: {e2}")
                return None



    def read_lora_metadata(self, lora_file: str):
        """
        读取LoRA模型的元数据信息
        """
        if lora_file == "None":
            return ("未选择LoRA文件",)

        # 构建完整文件路径
        if self.lora_dir:
            file_path = os.path.join(self.lora_dir, lora_file)
        else:
            file_path = lora_file

        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            return (f"错误：文件不存在\n路径：{file_path}",)

                # 读取元数据
        metadata = self.read_safetensors_metadata(file_path)
        
        if metadata is None:
            return (f"错误：无法读取元数据\n文件路径：{file_path}",)
        
        # 固定参数设置（始终显示完整信息）
        include_training_params = True
        include_tag_frequency = True
        include_dataset_info = True
        max_tags = 100  # 显示前100个标签
        
        # 提取基本信息
        model_name = os.path.splitext(lora_file)[0]
        display_name = metadata.get("ssmd_display_name", model_name) if metadata else model_name
        author = metadata.get("ssmd_author", "未知") if metadata else "未知"
        description = metadata.get("ssmd_description", "无描述") if metadata else "无描述"
        keywords = metadata.get("ssmd_keywords", "无关键词") if metadata else "无关键词"
        tags = metadata.get("ssmd_tags", "无标签") if metadata else "无标签"

        # 构建完整元数据字典，只添加有效数据
        full_metadata = {}
        
        # 基本信息部分 - 只添加非空值
        basic_info = {}
        if model_name and model_name != "None":
            basic_info["模型名称"] = model_name
        if display_name and display_name != model_name and display_name != "未知":
            basic_info["显示名称"] = display_name
        if author and author != "未知":
            basic_info["作者"] = author
        if description and description != "无描述":
            basic_info["描述"] = description
        if keywords and keywords != "无关键词":
            basic_info["关键词"] = keywords
        if tags and tags != "无标签":
            basic_info["标签"] = tags
            
        # 评分
        rating = metadata.get("ssmd_rating", "0") if metadata else "0"
        if rating and rating != "0":
            basic_info["评分"] = rating
            
        # 来源信息
        source = metadata.get("ssmd_source", "") if metadata else ""
        if source and source != "未知":
            basic_info["来源"] = source
            
        # 版本信息
        version = metadata.get("ssmd_version", "") if metadata else ""
        if version and version != "未知":
            basic_info["版本"] = version
            
        # 哈希值
        model_hash = metadata.get("sshs_model_hash", "") if metadata else ""
        legacy_hash = metadata.get("sshs_legacy_hash", "") if metadata else ""
        if model_hash and model_hash != "未知":
            basic_info["模型哈希"] = model_hash
        if legacy_hash and legacy_hash != "未知":
            basic_info["传统哈希"] = legacy_hash
            
        # 只有当基本信息非空时才添加
        if basic_info:
            full_metadata["基本信息"] = basic_info

        # 添加训练参数（ss_前缀的参数）
        if include_training_params and metadata:
            training_params = {}
            
            # 保留原始参数名，不翻译
            
            # 需要排除的内部参数
            excluded_params = {
                "ss_bucket_info",  # 内部分桶信息
                "ss_sd_scripts_commit_hash",  # 脚本版本哈希
                "ss_model_prediction_type",  # 模型预测类型
                "ss_loss_type",  # 损失类型
                "ss_huber_scale", "ss_huber_c", "ss_huber_schedule",  # Huber损失参数
                "ss_logit_mean", "ss_logit_std",  # Logit参数
                "ss_mode_scale",  # 模式缩放
                "ss_discrete_flow_shift",  # 离散流偏移
                "ss_weighting_scheme",  # 权重方案
                "ss_timestep_sampling",  # 时间步采样
                "ss_sigmoid_scale",  # Sigmoid缩放
                "ss_debiased_estimation",  # 去偏估计
                "ss_ip_noise_gamma_random_strength",  # IP噪声随机强度
                "ss_noise_offset_random_strength",  # 噪声偏移随机强度
                "ss_validation_split", "ss_validation_seed",  # 验证参数
                "ss_validate_every_n_steps", "ss_validate_every_n_epochs",  # 验证频率
                "ss_max_validation_steps",  # 最大验证步数
                "ss_num_validation_images",  # 验证图像数
                "ss_num_train_images",  # 训练图像数
                "ss_num_batches_per_epoch",  # 每轮批次数
                "ss_guidance_scale",  # 引导缩放
                "ss_apply_t5_attn_mask",  # T5注意力掩码
                "ss_sd_model_hash",  # SD模型哈希
                "ss_training_comment",  # 训练注释
                # 额外的内部参数
                "ss_steps",  # 如果和max_train_steps相同就排除
                "ss_num_epochs",  # 如果和epoch相同就排除
                "ss_v_parameterization",  # V参数化
                "ss_v_prediction",  # V预测
                "ss_dataset_split",  # 数据集分割
                "ss_augmentation",  # 增强
                "ss_optimizer_args",  # 优化器参数
                "ss_network_args",  # 网络参数
                "ss_max_bucket_resolution",  # 最大分桶分辨率
                "ss_min_bucket_resolution",  # 最小分桶分辨率
                "ss_cached_latents_epoch",  # 缓存潜变量轮数
            }
            
            # 分类整理训练参数
            network_params = {}  # 网络结构参数
            training_params_basic = {}  # 基础训练参数
            lr_params = {}  # 学习率相关
            dataset_params = {}  # 数据集相关
            optimization_params = {}  # 优化相关
            other_params = {}  # 其他参数
            
            for k, v in metadata.items():
                if k.startswith("ss_") and k not in ["ss_tag_frequency", "ss_dataset_dirs"]:
                    # 跳过排除的参数
                    if k in excluded_params:
                        continue
                    
                    # 使用原始参数名，去掉ss_前缀
                    param_name = k[3:] if k.startswith("ss_") else k
                    
                    # 格式化值
                    if isinstance(v, str):
                        # 处理布尔值
                        if v.lower() == "true":
                            v = True
                        elif v.lower() == "false":
                            v = False
                        elif v.lower() == "none":
                            continue  # 跳过None值
                        else:
                            # 尝试解析数字
                            try:
                                if '.' in v:
                                    v = float(v)
                                    # 如果是整数，转换为int
                                    if v.is_integer():
                                        v = int(v)
                                else:
                                    v = int(v)
                            except:
                                # 处理特殊格式
                                if v.startswith("(") and v.endswith(")") and "," in v:
                                    # 处理分辨率格式 "(1024, 1024)"
                                    try:
                                        v = v.strip("()").replace(" ", "")
                                    except:
                                        pass
                                # 处理优化器名称
                                elif "." in v and "optim" in v.lower():
                                    # 简化优化器名称
                                    parts = v.split(".")
                                    v = parts[-1] if parts else v
                    
                    # 处理时间戳
                    if k in ["ss_training_started_at", "ss_training_finished_at"] and isinstance(v, (int, float)):
                        # 转换Unix时间戳为可读格式
                        import datetime
                        try:
                            dt = datetime.datetime.fromtimestamp(v)
                            v = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    
                    # 格式化学习率
                    if k in ["ss_learning_rate", "ss_unet_lr", "ss_text_encoder_lr"]:
                        if isinstance(v, str):
                            try:
                                # 处理科学计数法
                                v = float(v)
                            except:
                                pass
                    
                    # 分类参数
                    if k in ["ss_network_module", "ss_network_dim", "ss_network_alpha", "ss_base_model_version"]:
                        network_params[param_name] = v
                    elif k in ["ss_learning_rate", "ss_unet_lr", "ss_text_encoder_lr", "ss_lr_scheduler", "ss_lr_warmup_steps"]:
                        lr_params[param_name] = v
                    elif k in ["ss_max_train_steps", "ss_steps", "ss_epoch", "ss_num_epochs", "ss_batch_size_per_device", 
                              "ss_total_batch_size", "ss_gradient_accumulation_steps", "ss_seed", 
                              "ss_training_started_at", "ss_training_finished_at"]:
                        training_params_basic[param_name] = v
                    elif k in ["ss_resolution", "ss_enable_bucket", "ss_bucket_no_upscale", "ss_min_bucket_reso", 
                              "ss_max_bucket_reso", "ss_caption_dropout_rate", "ss_caption_tag_dropout_rate",
                              "ss_shuffle_caption", "ss_keep_tokens", "ss_max_token_length", "ss_cache_latents",
                              "ss_flip_aug", "ss_color_aug", "ss_random_crop", "ss_face_crop_aug_range"]:
                        dataset_params[param_name] = v
                    elif k in ["ss_optimizer", "ss_mixed_precision", "ss_gradient_checkpointing", "ss_clip_skip",
                              "ss_noise_offset", "ss_min_snr_gamma", "ss_scale_weight_norms", "ss_full_fp16",
                              "ss_fp8_base", "ss_fp8_base_unet", "ss_max_grad_norm", "ss_lowram"]:
                        optimization_params[param_name] = v
                    else:
                        # 添加其他参数
                        other_params[param_name] = v
            
            # 构建分层的训练参数
            if network_params:
                training_params["Network Structure"] = network_params
            if training_params_basic:
                # 合并重复的参数
                # 保留epoch参数，因为它通常表示当前训练的epoch数，与num_epochs（总epoch数）不同
                # 只有当epoch和num_epochs完全相同时才删除epoch
                if "epoch" in training_params_basic and "num_epochs" in training_params_basic:
                    if str(training_params_basic["epoch"]) == str(training_params_basic["num_epochs"]):
                        del training_params_basic["epoch"]
                    
                if "max_train_steps" in training_params_basic and "steps" in training_params_basic:
                    if str(training_params_basic["max_train_steps"]) == str(training_params_basic["steps"]):
                        del training_params_basic["steps"]
                        
                training_params["Basic Parameters"] = training_params_basic
            if lr_params:
                training_params["Learning Rate"] = lr_params
            if dataset_params:
                training_params["Dataset Settings"] = dataset_params
            if optimization_params:
                training_params["Optimization Settings"] = optimization_params
            if other_params:
                training_params["Other Parameters"] = other_params
                
            if training_params:
                full_metadata["训练参数"] = training_params

        # 处理标签频率
        if include_tag_frequency and metadata and "ss_tag_frequency" in metadata:
            try:
                tag_frequency_raw = metadata["ss_tag_frequency"]
                if isinstance(tag_frequency_raw, str):
                    tag_frequency_data = json.loads(tag_frequency_raw)
                else:
                    tag_frequency_data = tag_frequency_raw
                
                # 合并所有目录的标签频率
                all_tags = {}
                dir_tag_counts = {}  # 记录每个目录的标签数
                
                for dir_name, frequencies in tag_frequency_data.items():
                    dir_tag_counts[dir_name] = len(frequencies)
                    for tag, count in frequencies.items():
                        tag = tag.strip()
                        if tag in all_tags:
                            all_tags[tag] += count
                        else:
                            all_tags[tag] = count
                
                # 按频率排序并限制数量
                if all_tags:
                    sorted_tags = dict(sorted(all_tags.items(), key=lambda x: x[1], reverse=True))
                    if len(sorted_tags) > max_tags:
                        sorted_tags = dict(list(sorted_tags.items())[:max_tags])
                    
                    # 创建标签频率的可视化数据
                    max_count = max(sorted_tags.values()) if sorted_tags else 1
                    
                    tag_freq_data = {
                        "标签总数": len(all_tags),
                        "显示数量": len(sorted_tags),
                        "目录数": len(dir_tag_counts),
                    }
                    
                    # 添加前N个高频标签
                    top_tags_list = []
                    for i, (tag, count) in enumerate(sorted_tags.items()):
                        if i < max_tags:  # 使用max_tags参数
                            percentage = (count / max_count) * 100
                            top_tags_list.append({
                                "标签": tag,
                                "出现次数": count,
                                "频率": f"{percentage:.1f}%"
                            })
                    
                    if top_tags_list:
                        tag_freq_data["高频标签"] = top_tags_list
                    
                    # 如果有多个目录，显示每个目录的标签数
                    if len(dir_tag_counts) > 1:
                        tag_freq_data["各目录标签数"] = dir_tag_counts
                    
                    full_metadata["标签频率分析"] = tag_freq_data
                    
            except Exception as e:
                logging.warning(f"解析标签频率失败: {e}")

        # 处理数据集信息
        if include_dataset_info and metadata and "ss_dataset_dirs" in metadata:
            try:
                dataset_dirs_raw = metadata["ss_dataset_dirs"]
                if isinstance(dataset_dirs_raw, str):
                    dataset_dirs_data = json.loads(dataset_dirs_raw)
                else:
                    dataset_dirs_data = dataset_dirs_raw
                
                dataset_info = {
                    "overview": {},
                    "details": []
                }
                
                total_imgs = 0
                total_weighted = 0
                max_repeats = 0
                min_repeats = float('inf')
                
                for dir_name, counts in dataset_dirs_data.items():
                    img_count = int(counts.get("img_count", 0))
                    n_repeats = int(counts.get("n_repeats", 1))
                    weighted_total = img_count * n_repeats
                    
                    total_imgs += img_count
                    total_weighted += weighted_total
                    max_repeats = max(max_repeats, n_repeats)
                    min_repeats = min(min_repeats, n_repeats)
                    
                    # 提取目录名中的概念（如果有）
                    dir_display = dir_name.split('\\')[-1].split('/')[-1]
                    
                    detail_item = {
                        "name": dir_display,
                        "image_count": img_count,
                        "repeats": f"×{n_repeats}",
                        "weighted_total": weighted_total
                    }
                    
                    # 只有当路径和显示名不同时才添加原始路径
                    if dir_name != dir_display and '\\' in dir_name or '/' in dir_name:
                        detail_item["original_path"] = dir_name
                        
                    dataset_info["details"].append(detail_item)
                
                # 计算数据集概览
                dataset_info["overview"] = {
                    "dataset_count": len(dataset_dirs_data),
                    "total_images": total_imgs,
                    "weighted_total_images": total_weighted,
                    "average_repeats": f"{total_weighted / total_imgs:.1f}" if total_imgs > 0 else "0",
                    "repeat_range": f"{min_repeats} - {max_repeats}" if min_repeats != float('inf') else "无"
                }
                
                # 按加权总数排序
                dataset_info["details"].sort(key=lambda x: x["weighted_total"], reverse=True)
                
                full_metadata["数据集信息"] = dataset_info
                    
            except Exception as e:
                logging.warning(f"解析数据集信息失败: {e}")

        # 添加其他有用信息
        if metadata:
            # 检查是否为Flux LoRA
            is_flux = False
            base_model = metadata.get("ss_base_model_version", "")
            if "flux" in base_model.lower():
                is_flux = True
            elif metadata.get("ss_sd_model_name", ""):
                if "flux" in metadata.get("ss_sd_model_name", "").lower():
                    is_flux = True
            
            # 添加模型类型信息
            model_info = {
                "模型类型": "Flux LoRA" if is_flux else "LoRA",
                "文件大小": f"{os.path.getsize(file_path) / (1024*1024):.1f} MB" if os.path.exists(file_path) else "未知",
            }
            
            # 添加触发词（如果有）
            trigger_words = []
            if metadata.get("ss_trigger_words"):
                trigger_words = metadata.get("ss_trigger_words", "").split(",")
            elif metadata.get("trigger_words"):
                trigger_words = metadata.get("trigger_words", "").split(",")
                
            if trigger_words:
                model_info["触发词"] = [word.strip() for word in trigger_words if word.strip()]
                
            full_metadata["模型信息"] = model_info
            
            # 添加未分类的其他元数据（排除已处理的和内部使用的）
            excluded_keys = {
                "ss_tag_frequency", "ss_dataset_dirs", "ssmd_display_name", "ssmd_author",
                "ssmd_description", "ssmd_keywords", "ssmd_tags", "ssmd_rating", "ssmd_source",
                "ssmd_version", "sshs_model_hash", "sshs_legacy_hash", "ss_trigger_words",
                "trigger_words"
            }
            
            # 处理ModelSpec元数据，去掉前缀
            
            modelspec_data = {}
            other_metadata = {}
            
            for k, v in metadata.items():
                if not k.startswith("ss_") and k not in excluded_keys:
                    # 只添加有意义的值
                    if v and str(v).strip() and str(v) not in ["None", "null", "undefined", ""]:
                        # 处理ModelSpec元数据
                        if k.startswith("modelspec."):
                            # 去掉modelspec.前缀
                            spec_name = k.replace("modelspec.", "")
                            modelspec_data[spec_name] = v
                        else:
                            other_metadata[k] = v
            
            # 添加ModelSpec信息
            if modelspec_data:
                full_metadata["模型规范"] = modelspec_data
                        
            # 只有当有其他元数据时才添加
            if other_metadata:
                full_metadata["其他元数据"] = other_metadata

        logging.info(f"成功读取LoRA元数据: {model_name}")
        
        # 格式化输出为可读的字符串
        output_lines = []
        
        # 基本信息
        output_lines.append("【Basic Info / 基本信息】")
        output_lines.append(f"Model Name: {model_name}")
        if basic_info:
            if "模型哈希" in basic_info:
                output_lines.append(f"Model Hash: {basic_info['模型哈希']}")
            if "传统哈希" in basic_info:
                output_lines.append(f"Legacy Hash: {basic_info['传统哈希']}")
            if "作者" in basic_info:
                output_lines.append(f"Author: {basic_info['作者']}")
            if "版本" in basic_info:
                output_lines.append(f"Version: {basic_info['版本']}")
        
        # 模型信息
        if "模型信息" in full_metadata:
            output_lines.append("\n【Model Info / 模型信息】")
            model_info = full_metadata["模型信息"]
            output_lines.append(f"Type: {model_info.get('模型类型', '未知')}")
            output_lines.append(f"File Size: {model_info.get('文件大小', '未知')}")
            if "触发词" in model_info:
                output_lines.append(f"Trigger Words: {', '.join(model_info['触发词'])}")
        
        # 训练参数（显示完整信息）
        if "训练参数" in full_metadata:
            output_lines.append("\n【Training Parameters / 训练参数】")
            training = full_metadata["训练参数"]
            
            # 网络结构
            if "Network Structure" in training:
                net = training["Network Structure"]
                output_lines.append("◆ Network Structure / 网络结构")
                for k, v in net.items():
                    output_lines.append(f"  - {k}: {v}")
            
            # 基础参数
            if "Basic Parameters" in training:
                basic = training["Basic Parameters"]
                output_lines.append("◆ Basic Parameters / 基础参数")
                for k, v in basic.items():
                    output_lines.append(f"  - {k}: {v}")
            
            # 学习率
            if "Learning Rate" in training:
                lr = training["Learning Rate"]
                output_lines.append("◆ Learning Rate / 学习率")
                for k, v in lr.items():
                    output_lines.append(f"  - {k}: {v}")
            
            # 数据集设置
            if "Dataset Settings" in training:
                ds_settings = training["Dataset Settings"]
                output_lines.append("◆ Dataset Settings / 数据集设置")
                for k, v in ds_settings.items():
                    output_lines.append(f"  - {k}: {v}")
            
            # 优化设置
            if "Optimization Settings" in training:
                opt_settings = training["Optimization Settings"]
                output_lines.append("◆ Optimization Settings / 优化设置")
                for k, v in opt_settings.items():
                    output_lines.append(f"  - {k}: {v}")
            
            # 其他参数
            if "Other Parameters" in training:
                other_params = training["Other Parameters"]
                output_lines.append("◆ Other Parameters / 其他参数")
                for k, v in other_params.items():
                    output_lines.append(f"  - {k}: {v}")
        
        # 数据集信息（完整显示）
        if "数据集信息" in full_metadata:
            dataset = full_metadata["数据集信息"]
            output_lines.append("\n【Dataset Info / 数据集信息】")
            
            if "overview" in dataset:
                overview = dataset["overview"]
                output_lines.append("◆ Overview / 概览")
                for k, v in overview.items():
                    output_lines.append(f"  - {k}: {v}")
            
            if "details" in dataset and len(dataset["details"]) > 0:
                output_lines.append("◆ Dataset Details / 数据集详情")
                for ds in dataset["details"]:  # 显示所有数据集
                    ds_line = f"  - {ds['name']}: {ds['image_count']} images {ds['repeats']} (weighted: {ds['weighted_total']})"
                    if "original_path" in ds:
                        ds_line += f"\n    path: {ds['original_path']}"
                    output_lines.append(ds_line)
        
        # 标签频率（显示前50个）
        if "标签频率分析" in full_metadata:
            tag_analysis = full_metadata["标签频率分析"]
            output_lines.append("\n【Tag Frequency / 标签频率】")
            output_lines.append(f"Total Tags: {tag_analysis.get('标签总数', 0)} | Showing: {tag_analysis.get('显示数量', 0)}")
            
            # 各目录标签数
            if "各目录标签数" in tag_analysis:
                output_lines.append("◆ Tags per Directory / 各目录标签数")
                for dir_name, count in tag_analysis["各目录标签数"].items():
                    output_lines.append(f"  - {dir_name}: {count} tags")
            
            # 高频标签列表
            if "高频标签" in tag_analysis:
                output_lines.append("◆ Top Tags / 高频标签")
                for i, tag in enumerate(tag_analysis["高频标签"][:50]):  # 显示前50个
                    output_lines.append(f"  {i+1}. {tag['标签']} ({tag['频率']})")
        
        # 模型规范
        if "模型规范" in full_metadata:
            spec = full_metadata["模型规范"]
            output_lines.append("\n【Model Specification / 模型规范】")
            for k, v in spec.items():
                output_lines.append(f"  - {k}: {v}")
        
        # 其他元数据
        if "其他元数据" in full_metadata:
            other = full_metadata["其他元数据"]
            output_lines.append("\n【Other Metadata / 其他元数据】")
            for k, v in other.items():
                output_lines.append(f"  - {k}: {v}")
        
        # 如果没有任何数据
        if len(output_lines) == 0:
            output_lines.append("未找到有效的元数据")
        
        return ("\n".join(output_lines),) 