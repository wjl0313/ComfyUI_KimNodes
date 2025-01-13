import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import os
import json

class ExtractDifferenceLora:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_org": ("model", {
                }),
                "model_tuned": ("model", {
                }),
                "save_path": ("STRING", {
                    "multiline": False,
                    "default": "models/loras/difference_lora.safetensors"
                }),
                "dim": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 320,
                    "step": 2,
                    "display": "number"
                }),
                "device": (["cuda", "cpu"],),
                "save_precision": (["float", "fp16", "bf16", "none"],),
                "clamp_quantile": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "extract_difference"
    CATEGORY = "🍊 Kim-Nodes/LoRA_DifferenceExtraction"

    def str_to_dtype(self, p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    def save_to_file(self, file_name, state_dict, metadata, dtype):
        if dtype is not None:
            for key in list(state_dict.keys()):
                if type(state_dict[key]) == torch.Tensor:
                    state_dict[key] = state_dict[key].to(dtype)
        
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        save_file(state_dict, file_name, metadata=metadata)

    def extract_difference(self, model_org, model_tuned, save_path, dim=8, 
                         device="cuda", save_precision="fp16", clamp_quantile=0.99):
        # 设置数据类型
        calc_dtype = torch.float
        save_dtype = self.str_to_dtype(save_precision if save_precision != "none" else None)
        store_device = "cpu"

        # 加载模型
        print(f"Loading original model from {model_org}")
        org_state_dict = load_file(model_org)
        print(f"Loading tuned model from {model_tuned}")
        tuned_state_dict = load_file(model_tuned)

        # 提取权重
        lora_weights = {}
        
        # 过滤需要处理的键
        keys = [key for key in org_state_dict.keys() 
                if "model" in key 
                and ".weight" in key 
                and not any(x in key for x in ["norm", "time_embed"])
                and key in tuned_state_dict]

        print("Extracting difference and creating LoRA weights...")
        for key in tqdm(keys):
            # 计算两个模型的权重差异
            mat = tuned_state_dict[key].to(calc_dtype) - org_state_dict[key].to(calc_dtype)
            if device:
                mat = mat.to(device)

            # 获取维度
            out_dim, in_dim = mat.size()[0:2]
            rank = min(dim, in_dim, out_dim)

            # 展平矩阵
            mat = mat.reshape(out_dim, -1)

            # SVD分解
            U, S, Vh = torch.linalg.svd(mat)

            # 取前rank个奇异值
            U = U[:, :rank]
            S = S[:rank]
            U = U @ torch.diag(S)
            Vh = Vh[:rank, :]

            # 裁剪值
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp_quantile)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            # 转换数据类型并存储
            U = U.to(store_device, dtype=save_dtype).contiguous()
            Vh = Vh.to(store_device, dtype=save_dtype).contiguous()

            lora_weights[key] = (U, Vh)
            del mat, U, S, Vh

        # 创建LoRA权重字典
        lora_sd = {}
        for key, (up_weight, down_weight) in lora_weights.items():
            lora_name = key.replace(".weight", "").replace(".", "_")
            lora_sd[f"lora_{lora_name}.up.weight"] = up_weight
            lora_sd[f"lora_{lora_name}.down.weight"] = down_weight
            lora_sd[f"lora_{lora_name}.alpha"] = torch.tensor(down_weight.size()[0])

        # 添加元数据
        metadata = {
            "ss_network_module": "networks.lora",
            "ss_network_dim": str(dim),
            "ss_network_alpha": str(float(dim)),
            "ss_network_args": json.dumps({}),
        }

        # 保存LoRA权重
        print(f"Saving difference LoRA weights to {save_path}")
        self.save_to_file(save_path, lora_sd, metadata, save_dtype)
        print("Done!")
        
        return (save_path,)