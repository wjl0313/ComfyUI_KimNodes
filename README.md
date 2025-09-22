# ComfyUI_KimNodes

A versatile toolbox offering image effects, icon layout processing, YOLO-based cropping, and more. This node library is tailored to meet diverse creative and functional needs.

---

## Features

### 🍒YOLO_Crop✀/ 🍒Crop_Paste✀

Cut out the parts detected by YOLO and paste them back onto the original picture.

#### Example:
<img width="1479" height="704" alt="QQ20250922-142055" src="https://github.com/user-attachments/assets/f660191c-74d3-4a02-a030-a55cc0313817" />

---

### 🍒IconDistributeByGrid

Distribute icons in a grid layout, following a structured sequence.

#### Example:
<img width="3000" height="1878" alt="Seamless_Template" src="https://github.com/user-attachments/assets/d6e59c2b-044f-4886-ac2d-4447eeb110fd" />

---

### 🍒Distribute_Icons

Randomly distribute icons within a specified area.

#### Example:
![Distribute_Icons](https://github.com/user-attachments/assets/c0a842a2-fc0f-4309-81a2-c6f0dae91e23)

---

### 🍒Text_Match

Evaluate whether the content of an image matches a given set of keywords and return a Boolean result.

#### Examples:
![QQ20241214-003531](https://github.com/user-attachments/assets/bb5e5dd0-18c7-48b0-a482-658f6bfa9ac7)

---

### 🍒YOLOWorld_Match

yolo-world calculates whether the contents of the image match the specified type and returns a Boolean result. And move to the specified folder by image classification.

#### Examples:
![YOLOWorld_Match🔍特征匹配](https://github.com/user-attachments/assets/f1966c74-6a93-499f-9ed7-224630632f42)

---

### 🍒Filter

Adjust images with tools like sharpening, defogging, contrast enhancement, natural saturation adjustment, and gamma correction.

#### Example:
![Filter](https://github.com/user-attachments/assets/c85ef0b9-4a61-4b45-9bda-90b4e2656693)

---

### 🍒HDR

#### Example:
![image](https://github.com/user-attachments/assets/591afb41-a2f6-4bab-bf8f-3436b0a431f5)

---

### 🍒Metadata

Image metadata compilation
1.It can be used to write workflow and image generation information into any image.
2.You can write the personal information or promotional information you need in the image to obtain more dissemination channels.
3.Edited images in Photoshop can also have metadata written into them to disguise them as original images.

#### Example:
![QQ20241020-145830](https://github.com/user-attachments/assets/fd448c3c-d078-4a93-87c5-41c90d93ca57)
![QQ20241020-145818](https://github.com/user-attachments/assets/2a28b95b-2361-4ca8-a323-42814bfe539f)

---

### 🍒Whitening_👧🏻

Beauty Function - Milk skin
Get rid of the oily AI texture and make your skin crystal clear and radiant.

#### Example:
![Milk skin](https://github.com/user-attachments/assets/e4bbc544-36a6-4d86-a709-c3880c4a0037)

---

### 🍒Bitch_Filter👧🏻

"Internet celebrity filter" Similar to Kodak Gold 200.

#### Example:
<img width="1413" height="718" alt="QQ20250922-142410" src="https://github.com/user-attachments/assets/a3e76d79-42cc-4cea-8db3-69acea34571a" />

---

### 🍒Pixelate_Filter🎮

Pixel art image filter.

#### Example:
<img width="2025" height="835" alt="Pixelate_Filter" src="https://github.com/user-attachments/assets/c59e266d-8a80-48bc-9c91-91eb764f2a63" />

---

### 🍒Image_Square_Pad

Extend the long side to form a square.

#### Example:
<img width="3583" height="1290" alt="Image_Square_Pad" src="https://github.com/user-attachments/assets/5fc02d0a-f6fc-407d-abba-13bf05d6c233" />

---

### 🍒Mask_Noise_Cleaner🧹

Repair the middle area of the mask.

#### Example:
<img width="2649" height="932" alt="Mask_Noise_Cleaner" src="https://github.com/user-attachments/assets/68cf652d-14ec-4873-ac23-000a722c8402" />

---

### 🍒Transparent_Area_Cropper✀

Trim the transparent area and maintain the square output.

#### Example:
<img width="2147" height="775" alt="Transparent_Area_Cropper" src="https://github.com/user-attachments/assets/fb1f7694-24a1-44f8-ae5b-8ab435626390" />

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ComfyUI_KimNodes.git
   ```

2. Install dependencies:
   
   **自动安装（推荐）**：
   - 重启 ComfyUI，依赖将自动安装
   
   **手动安装**：
   ```bash
   # 进入插件目录
   cd ComfyUI/custom_nodes/ComfyUI_KimNodes
   
   # 运行安装脚本
   python install_dependencies.py
   
   # 或者直接使用 pip
   pip install -r requirements.txt
   ```

3. Restart ComfyUI to load the nodes.

---

Usage
1.Launch ComfyUI.
2.Access the node library from the user interface.
3.Select and configure the desired node(s) based on your project requirements.

---

Community & Support
知识星球: https://t.zsxq.com/8fA0M

---

QQ 群: 852450267
For real-time discussions, join our QQ group. 
