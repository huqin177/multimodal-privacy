# -*- coding: utf-8 -*-
import cv2
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

print("=" * 60)
print("多模态个人隐私信息识别可视化系统")
print("=" * 60)

# ---------------------- 1. 模型初始化 ----------------------
print("\n[1] 初始化模型...")

# Bi-GRU
class BiGRUEncoder(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(5000, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_dim * 2, 256)

    def forward(self, x):
        x = self.embedding(x)
        out, h = self.gru(x)
        feat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(feat)

bigru_model = BiGRUEncoder()
bigru_model.eval()
print("    ✓ Bi-GRU 双向语义编码加载完成")

# 使用 EasyOCR
print("\n[2] 加载 EasyOCR 模型...")
import easyocr
ocr = easyocr.Reader(['en'], gpu=False)
print("    ✓ EasyOCR 加载成功")

# 视觉特征
cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("    ✓ ResNet18 视觉特征模型加载成功")

# ---------------------- 2. 完整版隐私信息提取函数 ----------------------
def extract_privacy_info(text):
    privacy = {
        "姓名": None,
        "性别": None,
        "国籍": None,
        "出生日期": None,
        "住址": None,
        "身份证号": None
    }
    
    # 1. 提取姓名
    name_match = re.search(r'NAME\s*\n\s*([A-Za-z]+)', text, re.IGNORECASE)
    if name_match:
        privacy["姓名"] = name_match.group(1).strip()
    
    # 2. 提取性别
    gender_match = re.search(r'GENDER\s*[;:]\s*[\']?\s*([A-Za-z]+)', text, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).strip()
        privacy["性别"] = '女' if gender.upper() == 'FEMALE' else gender
    
    # 3. 提取国籍
    nation_match = re.search(r'NATIONALITY\s*[:;]\s*([A-Za-z]+)', text, re.IGNORECASE)
    if nation_match:
        privacy["国籍"] = nation_match.group(1).strip().upper()
    
    # 4. 提取出生日期（处理换行）
    date_match = re.search(r'BIRTH\s*\n\s*([\d\.]+)\s*\.?\s*([\d\.]+)\s*\.?\s*([\d\.]+)', text, re.IGNORECASE)
    if date_match:
        year = date_match.group(1).strip().rstrip('.')
        month = date_match.group(2).strip().rstrip('.')
        day = date_match.group(3).strip().rstrip('.')
        privacy["出生日期"] = f"{year}.{month}.{day}"
    else:
        date_match2 = re.search(r'(\d{4}\.\d{1,2}\.\d{1,2})', text)
        if date_match2:
            privacy["出生日期"] = date_match2.group(1)
    
    # 5. 提取地址（关键修复：匹配 ADDRESS = 后面换行的内容）
    # 匹配 ADDRESS = 然后换行，然后捕获直到遇到下一个空行或数字行
    address_match = re.search(
        r'ADDRESS\s*[=:;]\s*\n\s*([A-Za-z0-9\s,\.\']+?)(?=\n\s*\n|\n\s*ID|\n\s*[A-Z]|\Z)', 
        text, 
        re.IGNORECASE | re.DOTALL
    )
    if not address_match:
        # 备用：匹配 ADDRESS 后直到行尾
        address_match = re.search(r'ADDRESS\s*[=:;]\s*([A-Za-z0-9\s,\.\']+)', text, re.IGNORECASE)
    
    if address_match:
        address = address_match.group(1).strip()
        # 清理换行和多余空格
        address = re.sub(r'\s+', ' ', address)
        # 清理特殊字符（保留字母、数字、空格、逗号、点）
        address = re.sub(r'[^\w\s,\.]', '', address)
        # 合并 Springfiled, 和 IL 之间的内容
        address = re.sub(r'\s+', ' ', address)
        privacy["住址"] = address[:200]
    
    # 6. 提取身份证号
    id_match = re.search(r'ID\s*Number\s*\n\s*(\d+)', text, re.IGNORECASE)
    if id_match:
        privacy["身份证号"] = id_match.group(1).strip()
    else:
        id_match2 = re.search(r'\b(\d{15,18})\b', text)
        if id_match2:
            privacy["身份证号"] = id_match2.group(1)
    
    return privacy

def text2vec(text, max_len=32):
    seq = [ord(c) % 5000 for c in text[:max_len]]
    seq += [0] * (max_len - len(seq))
    return torch.tensor([seq], dtype=torch.long)

def get_semantic_feature(text):
    with torch.no_grad():
        vec = bigru_model(text2vec(text))
    return vec.numpy()

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ---------------------- 3. 主流程 ----------------------
img_path = "B.png"
if not os.path.exists(img_path):
    fs = [f for f in os.listdir() if f.endswith(('png', 'jpg', 'jpeg'))]
    img_path = fs[0] if fs else None

if not img_path:
    print("错误：没有找到图片文件！")
    exit(1)

img = cv2.imread(img_path)
if img is None:
    print(f"错误：无法读取图片 {img_path}")
    exit(1)

print(f"\n[3] 读取图片：{img_path} (大小: {img.shape})")

img_enhanced = enhance_image(img)
print(f"[4] 图片增强完成")

print(f"[5] 开始 OCR 识别...")

full_text = ""
try:
    result = ocr.readtext(img_enhanced)
    if not result:
        result = ocr.readtext(img)
    
    if result:
        for line in result:
            full_text += line[1] + "\n"
        print(f"    ✅ 识别到 {len(result)} 个文本区域")
    else:
        print("    ⚠ 未识别到文字")
except Exception as e:
    print(f"    ⚠ OCR 识别失败: {e}")

if full_text:
    print(f"\n[6] OCR 完整文本:")
    print("-" * 40)
    print(full_text)
    print("-" * 40)
    
    # Bi-GRU 语义特征
    feat = get_semantic_feature(full_text)
    np.save("full_text_semantic.npy", feat)
    print(f"\n  ✅ 全文 Bi-GRU 语义向量已保存 (维度: {feat.shape})")
    
    # 提取隐私信息
    privacy = extract_privacy_info(full_text)
    
    print(f"\n[7] 提取的隐私信息:")
    print("-" * 40)
    for key, value in privacy.items():
        if value:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: 未检测到")
    
    # 保存结果
    with open("privacy_result.txt", "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("多模态隐私信息识别结果\n")
        f.write("=" * 50 + "\n\n")
        f.write("【OCR 原始文本】\n")
        f.write("-" * 30 + "\n")
        f.write(full_text)
        f.write("\n\n【提取的隐私信息】\n")
        f.write("-" * 30 + "\n")
        for key, value in privacy.items():
            f.write(f"{key}: {value if value else '未检测到'}\n")
    
    print(f"\n  ✅ 结果已保存到: privacy_result.txt")
else:
    print("\n❌ OCR 未能识别出任何文字")

print("\n" + "=" * 60)
print("程序运行完成！")
print("=" * 60)