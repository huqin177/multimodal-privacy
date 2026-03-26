# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import easyocr
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("升级版多模态隐私信息识别系统")
print("="*60)

# ---------------------- 1. Bi-GRU 模型 ----------------------
class BiGRUEncoder(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(5000, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_dim*2, 256)

    def forward(self, x):
        x = self.embedding(x)
        out, h = self.gru(x)
        feat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(feat)

bigru_model = BiGRUEncoder()
bigru_model.eval()

def text2vec(text, max_len=32):
    seq = [ord(c) % 5000 for c in text[:max_len]]
    seq += [0] * (max_len - len(seq))
    return torch.tensor([seq], dtype=torch.long)

def get_semantic_feature(text):
    with torch.no_grad():
        vec = bigru_model(text2vec(text))
    return vec.numpy()

# ---------------------- 2. 图像特征提取 ----------------------
cnn_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn_model.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_image_feature(crop):
    with torch.no_grad():
        tensor = transform(crop).unsqueeze(0)
        feat = cnn_model(tensor)
    return feat.numpy()

# ---------------------- 3. OCR 模型 ----------------------
ocr = easyocr.Reader(['en'], gpu=False)

# ---------------------- 4. YOLOv8 检测 ----------------------
yolo_model = YOLO("yolov8n.pt")  # 轻量版模型，需要下载权重

# ---------------------- 5. 图像增强 ----------------------
def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ---------------------- 6. 主流程 ----------------------
img_path = "B.png"
if not os.path.exists(img_path):
    fs = [f for f in os.listdir() if f.endswith(('png','jpg','jpeg'))]
    img_path = fs[0] if fs else None
if not img_path:
    print("错误：未找到图片文件！")
    exit(1)

img = cv2.imread(img_path)
img_enhanced = enhance_image(img)

# YOLO 检测敏感区域
results = yolo_model(img_enhanced)

all_entities = []

for r in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = map(int, r[:6])
    crop = img_enhanced[y1:y2, x1:x2]

    # OCR
    ocr_result = ocr.readtext(crop)
    text = " ".join([line[1] for line in ocr_result])

    # 文本特征
    text_vec = get_semantic_feature(text)

    # 图像特征
    img_vec = get_image_feature(crop)

    # 跨模态匹配（这里简单存储向量，后续可与文本全局向量匹配）
    entity = {
        "bbox": (x1, y1, x2, y2),
        "ocr_text": text,
        "text_vec": text_vec,
        "img_vec": img_vec
    }
    all_entities.append(entity)

# ---------------------- 7. 可视化 ----------------------
vis_img = img.copy()
for e in all_entities:
    x1, y1, x2, y2 = e["bbox"]
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0,0,255), 2)
    cv2.putText(vis_img, e["ocr_text"][:20], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

cv2.imwrite("result_visualization.png", vis_img)
print(f"✅ 可视化结果已保存: result_visualization.png")
print(f"检测到 {len(all_entities)} 个敏感实体")

# 可选：打印 OCR 文本和向量信息
for i, e in enumerate(all_entities):
    print(f"\n实体 {i+1}:")
    print(f" OCR: {e['ocr_text']}")
    print(f" 文本向量 shape: {e['text_vec'].shape}")
    print(f" 图像向量 shape: {e['img_vec'].shape}")