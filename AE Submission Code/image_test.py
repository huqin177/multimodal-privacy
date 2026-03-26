# -*- coding: utf-8 -*-
"""
多模态图像特征包生成器（优化版 - 含全图OCR）
修复 OCR 乱码问题：简化预处理、英文优先、支持语言选择
"""
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import easyocr
from ultralytics import YOLO
import numpy as np
import re
import os
import sys
from pathlib import Path
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# ---------------------- 配置日志 ----------------------
# 设置控制台输出编码为 UTF-8（Windows 下避免显示乱码）
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ---------------------- 配置类 ----------------------
@dataclass
class Config:
    """全局配置"""
    # OCR配置
    ocr_languages: List[str] = None
    ocr_gpu: bool = False
    ocr_preprocess_mode: str = 'light'  # 'light' 或 'heavy'

    # 图像预处理配置
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)
    blur_kernel: tuple = (3, 3)
    ocr_upscale_factor: int = 2

    # YOLO配置
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence: float = 0.5
    yolo_iou: float = 0.45
    target_labels: List[str] = None

    # 特征提取配置
    text_max_len: int = 64
    text_vocab_size: int = 5000
    text_embed_dim: int = 128
    text_hidden_dim: int = 256

    image_size: int = 224
    image_cnn_out_dim: int = 512
    image_rnn_hidden: int = 256

    # 输出配置
    save_individual: bool = True
    save_json: bool = True
    save_visualization: bool = True

    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ['en', 'ch_sim']
        if self.target_labels is None:
            self.target_labels = ['person', 'id_card', 'bank_card']


# ---------------------- 1. 图像预处理（增强版）-----------------------
class ImagePreprocessor:
    """图像预处理器，支持轻量级和重量级 OCR 预处理"""

    def __init__(self, config: Config):
        self.config = config
        self.clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_grid_size
        )

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """基础图像增强（用于特征提取）"""
        if img is None:
            return None
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        enhanced = self.clahe.apply(gray)
        denoised = cv2.GaussianBlur(enhanced, self.config.blur_kernel, 0)
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """根据配置选择轻量级或重量级预处理"""
        if crop is None or crop.size == 0:
            return None

        # 通用：放大过小的图像
        h, w = crop.shape[:2]
        if h < 100 or w < 100:
            scale = self.config.ocr_upscale_factor
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 根据模式处理
        if self.config.ocr_preprocess_mode == 'light':
            # 轻量级：仅转灰度 + 轻微去噪，不破坏文字
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()
            gray = cv2.GaussianBlur(gray, (1, 1), 0)  # 轻微模糊去噪
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            # 重量级：原有的二值化 + 形态学（可能过度处理）
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((1, 1), np.uint8)
            denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


# ---------------------- 2. Bi-GRU文本编码器 ------------------------
class BiGRUEncoder(nn.Module):
    """Bi-GRU文本编码器"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.text_vocab_size, config.text_embed_dim, padding_idx=0
        )
        self.gru = nn.GRU(
            config.text_embed_dim, config.text_hidden_dim,
            bidirectional=True, batch_first=True
        )
        self.proj = nn.Linear(config.text_hidden_dim * 2, 256)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        out, h = self.gru(x)
        # 取双向GRU的最终隐藏状态
        feat = torch.cat([h[-2], h[-1]], dim=-1)
        feat = self.proj(feat)
        return feat

    def text_to_tensor(self, text: str) -> torch.Tensor:
        """文本转张量"""
        seq = [ord(c) % self.config.text_vocab_size for c in text[:self.config.text_max_len]]
        seq += [0] * (self.config.text_max_len - len(seq))
        return torch.tensor([seq], dtype=torch.long)

    def encode(self, text: str) -> np.ndarray:
        """编码文本为特征向量"""
        if not text or len(text.strip()) == 0:
            return np.zeros(256)

        with torch.no_grad():
            tensor = self.text_to_tensor(text)
            feat = self.forward(tensor)
        return feat.numpy().flatten()


# ---------------------- 3. CNN-RNN图像特征提取器 ------------------------
class CNNRNNImageEncoder(nn.Module):
    """CNN-RNN图像特征提取器"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # CNN特征提取器
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # RNN捕捉空间关系
        self.rnn = nn.GRU(
            config.image_cnn_out_dim, config.image_rnn_hidden,
            bidirectional=True, batch_first=True
        )
        self.proj = nn.Linear(config.image_rnn_hidden * 2, 256)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # CNN提取特征
        cnn_feat = self.cnn(x)  # (batch, 512, H, W)
        cnn_feat = self.adaptive_pool(cnn_feat)  # (batch, 512, 4, 4)

        # 转换为序列
        b, c, h, w = cnn_feat.shape
        seq_feat = cnn_feat.view(b, c, -1).permute(0, 2, 1)  # (batch, 16, 512)

        # RNN捕捉空间关联
        rnn_out, _ = self.rnn(seq_feat)
        final_feat = torch.mean(rnn_out, dim=1)  # (batch, hidden*2)
        final_feat = self.proj(final_feat)
        final_feat = self.dropout(final_feat)

        return final_feat

    def encode(self, image: np.ndarray) -> np.ndarray:
        """编码图像为特征向量"""
        if image is None or image.size == 0:
            return np.zeros(256)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        with torch.no_grad():
            tensor = transform(image).unsqueeze(0)
            feat = self.forward(tensor)
        return feat.numpy().flatten()


# ---------------------- 4. 敏感实体识别器（增强版）-----------------------
class SensitiveEntityRecognizer:
    """敏感实体识别器（优化版，修复银行卡号误判）"""

    def __init__(self):
        # 常见 OCR 错误替换映射（可根据实际增加）
        self.ocr_fixes = {
            'GENDB': 'GENDER',
            'Fn': 'Female',
            'BIRTH ': 'BIRTH: ',
            'ID Number': 'ID Number',
        }

        # 精确的正则模式，银行卡号只匹配明确标签，避免全局数字匹配
        self.patterns = {
            "姓名": [
                r'NAME\s*[:：]?\s*([A-Za-z\s]+?)(?=\s*(?:GENDER|NATIONALITY|BIRTH|ADDRESS|ID|$))',
            ],
            "性别": [
                r'GENDER\s*[:：]?\s*([A-Za-z]+)(?=\s*(?:NATIONALITY|BIRTH|ADDRESS|ID|$))',
            ],
            "国籍": [
                r'NATIONALITY\s*[:：]?\s*([A-Za-z\s]+?)(?=\s*(?:BIRTH|ADDRESS|ID|$))',
            ],
            "身份证号": [
                # 匹配 15-18 位数字或 X，允许横线/空格（会被清洗）
                r'ID\s*Number\s*[:：]?\s*([\dXx]{15,18})',
            ],
            "手机号": [
                r'PHONE\s*[:：]?\s*([+\d\s\-\(\)]+)',
                r'\b(1[3-9]\d{9})\b',
            ],
            "银行卡号": [
                # 只匹配明确带有 CARD Number 标签的数字，避免误判身份证号
                r'CARD\s*Number\s*[:：]?\s*(\d{16,19})',
            ],
            "出生日期": [
                r'BIRTH\s*[:：]?\s*(\d{4}[-年\.]\d{1,2}[-月\.]\d{1,2})',
                r'(\d{4}\.\d{1,2}\.\d{1,2})',
                r'(\d{4}/\d{1,2}/\d{1,2})',
            ],
            "地址": [
                r'ADDRESS\s*[:：]?\s*(.*?)(?=\s*(?:ID|$))',
            ]
        }

    def _fix_ocr_text(self, text: str) -> str:
        """修正 OCR 常见错误"""
        for wrong, right in self.ocr_fixes.items():
            text = text.replace(wrong, right)
        return text

    def recognize(self, text: str) -> Dict[str, str]:
        """识别文本中的敏感实体"""
        if not text:
            return {}

        # 预处理：合并换行、修正 OCR 错误
        text = re.sub(r'[\n\r]+', ' ', text)
        text = self._fix_ocr_text(text)

        entities = {}
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    if not value:
                        continue

                    # 清洗多余空格
                    value = re.sub(r'\s+', ' ', value)

                    # 根据类型特殊处理
                    if entity_type == "身份证号":
                        # 只保留数字和 X，去除横线、空格等
                        value = re.sub(r'[^\dXx]', '', value)
                    elif entity_type == "银行卡号":
                        # 银行卡号只保留数字
                        value = re.sub(r'[^\d]', '', value)
                    elif entity_type == "姓名":
                        # 姓名通常只取第一个单词
                        parts = value.split()
                        if parts:
                            value = parts[0]
                    elif entity_type == "地址":
                        # 地址可能很长，去除尾部多余字符（如 'T'）
                        value = re.sub(r'[Tt]\d*$', '', value).strip()

                    if value:
                        entities[entity_type] = value[:200]
                        break

        # 后处理：如果身份证号和银行卡号内容相同，删除银行卡号（防止重复）
        if "身份证号" in entities and "银行卡号" in entities:
            if entities["身份证号"] == entities["银行卡号"]:
                del entities["银行卡号"]

        return entities

# ---------------------- 5. OCR管理器（增强版）-----------------------
class OCRManager:
    """OCR管理器（单例模式）"""

    _instance = None
    _reader = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self, config: Config):
        """初始化OCR（兼容不同版本的easyocr）"""
        if self._reader is not None:
            return self._reader

        logger.info(f"初始化OCR模块，语言: {config.ocr_languages}")
        try:
            # 新版 easyocr 可能不支持 paragraph 参数
            self._reader = easyocr.Reader(
                config.ocr_languages,
                gpu=config.ocr_gpu
            )
            logger.info("✅ OCR模块初始化成功")
        except TypeError as e:
            if "paragraph" in str(e):
                try:
                    self._reader = easyocr.Reader(
                        config.ocr_languages,
                        gpu=config.ocr_gpu,
                        paragraph=True
                    )
                    logger.info("✅ OCR模块初始化成功（旧版兼容模式）")
                except Exception as e2:
                    logger.error(f"❌ OCR模块初始化失败（旧版兼容模式也失败）: {e2}")
                    self._reader = None
            else:
                logger.error(f"❌ OCR模块初始化失败: {e}")
                self._reader = None
        except Exception as e:
            logger.error(f"❌ OCR模块初始化失败: {e}")
            self._reader = None
        return self._reader

    def get_reader(self):
        return self._reader

    def extract_text(self, image: np.ndarray, preprocessor: ImagePreprocessor) -> Tuple[str, List]:
        """提取图像中的文本"""
        if self._reader is None or image is None:
            return "", []

        try:
            processed = preprocessor.preprocess_for_ocr(image)
            if processed is None:
                processed = image
            result = self._reader.readtext(processed)
            texts = [item[1] for item in result]
            clean_text = " ".join(texts)
            return clean_text, result
        except Exception as e:
            logger.debug(f"OCR错误: {e}")
            return "", []


# ---------------------- 6. YOLO管理器 ------------------------
class YOLOManager:
    """YOLO管理器（单例模式）"""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self, config: Config):
        """初始化YOLO"""
        if self._model is not None:
            return self._model

        logger.info(f"初始化YOLO模型: {config.yolo_model_path}")
        try:
            self._model = YOLO(config.yolo_model_path)
            logger.info("✅ YOLO模型初始化成功")
        except Exception as e:
            logger.error(f"❌ YOLO模型初始化失败: {e}")
            self._model = None
        return self._model

    def get_model(self):
        return self._model

    def detect(self, image: np.ndarray, config: Config) -> List[Dict]:
        """检测敏感物体"""
        if self._model is None or image is None:
            return []

        results = self._model(image, conf=config.yolo_confidence, iou=config.yolo_iou)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls in zip(boxes, confs, classes):
                label = self._model.names[cls]

                # 检查是否为敏感物体（person + 其他）
                if 'person' in config.target_labels and label == 'person':
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "confidence": float(conf),
                        "area": (x2 - x1) * (y2 - y1)  # 面积
                    })
                # 可以扩展其他敏感物体检测

        # 按面积排序（大物体优先）
        detections.sort(key=lambda x: x.get('area', 0), reverse=True)
        return detections


# ---------------------- 7. 特征包生成器主类 ------------------------
class FeaturePackGenerator:
    """特征包生成器主类"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

        # 初始化各组件
        self.preprocessor = ImagePreprocessor(self.config)
        self.ocr_manager = OCRManager()
        self.yolo_manager = YOLOManager()
        self.entity_recognizer = SensitiveEntityRecognizer()

        # 初始化模型
        self._init_models()

    def _init_models(self):
        """初始化所有模型"""
        self.ocr_manager.init(self.config)
        self.yolo_manager.init(self.config)

        self.text_encoder = BiGRUEncoder(self.config)
        self.text_encoder.eval()

        self.image_encoder = CNNRNNImageEncoder(self.config)
        self.image_encoder.eval()

        logger.info("所有模型初始化完成")

    def process_image(self, img_path: str) -> Optional[List[Dict]]:
        """处理单张图片（包含全图OCR）"""
        logger.info(f"处理图片: {img_path}")

        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"无法读取图片: {img_path}")
            return None

        # 图像增强
        img_enhanced = self.preprocessor.enhance(img)

        # ========== 全图OCR ==========
        full_ocr_text, full_ocr_details = self.ocr_manager.extract_text(img_enhanced, self.preprocessor)
        full_entities = self.entity_recognizer.recognize(full_ocr_text)
        full_text_vec = self.text_encoder.encode(full_ocr_text)
        full_img_vec = self.image_encoder.encode(img_enhanced)

        # 全图特征包
        full_feature_pack = {
            "image_name": Path(img_path).name,
            "bbox": "full_image",
            "label": "full_image",
            "confidence": 1.0,
            "ocr_text": full_ocr_text,
            "entities": full_entities,
            "text_vector": full_text_vec.tolist(),
            "image_vector": full_img_vec.tolist(),
            "combined_vector": np.concatenate([full_text_vec, full_img_vec]).tolist(),
            "metadata": {
                "crop_size": img_enhanced.shape[:2],
                "ocr_detections": len(full_ocr_details),
                "timestamp": datetime.now().isoformat(),
                "type": "full_image"
            }
        }

        feature_packs = [full_feature_pack]  # 始终包含全图特征

        # YOLO检测
        detections = self.yolo_manager.detect(img_enhanced, self.config)
        logger.info(f"检测到 {len(detections)} 个敏感区域")

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            crop = img_enhanced[y1:y2, x1:x2]

            ocr_text, ocr_details = self.ocr_manager.extract_text(crop, self.preprocessor)
            entities = self.entity_recognizer.recognize(ocr_text)
            text_vec = self.text_encoder.encode(ocr_text)
            img_vec = self.image_encoder.encode(crop)

            feature_pack = {
                "image_name": Path(img_path).name,
                "bbox": det['bbox'],
                "label": det['label'],
                "confidence": det['confidence'],
                "ocr_text": ocr_text,
                "entities": entities,
                "text_vector": text_vec.tolist(),
                "image_vector": img_vec.tolist(),
                "combined_vector": np.concatenate([text_vec, img_vec]).tolist(),
                "metadata": {
                    "crop_size": crop.shape[:2],
                    "ocr_detections": len(ocr_details),
                    "timestamp": datetime.now().isoformat(),
                    "type": "yolo_crop"
                }
            }
            feature_packs.append(feature_pack)

            if entities:
                logger.info(f"  区域{i+1}: 识别到 {list(entities.keys())}")
            elif ocr_text:
                logger.info(f"  区域{i+1}: 文字 '{ocr_text[:50]}...'")
            else:
                logger.info(f"  区域{i+1}: 未识别到文字")

        # 打印全图识别结果
        if full_entities:
            logger.info(f"全图识别到敏感实体: {list(full_entities.keys())}")

        return feature_packs

    def batch_process(self, input_dir: str, output_dir: str = None) -> List[Dict]:
        """批量处理"""
        input_path = Path(input_dir)
        if output_dir is None:
            output_dir = input_path / "output"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取图片文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))
        logger.info(f"找到 {len(image_files)} 张图片")

        if not image_files:
            logger.warning("未找到图片文件")
            return []

        # 批量处理
        all_results = []
        for img_path in image_files:
            try:
                features = self.process_image(img_path)
                if features:  # 总是有全图特征，所以不会为None
                    result = {
                        "image": str(img_path),
                        "features": features
                    }
                    all_results.append(result)

                    # 保存单个结果
                    if self.config.save_individual:
                        output_file = output_dir / f"{img_path.stem}_features.npy"
                        np.save(output_file, features, allow_pickle=True)
                        logger.info(f"  已保存: {output_file}")
            except Exception as e:
                logger.error(f"处理失败 {img_path}: {e}")
                continue

        # 保存汇总结果
        if all_results and self.config.save_json:
            summary_file = output_dir / "all_features_summary.npy"
            np.save(summary_file, all_results, allow_pickle=True)

            # 保存JSON摘要
            json_file = output_dir / "all_features_summary.json"
            json_data = []
            for res in all_results:
                json_data.append({
                    "image": res["image"],
                    "feature_count": len(res["features"]),
                    "features": [
                        {
                            "label": f["label"],
                            "confidence": f["confidence"],
                            "ocr_text": f["ocr_text"][:100],
                            "entities": f["entities"]
                        }
                        for f in res["features"]
                    ]
                })
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ JSON摘要已保存: {json_file}")

        # 可视化
        if self.config.save_visualization:
            self.visualize_results(all_results, output_dir)

        logger.info(f"处理完成: 成功 {len(all_results)}/{len(image_files)}")
        return all_results

    def visualize_results(self, all_results: List[Dict], output_dir: Path):
        """可视化结果（跳过全图特征包）"""
        for res in all_results:
            img_path = res["image"]
            features = res["features"]

            img = cv2.imread(img_path)
            if img is None:
                continue

            for fp in features:
                # 跳过全图特征包（没有bbox）
                if fp.get('label') == 'full_image':
                    continue
                x1, y1, x2, y2 = fp['bbox']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                label_text = f"{fp['label']} {fp['confidence']:.2f}"
                cv2.putText(img, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if fp['entities']:
                    entities_text = ", ".join([f"{k}:{v}" for k, v in fp['entities'].items()])
                    cv2.putText(img, entities_text[:50], (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            vis_path = output_dir / f"{Path(img_path).stem}_visualized.png"
            cv2.imwrite(str(vis_path), img)

        logger.info(f"可视化完成，保存至: {output_dir}")


# ---------------------- 8. 命令行入口 ----------------------
def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='多模态图像特征包生成器')
    parser.add_argument('input_dir', type=str, help='输入图片目录')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='输出目录')
    parser.add_argument('--conf', '-c', type=float, default=0.5, help='YOLO置信度阈值')
    parser.add_argument('--gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--ocr-lang', type=str, default='en,ch_sim',
                        help='OCR语言，用逗号分隔，如 "en,ch_sim" 或 "en"')
    parser.add_argument('--preprocess-mode', choices=['light', 'heavy'], default='light',
                        help='OCR预处理模式：light（推荐）或 heavy（原方案）')

    args = parser.parse_args()

    # 更新配置
    config = Config()
    config.yolo_confidence = args.conf
    config.ocr_gpu = args.gpu
    config.ocr_preprocess_mode = args.preprocess_mode
    # 解析语言
    config.ocr_languages = [lang.strip() for lang in args.ocr_lang.split(',')]

    # 检查目录
    if not os.path.exists(args.input_dir):
        logger.error(f"目录不存在: {args.input_dir}")
        return

    # 运行
    generator = FeaturePackGenerator(config)
    generator.batch_process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()