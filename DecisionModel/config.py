"""
DecisionModel demo 全局配置：路径、编码器、阈值等。
"""

from pathlib import Path

# 项目根目录（DecisionModel 的父目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认筛选要素列表（可与 ProbingFactorGeneration 对齐）
DEFAULT_FILTER_FACTORS_JSON = PROJECT_ROOT / "ProbingFactorGeneration" / "configs" / "filter_factors_list.json"

# 文本编码器：sentence-level 或 CLIP
# 可选: "sentence-transformers/all-MiniLM-L6-v2" | "openai/clip-vit-base-patch32" 等
TEXT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM = 384  # all-MiniLM-L6-v2 输出维度；CLIP 需改为 512

# 相似度图：余弦相似度阈值，超过则连边
SIMILARITY_THRESHOLD = 0.75

# 聚类：若不用图连通分量，可用简单聚类（如 Agglomerative）
USE_GRAPH_COMPONENTS = True  # True = 阈值图 + 连通分量

# 视觉编码器
VISION_ENCODER_NAME = "resnet18"  # 或 "openai/clip-vit-base-patch32" 的 vision 部分
VISION_EMBED_DIM = 512
FREEZE_VISION_LAYERS = True
NUM_FROZEN_LAYERS = 3  # ResNet 前几层冻结

# 因子预测头
FACTOR_HEAD_HIDDEN_DIM = 256
FACTOR_HEAD_DROPOUT = 0.1

# 训练
BATCH_SIZE = 16
LR = 1e-3
NUM_EPOCHS = 10
DEVICE = "cpu"  # 或 "cuda"

# Demo 数据
DEMO_IMAGES_DIR = PROJECT_ROOT / "DecisionModel" / "demo_data" / "images"
DEMO_ANNOTATIONS_JSONL = PROJECT_ROOT / "DecisionModel" / "demo_data" / "annotations.jsonl"

# Qwen2-VL 因子预测（可选，替换 ResNet 骨干）
QWEN2VL_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"  # 或 7B，显存不足可用 2B
QWEN2VL_FREEZE_BACKBONE = True  # 仅训练多标签头时设为 True
QWEN2VL_MIN_PIXELS = 256 * 28 * 28  # 降低可省显存
QWEN2VL_MAX_PIXELS = 1024 * 28 * 28
QWEN2VL_LR_HEAD = 1e-3  # 仅训练 head 时的学习率
QWEN2VL_LR_FULL = 1e-5  # 全量微调时的学习率
