#!/usr/bin/env python3
"""
统筹脚本：在同一套数据与 train/test 划分下，分别训练并测试 ResNet+MLP 与 Qwen2-VL，
输出两份训练+测试结果并写入报告，便于对比两种模型。

使用前请在本文件顶部「需要补全」区域填写/修改配置。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# 需要补全 / CONFIG：请按你的环境修改以下变量
# =============================================================================

# 训练/测试数据
# - 训练+测试共用一份标注时：只填 ANNOTATIONS_PATH，脚本会按 TRAIN_RATIO 划分
ANNOTATIONS_PATH: Optional[Path] = Path("DecisionModel/demo_real_data/annotations.jsonl")  # 转换工具输出；None 则用 demo_data/annotations.jsonl
# - 若有单独测试集，可后续在代码里支持 TEST_ANNOTATIONS_PATH（当前版本用划分）

# 筛选要素列表（用于无 ANNOTATIONS_PATH 时生成合成数据）
FACTORS_JSON_PATH: Optional[Path] = None  # 例如 Path("ProbingFactorGeneration/configs/filter_factors_list.json")

# 划分与随机种子
TRAIN_RATIO: float = 0.8  # 训练集比例，剩余为测试集
RANDOM_SEED: int = 42

# 设备与训练规模
DEVICE: str = "cpu"  # 填 "cuda" 若有 GPU
NUM_EPOCHS: int = 10
BATCH_SIZE: int = 16

# 要跑哪两个模型（可只跑一个）
RUN_RESNET: bool = True
RUN_QWEN2VL: bool = False  # 需 transformers + 足够显存，无 GPU 可保持 False

# 报告输出路径（None 则只打印不写文件）
OUTPUT_REPORT_PATH: Optional[Path] = None  # 例如 Path("DecisionModel/demo_data/report_both_models.json")

# 无标注文件时，合成样本数量
SYNTHETIC_NUM_SAMPLES: int = 60  # 至少约 20+ 以便划分 train/test

# =============================================================================
# 以下为脚本逻辑，一般无需修改
# =============================================================================

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from DecisionModel.config import (
    DEMO_ANNOTATIONS_JSONL,
    DEFAULT_FILTER_FACTORS_JSON,
    SIMILARITY_THRESHOLD,
    TEXT_ENCODER_NAME,
    NUM_FROZEN_LAYERS,
    FACTOR_HEAD_HIDDEN_DIM,
    FACTOR_HEAD_DROPOUT,
    QWEN2VL_MODEL_NAME,
    QWEN2VL_FREEZE_BACKBONE,
    QWEN2VL_MIN_PIXELS,
    QWEN2VL_MAX_PIXELS,
    QWEN2VL_LR_HEAD,
    QWEN2VL_LR_FULL,
)
from DecisionModel.data.schemas import ImageFilterSample, LabeledSample
from DecisionModel.filter_factor_abstraction import TextEncoder, run_abstraction_pipeline
from DecisionModel.label_construction import build_labeled_samples
from DecisionModel.eval_metrics import compute_multilabel_metrics, predictions_to_multi_hot


def load_samples(
    annotations_path: Optional[Path],
    factors_json_path: Optional[Path],
    base_path: Path,
    num_synthetic: int,
) -> List[ImageFilterSample]:
    """加载或生成样本。"""
    def _load_jsonl(p: Path) -> List[ImageFilterSample]:
        out = []
        if not p or not Path(p).exists():
            return out
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.append(
                    ImageFilterSample(
                        image_path=obj["image_path"],
                        filter_factor_texts=obj.get("filter_factor_texts", []),
                        image_id=obj.get("image_id"),
                    )
                )
        return out

    def _load_factors(p: Optional[Path]) -> List[str]:
        if not p or not Path(p).exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f).get("filter_factors", [])

    if annotations_path and Path(annotations_path).exists():
        return _load_jsonl(Path(annotations_path))
    # 合成数据时使用的筛选要素列表（来自 config 或传入的 factors_json_path）
    factors_path = factors_json_path if factors_json_path and Path(factors_json_path).exists() else DEFAULT_FILTER_FACTORS_JSON
    factors = _load_factors(factors_path)
    if not factors:
        factors = [
            "clear and identifiable concept",
            "at least one visually salient object",
            "multiple objects of different semantic types",
            "challenging visual conditions (low light, blur, reflections)",
            "foreground-background distinction",
        ]
    samples = []
    for i in range(num_synthetic):
        n_pick = random.randint(1, min(5, len(factors)))
        chosen = random.sample(factors, n_pick)
        image_path = str(base_path / "demo_data" / "images" / f"sample_{i:03d}.jpg")
        samples.append(
            ImageFilterSample(image_path=image_path, filter_factor_texts=chosen, image_id=f"img_{i}")
        )
    return samples


def train_test_split(
    labeled: List[LabeledSample],
    train_ratio: float,
    seed: int,
) -> Tuple[List[LabeledSample], List[LabeledSample]]:
    """按比例划分训练/测试，保证两类都有样本。"""
    n = len(labeled)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    k = max(1, int(n * train_ratio))
    k = min(k, n - 1)  # 至少留 1 个测试
    train_idx = set(indices[:k])
    train_list = [labeled[i] for i in range(n) if i in train_idx]
    test_list = [labeled[i] for i in range(n) if i not in train_idx]
    return train_list, test_list


def evaluate_predictions(
    labeled: List[LabeledSample],
    predictions: List[List[str]],
    factor_ids_ordered: List[str],
) -> Dict[str, float]:
    """将预测的 factor_id 列表与 labeled 的 multi_hot 对比，返回多标签指标。"""
    import torch
    pred_multi = predictions_to_multi_hot(predictions, factor_ids_ordered)
    true_multi = torch.tensor([s.multi_hot for s in labeled], dtype=torch.float32)
    return compute_multilabel_metrics(pred_multi, true_multi, threshold=0.5)


def train_and_eval_resnet(
    train_labeled: List[LabeledSample],
    test_labeled: List[LabeledSample],
    abstraction,
    factor_ids_ordered: List[str],
    device: str,
    num_epochs: int,
    batch_size: int,
) -> Dict[str, object]:
    """训练 ResNet+MLP 并在测试集上评估，返回指标与元信息。"""
    import torch
    from torch.utils.data import DataLoader

    from DecisionModel.factor_prediction import VisionEncoder, FactorPredictionModel, multi_label_bce_loss

    num_factors = len(factor_ids_ordered)

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, labeled_list, image_size=(224, 224)):
            self.labeled_list = labeled_list
            self.image_size = image_size

        def __len__(self):
            return len(self.labeled_list)

        def __getitem__(self, idx):
            from PIL import Image
            from torchvision import transforms
            s = self.labeled_list[idx]
            try:
                img = Image.open(s.image_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", self.image_size, color=(128, 128, 128))
            img = img.resize(self.image_size)
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            x = T(img)
            y = torch.tensor(s.multi_hot, dtype=torch.float32)
            return x, y

    train_ds = _Dataset(train_labeled)
    test_ds = _Dataset(test_labeled)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=min(batch_size, len(test_ds)),
        shuffle=False,
        num_workers=0,
    )

    vision_encoder = VisionEncoder(
        backbone_name="resnet18",
        pretrained=True,
        num_frozen_layers=NUM_FROZEN_LAYERS,
    )
    model = FactorPredictionModel(
        vision_encoder=vision_encoder,
        num_factors=num_factors,
        hidden_dim=FACTOR_HEAD_HIDDEN_DIM,
        dropout=FACTOR_HEAD_DROPOUT,
    )
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = multi_label_bce_loss(out["factor_logits"], y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        # optional: print per-epoch train loss

    # Eval on test
    model.eval()
    threshold = 0.5
    predictions = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            probs = model.predict_probs(x)
            for i in range(probs.size(0)):
                pred_idx = (probs[i] > threshold).nonzero(as_tuple=True)[0].cpu().tolist()
                pred_ids = [factor_ids_ordered[j] for j in pred_idx]
                predictions.append(pred_ids)

    test_metrics = evaluate_predictions(test_labeled, predictions, factor_ids_ordered)
    return {
        "model": "ResNet18+MLP",
        "train_size": len(train_labeled),
        "test_size": len(test_labeled),
        "num_factors": num_factors,
        "test_metrics": test_metrics,
    }


def train_and_eval_qwen2vl(
    train_labeled: List[LabeledSample],
    test_labeled: List[LabeledSample],
    abstraction,
    factor_ids_ordered: List[str],
    device: str,
    num_epochs: int,
    batch_size: int,
) -> Optional[Dict[str, object]]:
    """训练 Qwen2-VL 并在测试集上评估。若依赖缺失则返回 None。"""
    try:
        import torch
        from torch.utils.data import DataLoader
        from DecisionModel.factor_prediction.qwen2vl_factor_model import (
            Qwen2VLFactorModel,
            build_qwen2vl_collate_fn,
        )
        from DecisionModel.factor_prediction.loss import multi_label_bce_loss
    except ImportError:
        return None

    num_factors = len(factor_ids_ordered)
    train_data = [(s.image_path, s.multi_hot) for s in train_labeled]
    test_data = [(s.image_path, s.multi_hot) for s in test_labeled]

    model = Qwen2VLFactorModel(
        model_name=QWEN2VL_MODEL_NAME,
        num_factors=num_factors,
        hidden_dim=FACTOR_HEAD_HIDDEN_DIM,
        dropout=FACTOR_HEAD_DROPOUT,
        freeze_backbone=QWEN2VL_FREEZE_BACKBONE,
        device=device,
        min_pixels=QWEN2VL_MIN_PIXELS,
        max_pixels=QWEN2VL_MAX_PIXELS,
    )
    collate_fn = build_qwen2vl_collate_fn(model.processor)
    train_loader = DataLoader(
        train_data,
        batch_size=min(batch_size, len(train_data)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=min(batch_size, len(test_data)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model.to(device)
    lr = QWEN2VL_LR_HEAD if QWEN2VL_FREEZE_BACKBONE else QWEN2VL_LR_FULL
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad()
            out = model(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            loss = multi_label_bce_loss(out["factor_logits"], labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

    model.eval()
    threshold = 0.5
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)
            probs = model.predict_probs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            for i in range(probs.size(0)):
                pred_idx = (probs[i] > threshold).nonzero(as_tuple=True)[0].cpu().tolist()
                pred_ids = [factor_ids_ordered[j] for j in pred_idx]
                predictions.append(pred_ids)

    test_metrics = evaluate_predictions(test_labeled, predictions, factor_ids_ordered)
    return {
        "model": "Qwen2-VL",
        "model_name": QWEN2VL_MODEL_NAME,
        "train_size": len(train_labeled),
        "test_size": len(test_labeled),
        "num_factors": num_factors,
        "test_metrics": test_metrics,
    }


def main() -> None:
    # 解析路径：未设置时使用 config 默认
    annotations_path = ANNOTATIONS_PATH
    if annotations_path is None:
        annotations_path = Path(DEMO_ANNOTATIONS_JSONL)
    factors_json_path = FACTORS_JSON_PATH or (DEFAULT_FILTER_FACTORS_JSON if DEFAULT_FILTER_FACTORS_JSON.exists() else None)

    print("======== 统筹：ResNet+MLP vs Qwen2-VL 训练+测试 ========\n")
    print("配置:")
    print(f"  ANNOTATIONS_PATH: {annotations_path}")
    print(f"  TRAIN_RATIO: {TRAIN_RATIO}, RANDOM_SEED: {RANDOM_SEED}")
    print(f"  DEVICE: {DEVICE}, NUM_EPOCHS: {NUM_EPOCHS}, BATCH_SIZE: {BATCH_SIZE}")
    print(f"  RUN_RESNET: {RUN_RESNET}, RUN_QWEN2VL: {RUN_QWEN2VL}")
    print()

    # 1) 加载样本 + 抽象 + 标签
    samples = load_samples(
        annotations_path,
        factors_json_path,
        base_path=ROOT,
        num_synthetic=SYNTHETIC_NUM_SAMPLES,
    )
    if not samples:
        print("未加载到任何样本，请检查 ANNOTATIONS_PATH 或 FACTORS_JSON_PATH。")
        return

    all_texts = []
    for s in samples:
        all_texts.extend(s.filter_factor_texts)
    encoder = TextEncoder(model_name=TEXT_ENCODER_NAME, device=DEVICE)
    abstraction = run_abstraction_pipeline(all_texts, encoder, similarity_threshold=SIMILARITY_THRESHOLD)
    labeled = build_labeled_samples(samples, abstraction)
    factor_ids_ordered = [p.factor_id for p in abstraction.primitives]
    num_primitives = len(factor_ids_ordered)

    train_labeled, test_labeled = train_test_split(labeled, TRAIN_RATIO, RANDOM_SEED)
    print(f"数据: 总样本 {len(labeled)}, 原语数 {num_primitives}, 训练 {len(train_labeled)}, 测试 {len(test_labeled)}\n")

    results: List[Dict[str, object]] = []

    # 2) ResNet+MLP
    if RUN_RESNET:
        print("--- ResNet18+MLP 训练与测试 ---")
        try:
            res = train_and_eval_resnet(
                train_labeled,
                test_labeled,
                abstraction,
                factor_ids_ordered,
                device=DEVICE,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
            )
            results.append(res)
            m = res["test_metrics"]
            print(f"  测试: BCE={m['bce_loss']:.4f}, SubsetAcc={m['subset_accuracy']:.4f}, F1_macro={m['f1_macro']:.4f}, F1_micro={m['f1_micro']:.4f}")
        except Exception as e:
            print(f"  ResNet 运行失败: {e}")
            results.append({"model": "ResNet18+MLP", "error": str(e)})
        print()

    # 3) Qwen2-VL
    if RUN_QWEN2VL:
        print("--- Qwen2-VL 训练与测试 ---")
        try:
            res = train_and_eval_qwen2vl(
                train_labeled,
                test_labeled,
                abstraction,
                factor_ids_ordered,
                device=DEVICE,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
            )
            if res is not None:
                results.append(res)
                m = res["test_metrics"]
                print(f"  测试: BCE={m['bce_loss']:.4f}, SubsetAcc={m['subset_accuracy']:.4f}, F1_macro={m['f1_macro']:.4f}, F1_micro={m['f1_micro']:.4f}")
            else:
                print("  Qwen2-VL 依赖缺失或加载失败，已跳过。")
                results.append({"model": "Qwen2-VL", "skipped": True})
        except Exception as e:
            print(f"  Qwen2-VL 运行失败: {e}")
            results.append({"model": "Qwen2-VL", "error": str(e)})
        print()

    # 4) 汇总表
    print("======== 汇总 ========")
    for r in results:
        name = r.get("model", "?")
        if "error" in r:
            print(f"  {name}: 错误 - {r['error']}")
        elif "skipped" in r:
            print(f"  {name}: 已跳过")
        else:
            m = r.get("test_metrics", {})
            print(f"  {name}: BCE={m.get('bce_loss', 0):.4f}, SubsetAcc={m.get('subset_accuracy', 0):.4f}, F1_macro={m.get('f1_macro', 0):.4f}, F1_micro={m.get('f1_micro', 0):.4f}")

    if OUTPUT_REPORT_PATH is not None:
        out_path = Path(OUTPUT_REPORT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": {
                "annotations_path": str(annotations_path),
                "train_ratio": TRAIN_RATIO,
                "random_seed": RANDOM_SEED,
                "device": DEVICE,
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
            },
            "data": {
                "total_samples": len(labeled),
                "train_size": len(train_labeled),
                "test_size": len(test_labeled),
                "num_factors": num_primitives,
            },
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    main()
