#!/usr/bin/env python3
"""
DecisionModel 最小可运行闭环：
筛选要素抽象 → 监督构造 → 因子预测模型 → 管线路由验证。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 项目根
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from DecisionModel.config import (
    SIMILARITY_THRESHOLD,
    TEXT_ENCODER_NAME,
    BATCH_SIZE,
    LR,
    NUM_EPOCHS,
    DEVICE,
    FACTOR_HEAD_HIDDEN_DIM,
    FACTOR_HEAD_DROPOUT,
    NUM_FROZEN_LAYERS,
    DEMO_ANNOTATIONS_JSONL,
    DEFAULT_FILTER_FACTORS_JSON,
)
from DecisionModel.data.schemas import ImageFilterSample
from DecisionModel.filter_factor_abstraction import TextEncoder, run_abstraction_pipeline
from DecisionModel.label_construction import build_labeled_samples
from DecisionModel.factor_prediction import (
    VisionEncoder,
    FactorPredictionModel,
    multi_label_bce_loss,
)
from DecisionModel.routing_sanity_check import Router, run_sanity_check, SanityCheckReport


def load_demo_samples(data_path: Path) -> list[ImageFilterSample]:
    """从 JSONL 加载 demo 样本：每行 image_path, filter_factor_texts[, image_id]。"""
    samples = []
    if not data_path.exists():
        return samples
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(
                ImageFilterSample(
                    image_path=obj["image_path"],
                    filter_factor_texts=obj.get("filter_factor_texts", []),
                    image_id=obj.get("image_id"),
                )
            )
    return samples


def load_filter_texts_from_json(json_path: Path) -> list[str]:
    """从 filter_factors_list.json 等格式加载筛选要素文本列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("filter_factors", [])


def build_synthetic_samples_if_needed(
    annotations_path: Path,
    factors_json_path: Path | None,
    base_path: Path | None = None,
    num_samples: int = 30,
) -> list[ImageFilterSample]:
    """若无 annotations.jsonl，则用 filter_factors 列表构造合成样本（用于 demo）。"""
    if annotations_path.exists():
        return load_demo_samples(annotations_path)
    factors = (
        load_filter_texts_from_json(factors_json_path)
        if factors_json_path and factors_json_path.exists()
        else []
    )
    if not factors:
        factors = [
            "clear and identifiable concept",
            "at least one visually salient object",
            "multiple objects of different semantic types",
            "challenging visual conditions (low light, blur, reflections)",
            "foreground-background distinction",
        ]
    import random
    samples = []
    for i in range(num_samples):
        n_pick = random.randint(1, min(5, len(factors)))
        chosen = random.sample(factors, n_pick)
        # 使用 base_path 使路径指向 DecisionModel/demo_data/images（便于从项目根运行）
        if base_path is not None:
            image_path = str(base_path / "demo_data" / "images" / f"sample_{i:03d}.jpg")
        else:
            image_path = f"demo_data/images/sample_{i:03d}.jpg"
        samples.append(
            ImageFilterSample(
                image_path=image_path,
                filter_factor_texts=chosen,
                image_id=f"img_{i}",
            )
        )
    return samples


def main() -> None:
    print("=== DecisionModel 最小可运行闭环 ===\n")

    # 路径（使用 config 中的 DEMO 与筛选要素列表）
    annotations_path = Path(DEMO_ANNOTATIONS_JSONL)
    factors_json = DEFAULT_FILTER_FACTORS_JSON if DEFAULT_FILTER_FACTORS_JSON.exists() else None

    # 1) 加载样本（images + 筛选要素文本）
    samples = build_synthetic_samples_if_needed(
        annotations_path,
        factors_json,
        base_path=ROOT,
        num_samples=40,
    )
    if not samples:
        print("未找到 demo 数据，请提供 demo_data/annotations.jsonl 或 filter_factors 列表。")
        return
    all_texts = []
    for s in samples:
        all_texts.extend(s.filter_factor_texts)
    print(f"样本数: {len(samples)}, 原始筛选要素文本种类: {len(set(all_texts))}")

    # 2) 筛选要素抽象
    print("\n--- 筛选要素抽象 ---")
    encoder = TextEncoder(model_name=TEXT_ENCODER_NAME, device=DEVICE)
    abstraction = run_abstraction_pipeline(
        all_texts,
        encoder,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    num_primitives = len(abstraction.primitives)
    print(f"抽象后原语数（factor_id 数）: {num_primitives}")
    for p in abstraction.primitives[:5]:
        print(f"  {p.factor_id}: {p.original_texts[:1]}")

    # 3) 监督构造：multi-hot 标签
    print("\n--- 监督构造 ---")
    labeled = build_labeled_samples(samples, abstraction)
    factor_ids_ordered = [p.factor_id for p in abstraction.primitives]
    factor_id_to_idx = {fid: i for i, fid in enumerate(factor_ids_ordered)}
    print(f"已构造 {len(labeled)} 条带 multi-hot 的样本, 标签维度: {num_primitives}")

    # 4) 因子预测模型：数据加载 + 训练
    print("\n--- 因子预测模型 ---")
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("跳过训练：未安装 torch。将使用随机预测做路由验证。")
        predictions = [
            labeled[i].factor_ids[:3] for i in range(len(labeled))
        ]  # 用 GT 前几个模拟预测
        router = Router(abstraction)
        report = run_sanity_check(
            predictions,
            labeled,
            router,
            factor_id_to_idx,
            top_k=5,
        )
        print_sanity_report(report)
        return

    class SimpleImageFactorDataset(Dataset):
        def __init__(self, labeled_list, num_factors, image_size=(224, 224)):
            self.labeled_list = labeled_list
            self.num_factors = num_factors
            self.image_size = image_size

        def __len__(self):
            return len(self.labeled_list)

        def __getitem__(self, idx):
            import torch
            from PIL import Image
            s = self.labeled_list[idx]
            try:
                img = Image.open(s.image_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", self.image_size, color=(128, 128, 128))
            img = img.resize(self.image_size)
            from torchvision import transforms
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            x = T(img)
            y = torch.tensor(s.multi_hot, dtype=torch.float32)
            return x, y

    dataset = SimpleImageFactorDataset(labeled, num_primitives)
    dataloader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=0,
    )
    vision_encoder = VisionEncoder(
        backbone_name="resnet18",
        pretrained=True,
        num_frozen_layers=NUM_FROZEN_LAYERS,
    )
    model = FactorPredictionModel(
        vision_encoder=vision_encoder,
        num_factors=num_primitives,
        hidden_dim=FACTOR_HEAD_HIDDEN_DIM,
        dropout=FACTOR_HEAD_DROPOUT,
    )
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = multi_label_bce_loss(out["factor_logits"], y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} loss: {total_loss/len(dataloader):.4f}")

    # 预测：sigmoid > 0.5 的 factor_id（逐样本，保证顺序与 labeled 一致）
    model.eval()
    threshold = 0.5
    inference_loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=False,
        num_workers=0,
    )
    predictions = []
    with torch.no_grad():
        for x, _ in inference_loader:
            x = x.to(DEVICE)
            probs = model.predict_probs(x)
            for i in range(probs.size(0)):
                pred_idx = (probs[i] > threshold).nonzero(as_tuple=True)[0].cpu().tolist()
                pred_ids = [factor_ids_ordered[j] for j in pred_idx]
                predictions.append(pred_ids)

    # 保存模型与抽象结果（供推理复用）
    checkpoint_dir = ROOT / "demo_data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "factor_ids_ordered": factor_ids_ordered,
            "num_factors": num_primitives,
        },
        checkpoint_dir / "factor_model.pt",
    )
    print(f"  已保存 checkpoint: {checkpoint_dir / 'factor_model.pt'}")

    # 5) 管线路由与验证
    print("\n--- 管线路由与验证 ---")
    router = Router(abstraction)
    report = run_sanity_check(
        predictions,
        labeled,
        router,
        factor_id_to_idx,
        top_k=5,
    )
    print_sanity_report(report)

    # 样例：对第一张图执行原语
    if labeled and predictions:
        first_ids = predictions[0]
        specs = router.get_primitive_specs(first_ids)
        print("\n首张图预测的筛选原语（可解释）:")
        for sp in specs[:5]:
            print(f"  {sp.factor_id}: {sp.description[:60]}...")


def print_sanity_report(report: SanityCheckReport) -> None:
    print(f"  样本数: {report.num_samples}, 原语数: {report.num_factors}")
    print(f"  可解释预测数: {report.interpretable_count}, 平均预测 factor 数: {report.avg_predicted_factors:.2f}")
    print(f"  通过: {report.passed}")
    for image_id, pred_ids, gt_ids in report.sample_predictions[:3]:
        print(f"    样例 {image_id}: pred={pred_ids[:3]}... gt={gt_ids[:3]}...")


if __name__ == "__main__":
    main()
