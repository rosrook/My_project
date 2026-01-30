#!/usr/bin/env python3
"""
DecisionModel + Qwen2-VL 训练：与 run_demo 相同的数据（图片 + 筛选要素 id 组合），
仅将骨干从 ResNet 换为 Qwen2-VL，监督仍为 multi-hot BCE。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from DecisionModel.config import (
    SIMILARITY_THRESHOLD,
    TEXT_ENCODER_NAME,
    DEMO_ANNOTATIONS_JSONL,
    DEFAULT_FILTER_FACTORS_JSON,
    QWEN2VL_MODEL_NAME,
    QWEN2VL_FREEZE_BACKBONE,
    QWEN2VL_MIN_PIXELS,
    QWEN2VL_MAX_PIXELS,
    QWEN2VL_LR_HEAD,
    QWEN2VL_LR_FULL,
    FACTOR_HEAD_HIDDEN_DIM,
    FACTOR_HEAD_DROPOUT,
    BATCH_SIZE,
    NUM_EPOCHS,
    DEVICE,
)
from DecisionModel.data.schemas import ImageFilterSample
from DecisionModel.filter_factor_abstraction import TextEncoder, run_abstraction_pipeline
from DecisionModel.label_construction import build_labeled_samples
from DecisionModel.factor_prediction.qwen2vl_factor_model import (
    Qwen2VLFactorModel,
    build_qwen2vl_collate_fn,
)
from DecisionModel.factor_prediction.loss import multi_label_bce_loss
from DecisionModel.routing_sanity_check import Router, run_sanity_check, SanityCheckReport


def load_demo_samples(data_path: Path) -> list[ImageFilterSample]:
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
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("filter_factors", [])


def build_synthetic_samples_if_needed(
    annotations_path: Path,
    factors_json_path: Path | None,
    base_path: Path | None = None,
    num_samples: int = 30,
) -> list[ImageFilterSample]:
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
        if base_path is not None:
            image_path = str(base_path / "demo_data" / "images" / f"sample_{i:03d}.jpg")
        else:
            image_path = f"demo_data/images/sample_{i:03d}.jpg"
        samples.append(
            ImageFilterSample(image_path=image_path, filter_factor_texts=chosen, image_id=f"img_{i}")
        )
    return samples


def main() -> None:
    print("=== DecisionModel + Qwen2-VL 训练（图片 + 筛选要素 id，BCE 监督）===\n")

    annotations_path = Path(DEMO_ANNOTATIONS_JSONL)
    factors_json = DEFAULT_FILTER_FACTORS_JSON if DEFAULT_FILTER_FACTORS_JSON.exists() else None
    samples = build_synthetic_samples_if_needed(
        annotations_path, factors_json, base_path=ROOT, num_samples=40
    )
    if not samples:
        print("未找到 demo 数据。")
        return

    all_texts = []
    for s in samples:
        all_texts.extend(s.filter_factor_texts)
    print(f"样本数: {len(samples)}, 原始筛选要素种类: {len(set(all_texts))}")

    # 抽象 + 标签（与 run_demo 一致）
    encoder = TextEncoder(model_name=TEXT_ENCODER_NAME, device=DEVICE)
    abstraction = run_abstraction_pipeline(all_texts, encoder, similarity_threshold=SIMILARITY_THRESHOLD)
    num_primitives = len(abstraction.primitives)
    labeled = build_labeled_samples(samples, abstraction)
    factor_ids_ordered = [p.factor_id for p in abstraction.primitives]
    factor_id_to_idx = {fid: i for i, fid in enumerate(factor_ids_ordered)}
    print(f"抽象后原语数: {num_primitives}")

    # 数据集：list of (image_path, multi_hot)
    dataset = [(s.image_path, s.multi_hot) for s in labeled]

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("未安装 torch，跳过 Qwen2-VL 训练。")
        return

    # 先建 model 以拿到 processor，再建 collate
    print("\n加载 Qwen2-VL 模型与 processor...")
    model = Qwen2VLFactorModel(
        model_name=QWEN2VL_MODEL_NAME,
        num_factors=num_primitives,
        hidden_dim=FACTOR_HEAD_HIDDEN_DIM,
        dropout=FACTOR_HEAD_DROPOUT,
        freeze_backbone=QWEN2VL_FREEZE_BACKBONE,
        device=DEVICE,
        min_pixels=QWEN2VL_MIN_PIXELS,
        max_pixels=QWEN2VL_MAX_PIXELS,
    )
    collate_fn = build_qwen2vl_collate_fn(model.processor)
    dataloader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model.to(DEVICE)
    lr = QWEN2VL_LR_HEAD if QWEN2VL_FREEZE_BACKBONE else QWEN2VL_LR_FULL
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    print("\n开始训练（BCE 多标签）...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            image_grid_thw = batch["image_grid_thw"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            opt.zero_grad()
            out = model(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            loss = multi_label_bce_loss(out["factor_logits"], labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS} loss: {total_loss/len(dataloader):.4f}")

    # 保存
    checkpoint_dir = ROOT / "demo_data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "factor_ids_ordered": factor_ids_ordered,
            "num_factors": num_primitives,
            "model_name": QWEN2VL_MODEL_NAME,
        },
        checkpoint_dir / "factor_model_qwen2vl.pt",
    )
    print(f"已保存: {checkpoint_dir / 'factor_model_qwen2vl.pt'}")

    # 预测 + sanity check
    model.eval()
    threshold = 0.5
    predictions = []
    inference_loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    with torch.no_grad():
        for batch in inference_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            image_grid_thw = batch["image_grid_thw"].to(DEVICE)
            probs = model.predict_probs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            for i in range(probs.size(0)):
                pred_idx = (probs[i] > threshold).nonzero(as_tuple=True)[0].cpu().tolist()
                pred_ids = [factor_ids_ordered[j] for j in pred_idx]
                predictions.append(pred_ids)

    router = Router(abstraction)
    report = run_sanity_check(predictions, labeled, router, factor_id_to_idx, top_k=5)
    print("\n--- 管线路由与验证 ---")
    print(f"  样本数: {report.num_samples}, 原语数: {report.num_factors}")
    print(f"  可解释预测数: {report.interpretable_count}, 平均预测 factor 数: {report.avg_predicted_factors:.2f}")
    print(f"  通过: {report.passed}")


if __name__ == "__main__":
    main()
