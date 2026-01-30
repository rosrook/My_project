# DecisionModel — 筛选要素闭环 Demo

**面向导师的设计与目的说明**：见 [docs/DESIGN_AND_PURPOSE.md](docs/DESIGN_AND_PURPOSE.md)，包含模块目的、整体设计、四段流程、与项目其他部分的关系及两种模型方案说明。

---

最小可运行闭环，依次包含：

1. **filter_factor_abstraction** — 筛选要素抽象  
   自然语言筛选要素 → 冻结文本编码器 → 向量表示 → 余弦相似度图 → 阈值/聚类 → 离散筛选原语，分配 `factor_id`，保留 `factor_id → 原始文本列表` 映射。

2. **label_construction** — 监督构造  
   每张图片关联的原始筛选要素文本集合 → 映射为 `factor_id` 集合 → 编码为 multi-hot 向量作为监督信号。

3. **factor_prediction** — 因子预测模型  
   冻结/部分冻结视觉编码器 + 多标签预测头（线性层或小型 MLP），学习 image features → factor_id 概率，训练目标为多标签损失（如 BCE）。

4. **routing_sanity_check** — 管线路由与验证  
   根据预测的 factor_id 组合调用/模拟筛选原语执行逻辑，验证模型能否为不同图像生成合理、可解释的筛选要素组合，作为动态筛选管线与 VQA 构造的接口。

## 文件组织

```
DecisionModel/
├── __init__.py
├── config.py
├── filter_factor_abstraction/   # 筛选要素抽象
│   ├── __init__.py
│   ├── text_encoder.py
│   ├── similarity_graph.py
│   └── factor_primitives.py
├── label_construction/         # 监督构造
│   ├── __init__.py
│   └── multi_hot_labels.py
├── factor_prediction/          # 因子预测模型
│   ├── __init__.py
│   ├── vision_encoder.py
│   ├── model.py
│   └── loss.py
├── routing_sanity_check/       # 管线路由与验证
│   ├── __init__.py
│   ├── router.py
│   └── sanity_check.py
├── data/
│   └── schemas.py
├── training_example/           # Qwen2VL+GRPO 示例（可选借鉴，见下）
│   ├── README.md               # 与主流程对比说明
│   ├── config.yaml
│   ├── grpo_trainer.py
│   └── grpo.py
└── run_demo.py                 # 最小可运行闭环入口
```

## Qwen2-VL 因子预测（可选）

在**同一套数据**（图片 + 筛选要素 id 组合）和**同一监督方式**（multi-hot BCE）下，可将骨干从 ResNet 换为 Qwen2-VL：

- **模型**：`factor_prediction/qwen2vl_factor_model.py` — Qwen2-VL 骨干 + 多标签头，`get_image_features` → 线性层 → factor logits。
- **训练**：`run_demo_qwen2vl.py` — 与 `run_demo.py` 相同流程（抽象 → 标签 → 训练 → 保存 → 路由验证），仅模型改为 Qwen2VLFactorModel。
- **配置**：`config.py` 中 `QWEN2VL_*`（模型名、是否冻结骨干、min/max_pixels、学习率等）。

运行（需安装 `transformers` 与足够显存）：

```bash
python -m DecisionModel.run_demo_qwen2vl
```

默认使用 `Qwen/Qwen2-VL-2B-Instruct`，显存不足可改 config 中 `QWEN2VL_MODEL_NAME` 或减小 `QWEN2VL_MAX_PIXELS`。

## 统筹脚本：两种模型训练+测试对比（run_both_models.py）

在同一套数据与 **train/test 划分** 下，分别训练并测试 **ResNet+MLP** 与 **Qwen2-VL**，输出可对比的测试指标并可选写入 JSON 报告。

**运行**（在项目根目录）：

```bash
python -m DecisionModel.run_both_models
```

**需要补全 / 可配置项**（在 `DecisionModel/run_both_models.py` 顶部「需要补全 / CONFIG」区域）：

| 变量 | 说明 | 示例 / 默认 |
|------|------|-------------|
| `ANNOTATIONS_PATH` | 标注 JSONL 路径（训练+测试共用时按比例划分）；不填则用 config 默认，若无文件则用合成数据 | `None` 或 `Path("DecisionModel/demo_data/annotations.jsonl")` |
| `FACTORS_JSON_PATH` | 筛选要素列表 JSON（无标注时生成合成数据用）；不填则用 config 默认 | `None` 或 `Path("ProbingFactorGeneration/configs/filter_factors_list.json")` |
| `TRAIN_RATIO` | 训练集比例，剩余为测试集 | `0.8` |
| `RANDOM_SEED` | 划分与合成数据随机种子 | `42` |
| `DEVICE` | 设备 | `"cpu"` 或 `"cuda"` |
| `NUM_EPOCHS` / `BATCH_SIZE` | 训练轮数与批大小 | `10`, `16` |
| `RUN_RESNET` / `RUN_QWEN2VL` | 是否跑 ResNet 与 Qwen2-VL（无 GPU/transformers 时可只跑 ResNet） | `True`, `False` |
| `OUTPUT_REPORT_PATH` | 报告 JSON 输出路径；不填则只打印不写文件 | `None` 或 `Path("DecisionModel/demo_data/report_both_models.json")` |
| `SYNTHETIC_NUM_SAMPLES` | 无标注文件时合成样本数 | `60` |

**输出**：控制台打印各模型在测试集上的 BCE、SubsetAcc、F1_macro、F1_micro；若设置了 `OUTPUT_REPORT_PATH`，会写入包含 config、数据规模与各模型 `test_metrics` 的 JSON。

**评估指标**（`eval_metrics.py`）：`bce_loss`、`subset_accuracy`、`f1_macro`、`f1_micro`、`precision_micro`、`recall_micro`。

## 数据格式示例（一条数据 + 两个筛选要素）

下面用「只有一条样本、且该样本对应两个筛选要素」的最小例子说明 `ANNOTATIONS_PATH`（JSONL）和 `FACTORS_JSON_PATH`（JSON）各自的数据结构。示例文件在 `DecisionModel/demo_data/` 下：`annotations_one_example.jsonl`、`filter_factors_two_example.json`。

### ANNOTATIONS_PATH（JSONL）

- **含义**：每行一个 JSON 对象，表示「一张图 + 该图关联的筛选要素文本列表」。
- **字段**：
  - `image_path`（必填）：图片路径（相对项目根或绝对路径）。
  - `filter_factor_texts`（必填）：该图对应的筛选要素**自然语言描述**列表（字符串数组）。
  - `image_id`（可选）：样本 ID，便于日志与报告。

**一条数据、两个筛选要素时的单行示例**：

```json
{"image_path": "demo_data/images/sample_001.jpg", "image_id": "img_0", "filter_factor_texts": ["clear and identifiable concept", "at least one visually salient object"]}
```

多条数据时，每行一条 JSON，例如：

```json
{"image_path": "demo_data/images/sample_001.jpg", "image_id": "img_0", "filter_factor_texts": ["clear and identifiable concept", "at least one visually salient object"]}
{"image_path": "demo_data/images/sample_002.jpg", "image_id": "img_1", "filter_factor_texts": ["at least one visually salient object"]}
```

### FACTORS_JSON_PATH（JSON）

- **含义**：所有可能出现的筛选要素的**自然语言描述**列表（候选池）。用于无标注文件时生成合成数据；也可仅作参考，抽象阶段会从标注里出现的文本自动收集。
- **结构**：根节点需包含 `filter_factors` 数组，数组元素为字符串。

**仅两个筛选要素时的示例**：

```json
{
    "filter_factors_schema_version": "v1.0",
    "description": "示例：仅包含两条筛选要素。",
    "filter_factors": [
        "clear and identifiable concept",
        "at least one visually salient object"
    ]
}
```

与上面「一条数据 + 两个筛选要素」的对应关系：该条标注里的 `filter_factor_texts` 的两个字符串，应出现在 `filter_factors` 列表中（或与列表中某条语义一致，抽象阶段会做聚类映射）。实际使用时，`filter_factors` 可以很长，每张图只带其中一部分作为 `filter_factor_texts`。

## 两种模型：训练时间与数据量预估

以下为**经验性预估**，实际取决于硬件、分辨率、batch 大小和任务难度，仅供规划参考。

### ResNet18 + MLP

| 项目 | 预估 |
|------|------|
| **单 epoch 时间** | CPU：约 1–3 分钟/500 样本；GPU（单卡）：约 10–30 秒/500 样本。 |
| **10 epoch 总时间** | 500 样本：CPU 约 10–30 分钟，GPU 约 2–5 分钟；5000 样本：CPU 约 1.5–3 小时，GPU 约 15–30 分钟。 |
| **建议数据量** | **最小可跑**：100–200 条（仅验证闭环）；**demo/小规模**：500–2000 条；**效果更稳**：5000–2 万条。原语数（factor 类别数）较多时，建议至少每类几十条以上。 |
| **显存/内存** | CPU 可跑，内存约 2–4 GB；GPU 单卡 2–4 GB 足够。 |

### Qwen2-VL（2B，冻结骨干 + 多标签头）

| 项目 | 预估 |
|------|------|
| **单 epoch 时间** | 需 GPU；batch=4、500 样本：约 5–15 分钟/epoch；batch=8、1000 样本：约 15–30 分钟/epoch（视分辨率与 max_pixels 而定）。 |
| **10 epoch 总时间** | 500 样本：约 1–2.5 小时；1000 样本：约 2.5–5 小时。 |
| **建议数据量** | **最小可跑**：100–300 条；**推荐**：500–2000 条（冻结骨干时）；若全量微调，建议 3000+ 条并减小学习率、增加 epoch。原语数多时适当增加样本。 |
| **显存** | 2B + 冻结骨干：单卡约 8–12 GB；若提高分辨率或全量微调，约 16–24 GB。 |

### 对比小结

| 维度 | ResNet18+MLP | Qwen2-VL（2B，冻骨干） |
|------|----------------|-------------------------|
| 训练速度 | 快（CPU 也可用） | 慢，需 GPU |
| 单卡显存 | 2–4 GB | 约 8–12 GB |
| 最小数据量 | 约 100–200 条 | 约 100–300 条 |
| 推荐数据量 | 500–5000 条 | 500–2000 条（冻骨干） |
| 10 epoch、约 500 样本 | CPU 约 10–30 分钟，GPU 约 2–5 分钟 | GPU 约 1–2.5 小时 |

**说明**：上述时间为同一配置（如 `NUM_EPOCHS=10`、`BATCH_SIZE=16` 或 4）下的量级估计；实际以你本机跑一次为准。数据量“建议”以保证基本泛化与稳定收敛为主；若仅做 pipeline 连通性验证，可用几十条合成数据快速跑通。

## training_example 与主流程训练的关系

`training_example/` 下是 **Qwen2-VL + GRPO** 的训练示例（生成式「报告 → Agent 选择」），与主流程中的**因子预测模型**（判别式「图像 → factor_id 多标签」）是**不同任务**：

- **主流程训练**（`run_demo.py` + `factor_prediction`）：ResNet + MLP，监督 BCE，**不需要**改成 Qwen2VL 或 GRPO。
- **可借鉴**：若未来增加「VLM 根据报告 + factor_ids 生成 Agent 选择」阶段，可参考 `training_example` 的 GRPO 与 reward 设计；或统一使用 YAML/dataclass 配置风格。

详见 [training_example/README.md](training_example/README.md)。

## 运行

在**项目根目录**下安装依赖并执行（推荐）：

```bash
cd /path/to/My_project
pip install -r DecisionModel/requirements.txt
python -m DecisionModel.run_demo
```

或在 DecisionModel 目录下：

```bash
cd DecisionModel
pip install -r requirements.txt
cd .. && python -m DecisionModel.run_demo
```

输入：`images` + 自然语言描述的筛选要素集合（例如每张图对应一组 filter factor 文本）。  
- 若存在 `DecisionModel/demo_data/annotations.jsonl`，则从该文件读取样本；  
- 否则使用 `ProbingFactorGeneration/configs/filter_factors_list.json` 中的筛选要素列表生成合成样本（占位图片路径），用于验证闭环。

输出：抽象后的 factor_id 表、multi-hot 标签、训练好的因子预测模型（保存至 `demo_data/checkpoints/factor_model.pt`），以及路由与 sanity check 报告。

---

## 模块完整性与需补全项

### 已实现（闭环完整）

| 模块 | 状态 | 说明 |
|------|------|------|
| filter_factor_abstraction | ✅ | 文本编码（sentence-transformers/CLIP）、相似度图、连通分量、factor_id 与可解释映射 |
| label_construction | ✅ | 原始文本 → factor_id → multi-hot |
| factor_prediction | ✅ | ResNet18 视觉编码器 + 多标签头、BCE 损失、训练与预测 |
| routing_sanity_check | ✅ | Router 原语执行/模拟、SanityCheckReport |
| run_demo.py | ✅ | 端到端：加载数据 → 抽象 → 标签 → 训练 → 保存 → 路由验证 |
| inference.py | ✅ | 模型保存/加载、单张图/批量 predict_factor_ids |
| config | ✅ | 路径、编码器名、阈值、训练超参；run_demo 已使用 DEMO_ANNOTATIONS_JSONL、DEFAULT_FILTER_FACTORS_JSON |

### 建议补全或可选增强

1. **数据与路径**
   - 合成样本的 `image_path` 已使用 `base_path`（DecisionModel 目录）指向 `demo_data/images/`；若目录下无真实图片，会回退为灰图，闭环仍可跑通。
   - 可选：在 `demo_data/annotations.jsonl` 中填写真实 `image_path`（项目内或绝对路径），或提供从 parquet/其他格式加载的脚本。

2. **模型持久化与推理**
   - 训练结束会保存 `demo_data/checkpoints/factor_model.pt`（含 state_dict、factor_ids_ordered、num_factors）。
   - 推理：使用 `DecisionModel.inference.load_model_from_checkpoint` + `predict_factor_ids` 即可对单张图或 batch 得到 factor_id 列表。

3. **视觉编码器**
   - 当前仅实现并默认使用 `resnet18`；config 中 `VISION_ENCODER_NAME` 未在 run_demo 中读取，扩展其他 backbone（如 ResNet50、CLIP vision）时需在 `vision_encoder.py` 与 run_demo 中统一使用 config。

4. **原语真实执行**
   - `Router.run_primitives` 与 `PrimitiveExecutor` 当前为“模拟”（无真实 VLM/规则调用）。若要对接到 FactorFilterAgent 的 VLM 或 Rule_based，需在构造 `Router(abstraction, executors=...)` 时传入带 `execute_fn` 的 `PrimitiveExecutor`，在 `execute_fn` 内调用现有管线。

5. **评估指标**
   - Sanity check 仅统计可解释数、平均预测 factor 数、通过与否；未计算多标签分类指标（如 mAP、F1、subset accuracy）。可选：在 `routing_sanity_check` 或 run_demo 中增加上述指标。

6. **依赖**
   - `requirements.txt` 已包含 sentence-transformers；若使用 CLIP 文本编码器，需额外安装 `transformers`。

7. **测试**
   - 当前无单元测试；可选：为 abstraction、label_construction、predict_factor_ids 等增加 `tests/` 与 pytest。
