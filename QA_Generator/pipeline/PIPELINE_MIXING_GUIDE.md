# Pipeline 数据分类和调配指南

## 功能概述

QA Generator 现在支持两个新功能：

1. **按 Pipeline 类型分类输出**：自动将生成的数据按 `pipeline_name` 分类保存到不同的 JSON 文件
2. **按比例调配数据**：可以自由调控不同类型的数据比例，生成最终的 VQA 数据集

## 功能 1: 按 Pipeline 分类输出

### 说明

运行 `pipeline.py` 时，会自动按 `pipeline_name` 将成功的数据分类保存：

- **分类文件**：`vqa_dataset_successful_{pipeline_name}_{timestamp}.json`
  - 例如：`vqa_dataset_successful_object_absence_20240101_120000.json`
  - 例如：`vqa_dataset_successful_object_counting_20240101_120000.json`

- **合并文件**：`vqa_dataset_successful_all_{timestamp}.json`
  - 包含所有 pipeline 的合并数据（向后兼容）

### 输出文件结构

```
output/
├── vqa_dataset_successful_object_absence_20240101_120000.json    # object_absence 类型
├── vqa_dataset_successful_object_counting_20240101_120000.json  # object_counting 类型
├── vqa_dataset_successful_question_20240101_120000.json         # question 类型
├── vqa_dataset_successful_all_20240101_120000.json              # 全部合并
├── meta.json                                                      # 元数据（包含pipeline文件路径）
└── ...
```

### meta.json 内容

`meta.json` 文件现在包含以下额外字段：

```json
{
  "pipeline_output_files": {
    "object_absence": "output/vqa_dataset_successful_object_absence_20240101_120000.json",
    "object_counting": "output/vqa_dataset_successful_object_counting_20240101_120000.json",
    ...
  },
  "pipeline_counts": {
    "object_absence": 150,
    "object_counting": 200,
    ...
  }
}
```

## 功能 2: 按比例调配数据

### 使用方法

#### 方法 1: 从目录加载，命令行指定比例

```bash
python QA_Generator/pipeline/mix_pipelines.py \
    --input-dir ./output \
    --ratios object_absence:0.3 object_counting:0.2 question:0.5 \
    --output mixed_dataset.json
```

#### 方法 2: 从 meta.json 加载（推荐）

```bash
python QA_Generator/pipeline/mix_pipelines.py \
    --meta ./output/meta.json \
    --ratios object_absence:0.3 object_counting:0.2 question:0.5 \
    --output mixed_dataset.json
```

#### 方法 3: 使用配置文件

创建配置文件 `mix_config.json`：

```json
{
  "input_dir": "./output",
  "ratios": {
    "object_absence": 0.3,
    "object_counting": 0.2,
    "question": 0.5
  },
  "total_samples": 1000,
  "seed": 42
}
```

然后运行：

```bash
python QA_Generator/pipeline/mix_pipelines.py \
    --config mix_config.json \
    --output mixed_dataset.json
```

### 参数说明

- `--input-dir`: 输入目录（包含按 pipeline 分类的 JSON 文件）
- `--meta`: meta.json 文件路径（从中读取 pipeline 文件路径，推荐使用）
- `--config`: 配置文件路径（JSON 格式）
- `--ratios`: Pipeline 比例列表，格式：`pipeline1:ratio1 pipeline2:ratio2 ...`
- `--total-samples`: 总样本数（可选，如果指定则按比例采样；否则使用所有可用数据）
- `--seed`: 随机种子（可选，用于可重复的采样结果）
- `--output`: 输出 JSON 文件路径

### 比例说明

- 比例可以是任意正数，脚本会自动归一化
- 例如：`object_absence:0.3 object_counting:0.2 question:0.5` 会被归一化为 `30% : 20% : 50%`
- 如果指定了 `--total-samples`，会按比例采样；否则使用所有可用数据

### 示例

#### 示例 1: 基本使用（使用所有数据）

```bash
python QA_Generator/pipeline/mix_pipelines.py \
    --meta ./output/meta.json \
    --ratios object_absence:1 object_counting:1 question:2 \
    --output balanced_dataset.json
```

结果：`object_absence` 25%，`object_counting` 25%，`question` 50%

#### 示例 2: 指定总样本数

```bash
python QA_Generator/pipeline/mix_pipelines.py \
    --meta ./output/meta.json \
    --ratios object_absence:0.3 object_counting:0.2 question:0.5 \
    --total-samples 1000 \
    --seed 42 \
    --output mixed_1000_samples.json
```

结果：从各 pipeline 中按比例采样，总共 1000 个样本

#### 示例 3: 使用配置文件

```bash
# 创建配置文件
cat > my_mix_config.json << EOF
{
  "meta": "./output/meta.json",
  "ratios": {
    "object_absence": 0.4,
    "object_counting": 0.3,
    "question": 0.3
  },
  "total_samples": 5000,
  "seed": 123
}
EOF

# 运行
python QA_Generator/pipeline/mix_pipelines.py \
    --config my_mix_config.json \
    --output final_dataset.json
```

## 完整工作流程

### Step 1: 运行 QA Generator

```bash
python QA_Generator/pipeline/pipeline.py input.json \
    --output-dir ./output \
    --batch-size 1000
```

这会生成：
- `vqa_dataset_successful_{pipeline_name}_{timestamp}.json`（按类型分类）
- `vqa_dataset_successful_all_{timestamp}.json`（全部合并）
- `meta.json`（包含元数据）

### Step 2: 按比例调配数据

```bash
python QA_Generator/pipeline/mix_pipelines.py \
    --meta ./output/meta.json \
    --ratios object_absence:0.3 object_counting:0.2 question:0.5 \
    --total-samples 10000 \
    --output final_vqa_dataset.json
```

## 注意事项

1. **Pipeline 名称匹配**：确保 `--ratios` 中指定的 pipeline 名称与生成的文件中的 `pipeline_name` 字段一致
2. **数据量限制**：如果某个 pipeline 的数据量不足，会使用所有可用数据
3. **随机性**：使用 `--seed` 参数可以确保结果可重复
4. **文件格式**：输入文件应为 JSON 数组格式，每个元素包含 `pipeline_name` 字段

## 故障排除

### 问题：找不到 pipeline 文件

**原因**：Pipeline 名称不匹配

**解决**：
1. 检查 `meta.json` 中的 `pipeline_output_files` 字段
2. 确认 `--ratios` 中使用的 pipeline 名称与文件中的一致

### 问题：采样数量不符合预期

**原因**：某个 pipeline 的数据量不足

**解决**：
- 脚本会自动调整，使用所有可用数据
- 检查输出日志中的警告信息

### 问题：比例总和不为 1

**说明**：这是正常的，脚本会自动归一化比例

**示例**：
- 输入：`object_absence:3 object_counting:2 question:5`
- 归一化后：`30% : 20% : 50%`
