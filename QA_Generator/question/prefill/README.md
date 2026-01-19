# QA_Generator/question/prefill

本目录是“预填充对象版本”的问题生成模块，支持在输入中显式提供对象或 claim，
从而保证问题围绕指定对象生成。

## 直接使用方式（CLI）

```bash
python QA_Generator/question/prefill/main.py input.json output.json \
  --config QA_Generator/question/config/question_config.json \
  --pipelines question object_counting \
  -n 100
```

- `input.json`：输入样本列表，**必须包含 `prefill` 字段**
- `output.json`：问题生成结果
- `--config`：可选，问题生成配置
- `--pipelines`：可选，只使用指定 pipeline
- `-n/--max-samples`：可选，限制处理数量

### 输入格式（关键）

**方式 1：claim**
```json
{
  "source_a": {...},
  "prefill": {
    "claim": "图片中有一个红色的小汽车停在路边"
  }
}
```

**方式 2：target_object**
```json
{
  "source_a": {...},
  "prefill": {
    "target_object": "car",
    "target_object_info": {
      "name": "car",
      "category": "vehicle"
    }
  }
}
```

## 在完整 QA 流程中的接口形式

在 `QA_Generator/pipeline/pipeline.py` 中，问题生成阶段会：

- 读取输入记录（包含 `prefill`）
- 使用 `VQAGeneratorPrefill.process_data_file(...)` 或 `process_data_file_async(...)`
- 产出含 `question / question_type / image_base64 / pipeline_name` 等字段的列表

这些结果会进入答案生成模块。

## 文件说明

### `vqa_generator.py`
**问题生成主流程（预填充版本）**：
- 负责读取输入、调用预填充、槽位填充、问题生成
- 输出每条记录的 `question` 相关字段

### `question_generator_prefill.py`
**具体的问题生成器**：
- 接收 `prefill_object` + `slots` + `pipeline_config` 生成问题
- 强制围绕预填充对象/claim 生成，避免漂移

### `object_prefill.py`
**预填充对象处理**：
- 解析 `prefill` 字段（claim 或 target_object）
- 对 claim 模式直接保留文本，不强行抽取对象

### `prefill_processor.py`
**预填充处理器（备用/早期版本）**：
- 功能与 `object_prefill.py` 类似
- 当前流程主要使用 `object_prefill.PrefillProcessor`

### `main.py`
**命令行入口**：
- 读取输入 JSON → 生成问题 → 写出结果

---

## 重复文件说明

此前存在 `question_generator.py` 与 `question_generator_prefill.py` 内容重复的问题。  
目前已保留 `question_generator_prefill.py`，并删除重复的 `question_generator.py`，避免维护分叉。
