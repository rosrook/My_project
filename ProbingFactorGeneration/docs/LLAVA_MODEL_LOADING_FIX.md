# LLaVA OneVision 模型加载修复说明

根据 `eval_vqa_hardness/vqa_evaluator.py` 的正确实现，需要修复 `AsyncLLaVAClient` 的模型加载和推理逻辑。

## 关键修改点

1. **模型加载**：
   - 使用 `AutoModelForCausalLM.from_pretrained()` 而不是 `LlavaNextForConditionalGeneration`
   - 必须添加 `trust_remote_code=True`
   - 使用 `torch_dtype="auto"` 而不是手动指定

2. **Processor 加载**：
   - 使用 `AutoProcessor.from_pretrained()` 而不是 `LlavaNextProcessor`
   - 必须添加 `trust_remote_code=True`
   - 必须添加 `max_pixels=3240000` 和 `min_pixels=200704` 参数

3. **依赖库**：
   - 需要 `qwen_vl_utils` 库（`pip install qwen-vl-utils`）
   - 使用 `qwen_vl_utils.process_vision_info` 处理视觉信息

4. **推理过程**：
   - 使用 `processor.apply_chat_template()` 构建 prompt
   - 使用 `process_vision_info()` 处理视觉信息
   - 图片需要保存为临时文件，并使用 `file://` URL 格式
   - 生成后需要 trim input_ids（只保留新生成的部分）

## 参考实现

参考文件：`eval_vqa_hardness/vqa_evaluator.py`
- `load_model()` 方法：第 44-288 行
- `query_model()` 方法：第 341-440 行
