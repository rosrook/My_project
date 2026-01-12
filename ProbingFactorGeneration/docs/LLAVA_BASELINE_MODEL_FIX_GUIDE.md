# LLaVA Baseline 模型调用方式修改指南

根据 `eval_vqa_hardness/vqa_evaluator.py` 的正确实现，需要修改 `ProbingFactorGeneration/utils/async_llava_client.py` 中的模型加载和推理逻辑。

## 修改范围

**仅修改文件：**
- `ProbingFactorGeneration/utils/async_llava_client.py`

**需要修改的方法：**
1. `_load_model()` 方法（第 133-202 行）
2. `_generate_response()` 方法（第 241-323 行）

**其他文件不需要修改**（BaselineModel、Pipeline 等调用 AsyncLLaVAClient 的方式保持不变）

---

## 修改步骤

### 步骤 1: 添加依赖库检查（在文件顶部导入部分）

**位置：** 在 `_load_model()` 方法开始处（约第 133 行）

**需要添加：**
```python
# 检查是否安装了 qwen_vl_utils（必需依赖）
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "qwen_vl_utils 未安装！请运行: pip install qwen-vl-utils\n"
        "这是 LLaVA-OneVision / Qwen2-VL 模型的必需依赖。"
    )
```

**参考：** `eval_vqa_hardness/vqa_evaluator.py` 第 59-66 行

---

### 步骤 2: 修改 `_load_model()` 方法 - 模型加载部分

**位置：** 第 153-199 行（模型加载部分）

**当前代码：**
```python
# Prepare model loading kwargs
model_kwargs = {
    "torch_dtype": self.torch_dtype,
    "device_map": "auto" if self.device == "cuda" else None,
}

# Load model - use AutoModelForCausalLM for compatibility
try:
    # Try LLaVA Next model first (if available)
    if LlavaNextForConditionalGeneration is not None:
        self.model = LlavaNextForConditionalGeneration.from_pretrained(...)
    else:
        self.model = AutoModelForCausalLM.from_pretrained(...)
except Exception as e:
    self.model = AutoModelForCausalLM.from_pretrained(...)
```

**修改为：**
```python
# 使用 AutoModelForCausalLM + trust_remote_code=True（按照 eval_vqa_hardness 的方式）
use_device_map = "auto" if self.device == "cuda" else None

self.model = AutoModelForCausalLM.from_pretrained(
    self.model_path,
    torch_dtype="auto",  # 使用 "auto" 让 transformers 自动选择
    device_map=use_device_map,
    trust_remote_code=True  # 必须添加
)

# 如果未使用device_map，手动移动到指定设备
if use_device_map is None and self.device != "cpu":
    self.model = self.model.to(self.device)

self.model.eval()
```

**参考：** `eval_vqa_hardness/vqa_evaluator.py` 第 155-168 行

---

### 步骤 3: 修改 `_load_model()` 方法 - Processor 加载部分

**位置：** 第 141-151 行（Processor 加载部分）

**当前代码：**
```python
# Load processor - use AutoProcessor for compatibility
try:
    if LlavaNextProcessor is not None:
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
    else:
        self.processor = AutoProcessor.from_pretrained(self.model_path)
except Exception as e:
    self.processor = AutoProcessor.from_pretrained(self.model_path)
```

**修改为：**
```python
# 使用 AutoProcessor + trust_remote_code=True + max_pixels/min_pixels
max_pixels = 3240000
min_pixels = 200704
self.processor = AutoProcessor.from_pretrained(
    self.model_path,
    trust_remote_code=True,  # 必须添加
    max_pixels=max_pixels,   # 必须添加
    min_pixels=min_pixels    # 必须添加
)

# 保存 qwen_vl_utils 的引用供后续使用
self.process_vision_info = process_vision_info
```

**参考：** `eval_vqa_hardness/vqa_evaluator.py` 第 269-282 行

---

### 步骤 4: 修改 `_generate_response()` 方法 - 推理逻辑

**位置：** 第 241-323 行（整个 `_generate_response()` 方法）

**当前代码：**
```python
def _generate_response(self, text: str, image: Image.Image, ...):
    # 使用简单的 prompt 格式
    prompt = f"USER: <image>\n{text}\nASSISTANT:"
    inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
    outputs = self.model.generate(**inputs, ...)
    generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
    # 提取 response
```

**修改为：**
```python
def _generate_response(self, text: str, image: Image.Image, ...):
    # 使用 qwen_vl_utils 的方式（按照 eval_vqa_hardness）
    from tempfile import NamedTemporaryFile
    
    # 步骤1: 保存图片为临时文件
    temp_file = NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_image_path = temp_file.name
    image.save(temp_image_path, 'JPEG')
    temp_file.close()
    
    try:
        # 步骤2: 构建消息（file:// URL 格式）
        image_url = f"file://{temp_image_path}"
        content_list = [
            {"type": "image", "image": image_url},
            {"type": "text", "text": text}
        ]
        messages = [{"role": "user", "content": content_list}]
        
        # 步骤3: 使用 processor.apply_chat_template
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 步骤4: 使用 qwen_vl_utils.process_vision_info
        image_inputs, video_inputs = self.process_vision_info(messages)
        
        # 步骤5: 调用 processor
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # 步骤6: 生成答案
        generate_kwargs = {"max_new_tokens": max_tokens}
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        
        # 步骤7: Trim input_ids（只保留新生成的部分）
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 步骤8: 解码
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except:
                pass
    
    return response
```

**参考：** `eval_vqa_hardness/vqa_evaluator.py` 第 361-440 行

---

### 步骤 5: 更新 requirements.txt

**位置：** `ProbingFactorGeneration/requirements.txt`

**添加：**
```
qwen-vl-utils>=0.0.1  # Required for LLaVA OneVision models
```

---

## 修改后的关键变化

1. **模型加载：**
   - ✅ 使用 `AutoModelForCausalLM`（不再尝试 `LlavaNextForConditionalGeneration`）
   - ✅ 添加 `trust_remote_code=True`
   - ✅ 使用 `torch_dtype="auto"`

2. **Processor 加载：**
   - ✅ 使用 `AutoProcessor`（不再尝试 `LlavaNextProcessor`）
   - ✅ 添加 `trust_remote_code=True`
   - ✅ 添加 `max_pixels` 和 `min_pixels` 参数

3. **推理过程：**
   - ✅ 使用 `qwen_vl_utils.process_vision_info`
   - ✅ 使用 `processor.apply_chat_template`
   - ✅ 使用临时文件路径（file:// URL 格式）
   - ✅ Trim input_ids（只保留新生成的部分）

---

## 测试建议

修改完成后，测试：

1. **导入测试：**
```python
from ProbingFactorGeneration.utils.async_llava_client import AsyncLLaVAClient
```

2. **模型加载测试：**
```python
async with AsyncLLaVAClient(
    model_path="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model",
    gpu_id=0
) as client:
    # 应该能成功加载
    pass
```

3. **推理测试：**
```python
from PIL import Image
image = Image.new("RGB", (100, 100), color="red")
response = await client.analyze_image_async(image, "What is in the image?")
print(response)
```

---

## 注意事项

1. **不修改的部分：**
   - `BaselineModel` 类（`models/baseline.py`）的接口保持不变
   - `Pipeline` 类（`pipeline.py`）的调用方式保持不变
   - 其他组件不需要修改

2. **向后兼容性：**
   - `AsyncLLaVAClient` 的公共接口（`analyze_image_async`, `chat.completions.create` 等）保持不变
   - 只有内部实现（`_load_model`, `_generate_response`）被修改

3. **错误处理：**
   - 如果 `qwen_vl_utils` 未安装，会在 `_load_model()` 时抛出明确的错误信息
   - 临时文件清理使用 `try-finally` 确保清理
