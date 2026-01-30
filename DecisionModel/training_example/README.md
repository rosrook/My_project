# training_example：Qwen2VL + GRPO 训练示例

本目录是 **Qwen2-VL + GRPO（Group Relative Policy Optimization）** 的训练示例，用于**生成式**的「根据报告选择筛选 Agent 并生成 prompt」任务，与 DecisionModel 主流程中的**判别式**「图像 → factor_id 多标签预测」是**不同任务**，可按需借鉴部分设计，但不必直接替换主流程训练。

---

## training_example 在做什么

| 项目 | 说明 |
|------|------|
| **模型** | Qwen2-VL-7B-Instruct（大参数量 VLM） |
| **任务** | 输入：文本报告（+ 可选图像）→ 输出：**生成** XML 格式的 Agent 选择（`<reasoning>`, `<agents>`, `<name>`, `<prompt>`） |
| **训练方式** | GRPO：每个 prompt 生成多条 completion → 用 reward 函数打分 → advantage + KL 惩罚 → 策略梯度 |
| **Reward** | `agent_selection`（选对 Agent 的 F1）、`filtering_quality`（过滤结果与 GT 的 F1）、`format`（XML 格式）、`combined` |
| **数据** | 每条样本：`report`、`ground_truth_agents`、`ground_truth_filtered_data`；prompt 为对话（system + user 报告） |

---

## DecisionModel 主流程在做什么

| 项目 | 说明 |
|------|------|
| **模型** | ResNet18 视觉编码器 + 小型 MLP 多标签头（轻量） |
| **任务** | 输入：**图像** → 输出：**固定集合**上的 factor_id 多标签（概率/二值），用于路由到筛选原语 |
| **训练方式** | **监督学习**：image + multi-hot 标签 → BCE 损失，无生成、无 reward |
| **数据** | 每条样本：`image_path`、`filter_factor_texts` → 经抽象与标签构造得到 `multi_hot` |

---

## 是否需要借鉴 training_example？

### 结论（简要）

- **当前 DecisionModel 的「因子预测模型」训练：不需要**把 Qwen2VL/GRPO 直接搬过来。  
  主流程是「图像 → factor_id 多标签」的**判别式、监督**任务，和「报告 → 生成 Agent 选择」的**生成式、强化学习**任务不同，用现有 ResNet+MLP + BCE 即可。
- **可以按需借鉴**的，主要是**工程与扩展**层面（见下）。

### 适合借鉴的情况

1. **统一工程风格**  
   - 若希望和 training_example 一致：可采用 **YAML + dataclass** 的 config（如 `config.yaml` + `ScriptArguments`），用 `TrlParser` 或类似方式解析；  
   - 当前 DecisionModel 已用 `config.py`，可保持现状，仅在做「大训练脚本」时再考虑 YAML。

2. **未来增加「VLM 阶段」**  
   - 若设计成：**先**用现有模型得到 image → factor_ids，**再**用 VLM 根据「报告 + factor_ids」**生成** Agent 选择与 prompt；  
   - 则 **training_example 的 Qwen2VL + GRPO 流程**（`grpo_trainer.py`、reward 设计、数据格式）可直接作为这一 VLM 阶段的参考，而不是替换当前的因子预测模型。

3. **数据与日志**  
   - 可借鉴：HuggingFace `datasets` 的封装、`Trainer` 的 logging/checkpoint 约定；  
   - 当前 `run_demo.py` 已能跑通闭环，可按需逐步对齐。

### 不需要借鉴的部分

- **GRPO 算法、reward 函数、生成逻辑**：与「图像 → factor_id 多标签」无关，当前主流程不必引入。
- **Qwen2-VL 作为因子预测骨干**：会变成「大 VLM 生成再解析」，成本高、与现有「轻量中间表示」设计不一致；除非明确要改成「生成式 factor 描述」，否则不必替换现有 ResNet+MLP。

---

## 总结表

| 维度 | training_example (Qwen2VL+GRPO) | DecisionModel 因子预测 (run_demo) |
|------|--------------------------------|-----------------------------------|
| 任务形态 | 生成式（报告 → Agent 选择 XML） | 判别式（图像 → factor_id 多标签） |
| 模型 | Qwen2-VL 7B | ResNet18 + MLP |
| 训练 | GRPO + reward | 监督 BCE |
| 是否必须借鉴 | 否 | — |
| 可借鉴点 | Config/数据/未来 VLM 阶段 | — |

**简单说**：  
- 现有 DecisionModel 的模型训练**不需要**改成 Qwen2VL 或 GRPO。  
- training_example 适合作为**另一条线**（VLM 做 Agent 选择）或**工程风格**的参考，按需选用即可。
