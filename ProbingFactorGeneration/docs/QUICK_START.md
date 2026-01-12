# 快速开始指南

## 使用 Qwen 作为 Judge 模型

根据你的配置（`debug_qwen3vl.py`），你需要设置以下环境变量：

```bash
export SERVICE_NAME="mediak8s-editprompt-qwen235b"
export ENV="prod"
export API_KEY="1"
export USE_LB_CLIENT="true"
```

然后运行：

```bash
python ProbingFactorGeneration/examples/run_complete_pipeline.py \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --sample_size 10 \
    --judge_model_name /workspace/Qwen3-VL-235B-A22B-Instruct \
    --output_dir ./output
```

## 为什么需要这些环境变量？

代码使用 `LBOpenAIAsyncClient`（异步版本），需要：
- `SERVICE_NAME`: 服务名称（从你的 debug_qwen3vl.py 中获取）
- `ENV`: 环境（prod/staging）
- `API_KEY`: API 密钥
- `USE_LB_CLIENT="true"`: 启用 LBOpenAIAsyncClient（这样就不需要 BASE_URL）

如果 `USE_LB_CLIENT="false"` 或 `redeuler` 库未安装，代码会回退到需要 `BASE_URL` 的模式。

## 检查依赖

确保 `redeuler` 已安装：

```bash
pip install redeuler
```

或让脚本自动安装（如果代码支持）。
