# Examples 使用说明

## 运行方式

### 方式 1: 从 My_project 目录运行（推荐）

```bash
# 在 My_project 目录下运行
cd ~/My_project
python ProbingFactorGeneration/examples/run_complete_pipeline.py \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --sample_size 10 \
    --output_dir ./output
```

### 方式 2: 从 ProbingFactorGeneration 目录运行

```bash
# 在 ProbingFactorGeneration 目录下运行
cd ~/My_project/ProbingFactorGeneration
PYTHONPATH=..:$PYTHONPATH python examples/run_complete_pipeline.py \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --sample_size 10 \
    --output_dir ./output
```

### 方式 3: 使用 -m 模块方式运行

```bash
cd ~/My_project
python -m ProbingFactorGeneration.examples.run_complete_pipeline \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --sample_size 10 \
    --output_dir ./output
```

## 常见问题

### ModuleNotFoundError: No module named 'ProbingFactorGeneration'

**原因**: Python 无法找到 `ProbingFactorGeneration` 包。

**解决方案**:
1. 确保从 `My_project` 目录运行脚本
2. 或者设置 `PYTHONPATH` 环境变量:
   ```bash
   export PYTHONPATH=$PYTHONPATH:~/My_project
   ```
3. 或者使用方式 2 或方式 3 的运行方法

### 相对路径问题

如果配置文件路径有问题，使用绝对路径：
```bash
python ProbingFactorGeneration/examples/run_complete_pipeline.py \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --claim_template_config $(pwd)/ProbingFactorGeneration/configs/claim_template.example_v1_1.json \
    --sample_size 10 \
    --output_dir ./output
```
