#!/bin/bash
# ===============================
# 恢复 My_project 环境脚本
# 说明：
# 1. 自动创建 conda 环境
# 2. 安装关键依赖
# 3. 配置代理（如需）
# 4. clone 项目代码
# 5. 设置 pip 源
# 6. 安装 Python 包
# ===============================

# ---------- 配置变量 ----------
ENV_NAME="my_project"
PYTHON_VER="3.10"
GITHUB_REPO="https://github.com/rosrook/My_project.git"
PROXY_HTTP="10.140.24.177:3128"   # 如不需要代理可注释掉
PROXY_HTTPS="10.140.24.177:3128"  # 如不需要代理可注释掉

# ---------- 1. 创建并激活 conda 环境 ----------
conda create -n $ENV_NAME python=$PYTHON_VER -y
conda activate $ENV_NAME

# ---------- 2. 配置代理（可选） ----------
export http_proxy=$PROXY_HTTP
export https_proxy=$PROXY_HTTPS

# ---------- 3. 克隆代码 ----------
if [ ! -d "My_project" ]; then
    git clone $GITHUB_REPO
else
    echo "目录 My_project 已存在，跳过 clone"
fi

# ---------- 4. 升级 pip ----------
python -m pip install --upgrade pip

# ---------- 5. 安装核心依赖 ----------
# PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 图像处理
pip install "Pillow>=10.0.0"

# 数据与配置管理
pip install "pyyaml>=6.0" "pandas>=1.5.0" "pyarrow>=10.0.0" "packaging>=21.0"

# 辅助工具
pip install "tqdm>=4.65.0" "aiohttp>=3.8.0" "nest-asyncio>=1.5.0"

# 模型和科学计算
pip install "torch>=2.0.0" "transformers>=4.30.0" "numpy>=1.24.0"

# Qwen-VL 工具
pip install "qwen-vl-utils>=0.0.1"

# 内部 PyPI 源配置（可选）
pip config set global.index-url http://pypi.devops.xiaohongshu.com/simple/
pip config set install.trusted-host pypi.devops.xiaohongshu.com

# 安装内部包
pip install redeuler

# ---------- 6. 系统工具（tmux） ----------
# 注意：仅当你有 sudo 权限才可执行
if command -v sudo &> /dev/null; then
    sudo apt update
    sudo apt install tmux -y
else
    echo "未检测到 sudo 权限，tmux 安装跳过"
fi

# ---------- 完成 ----------
echo "恢复完成！请使用：conda activate $ENV_NAME 进入环境"
