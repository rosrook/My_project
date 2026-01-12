"""
快速探索 Parquet 文件的简化版本 - 只显示关键信息，避免刷屏
"""

import pandas as pd
import pyarrow.parquet as pq
import sys
from pathlib import Path
from datetime import datetime

class TeeOutput:
    """同时输出到终端和文件的类"""
    def __init__(self, file_path, mode='w', encoding='utf-8'):
        self.terminal = sys.stdout
        self.file = open(file_path, mode, encoding=encoding)
        self.file_path = file_path
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # 确保及时写入
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()
            print(f"\n输出已保存到: {self.file_path}")

def explore_parquet_quick(file_path, output_file=None):
    """快速探索 Parquet 文件，只显示关键信息"""
    
    # 设置输出文件
    tee = None
    original_stdout = sys.stdout
    
    if output_file:
        tee = TeeOutput(output_file)
        sys.stdout = tee
    
    try:
        print("=" * 80)
        print(f"快速探索: {file_path}")
        if output_file:
            print(f"输出文件: {output_file}")
        print("=" * 80)
        
        # 1. 使用 PyArrow 读取元数据（不加载实际数据）
        parquet_file = pq.ParquetFile(file_path)
        
        print("\n【文件基本信息】")
        print(f"行数: {parquet_file.metadata.num_rows:,}")
        print(f"列数: {parquet_file.metadata.num_columns}")
        print(f"文件大小: {parquet_file.metadata.serialized_size / 1024 / 1024:.2f} MB")
        
        # 2. Schema 信息
        print("\n【列信息】")
        schema = parquet_file.schema_arrow
        for i, field in enumerate(schema):
            print(f"  {i+1}. {field.name}: {field.type}")
        
        # 3. 只读取前5行进行预览
        print("\n【数据预览（前3行，只显示关键列）】")
        df = pd.read_parquet(file_path, nrows=3)
        
        # 只显示前5列
        display_cols = df.columns[:5].tolist()
        preview_df = df[display_cols].copy()
        
        # 截断长文本
        for col in preview_df.columns:
            if preview_df[col].dtype == 'object':
                preview_df[col] = preview_df[col].apply(lambda x: str(x)[:50] + "..." if len(str(x)) > 50 else str(x))
        
        print(preview_df.to_string())
        if len(df.columns) > 5:
            print(f"\n(共 {len(df.columns)} 列，只显示前5列)")
        
        # 4. 列名列表
        print(f"\n【所有列名（共{len(df.columns)}列）】")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            print(f"  {i:2d}. {col} ({dtype})")
        
        print("\n" + "=" * 80)
        print("快速预览完成！如需详细分析，请使用 explore_parquet.py")
        print("=" * 80)
    
    finally:
        # 恢复标准输出
        if tee:
            sys.stdout = original_stdout
            tee.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='快速探索 Parquet 文件')
    parser.add_argument('file_path', nargs='?',
                       default="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/part_0.parquet",
                       help='Parquet 文件路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出文件路径（如果不指定，会使用默认名称：<原文件名>_quick_explore_<时间戳>.txt）')
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，生成默认文件名
    if args.output is None:
        input_path = Path(args.file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = input_path.parent / f"{input_path.stem}_quick_explore_{timestamp}.txt"
    else:
        output_file = args.output
    
    explore_parquet_quick(args.file_path, output_file=str(output_file))
