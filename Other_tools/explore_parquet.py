import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from collections import Counter
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

def truncate_text(text, max_length=100):
    """截断长文本"""
    text_str = str(text)
    if len(text_str) > max_length:
        return text_str[:max_length] + "..."
    return text_str

def explore_parquet(file_path, sample_rows=5, show_stats=True, max_sample_length=100, preview_cols=5, output_file=None):
    """
    探索 Parquet 文件的内容和结构
    
    参数:
        file_path: Parquet 文件路径
        sample_rows: 显示的样本行数
        show_stats: 是否显示详细统计信息
        max_sample_length: 样本值的最大显示长度
        preview_cols: 预览时显示的列数（避免输出过多）
        output_file: 输出文件路径（如果为None，只输出到终端）
    """
    
    # 设置输出文件
    tee = None
    original_stdout = sys.stdout
    
    if output_file:
        tee = TeeOutput(output_file)
        sys.stdout = tee
    
    try:
        print("=" * 80)
        print(f"探索 Parquet 文件: {file_path}")
        if output_file:
            print(f"输出文件: {output_file}")
        print("=" * 80)
        
        # 1. 使用 PyArrow 读取元数据（不加载实际数据，快速）
        parquet_file = pq.ParquetFile(file_path)
        
        print("\n【文件基本信息】")
        print(f"行数: {parquet_file.metadata.num_rows:,}")
        print(f"列数: {parquet_file.metadata.num_columns}")
        print(f"行组数: {parquet_file.metadata.num_row_groups}")
        print(f"文件大小: {parquet_file.metadata.serialized_size / 1024 / 1024:.2f} MB")
        
        # 2. Schema 信息
        print("\n【Schema 信息】")
        schema = parquet_file.schema_arrow
        for i, field in enumerate(schema):
            print(f"{i+1}. {field.name}: {field.type}")
        
        # 3. 使用 Pandas 读取数据进行详细分析
        print("\n正在加载数据...")
        df = pd.read_parquet(file_path)
        print(f"数据加载完成: {len(df):,} 行 x {len(df.columns)} 列")
        
        # 使用临时选项来控制输出
        with pd.option_context(
            'display.max_columns', preview_cols,
            'display.max_rows', sample_rows,
            'display.max_colwidth', 50,
            'display.width', 120,
            'display.max_seq_items', 5,
            'display.show_dimensions', False
        ):
            print("\n【数据预览】")
            print(f"前 {sample_rows} 行 (只显示前 {preview_cols} 列):")
            preview_df = df.iloc[:sample_rows, :preview_cols].copy()
            
            # 对每列进行截断处理
            for col in preview_df.columns:
                if preview_df[col].dtype == 'object':  # 字符串类型
                    preview_df[col] = preview_df[col].apply(lambda x: truncate_text(x, 50))
            
            print(preview_df.to_string())
            if len(df.columns) > preview_cols:
                print(f"\n(省略 {len(df.columns) - preview_cols} 列，共 {len(df.columns)} 列)")
        
        # 4. 每列的详细信息
        print("\n【字段详细分析】")
        print("=" * 80)
        
        for col in df.columns:
            print(f"\n字段名: {col}")
            print(f"数据类型: {df[col].dtype}")
            print(f"非空值数量: {df[col].count():,} / {len(df):,}")
            print(f"空值数量: {df[col].isna().sum():,}")
            print(f"空值比例: {df[col].isna().sum() / len(df) * 100:.2f}%")
            
            if show_stats:
                # 数值型字段统计
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"数值统计:")
                    print(f"  最小值: {df[col].min()}")
                    print(f"  最大值: {df[col].max()}")
                    print(f"  平均值: {df[col].mean():.2f}")
                    print(f"  中位数: {df[col].median():.2f}")
                    print(f"  标准差: {df[col].std():.2f}")
                
                # 分类/文本型字段统计
                else:
                    unique_count = df[col].nunique()
                    print(f"唯一值数量: {unique_count:,}")
                    
                    if unique_count <= 20:  # 如果唯一值不多,显示所有值的分布
                        print(f"值分布:")
                        value_counts = df[col].value_counts()
                        for val, count in value_counts.items():
                            percentage = count / len(df) * 100
                            val_str = truncate_text(val, max_sample_length)
                            print(f"  {val_str}: {count:,} ({percentage:.2f}%)")
                    else:  # 如果唯一值很多,只显示 Top 10
                        print(f"Top 10 最常见的值:")
                        value_counts = df[col].value_counts().head(10)
                        for val, count in value_counts.items():
                            percentage = count / len(df) * 100
                            val_str = truncate_text(val, max_sample_length)
                            print(f"  {val_str}: {count:,} ({percentage:.2f}%)")
                
                # 样本值（截断长文本，只显示3个样本）
                sample_values = df[col].dropna().head(3).tolist()
                print(f"样本值 (前3个，最多{max_sample_length}字符):")
                for i, val in enumerate(sample_values, 1):
                    val_str = truncate_text(val, max_sample_length)
                    print(f"  {i}. {val_str}")
            
            print("-" * 80)
        
        # 5. 内存使用情况
        print("\n【内存使用情况】")
        memory_usage = df.memory_usage(deep=True)
        print(f"总内存使用: {memory_usage.sum() / 1024 / 1024:.2f} MB")
        print("各列内存使用 (Top 10):")
        sorted_mem = memory_usage.sort_values(ascending=False)
        for col, mem in sorted_mem.head(10).items():
            if col != 'Index':
                print(f"  {col}: {mem / 1024 / 1024:.2f} MB")
        
        # 6. 数据质量概览
        print("\n【数据质量概览】")
        print(f"总行数: {len(df):,}")
        print(f"完全重复的行数: {df.duplicated().sum():,}")
        print(f"至少有一个空值的行数: {df.isna().any(axis=1).sum():,}")
        
        return df
    
    finally:
        # 恢复标准输出
        if tee:
            sys.stdout = original_stdout
            tee.close()


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='探索 Parquet 文件')
    parser.add_argument('file_path', nargs='?', 
                       default="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/part_0.parquet",
                       help='Parquet 文件路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出文件路径（如果不指定，会使用默认名称：<原文件名>_explore_<时间戳>.txt）')
    parser.add_argument('--sample_rows', type=int, default=2,
                       help='预览行数（默认：2）')
    parser.add_argument('--preview_cols', type=int, default=3,
                       help='预览列数（默认：3）')
    parser.add_argument('--show_stats', action='store_true',
                       help='显示详细统计信息（默认：False）')
    parser.add_argument('--max_sample_length', type=int, default=50,
                       help='样本值最大长度（默认：50）')
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，生成默认文件名（保存在当前目录，不在数据源目录）
    if args.output is None:
        input_path = Path(args.file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 输出到当前目录，避免写入数据源目录
        output_file = Path.cwd() / f"{input_path.stem}_explore_{timestamp}.txt"
    else:
        output_file = Path(args.output)
        # 确保输出文件不在数据源目录
        input_path = Path(args.file_path)
        if output_file.parent == input_path.parent:
            # 如果输出目录和数据源目录相同，改为当前目录
            output_file = Path.cwd() / output_file.name
            print(f"警告: 输出文件不能在数据源目录，已改为: {output_file}")
    
    # 探索文件
    df = explore_parquet(
        args.file_path,
        sample_rows=args.sample_rows,
        show_stats=args.show_stats,
        max_sample_length=args.max_sample_length,
        preview_cols=args.preview_cols,
        output_file=str(output_file.resolve())
    )
    
    # 如果需要进一步分析,可以继续使用返回的 DataFrame
    # 例如:
    # print(df.describe())
    # print(df.info())
