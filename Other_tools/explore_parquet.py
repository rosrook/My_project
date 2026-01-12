import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from collections import Counter

def explore_parquet(file_path, sample_rows=5, show_stats=True):
    """
    探索 Parquet 文件的内容和结构
    
    参数:
        file_path: Parquet 文件路径
        sample_rows: 显示的样本行数
        show_stats: 是否显示详细统计信息
    """
    
    print("=" * 80)
    print(f"探索 Parquet 文件: {file_path}")
    print("=" * 80)
    
    # 1. 使用 PyArrow 读取元数据
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
    df = pd.read_parquet(file_path)
    
    print("\n【数据预览】")
    print(f"\n前 {sample_rows} 行:")
    print(df.head(sample_rows))
    
    print(f"\n后 {sample_rows} 行:")
    print(df.tail(sample_rows))
    
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
                print(f"\n数值统计:")
                print(f"  最小值: {df[col].min()}")
                print(f"  最大值: {df[col].max()}")
                print(f"  平均值: {df[col].mean():.2f}")
                print(f"  中位数: {df[col].median():.2f}")
                print(f"  标准差: {df[col].std():.2f}")
                print(f"  四分位数:")
                print(f"    25%: {df[col].quantile(0.25):.2f}")
                print(f"    50%: {df[col].quantile(0.50):.2f}")
                print(f"    75%: {df[col].quantile(0.75):.2f}")
            
            # 分类/文本型字段统计
            else:
                unique_count = df[col].nunique()
                print(f"唯一值数量: {unique_count:,}")
                
                if unique_count <= 20:  # 如果唯一值不多,显示所有值的分布
                    print(f"\n值分布:")
                    value_counts = df[col].value_counts()
                    for val, count in value_counts.items():
                        percentage = count / len(df) * 100
                        print(f"  {val}: {count:,} ({percentage:.2f}%)")
                else:  # 如果唯一值很多,只显示 Top 10
                    print(f"\nTop 10 最常见的值:")
                    value_counts = df[col].value_counts().head(10)
                    for val, count in value_counts.items():
                        percentage = count / len(df) * 100
                        print(f"  {val}: {count:,} ({percentage:.2f}%)")
            
            # 样本值
            sample_values = df[col].dropna().head(5).tolist()
            print(f"\n样本值: {sample_values}")
        
        print("-" * 80)
    
    # 5. 内存使用情况
    print("\n【内存使用情况】")
    memory_usage = df.memory_usage(deep=True)
    print(f"总内存使用: {memory_usage.sum() / 1024 / 1024:.2f} MB")
    print("\n各列内存使用:")
    for col, mem in memory_usage.items():
        if col != 'Index':
            print(f"  {col}: {mem / 1024 / 1024:.2f} MB")
    
    # 6. 数据质量概览
    print("\n【数据质量概览】")
    print(f"总行数: {len(df):,}")
    print(f"完全重复的行数: {df.duplicated().sum():,}")
    print(f"至少有一个空值的行数: {df.isna().any(axis=1).sum():,}")
    
    return df


# 使用示例
if __name__ == "__main__":
    # 替换为你的 Parquet 文件路径
    file_path = "your_file.parquet"
    
    # 探索文件
    df = explore_parquet(file_path, sample_rows=5, show_stats=True)
    
    # 如果需要进一步分析,可以继续使用返回的 DataFrame
    # 例如:
    # print(df.describe())
    # print(df.info())