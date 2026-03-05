import os
import re
from pathlib import Path

def analyze_document_lengths(file_paths, encoding='utf-8'):
    """
    统计文档长度分布
    
    Args:
        file_paths: 文件路径列表
        encoding: 文件编码
    
    Returns:
        统计结果字典
    """
    lengths = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for content in f:
                    lengths.append(len(re.split(r"\s+", content)))
        except Exception as e:
            print(f"读取 {file_path} 失败: {e}")

    print("data loaded")
    
    # 排序
    lengths.sort()
    total_docs = len(lengths)
    
    if total_docs == 0:
        return None
    
    # 计算百分位数
    p99_index = int(total_docs * 0.99)
    p95_index = int(total_docs * 0.95)
    p50_index = int(total_docs * 0.50)
    
    return {
        'total_documents': total_docs,
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_length': sum(lengths) / total_docs,
        'median_length': lengths[p50_index],
        'p95_length': lengths[p95_index],
        'p99_length': lengths[p99_index],
        'docs_below_p99': p99_index,
        'docs_above_p99': total_docs - p99_index,
        'lengths': lengths
    }

if __name__ == "__main__":
    print(analyze_document_lengths(["data/court_considerations.csv"]))