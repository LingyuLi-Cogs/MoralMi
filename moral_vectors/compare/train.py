import pandas as pd
import numpy as np
import os

def split_and_expand(input_path, output_dir, train_ratio=0.8, val_ratio=0.1):
    print(f"开始处理原始数据: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. 以 id 为基准进行划分
    unique_ids = df['id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_ids)
    
    n = len(unique_ids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_id_set = set(unique_ids[:train_end])
    val_id_set = set(unique_ids[train_end:val_end])
    test_id_set = set(unique_ids[val_end:])
    
    # 2. 处理维度展开逻辑
    df['rot-moral-foundations'] = df['rot-moral-foundations'].fillna('neutral')
    df['foundations_list'] = df['rot-moral-foundations'].apply(lambda x: x.split('|'))
    df_expanded = df.explode('foundations_list')
    df_expanded = df_expanded.rename(columns={'foundations_list': 'target_dimension'})
    
    # 3. 分配到对应的子集
    def get_split(row_id):
        if row_id in train_id_set: return 'train'
        if row_id in val_id_set: return 'val'
        return 'test'
    
    df_expanded['split'] = df_expanded['id'].apply(get_split)
    
    # 4. 导出三个文件
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ['train', 'val', 'test']:
        subset = df_expanded[df_expanded['split'] == split_name].copy()
        subset = subset.reset_index(drop=True)
        # 为每个子集分配局部 train_id（其实叫 row_id 更准确）
        subset.insert(0, 'row_id', subset.index) 
        
        output_file = os.path.join(output_dir, f"social_chem_{split_name}_expanded.csv")
        subset.to_csv(output_file, index=False)
        print(f"已导出 {split_name} 集: {len(subset)} 条数据 (原始 ID 数: {len(subset['id'].unique())})")

if __name__ == "__main__":
    split_and_expand("./moral_vectors/compare/human-representation-vectors.tsv", "./moral_vectors/compare/train-val-test-split")