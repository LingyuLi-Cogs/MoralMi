import pandas as pd
import ast
import json
import random

def sample_moral_vectors_stratified(input_path, output_path):
    
    df = pd.read_csv(input_path)
    df['moral_vector'] = df['moral_vector'].apply(ast.literal_eval)
    
    dimensions = [
        'care-harm',
        'fairness-cheating', 
        'loyalty-betrayal',
        'authority-subversion',
        'sanctity-degradation'
    ]
    
    dim_to_indices = {
        'care-harm': (0, 1),
        'fairness-cheating': (2, 3),
        'loyalty-betrayal': (4, 5),
        'authority-subversion': (6, 7),
        'sanctity-degradation': (8, 9)
    }
    
    # 归属度分层：非典型 vs 典型
    non_typical_scores = [0.125, 0.25, 0.375, 0.5, 0.75]
    typical_score = 1.0
    
    # 采样配额
    per_score_sample = 300
    typical_sample = 150
    neutral_sample_size = 300
    
    # 展开记录
    expanded_records = []
    for _, row in df.iterrows():
        foundations = row['rot-moral-foundations'].split('|')
        for foundation in foundations:
            foundation = foundation.strip()
            if foundation in dimensions:
                expanded_records.append({
                    'id': row['id'],
                    'action': row['action'],
                    'rot-moral-foundations': row['rot-moral-foundations'],
                    'moral_vector': row['moral_vector'],
                    'm_virtue': row['m_virtue'],
                    'm_vice': row['m_vice'],
                    'dimension': foundation
                })
    
    print(f"展开后共 {len(expanded_records)} 条记录\n")
    
    sampled_data = []
    
    for dim in dimensions:
        print(f"=== {dim} ===")
        pos_idx, neg_idx = dim_to_indices[dim]
        dim_records = [r for r in expanded_records if r['dimension'] == dim]
        
        # 分离正性、负性、中性记录
        virtue_records = [r for r in dim_records if r['moral_vector'][pos_idx] > 0]
        vice_records = [r for r in dim_records if r['moral_vector'][neg_idx] > 0]
        neutral_records = [r for r in dim_records if r['moral_vector'][pos_idx] == 0 and r['moral_vector'][neg_idx] == 0]
        
        dim_sampled = 0
        
        # 正性事件按归属度分层采样
        for score in non_typical_scores:
            score_records = [r for r in virtue_records if abs(r['moral_vector'][pos_idx] - score) < 1e-6]
            sample = random.sample(score_records, min(per_score_sample, len(score_records)))
            for r in sample:
                r['sample_type'] = 'virtue'
                r['sampled_dimension'] = dim
                r['score_stratum'] = score
            sampled_data.extend(sample)
            dim_sampled += len(sample)
            print(f"  Virtue score={score}: {len(score_records)} available, sampled {len(sample)}")
        
        # 正性典型事件
        typical_virtue = [r for r in virtue_records if abs(r['moral_vector'][pos_idx] - typical_score) < 1e-6]
        sample = random.sample(typical_virtue, min(typical_sample, len(typical_virtue)))
        for r in sample:
            r['sample_type'] = 'virtue_typical'
            r['sampled_dimension'] = dim
            r['score_stratum'] = typical_score
        sampled_data.extend(sample)
        dim_sampled += len(sample)
        print(f"  Virtue typical (1.0): {len(typical_virtue)} available, sampled {len(sample)}")
        
        # 负性事件按归属度分层采样
        for score in non_typical_scores:
            score_records = [r for r in vice_records if abs(r['moral_vector'][neg_idx] - score) < 1e-6]
            sample = random.sample(score_records, min(per_score_sample, len(score_records)))
            for r in sample:
                r['sample_type'] = 'vice'
                r['sampled_dimension'] = dim
                r['score_stratum'] = score
            sampled_data.extend(sample)
            dim_sampled += len(sample)
            print(f"  Vice score={score}: {len(score_records)} available, sampled {len(sample)}")
        
        # 负性典型事件
        typical_vice = [r for r in vice_records if abs(r['moral_vector'][neg_idx] - typical_score) < 1e-6]
        sample = random.sample(typical_vice, min(typical_sample, len(typical_vice)))
        for r in sample:
            r['sample_type'] = 'vice_typical'
            r['sampled_dimension'] = dim
            r['score_stratum'] = typical_score
        sampled_data.extend(sample)
        dim_sampled += len(sample)
        print(f"  Vice typical (1.0): {len(typical_vice)} available, sampled {len(sample)}")
        
        # 中性事件
        sample = random.sample(neutral_records, min(neutral_sample_size, len(neutral_records)))
        for r in sample:
            r['sample_type'] = 'neutral'
            r['sampled_dimension'] = dim
            r['score_stratum'] = 0.0
        sampled_data.extend(sample)
        dim_sampled += len(sample)
        print(f"  Neutral: {len(neutral_records)} available, sampled {len(sample)}")
        
        print(f"  维度合计: {dim_sampled} 条\n")
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in sampled_data:
            output_record = {
                'id': record['id'],
                'action': record['action'],
                'rot-moral-foundations': record['rot-moral-foundations'],
                'moral_vector': record['moral_vector'],
                'm_virtue': record['m_virtue'],
                'm_vice': record['m_vice'],
                'sampled_dimension': record['sampled_dimension'],
                'sample_type': record['sample_type'],
                'score_stratum': record['score_stratum']
            }
            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    
    print(f"共采样 {len(sampled_data)} 条记录")
    print(f"保存至 {output_path}")
    
    # 输出统计摘要
    print("\n=== 采样统计 ===")
    for dim in dimensions:
        dim_data = [r for r in sampled_data if r['sampled_dimension'] == dim]
        virtue_cnt = len([r for r in dim_data if r['sample_type'] == 'virtue'])
        virtue_typical_cnt = len([r for r in dim_data if r['sample_type'] == 'virtue_typical'])
        vice_cnt = len([r for r in dim_data if r['sample_type'] == 'vice'])
        vice_typical_cnt = len([r for r in dim_data if r['sample_type'] == 'vice_typical'])
        neutral_cnt = len([r for r in dim_data if r['sample_type'] == 'neutral'])
        print(f"{dim}: Virtue={virtue_cnt}, Virtue_typical={virtue_typical_cnt}, Vice={vice_cnt}, Vice_typical={vice_typical_cnt}, Neutral={neutral_cnt}")

if __name__ == "__main__":
    random.seed(42)
    input_path = "./data/compare/human-representation-vectors.tsv"
    output_path = "./data/compare/sampled-human-representation-vectors.jsonl"
    sample_moral_vectors_stratified(input_path, output_path)