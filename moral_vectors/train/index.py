import pandas as pd
import json
from collections import defaultdict

def build_split_buckets(csv_path, output_json):
    df = pd.read_csv(csv_path)
    dimensions = ['care-harm', 'fairness-cheating', 'loyalty-betrayal', 
                  'authority-subversion', 'sanctity-degradation']
    
    buckets = {dim: {"virtue": defaultdict(list), "vice": defaultdict(list), "neutral": defaultdict(list)} 
               for dim in dimensions}

    for _, row in df.iterrows():
        rid = int(row['row_id'])
        dim = row['target_dimension']
        if dim not in dimensions: continue
            
        v_score, a_score = float(row['m_virtue']), float(row['m_vice'])
        
        if v_score > 0:
            pol, score = "virtue", v_score
        elif a_score > 0:
            pol, score = "vice", a_score
        else:
            pol, score = "neutral", 0.0
            
        buckets[dim][pol][str(score)].append(rid)

    # 转换为标准 dict
    final_buckets = {d: {p: dict(buckets[d][p]) for p in ["virtue", "vice", "neutral"]} for d in dimensions}
    
    with open(output_json, 'w') as f:
        json.dump(final_buckets, f)
    print(f"索引桶已保存至: {output_json}")

# 为 Train 和 Val 分别构建桶
build_split_buckets("./data/train/social_chem_train_expanded.csv", "./data/train/train_buckets.json")
build_split_buckets("./data/train/social_chem_val_expanded.csv", "./data/train/val_buckets.json")