import pandas as pd
import numpy as np
import ast

def compute_human_moral_vectors(clean_data_path, output_path):
    
    df = pd.read_csv(clean_data_path)
    
    # 为每条数据生成唯一ID
    df['id'] = [f"socialchem_{i:06d}" for i in range(len(df))]
    
    # "[1, 1, 0...]" -> [1, 1, 0...]
    df['mft-vector'] = df['mft-vector'].apply(ast.literal_eval)
    # action-moral-judgment: [-2, 2] -> v_score: [-1, 1]
    v_score = df['action-moral-judgment'] / 2.0    
    # action-agree: [0, 4] -> w_conf: [0, 1]
    w_conf = df['action-agree'] / 4.0
    # ReLU
    m_virtue = (v_score.apply(lambda x: max(0, x))) * w_conf
    m_vice = (v_score.apply(lambda x: max(0, -x))) * w_conf

    def apply_scores_to_vector(row):
        base_vec = row['mft-vector']
        m_p = row['m_virtue']
        m_n = row['m_vice']        
        final_vec = []
        for i, val in enumerate(base_vec):
            if val == 0:
                final_vec.append(0.0)
                continue            
            # for even indices (0, 2, 4, 6, 8), apply Positive Score
            if i % 2 == 0:
                final_vec.append(m_p)
            # for odd indices (1, 3, 5, 7, 9), apply Negative Score
            else:
                final_vec.append(m_n)
        return final_vec

    df['m_virtue'] = m_virtue
    df['m_vice'] = m_vice
    df['moral_vector'] = df.apply(apply_scores_to_vector, axis=1)
    
    final_df = df[['id', 'action', 'rot-moral-foundations', 'moral_vector', 'm_virtue', 'm_vice']]
    final_df.to_csv(output_path, index=False)
    print(f"Computed vectors for {len(final_df)} rows.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    clean_data_path = "./data/social-chem-cleaned.tsv"
    output_path = "./data/compare/human-representation-vectors.tsv"
    compute_human_moral_vectors(clean_data_path, output_path)