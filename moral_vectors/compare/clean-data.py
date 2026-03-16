import pandas as pd
import numpy as np

def clean_social_chem_data(original_data_path, clean_data_path):

    df = pd.read_csv(original_data_path, sep='\t')
    original_lens = len(df)
    df = df[df['rot-bad'] == 0] # remove rows with bad rule-of-thumb
    df = df[df['m'] == 1] # keep only rows with one worker completing the judgment
    # remove rows with missing required columns
    required_cols = ['action', 'rot-moral-foundations', 'action-moral-judgment', 'action-agree']
    df = df.dropna(subset=required_cols)

    # map moral foundations
    mft_mapping = {
        'care-harm': [0, 1],
        'fairness-cheating': [2, 3],
        'loyalty-betrayal': [4, 5],
        'authority-subversion': [6, 7],
        'sanctity-degradation': [8, 9],
    }
    def get_mft_vector(foundation_str):
        vec = [0] * 10
        if isinstance(foundation_str, str):
            parts = foundation_str.split('|')
            for part in parts:
                part = part.strip()
                if part in mft_mapping:
                    indices = mft_mapping[part]
                    vec[indices[0]] = 1
                    vec[indices[1]] = 1
        return vec

    df['mft-vector'] = df['rot-moral-foundations'].apply(get_mft_vector)
    df.to_csv(clean_data_path, index=False)
    print(f"After cleaning, {len(df)} rows remain out of {original_lens} original rows")
    print(f"Cleaned data saved to {clean_data_path}")

if __name__ == "__main__":
    original_data_path = "./data/compare/social-chem-101.v1.0.tsv"
    clean_data_path = "./data/social-chem-cleaned.tsv"
    clean_social_chem_data(original_data_path, clean_data_path)
