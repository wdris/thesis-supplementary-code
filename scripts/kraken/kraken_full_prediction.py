import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

files = {
    'cv1': 'kraken_cv1_evaluation.csv',
    'cv2': 'kraken_cv2_evaluation.csv',
    'cv3': 'kraken_cv3_evaluation.csv',
    'cv4': 'kraken_cv4_evaluation.csv'
}

all_data = []

print(f"Reading files from directory: {DATA_DIR}")

for fold_name, file_name in files.items():
    full_path = os.path.join(DATA_DIR, file_name)
    
    if os.path.exists(full_path):
        try:
            df = pd.read_csv(full_path)
            df['Source_Fold'] = fold_name
            all_data.append(df)
            print(f"Successfully read: {file_name} ({len(df)} lines)")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    else:
        print(f"File not found: {full_path}")

if not all_data:
    exit()

full_df = pd.concat(all_data, ignore_index=True)

cols = ['Image_ID', 'Source_Fold', 'Ground_Truth', 'Prediction', 'CER']

if 'Prediction' in full_df.columns:
    final_df = full_df[cols]
else:
    print("\nWarning: Column names do not match expected structure.")
    final_df = full_df

output_path = os.path.join(DATA_DIR, 'kraken_cross_validation_full.csv')
final_df.to_csv(output_path, index=False)


print(f"Average CER: {final_df['CER'].mean():.4f}")