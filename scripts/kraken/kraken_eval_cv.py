import os
import pandas as pd
from PIL import Image
from kraken.lib import models
from kraken.rpred import rpred
from kraken.containers import Segmentation
import unicodedata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'kraken_models', 'cv1', 'kraken_cv1__best.mlmodel')
TEST_FILE = os.path.join(PROJECT_ROOT, 'sets', 'cv1', 'kraken_cv1_test_clean.txt')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'kraken_cv1_evaluation.csv')

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


try:
    model = models.load_any(MODEL_PATH, device='cpu')
    print("Model loaded successfully")
except Exception as e:
    exit()

data_list = []
if not os.path.exists(TEST_FILE):
    exit()

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if '\t' in line:
            p, t = line.split('\t', 1)
            data_list.append((p, t))
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                data_list.append((parts[0], parts[1]))

print(f"Read {len(data_list)} test entries")

total_chars = 0
total_edits = 0
correct_count = 0
results = []


for i, (img_path, gt_text) in enumerate(data_list):
    pred_text = ""
    try:
        if not os.path.exists(img_path):
            continue

        im = Image.open(img_path)
        w, h = im.size
        
        bounds = Segmentation(
            type='bbox',
            lines=[{
                'id': 'line_1', 
                'bbox': (0, 0, w, h),
                'tags': {'type': 'default'}
            }],
            text_direction='horizontal-lr',
            script_detection=False,
            imagename=os.path.basename(img_path)
        )
        
        pred = rpred(model, im, bounds)
        for record in pred:
            if record.prediction:
                pred_text += record.prediction
        
        pred_text = pred_text.strip()
        
        gt_norm = unicodedata.normalize('NFC', gt_text)
        pred_norm = unicodedata.normalize('NFC', pred_text)
        
        dist = levenshtein_distance(gt_norm, pred_norm)
        length = len(gt_norm)
        
        total_edits += dist
        total_chars += length
        
        if gt_norm == pred_norm:
            correct_count += 1
        
        results.append({
            'Image_ID': os.path.basename(img_path),
            'Ground_Truth': gt_norm,
            'Prediction': pred_norm,
            'CER': dist / length if length > 0 else 1.0
        })
        
        if (i+1) % 50 == 0:
            print(f"Progress: {i+1}/{len(data_list)}")
            
    except Exception as e:
        print(f"Error identifying {os.path.basename(img_path)}: {e}")

print("\n" + "="*40)
if total_chars > 0:
    final_cer = total_edits / total_chars
    final_acc = correct_count / len(data_list)
    
    print(f"Evaluation Complete (CV1)")
    print(f"Character Error Rate (CER): {final_cer:.4%}")
    print(f"Accuracy (Exact Match):     {final_acc:.4%}")
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nEvaluation results saved to: {OUTPUT_CSV}")
else:
    print("Calculation failed: No valid data processed")
print("="*40)