import os
import json
import glob
import re
import pandas as pd
import jiwer
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_FOLDER = os.path.join(PROJECT_ROOT, "data", "line_images_normalized")
PREDICTION_FOLDER = os.path.join(PROJECT_ROOT, "results", "gemini_flash_output_mode4")


output_name = os.path.basename(PREDICTION_FOLDER) + "_comprehensive_report.csv"
REPORT_SAVE_PATH = os.path.join(PROJECT_ROOT, "results", output_name)



def get_gt_path(json_filename):
    base_name = os.path.basename(json_filename)
    file_id = base_name.replace(".json", "")
    gt_filename = f"{file_id}.gt.txt"
    return os.path.join(GT_FOLDER, gt_filename)

def normalize_text(text, mode):
    if text is None: text = ""
    text = str(text).strip()
    
    if mode == 'baseline':
        return text
    
    elif mode == 'lower':
        return text.lower()
    
    elif mode == 'nospace':
        return text.replace(" ", "").replace("\t", "").replace("\n", "")
    
    elif mode == 'alnum':
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    elif mode == 'alnumlowernospace':
        text = text.lower()
        text = re.sub(r'[^a-z0-9]', '', text) 
        return text
    
    else:
        return text

def calculate_comprehensive_metrics():
    
    json_files = glob.glob(os.path.join(PREDICTION_FOLDER, "*.json"))
    if not json_files:
        return

    # define 5 dimensions
    metric_types = ['baseline', 'lower', 'nospace', 'alnum', 'alnumlowernospace']
    

    totals = {m: {'dist': 0, 'len': 0} for m in metric_types}
    
    row_data = []

    for json_file in tqdm(json_files, desc="Processing"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            continue

        file_name = data.get("file_name", os.path.basename(json_file).replace(".json", ".png"))
        raw_pred = data.get("prediction", "")
        
        gt_path = get_gt_path(json_file)
        if not os.path.exists(gt_path): continue
        with open(gt_path, "r", encoding="utf-8") as f:
            raw_gt = f.read().strip()


        file_metrics = {"Filename": file_name, "Raw_GT": raw_gt, "Raw_Pred": raw_pred}
        
        for m_type in metric_types:
            norm_gt = normalize_text(raw_gt, m_type)
            norm_pred = normalize_text(raw_pred, m_type)
            
            length = len(norm_gt)
            if length == 0:
                cer = 1.0 if len(norm_pred) > 0 else 0.0
                dist = len(norm_pred)
            else:
                cer = jiwer.cer(norm_gt, norm_pred)
                dist = int(round(cer * length))
            
    
            file_metrics[f"CER_{m_type}"] = round(cer, 4)
            
            totals[m_type]['dist'] += dist
            totals[m_type]['len'] += length
            
        row_data.append(file_metrics)

  
    df = pd.DataFrame(row_data)
    
    df.to_csv(REPORT_SAVE_PATH, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*60)
    print("-" * 60)
    print(f"{'Metric Type':<20}, {'Micro CER':<15}, {'Macro CER':<15}")
    print("-" * 60)
    
    summary_stats = []
    
    for m_type in metric_types:

        t_dist = totals[m_type]['dist']
        t_len = totals[m_type]['len']
        micro = t_dist / t_len if t_len > 0 else 0.0
        
        macro = df[f"CER_{m_type}"].mean()
        
        print(f"{m_type:<20} | {micro:.4%}        | {macro:.4%}")
        
        summary_stats.append({
            "Metric": m_type,
            "Micro_CER": micro,
            "Macro_CER": macro
        })
        
    print("-" * 60)
    print(f" report saved {REPORT_SAVE_PATH}")
    print("="*60)

if __name__ == "__main__":
    calculate_comprehensive_metrics()