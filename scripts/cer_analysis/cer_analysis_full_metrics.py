import pandas as pd
import unicodedata
import os



def get_levenshtein_distance(s1, s2):
    """
    Calculates Levenshtein distance. 
    """
    try:
        import editdistance
        return editdistance.eval(s1, s2)
    except ImportError:
        if len(s1) < len(s2):
            return get_levenshtein_distance(s2, s1)
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


def filter_nospace(text):
    """Removes all Unicode Category Z (Separator) characters."""
    return "".join([c for c in text if not unicodedata.category(c).startswith("Z")])

def filter_lower(text):
    """Converts text to lowercase."""
    return text.lower()

def filter_alnum(text):
    """Retains only Unicode L (Letter), N (Number), and M (Mark) categories."""
    allowed = ('L', 'N', 'M')
    return "".join([c for c in text if unicodedata.category(c).startswith(allowed)])

def filter_combined(text):
    """Combined filter: Alphanumeric + Lowercase + No Whitespace."""
    t = filter_alnum(text)
    t = filter_lower(t)
    t = filter_nospace(t)
    return t



def calculate_cer(gt, pred):
    dist = get_levenshtein_distance(gt, pred)
    length = len(gt)
    return dist, length, (dist / length if length > 0 else 0.0)

def analyze_single_row(gt, pred):
    """Calculates all CER metric dimensions."""
    gt = str(gt) if pd.notna(gt) else ""
    pred = str(pred) if pd.notna(pred) else ""

    results = {}

    # Base
    d, l, cer_val = calculate_cer(gt, pred)
    results['CER_Base'] = cer_val
    results['Len_Base'] = l
    results['Dist_Base'] = d

    # NoSpace Metrics
    gt_ns, pred_ns = filter_nospace(gt), filter_nospace(pred)
    _, _, cer_ns = calculate_cer(gt_ns, pred_ns)
    results['CER_NoSpace'] = cer_ns

    # Lowercase Metrics
    gt_lo, pred_lo = filter_lower(gt), filter_lower(pred)
    _, _, cer_lo = calculate_cer(gt_lo, pred_lo)
    results['CER_Lower'] = cer_lo

    # Alphanumeric Metrics
    gt_al, pred_al = filter_alnum(gt), filter_alnum(pred)
    d_al, l_al, cer_al = calculate_cer(gt_al, pred_al)
    results['CER_Alnum'] = cer_al
    results['Len_Alnum'] = l_al
    results['Dist_Alnum'] = d_al

    # Combined 
    gt_co, pred_co = filter_combined(gt), filter_combined(pred)
    _, _, cer_co = calculate_cer(gt_co, pred_co)
    results['CER_Alnumlowernospace'] = cer_co

   
    if results['Dist_Base'] == 0:
        results['Error_Type'] = "Perfect"
    elif results['Dist_Alnum'] == 0:
        results['Error_Type'] = "Formatting Error"
    else:
        results['Error_Type'] = "Content Error"

    return results




def main():

    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        project_root = os.getcwd()
    
    print(f"Working Directory: {project_root}")

  
    files_config = [
        {'filename': 'kraken_cross_validation_full.csv', 'tool': 'Kraken', 'cols': {'gt': 'Ground_Truth', 'pred': 'Prediction', 'id': 'Image_ID'}},
        {'filename': 'ppocr_zs_with_cer.csv', 'tool': 'PPOCR', 'cols': {'gt': 'GroundTruth', 'pred': 'Prediction', 'id': 'Filename'}},
        {'filename': 'trocr_with_cer.csv', 'tool': 'TrOCR', 'cols': {'gt': 'Ground_Truth', 'pred': 'TrOCR_Prediction', 'id': 'Image_ID'}}
    ]

    all_rows = []
    global_stats = {} 

    for config in files_config:
        file_path = os.path.join(project_root, config['filename'])
        tool = config['tool']
        
        if not os.path.exists(file_path):
            print(f"Skipping: {config['filename']} not found.")
            continue
            
        print(f"Processing Tool: {tool}...")
        df = pd.read_csv(file_path)
        
      
        global_stats[tool] = {m: {'dist': 0, 'len': 0} for m in ['Base', 'NoSpace', 'Lower', 'Alnum', 'Alnumlowernospace']}

        for idx, row in df.iterrows():
            gt = row.get(config['cols']['gt'], "")
            pred = row.get(config['cols']['pred'], "")
            img_id = row.get(config['cols']['id'], f"row_{idx}")

            gt_str = str(gt) if pd.notna(gt) else ""
            pred_str = str(pred) if pd.notna(pred) else ""
            


            metrics_mapping = {
                'Base': (gt_str, pred_str),
                'NoSpace': (filter_nospace(gt_str), filter_nospace(pred_str)),
                'Lower': (filter_lower(gt_str), filter_lower(pred_str)),
                'Alnum': (filter_alnum(gt_str), filter_alnum(pred_str)),
                'Alnumlowernospace': (filter_combined(gt_str), filter_combined(pred_str))
            }

            for m_name, (g_proc, p_proc) in metrics_mapping.items():
                d, l, _ = calculate_cer(g_proc, p_proc)
                global_stats[tool][m_name]['dist'] += d
                global_stats[tool][m_name]['len'] += l

            # generate record for CSV export
            res = analyze_single_row(gt, pred)
            record = {
                'Tool': tool,
                'Image_ID': img_id,
                'Ground_Truth': gt,
                'Prediction': pred,
                'CER_Base': res['CER_Base'],
                'CER_NoSpace': res['CER_NoSpace'],
                'CER_Lower': res['CER_Lower'],
                'CER_Alnum': res['CER_Alnum'],
                'CER_Alnumlowernospace': res['CER_Alnumlowernospace'],
                'Error_Type': res['Error_Type']
            }
            all_rows.append(record)

  
    if all_rows:
        out_df = pd.DataFrame(all_rows)
        save_path = os.path.join(project_root, "cer_full_metrics_detailed.csv")
        out_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\nDetailed analysis saved to: {save_path}")

       
        print("\n" + "="*70)
        header = f"{'TOOL':<10} {'BASE':<10} {'NOSPACE':<10} {'LOWER':<10} {'ALNUM':<10} {'COMBINED':<10}"
        print(header)
        print("-" * 70)
        
        summary_rows = []
        for tool, metrics in global_stats.items():
            row_str = f"{tool:<10} "
            row_data = {'Tool': tool}
            for m_name in ['Base', 'NoSpace', 'Lower', 'Alnum', 'Alnumlowernospace']:
                d = metrics[m_name]['dist']
                l = metrics[m_name]['len']
                cer_val = d / l if l > 0 else 0.0
                row_str += f"{cer_val:.2%}      "
                row_data[m_name] = cer_val
            print(row_str)
            summary_rows.append(row_data)
        print("="*70)
        

        pd.DataFrame(summary_rows).to_csv(os.path.join(project_root, "cer_summary_report.csv"), index=False)
    else:
        print("Error: No data was processed.")

if __name__ == "__main__":
    main()