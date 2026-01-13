import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'line_images_normalized')
NORMALIZED_GT_FILE = os.path.join(DATA_DIR, 'ground_truth', 'transcription_normalized.txt')
SETS_DIR = os.path.join(PROJECT_ROOT, 'sets')

def load_gt_dict():
    """
    Reads the normalized GT file and loads it into a dictionary.
    Format: { '270-01': 'Letters, Orders and Instructions...' }
    """
    gt_map = {}
    if not os.path.exists(NORMALIZED_GT_FILE):
        print(f" Error: GT file not found at {NORMALIZED_GT_FILE}")
        return {}

    with open(NORMALIZED_GT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
    
            parts = line.split(' ', 1)
            if len(parts) == 2:
                gt_map[parts[0]] = parts[1]
    
    return gt_map

def generate_list_for_split(cv_folder, split_name, gt_map):
    """
    Reads sets/cvX/train.txt (etc.), and generates a Kraken-compatible list file.
    Kraken Format: Absolute_Image_Path [TAB] Text_Content
    """

    id_file_path = os.path.join(cv_folder, f'{split_name}.txt')
    
    output_path = os.path.join(cv_folder, f'kraken_{split_name}.txt')
    
    if not os.path.exists(id_file_path):
        print(f" Skipping: ID file not found at {id_file_path}")
        return

    valid_entries = []
    missing_img_count = 0
    missing_gt_count = 0

    with open(id_file_path, 'r', encoding='utf-8') as f:

        ids = [line.strip() for line in f if line.strip()]

    for line_id in ids:
        img_name_png = f"{line_id}.png"
        img_path = os.path.join(IMAGES_DIR, img_name_png)
        
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGES_DIR, f"{line_id}.jpg")
        

        if not os.path.exists(img_path):
            missing_img_count += 1
            continue



        if line_id not in gt_map:
            missing_gt_count += 1
            print(f"  -> Missing GT: {line_id}")
            continue

        text = gt_map[line_id]
        entry = f"{os.path.abspath(img_path)}\t{text}"
        valid_entries.append(entry)



    if valid_entries:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_entries))
        
        print(f" Generated: {os.path.basename(output_path)} | Contains: {len(valid_entries)} lines")
        if missing_img_count > 0 or missing_gt_count > 0:
            print(f" (Warning: {missing_img_count} images missing, {missing_gt_count} GT lines missing)")
    else:
        print(f" Failed: {os.path.basename(output_path)} generated no valid data")

def main():
    print(" Starting generation of Kraken training lists...")
    
    gt_map = load_gt_dict()
    if not gt_map:
        return
    print(f" Loaded Ground Truth: {len(gt_map)} entries")

    for i in range(1, 5):
        cv_name = f'cv{i}'
        cv_path = os.path.join(SETS_DIR, cv_name)
        
        if os.path.exists(cv_path):
            print(f"\n Processing {cv_name} ...")
            generate_list_for_split(cv_path, 'train', gt_map)
            generate_list_for_split(cv_path, 'valid', gt_map)
            generate_list_for_split(cv_path, 'test', gt_map)
        else:
            print(f" Warning: Folder not found {cv_path}")


if __name__ == '__main__':
    main()