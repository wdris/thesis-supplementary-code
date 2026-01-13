import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'line_images_normalized')
NORMALIZED_GT_FILE = os.path.join(DATA_DIR, 'ground_truth', 'transcription_normalized.txt')
SETS_DIR = os.path.join(PROJECT_ROOT, 'sets')

def main():
    print("Starting Ketos training data preparation")

    print("--> Generating .gt.txt files")
    gt_count = 0
    with open(NORMALIZED_GT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split(' ', 1)
            if len(parts) < 2: continue
            
            line_id = parts[0]
            text = parts[1]
            
            gt_filename = f"{line_id}.gt.txt"
            gt_path = os.path.join(IMAGES_DIR, gt_filename)
            
            with open(gt_path, 'w', encoding='utf-8') as gt_f:
                gt_f.write(text)
            gt_count += 1
            
    print(f"Created {gt_count} .gt.txt files in {IMAGES_DIR}.")

    print("--> Generating path lists")
    for i in range(1, 5):
        cv_folder = os.path.join(SETS_DIR, f'cv{i}')
        if not os.path.exists(cv_folder): continue
        
        for split in ['train', 'valid', 'test']:
            id_file = os.path.join(cv_folder, f'{split}.txt')
            out_file = os.path.join(cv_folder, f'ketos_{split}.txt')
            
            if not os.path.exists(id_file): continue
            
            paths = []
            with open(id_file, 'r') as f:
                ids = [l.strip() for l in f if l.strip()]
            
            for lid in ids:
                img_path = os.path.join(IMAGES_DIR, f"{lid}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(IMAGES_DIR, f"{lid}.jpg")
                
                if os.path.exists(img_path):
                    paths.append(img_path)
            
            with open(out_file, 'w') as f:
                f.write('\n'.join(paths))


    print("\nData preparation complete. Ready for ketos training.")

if __name__ == '__main__':
    main()