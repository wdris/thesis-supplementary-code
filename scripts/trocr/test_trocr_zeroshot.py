import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from jiwer import cer
from tqdm import tqdm
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS_DIR = os.path.join(PROJECT_ROOT, "sets")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "line_images_normalized")
MODEL_NAME = "microsoft/trocr-base-handwritten"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clean_txt(txt_path):
    data_list = []

    
    if not os.path.exists(txt_path):
        return []

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    valid_count = 0
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            img_path, label = parts
            data_list.append({'file_name': img_path, 'text': label})
            valid_count += 1
            
    print(f"Loaded {valid_count} valid samples")
    return data_list


processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)

all_fold_results = []
folds = [1, 2, 3, 4]




for i in folds:
    fold_name = f"cv{i}"
    filename = f"kraken_{fold_name}_test_clean.txt"
    txt_path = os.path.join(SETS_DIR, fold_name, filename)
    
    print(f"\nProcessing Fold: {fold_name}")
    data = load_clean_txt(txt_path)
    
    if not data:
        print(f"Skipping {fold_name} (No data)")
        continue

    total_cer = 0
    valid_count = 0
    pbar = tqdm(data, desc=f"Testing {fold_name}", leave=True)
    
    for item in pbar:
        img_path = item['file_name']
        ground_truth = str(item['text'])

        try:
            if not os.path.exists(img_path):
                basename = os.path.basename(img_path)
                img_path = os.path.join(IMAGES_DIR, basename)
                if not os.path.exists(img_path):
                    continue

            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            error = cer(ground_truth, generated_text)
            total_cer += error
            valid_count += 1
            pbar.set_postfix({'cer': f"{total_cer/valid_count:.4f}"})

        except Exception:
            pass

    if valid_count > 0:
        avg_cer = total_cer / valid_count
        print(f"{fold_name} Complete. CER = {avg_cer:.4f}")
        all_fold_results.append({
            'fold': fold_name,
            'cer': avg_cer,
            'samples': valid_count
        })

print("\n" + "="*40)
if len(all_fold_results) > 0:
    df_res = pd.DataFrame(all_fold_results)
    print("TrOCR Zero-shot Final Results:")
    print(df_res)
    
    mean_cer = df_res['cer'].mean()
    std_cer = df_res['cer'].std()
    
    print("-" * 40)
    print(f"Final Average CER: {mean_cer:.4f} (±{std_cer:.4f})")
    print("-" * 40)
    
    save_path = os.path.join(PROJECT_ROOT, "trocr_zeroshot_final_results.csv")
    df_res.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")
else:
    print("No results generated. Check input files.")