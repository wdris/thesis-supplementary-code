import os
import pandas as pd
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm import tqdm
import jiwer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV_PATH = os.path.join(PROJECT_ROOT, 'gold_standard_full.csv')
MODEL_FOLDER_PATH = os.path.join(PROJECT_ROOT, 'trocr_line_output', 'final_best_model')
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, 'trocr_predictions_result.csv')

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Metal Performance Shaders (MPS) detected, enabling hardware acceleration")
        return torch.device("mps")
    else:
        print("No GPU detected, using CPU")
        return torch.device("cpu")

def run_evaluation():
    device = get_device()
    
    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV_PATH}")
        

    df = pd.read_csv(INPUT_CSV_PATH)
    
    if not os.path.exists(MODEL_FOLDER_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_FOLDER_PATH}")
        
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_FOLDER_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_FOLDER_PATH).to(device)
    except OSError as e:
        print(f"Error: Could not load model. Ensure {MODEL_FOLDER_PATH} contains config.json and pytorch_model.bin.")
        raise e

    model.eval()
    


    predictions = []
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row['Image_Path']
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found, skipping: {image_path}")
            predictions.append("")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    max_length=128,
                    early_stopping=True
                )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predictions.append(generated_text)
            
        except Exception as e:
            predictions.append("")

    df['TrOCR_Prediction'] = predictions
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Inference complete. Results saved to: {OUTPUT_CSV_PATH}")

    print("\n" + "="*30)
    print("CER Evaluation Report")
    print("="*30)
    
    ground_truths = df['Ground_Truth'].fillna('').astype(str).tolist()
    preds = df['TrOCR_Prediction'].fillna('').astype(str).tolist()
    
    overall_cer = jiwer.cer(ground_truths, preds)
    print(f"Overall CER: {overall_cer:.4f}")
    

    if 'Source_Fold' in df.columns:
        for fold, group in df.groupby('Source_Fold'):
            g_truths = group['Ground_Truth'].fillna('').astype(str).tolist()
            g_preds = group['TrOCR_Prediction'].fillna('').astype(str).tolist()
            fold_cer = jiwer.cer(g_truths, g_preds)
            print(f"Fold {fold}: CER = {fold_cer:.4f}")
    else:
        print("Source_Fold column not found, skipping group evaluation.")
    print("="*30)

if __name__ == "__main__":
    run_evaluation()