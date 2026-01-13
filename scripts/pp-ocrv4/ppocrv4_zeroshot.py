import os
import sys
import zipfile
import glob
import pandas as pd
import jiwer
from tqdm import tqdm
from paddleocr import PaddleOCR


def install_dependencies():

    !pip install "numpy==1.26.4" "opencv-python-headless==4.8.0.74" -q

    !python -m pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -q

    !pip install "paddleocr==2.7.3" jiwer shapely -q


ZIP_PATH = '/content/drive/MyDrive/PPOCR_Project/dataset_normalized.zip'
EXTRACT_PATH = '/content/gw_data'
MODEL_BASE = "/content/models/en_PP-OCRv4_rec"
REC_MODEL_DIR = os.path.join(MODEL_BASE, "en_PP-OCRv4_rec_infer")
OUTPUT_CSV = '/content/drive/MyDrive/PPOCR_Project/gw_zero_shot_results.csv'

def main():
    if not os.path.exists('/content/drive'):
        from google.colab import drive
        drive.mount('/content/drive')

    if not os.path.exists(REC_MODEL_DIR):
        print("Downloading inference model.")
        os.makedirs(MODEL_BASE, exist_ok=True)
        os.system(f"wget -P {MODEL_BASE} https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar")
        os.system(f"tar -xf {MODEL_BASE}/en_PP-OCRv4_rec_infer.tar -C {MODEL_BASE}")

    if not os.path.exists(EXTRACT_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)

    image_dir = None
    for root, dirs, files in os.walk(EXTRACT_PATH):
        if any(f.endswith('.png') for f in files):
            image_dir = root
            break

    if not image_dir:
        return


    ocr = PaddleOCR(use_angle_cls=False, lang='en', rec_model_dir=REC_MODEL_DIR, use_gpu=True, show_log=False)

    image_files = glob.glob(os.path.join(image_dir, "*.png"))

    ground_truths = []
    hypothesis = []
    detailed_results = []

    for img_path in tqdm(image_files):
        gt_path = img_path.replace('.png', '.gt.txt')
        if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_text = f.read().strip()

            result = ocr.ocr(img_path, cls=False, det=False)

            if result and result[0]:
                pred_text = result[0][0][0]
                confidence = result[0][0][1]
            else:
                pred_text = ""
                confidence = 0.0

            ground_truths.append(gt_text)
            hypothesis.append(pred_text)

            file_name = os.path.basename(img_path)
            detailed_results.append({
                "Filename": file_name,
                "GroundTruth": gt_text,
                "Prediction": pred_text,
                "Confidence": confidence
            })

    if len(ground_truths) > 0:
        final_cer = jiwer.cer(ground_truths, hypothesis)

        print("\n" + "="*30)
        print(f"Zero-Shot CER: {final_cer:.4f}")
        print("="*30)

        df = pd.DataFrame(detailed_results)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf_8_sig')
        print(f"Detailed results saved to: {OUTPUT_CSV}")
    else:
        print("Error: No valid data found for evaluation.")

if __name__ == "__main__":
    try:
        import paddle
        main()
    except ImportError:
        install_dependencies()