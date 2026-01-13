import os
import sys
import zipfile
import glob
import math
import yaml
import pandas as pd
import cv2
import numpy as np
import paddle
from google.colab import drive

PROJECT_DIR = "/content/drive/My Drive/PPOCR_Project"
CSV_PATH = os.path.join(PROJECT_DIR, "gold_standard_full.csv")
ZIP_PATH = os.path.join(PROJECT_DIR, "dataset_normalized.zip")
CONFIG_PATH = os.path.join(PROJECT_DIR, "output/config.yml")
MODEL_PARAMS = os.path.join(PROJECT_DIR, "output/latest.pdparams")
OUTPUT_REPORT = os.path.join(PROJECT_DIR, "FINAL_EVAL_REPORT.csv")



DATA_EXTRACT_DIR = "/content/data_temp_final"
PADDLE_REPO_DIR = "/content/PaddleOCR"
FIXED_DICT_PATH = "/content/final_corrected_dict.txt"


IMAGE_COL = 'Image_Path'
GT_COL = 'Ground_Truth'


def setup_environment():
    drive.mount('/content/drive')
    
    if not os.path.exists(PADDLE_REPO_DIR):
        !git clone https://github.com/PaddlePaddle/PaddleOCR.git
    
    %cd {PADDLE_REPO_DIR}
    !pip install -r requirements.txt -q
    !pip install paddlepaddle-gpu jiwer -q
    sys.path.append(PADDLE_REPO_DIR)

def prepare_data():
    if not os.path.exists(DATA_EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_EXTRACT_DIR)
    
    image_index = {}
    for root, _, files in os.walk(DATA_EXTRACT_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_index[file] = os.path.join(root, file)
    return image_index

def align_dictionary():

    try:
        state_dict = paddle.load(MODEL_PARAMS)
        weight_key = "head.ctc_head.fc.weight"
        if weight_key not in state_dict:
            weight_key = "head.fc.weight"
        
        num_classes = state_dict[weight_key].shape[1]
       
        required_chars = num_classes - 2 
        print(f"Model requirement detected: {required_chars} characters.")
    except Exception as e:
        print(f"Warning: Could not detect weights, using default 70. Error: {e}")
        required_chars = 70

   
    chars = set()
    for root, _, files in os.walk(DATA_EXTRACT_DIR):
        if "__MACOSX" in root: continue
        for file in files:
            if file.endswith(".txt") and not file.startswith("._"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        label = content.split('\t')[-1] if '\t' in content else content
                        for c in label: chars.add(c)
                except: continue




    sorted_chars = sorted(list(chars))
    if len(sorted_chars) > required_chars:
        sorted_chars = sorted_chars[:required_chars]
    elif len(sorted_chars) < required_chars:
        for i in range(required_chars - len(sorted_chars)):
            sorted_chars.append(f"<pad_{i}>")

    # insert dummy start to handle index shifting
    sorted_chars[0] = "<dummy_start>"

    with open(FIXED_DICT_PATH, 'w', encoding='utf-8') as f:
        for c in sorted_chars:
            f.write(c + '\n')
    return FIXED_DICT_PATH

def resize_norm_img(img, image_shape=[3, 48, 320]):
    imgC, imgH, imgW = image_shape
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH)).astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def run_evaluation(image_index, dict_path):
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    from ppocr.utils.save_load import load_model
    from jiwer import cer

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config['Global']['character_dict_path'] = dict_path
    config['Global']['use_space_char'] = True
    config['Global']['pretrained_model'] = MODEL_PARAMS.replace('.pdparams', '')
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        char_count = len(f.readlines())
    
    out_channels = char_count + 2
    
    if config['Architecture']['Head']['name'] == 'MultiHead':
        config['Architecture']['Head']['out_channels_list'] = {
            'CTCLabelDecode': out_channels, 'SARLabelDecode': out_channels,
            'NRTRLabelDecode': out_channels, 'CTCHead': out_channels
        }
    config['PostProcess']['character_dict_path'] = dict_path

    model = build_model(config['Architecture'])
    load_model(config, model)
    model.eval()
    post_process_class = build_post_process(config['PostProcess'], config['Global'])

    df = pd.read_csv(CSV_PATH)
    results = []

    for _, row in df.iterrows():
        file_name = os.path.basename(str(row[IMAGE_COL]).strip())
        gt_text = str(row[GT_COL]).strip()
        real_path = image_index.get(file_name)

        pred_text = ""
        if real_path and os.path.exists(real_path):
            img = cv2.imread(real_path)
            if img is not None:
                img_tensor = paddle.to_tensor(resize_norm_img(img)[np.newaxis, :])
                try:
                    preds = model(img_tensor)
                    ctc_preds = preds['ctc'] if isinstance(preds, dict) else preds
                    rec_result = post_process_class(ctc_preds)
                    if rec_result:
                        pred_text = rec_result[0][0]
                except: pass

        gt_text = " " if not gt_text or gt_text == "nan" else gt_text
        pred_text = " " if not pred_text else pred_text
        
        results.append({
            'Image_Name': file_name,
            'Ground_Truth': gt_text,
            'Prediction': pred_text,
            'CER': cer(gt_text, pred_text)
        })

    report_df = pd.DataFrame(results)
    report_df.to_csv(OUTPUT_REPORT, index=False)
    

    print(f"Average CER: {report_df['CER'].mean():.4f}")
    print(f"Full report saved to: {OUTPUT_REPORT}")

setup_environment()
idx = prepare_data()
final_dict = align_dictionary()
run_evaluation(idx, final_dict)