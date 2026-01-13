drive_project_folder = "/content/drive/MyDrive/PPOCR_Project"
zip_filename = "dataset_normalized.zip"
epoch_num = 50


import os
import shutil
import zipfile
import yaml
import random
from google.colab import drive

def setup_environment():
    !pip install "numpy<2.0.0" "protobuf==3.20.3" -q
    !pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -q
    !pip install paddleocr>=2.0.1 imgaug==0.4.0 pyclipper lmdb tqdm shapely scikit-image \
                 opencv-python-headless rapidfuzz attrdict visualdl python-Levenshtein -q
    print("Environment setup complete.")

def prepare_data():
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')
    
    local_zip = f"/content/{zip_filename}"
    drive_zip = os.path.join(drive_project_folder, zip_filename)
    data_root = "/content/my_data"

    if not os.path.exists("/content/paddle_norm_train.txt"):
        if not os.path.exists(local_zip):
            if os.path.exists(drive_zip):
                shutil.copy(drive_zip, local_zip)
            else:
                raise FileNotFoundError(f"Error: {zip_filename} not found in Google Drive")

        if os.path.exists(data_root): shutil.rmtree(data_root)
        with zipfile.ZipFile(local_zip, 'r') as z:
            z.extractall(data_root)

        # generate label lists and dictionary
        data_lines = []
        vocab = set()
        for root, dirs, files in os.walk(data_root):
            if '__MACOSX' in dirs: dirs.remove('__MACOSX')
            for file in files:
                if not file.lower().endswith(('.jpg', '.png')): continue
                base = os.path.splitext(file)[0]
                gt_path = os.path.join(root, base + ".gt.txt")
                if not os.path.exists(gt_path): gt_path = os.path.join(root, base + ".txt")
                
                if os.path.exists(gt_path):
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        lbl = f.read().strip()
                    if lbl:
                        data_lines.append(f"{os.path.abspath(os.path.join(root, file))}\t{lbl}")
                        for char in lbl: vocab.add(char)

        random.seed(42)
        random.shuffle(data_lines)
        split = int(len(data_lines) * 0.9)
        
        with open("/content/paddle_norm_train.txt", 'w') as f: f.write('\n'.join(data_lines[:split]))
        with open("/content/paddle_norm_val.txt", 'w') as f: f.write('\n'.join(data_lines[split:]))
        with open("/content/paddle_norm_dict.txt", 'w') as f: f.write('\n'.join(sorted(list(vocab))))


def configure_ocr():
    if not os.path.exists("/content/PaddleOCR"):
        !git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
    
    os.chdir("/content/PaddleOCR")
    !wget -O final_config.yml -q https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml

    with open("final_config.yml", 'r') as f:
        config = yaml.safe_load(f)

    save_dir = os.path.join(drive_project_folder, "output")
    os.makedirs(save_dir, exist_ok=True)

    config['Global']['save_model_dir'] = save_dir
    config['Global']['use_gpu'] = True
    config['Global']['epoch_num'] = epoch_num
    config['Global']['save_epoch_step'] = 1
    config['Global']['character_dict_path'] = "/content/paddle_norm_dict.txt"
    config['Global']['use_space_char'] = True
    
    config['Train']['dataset']['label_file_list'] = ["/content/paddle_norm_train.txt"]
    config['Train']['dataset']['data_dir'] = "/content/"
    config['Eval']['dataset']['label_file_list'] = ["/content/paddle_norm_val.txt"]
    config['Eval']['dataset']['data_dir'] = "/content/"
    
    if 'lr' in config['Optimizer']:
        config['Optimizer']['lr']['learning_rate'] = 0.0005

    os.makedirs("./pretrain_models", exist_ok=True)
    if not os.path.exists("./pretrain_models/en_PP-OCRv4_rec_train"):
        !wget -P ./pretrain_models/ -q https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar
        !tar -xf ./pretrain_models/en_PP-OCRv4_rec_train.tar -C ./pretrain_models/
    config['Global']['pretrained_model'] = "./pretrain_models/en_PP-OCRv4_rec_train/best_accuracy"

    with open("final_config.yml", 'w') as f:
        yaml.dump(config, f)
    print("Configuration complete.")

def start_training():
    os.chdir("/content/PaddleOCR")
    os.environ['PYTHONPATH'] = os.getcwd()
    
    resume_path = os.path.join(drive_project_folder, "output/latest")
    
    if os.path.exists(resume_path + ".pdparams"):
        print(f"Checkpoint detected. Resuming from {resume_path}")
        !python3 tools/train.py -c final_config.yml -o Global.checkpoints="{resume_path}"
    else:
        print("No checkpoint found. Starting fresh training")
        !python3 tools/train.py -c final_config.yml


setup_environment() 
prepare_data()
configure_ocr()
start_training()