import os
import json
import time
import glob
import sys
import google.generativeai as genai
from PIL import Image
from tqdm import tqdm
import google.ai.generativelanguage as glm




API_KEY = ""


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "line_images_subset_50")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "gemini_flash_output_mode4")


SAMPLE_ONE_SHOT_PATH = "/Users/ad168613/Desktop/GW_Project/data/line_images_normalized/276-04.png"
SAMPLE_TWO_SHOT_1_PATH = "/Users/ad168613/Desktop/GW_Project/data/line_images_normalized/273-03.png"
SAMPLE_TWO_SHOT_2_PATH = "/Users/ad168613/Desktop/GW_Project/data/line_images_normalized/302-18.png"



PROMPT_MODE = 4


MODEL_NAME = "gemini-2.5-flash"



genai.configure(api_key=API_KEY)

FEW_SHOT_CACHE = {}

def load_sample_data(img_path):
    if not os.path.exists(img_path):
        print(f" ERROR: sample image not found at {img_path}")
        sys.exit(1)
        
    gt_path = img_path.replace(".png", ".gt.txt")
    if not os.path.exists(gt_path):
        print(f" ERROR: sample ground truth not found at {gt_path}")
        sys.exit(1)
        
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.read().strip()
    
    return {"img_path": img_path, "text": gt_text}

def load_few_shot_samples():
 
    
    if PROMPT_MODE == 3:
        FEW_SHOT_CACHE["one_shot"] = load_sample_data(SAMPLE_ONE_SHOT_PATH)
        print(f"One-shot sample loaded")

    elif PROMPT_MODE == 4:
        FEW_SHOT_CACHE["two_shot_1"] = load_sample_data(SAMPLE_TWO_SHOT_1_PATH)
        FEW_SHOT_CACHE["two_shot_2"] = load_sample_data(SAMPLE_TWO_SHOT_2_PATH)
        print(f"Two-shot sample loaded")

def get_prompt_content(mode, target_image_path):
    target_image = Image.open(target_image_path)

    # Mode 1: Zero-shot Basic
    if mode == 1:
        return [
            "Please transcribe the handwritten text in the following image. Output the result strictly as a valid JSON object with a single key 'text'.",
            target_image
        ]

    # Mode 2: Zero-shot Expert
    elif mode == 2:
        return [
            """You are an expert transcriber specializing in 18th-century handwritten manuscripts. 
            Your task is to transcribe the handwritten text in the provided line-level image with absolute diplomatic fidelity.

            Strictly adhere to the following transcription rules:
            1. **Diplomatic Transcription:** Transcribe exactly what you see visually. Do not correct spelling or grammar. Retain syntax, capitalization, punctuation and line breaks.
            2. **No Expansion:** Do not expand abbreviations.
            3. **Visual Evidence:** Rely on visual strokes, not context.
            4. **Character Specifics:** Transcribe the archaic long 's' character (resembling an 'f') strictly as 's'.

            Output Format:
            Output the result strictly as a valid JSON object with a single key "text".
            Example: {"text": "Your transcription here"}""",
            target_image
        ]

    # Mode 3: One-shot
    elif mode == 3:
        if "one_shot" not in FEW_SHOT_CACHE: raise ValueError("Samples not loaded.")
        s1 = FEW_SHOT_CACHE["one_shot"]
        img1 = Image.open(s1["img_path"])
        return [
            "You are an expert transcriber. Learn the transcription style from the example below. Note specifically that the long 's' is transcribed as 's'. Output strictly as JSON.",
            "Example Image:", img1,
            "Example Transcription:", json.dumps({"text": s1["text"]}),
            "Now, transcribe the following image using the same style.",
            target_image
        ]

    # Mode 4: Two-shot
    elif mode == 4:
        if "two_shot_1" not in FEW_SHOT_CACHE: raise ValueError("Samples not loaded.")
        s1 = FEW_SHOT_CACHE["two_shot_1"]
        s2 = FEW_SHOT_CACHE["two_shot_2"]
        img1 = Image.open(s1["img_path"])
        img2 = Image.open(s2["img_path"])
        return [
            "You are an expert transcriber. Learn the style from these examples. Note specifically that the long 's' is transcribed as 's'.",
            "Example 1 Image:", img1,
            "Example 1 Transcription:", json.dumps({"text": s1["text"]}),
            "Example 2 Image:", img2,
            "Example 2 Transcription:", json.dumps({"text": s2["text"]}),
            "Now, transcribe the following image using the same style.",
            target_image
        ]
    
    else:
        raise ValueError(f"Invalid Mode: {mode}")

def clean_json_response(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

def process_images():

    if PROMPT_MODE in [3, 4]:
        load_few_shot_samples()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"create output directory: {OUTPUT_DIR}")

 
    image_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    samples_to_exclude = [SAMPLE_ONE_SHOT_PATH, SAMPLE_TWO_SHOT_1_PATH, SAMPLE_TWO_SHOT_2_PATH]
    samples_to_exclude = [os.path.abspath(p) for p in samples_to_exclude]
    
    test_files = []
    for f in image_files:
        if os.path.abspath(f) not in samples_to_exclude:
            test_files.append(f)


    model = genai.GenerativeModel(MODEL_NAME)

    for img_path in tqdm(test_files, desc="Processing"):
        filename = os.path.basename(img_path)
        json_output_path = os.path.join(OUTPUT_DIR, filename.replace(".png", ".json"))

        if os.path.exists(json_output_path):
            continue

        retries = 3
        while retries > 0:
            try:
                prompt_content = get_prompt_content(PROMPT_MODE, img_path)
                
                
                if not response.parts:
                    raise ValueError(f"Response blocked or empty. Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")

                raw_text = clean_json_response(response.text)
                parsed_json = json.loads(raw_text)
                
                final_output = {
                    "file_name": filename,
                    "prompt_mode": PROMPT_MODE,
                    "model": MODEL_NAME,
                    "prediction": parsed_json.get("text", "")
                }

                with open(json_output_path, "w", encoding='utf-8') as f:
                    json.dump(final_output, f, indent=4, ensure_ascii=False)
                
                time.sleep(1) 
                break 

            except Exception as e:
                retries -= 1
                error_msg = str(e)
                
                if "429" in error_msg:
                    print(f"\n rate limit triggered sleeping for 10 seconds. retries left: {retries})")
                    time.sleep(10)
                else:
                    print(f"\n error {filename} - {error_msg}")
                    time.sleep(1)
                
                if retries == 0:
                    with open(json_output_path, "w", encoding='utf-8') as f:
                        json.dump({"file_name": filename, "error": error_msg}, f, indent=4)

if __name__ == "__main__":
    process_images()