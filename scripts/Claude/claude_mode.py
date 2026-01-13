import os
import json
import time
import glob
import sys
import base64
import anthropic
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = ""

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "line_images_subset_50")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "claude_output")

SAMPLE_ONE_SHOT_PATH = os.path.join(PROJECT_ROOT, "data", "line_images_normalized", "276-04.png")
SAMPLE_TWO_SHOT_1_PATH = os.path.join(PROJECT_ROOT, "data", "line_images_normalized", "273-03.png")
SAMPLE_TWO_SHOT_2_PATH = os.path.join(PROJECT_ROOT, "data", "line_images_normalized", "302-18.png")




PROMPT_MODE = 4  


MODEL_NAME = "claude-opus-4-5-20251101"
MAX_WORKERS = 1

FEW_SHOT_CACHE = {}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_media_type(filename):
    ext = filename.lower().split('.')[-1]
    if ext == 'png': return 'image/png'
    if ext in ['jpg', 'jpeg']: return 'image/jpeg'
    return 'image/jpeg'

def load_few_shot_samples():
    """
    Preloads required samples into cache based on the current mode.
    """
    
    if PROMPT_MODE == 3:
        FEW_SHOT_CACHE["one_shot"] = load_sample_data(SAMPLE_ONE_SHOT_PATH)
        print(f"One-shot sample loaded:{os.path.basename(SAMPLE_ONE_SHOT_PATH)}")

    elif PROMPT_MODE == 4:
        FEW_SHOT_CACHE["two_shot_1"] = load_sample_data(SAMPLE_TWO_SHOT_1_PATH)
        FEW_SHOT_CACHE["two_shot_2"] = load_sample_data(SAMPLE_TWO_SHOT_2_PATH)
        print(f"Two-shot sample loaded: {os.path.basename(SAMPLE_TWO_SHOT_1_PATH)}, {os.path.basename(SAMPLE_TWO_SHOT_2_PATH)}")

def build_claude_payload(mode, target_image_path):
    """
    Similar to Gemini, all context is placed within a single User message.
    """
    target_b64 = encode_image(target_image_path)
    target_media = get_media_type(target_image_path)


    system_prompt = "You are an expert transcriber. Output strictly as JSON."

    

    content_list = []

    # Mode 1: Zero-shot Basic
    if mode == 1:
        content_list.append({"type": "text", "text": "Please transcribe the handwritten text in this image. Output the result strictly as a valid JSON object with a single key 'text'."})
        content_list.append({"type": "image", "source": {"type": "base64", "media_type": target_media, "data": target_b64}})

    # Mode 2: Zero-shot Expert
    elif mode == 2:
        expert_instruction = """You are an expert transcriber specializing in 18th-century handwritten manuscripts. 
        Your task is to transcribe the handwritten text in the provided line-level image with absolute diplomatic fidelity.

        Strictly adhere to the following transcription rules:
        1. **Diplomatic Transcription:** Transcribe exactly what you see visually. Do not correct spelling or grammar. Retain syntax, capitalization, punctuation and line breaks.
        2. **No Expansion:** Do not expand abbreviations.
        3. **Visual Evidence:** Rely on visual strokes, not context.
        4. **Character Specifics:** Transcribe the archaic long 's' character (resembling an 'f') strictly as 's'.

        Output Format:
        Output the result strictly as a valid JSON object with a single key "text".
        Example: {"text": "Your transcription here"}"""
        
        content_list.append({"type": "text", "text": expert_instruction})
        content_list.append({"type": "image", "source": {"type": "base64", "media_type": target_media, "data": target_b64}})

    # Mode 3: One-shot
    elif mode == 3:
        if "one_shot" not in FEW_SHOT_CACHE: raise ValueError("Sample not loaded.")
        s1 = FEW_SHOT_CACHE["one_shot"]
        
        content_list.append({"type": "text", "text": "You are an expert transcriber. Learn the transcription style from the example below. Note specifically that original spelling is preserved and the long 's' is transcribed as 's' ."})
        
      
        content_list.append({"type": "text", "text": "Example Image:"})
        content_list.append({"type": "image", "source": {"type": "base64", "media_type": s1["media_type"], "data": s1["base64"]}})
        
        s1_json = json.dumps({"text": s1["text"]}, ensure_ascii=False)
        content_list.append({"type": "text", "text": f"Example Transcription:\n{s1_json}"})
    
        content_list.append({"type": "text", "text": "Now, transcribe the following image using the same style. Output strictly as JSON."})
        content_list.append({"type": "image", "source": {"type": "base64", "media_type": target_media, "data": target_b64}})

    # Mode 4: Two-shot
    elif mode == 4:
        if "two_shot_1" not in FEW_SHOT_CACHE: raise ValueError("Samples not loaded.")
        s1 = FEW_SHOT_CACHE["two_shot_1"]
        s2 = FEW_SHOT_CACHE["two_shot_2"]
        
     
        content_list.append({"type": "text", "text": "You are an expert transcriber. Learn the transcription style from the following examples. Note specifically that original spelling is preserved and the long 's' is transcribed as 's'."})
        
        content_list.append({"type": "text", "text": "Example 1 Image:"})
        content_list.append({"type": "image", "source": {"type": "base64", "media_type": s1["media_type"], "data": s1["base64"]}})
        s1_json = json.dumps({"text": s1["text"]}, ensure_ascii=False)
        content_list.append({"type": "text", "text": f"Example 1 Transcription:\n{s1_json}"})

        content_list.append({"type": "text", "text": "Example 2 Image:"})
        content_list.append({"type": "image", "source": {"type": "base64", "media_type": s2["media_type"], "data": s2["base64"]}})
        s2_json = json.dumps({"text": s2["text"]}, ensure_ascii=False)
        content_list.append({"type": "text", "text": f"Example 2 Transcription:\n{s2_json}"})
        
        content_list.append({"type": "text", "text": "Now, transcribe the following image using the same style. Output strictly as JSON."})
        content_list.append({"type": "image", "source": {"type": "base64", "media_type": target_media, "data": target_b64}})

    messages = [
        {
            "role": "user",
            "content": content_list
        }
    ]
    
    return system_prompt, messages

def clean_json_response(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]


    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return text.strip()

def process_single_image(img_path):
    filename = os.path.basename(img_path)
    json_output_path = os.path.join(OUTPUT_DIR, filename.replace(".png", ".json"))


    if os.path.exists(json_output_path):
        return f"SKIP: {filename}"

    client = anthropic.Anthropic(api_key=API_KEY)
    
    retries = 3
    while retries > 0:
        try:
            system_instruction, message_payload = build_claude_payload(PROMPT_MODE, img_path)
            
            start_time = time.time()
            
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                system=system_instruction,
                messages=message_payload,
                temperature=0.0
            )
            
            raw_text = response.content[0].text
            cleaned_text = clean_json_response(raw_text)
            
            try:
                parsed_json = json.loads(cleaned_text)
                prediction_text = parsed_json.get("text", "")
            except json.JSONDecodeError:
                prediction_text = raw_text
            
            final_output = {
                "file_name": filename,
                "prompt_mode": PROMPT_MODE,
                "model": MODEL_NAME,
                "prediction": prediction_text,
                "raw_response": raw_text 
            }
            

            with open(json_output_path, "w", encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)

            return f"DONE: {filename}"

        except Exception as e:
            retries -= 1
            error_msg = str(e)
            
            if "429" in error_msg:
                print(f"rate limit reached({filename}), sleeping for 20 seconds.")
                time.sleep(20) 
            elif "503" in error_msg or "529" in error_msg: 
                print(f"server overloaded({filename}), sleeping for 30 seconds.")
                time.sleep(30)
            else:
                time.sleep(2)
            
            if retries == 0:
                with open(json_output_path, "w", encoding='utf-8') as f:
                    json.dump({"file_name": filename, "error": error_msg}, f, indent=4)
                return f"FAIL: {filename} ({error_msg[:30]}...)"



def run_parallel_processing():

    if PROMPT_MODE in [3, 4]:
        load_few_shot_samples()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    image_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    
    samples_to_exclude = [SAMPLE_ONE_SHOT_PATH, SAMPLE_TWO_SHOT_1_PATH, SAMPLE_TWO_SHOT_2_PATH]
    samples_to_exclude = [os.path.abspath(p) for p in samples_to_exclude]
    
    test_files = []
    for f in image_files:
        if os.path.abspath(f) not in samples_to_exclude:
            test_files.append(f)


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_image, img_path): img_path for img_path in test_files}
        
        for future in tqdm(as_completed(future_to_file), total=len(test_files), desc="Progress"):
            try:
                result_msg = future.result()
            except Exception as exc:
                print(f"\nThread exception occured: {exc}")

    print("tasks completed")

if __name__ == "__main__":
    run_parallel_processing()