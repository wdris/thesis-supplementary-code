import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_GT_PATH = os.path.join(PROJECT_ROOT, 'data', 'ground_truth', 'transcription.txt')
OUTPUT_GT_PATH = os.path.join(PROJECT_ROOT, 'data', 'ground_truth', 'transcription_normalized.txt')

TOKEN_MAP = {     
    r's_GW': 'G. W.',              
    r's_s': 's',                              
    

    r's_sq': ';',
    r's_pt': '.',
    r's_cm': ',',
    r's_qo': '"',
    r's_qt': "'",
    r's_mi': '-',
    r's_sl': '/',
    r's_bl': '(',
    r's_br': ')',
    r's_et': '&',
    r's_as': '*',
    r's_cl': ':',
    r's_ex': '!',
    r's_qm': '?',
    r's_lb': '£',
  

    
    r's_1st': '1st',          
    r's_2nd': '2nd',            
    r's_3rd': '3rd',            
    r's_0th': '0th',            
    r's_1th': '1th',            
    r's_4th': '4th',            
    r's_5th': '5th',            
    r's_6th': '6th',            
    r's_7th': '7th',            
    r's_8th': '8th',            
    r's_9th': '9th',                        


    r's_0': '0',
    r's_1': '1',
    r's_2': '2',
    r's_3': '3',
    r's_4': '4',
    r's_5': '5',
    r's_6': '6',
    r's_7': '7',
    r's_8': '8',
    r's_9': '9',
}

def normalize_line_structure(raw_text):
    raw_text = raw_text.replace('s_et-c-s_pt', 'etc.')
    raw_words = raw_text.split('|')

    clean_words = []
    for word in raw_words:
        if not word: continue

        tokens = word.split('-')
        
        clean_tokens = []
        for token in tokens:
            if not token: continue

            token_key = token.strip()
            
            if token_key in TOKEN_MAP:
                clean_tokens.append(TOKEN_MAP[token_key])
            else:
                if token_key.startswith('s_') and len(token_key) > 2:
                    pass 
                clean_tokens.append(token_key)

        clean_word = "".join(clean_tokens)
        clean_words.append(clean_word)

    final_line = " ".join(clean_words)
    return final_line   

        

def main():
    print(f"Processing GT file: {INPUT_GT_PATH}")

    if not os.path.exists(INPUT_GT_PATH):
        print(f"Error: GT file not found at {INPUT_GT_PATH}")
        return
    
    normalized_lines = []
    error_count = 0

    try:
        with open(INPUT_GT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                parts = line.split(' ', 1)

                if len(parts) < 2: continue
                line_id = parts[0]
                raw_text = parts[1]

                clean_text = normalize_line_structure(raw_text)
                normalized_lines.append(f"{line_id} {clean_text}")

        with open(OUTPUT_GT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(normalized_lines))

        print(f"GT Normalization complete! Processed {len(normalized_lines)} lines.")
        print(f"Output file: {OUTPUT_GT_PATH}")

        print("\n--- Preview first 5 lines (Check Long S, Ordinals, Dates) ---")
        preview_count = 0
        for line in normalized_lines:
            print(line)
            preview_count += 1
            if preview_count >= 5:
                break

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()            