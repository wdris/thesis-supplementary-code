import os
import torch
import numpy as np 
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from jiwer import wer, cer



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "trocr_line_train.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "trocr_line_val.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "trocr_line_test.csv")

MODEL_NAME = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = "./trocr_line_output"

def load_local_datasets(train_csv, val_csv, test_csv):
    data_files = {'train': train_csv, 'validation': val_csv, 'test': test_csv}
    return load_dataset('csv', data_files=data_files)

datasets = load_local_datasets(TRAIN_CSV, VAL_CSV, TEST_CSV)
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

def prepare_features(examples):
    labels = processor.tokenizer(
        examples['text'], 
        max_length=processor.tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True
    ).input_ids
    labels = [[(l if l != processor.tokenizer.pad_token_id else -100) for l in labels_seq] for labels_seq in labels]
    return {'labels': labels, 'text': examples['text'], 'file_name': examples['file_name']}

def image_loader(example):
    from PIL import Image
    try:
        image = Image.open(example['file_name']).convert("RGB")
        example['pixel_values'] = processor.image_processor(image, return_tensors="pt").pixel_values.squeeze()
        return example
    except Exception as e:
        print(f"Error loading {example['file_name']}: {e}")
        return None

datasets = datasets.map(prepare_features, batched=True, remove_columns=['file_name'])
datasets = datasets.map(image_loader)
datasets = datasets.filter(lambda example: example is not None)
datasets.set_format(type="torch", columns=['pixel_values', 'labels'])

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = 64
model.config.early_stopping = True
model.config.num_beams = 4
model.gradient_checkpointing_enable() 
model.config.use_cache = False 



def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    labels_ids = np.where(labels_ids != -100, labels_ids, processor.tokenizer.pad_token_id)

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    return {"wer": wer(label_str, pred_str), "cer": cer(label_str, pred_str)}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    predict_with_generate=True,      
    learning_rate=4e-5,
    num_train_epochs=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=datasets['train'],
    eval_dataset=datasets['validation'],
    data_collator=default_data_collator,
)


trainer.train()
results = trainer.evaluate(datasets['test'])
print(f"Test Results: {results}")

final_path = os.path.join(OUTPUT_DIR, "final_best_model")
model.save_pretrained(final_path)
processor.save_pretrained(final_path)