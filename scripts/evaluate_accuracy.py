#!/usr/bin/env python3
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate

print("="*60)
print("Evaluating Model Accuracy (WER/CER)")
print("="*60)
print()

model_path = os.path.join(PROJECT_ROOT, "output", "final")
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if torch.cuda.is_available():
    model = model.cuda()

print(f"✅ Model loaded: {model_path}")
print(f"✅ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print()

# 加载测试数据
print("Loading test dataset...")
test_dataset = load_dataset("librispeech_asr", "clean", split="test[:100]")
print(f"✅ Loaded {len(test_dataset)} test samples")
print()

# 初始化评估指标
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# 预测并计算指标
predictions = []
references = []

print("Running inference...")
for i, sample in enumerate(test_dataset):
    audio = sample["audio"]["array"]
    text = sample["text"]
    
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    if torch.cuda.is_available():
        input_features = input_features.cuda().half()
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="en", task="transcribe")
    
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    
    predictions.append(transcription)
    references.append(text)
    
    if (i + 1) % 20 == 0:
        print(f"  Processed {i+1}/{len(test_dataset)} samples...")

print()
print("Calculating metrics...")
wer = wer_metric.compute(predictions=predictions, references=references) * 100
cer = cer_metric.compute(predictions=predictions, references=references) * 100

print()
print("="*60)
print("Accuracy Evaluation Results")
print("="*60)
print()
print(f"📊 Word Error Rate (WER): {wer:.2f}%")
print(f"📊 Character Error Rate (CER): {cer:.2f}%")
print()
print(f"✅ Word Accuracy: {(100 - wer):.2f}%")
print(f"✅ Character Accuracy: {(100 - cer):.2f}%")
print()
print("="*60)
print("Sample Predictions:")
print("="*60)
print()

for i in range(min(5, len(predictions))):
    print(f"Sample {i+1}:")
    print(f"  Reference: {references[i]}")
    print(f"  Prediction: {predictions[i]}")
    print()

# 保存评估结果
result = {
    "wer_percent": wer,
    "cer_percent": cer,
    "word_accuracy_percent": 100 - wer,
    "character_accuracy_percent": 100 - cer,
    "num_samples": len(test_dataset),
    "predictions": predictions,
    "references": references
}

import json
output_file = os.path.join(PROJECT_ROOT, "output", "accuracy_results.json")
with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"✅ Results saved to: {output_file}")
