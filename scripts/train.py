#!/usr/bin/env python3
"""
Whisper ASR Training Script
Main training pipeline with complete functionality
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def main():
    print("=" * 60)
    print("  Whisper ASR Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    quick_mode = os.environ.get('QUICK_TEST_MODE', 'true').lower() == 'true'
    num_workers = int(os.environ.get('NUM_WORKERS', '4'))
    max_steps = int(os.environ.get('MAX_TRAIN_STEPS', '500'))
    eval_steps = int(os.environ.get('EVAL_STEPS', '250'))
    save_steps = int(os.environ.get('SAVE_STEPS', '500'))
    
    model_path = os.environ.get('MODEL_LOCAL_DIR', './models/whisper-small')
    output_dir = os.environ.get('OUTPUT_DIR', './output')
    compute_wer = os.environ.get('COMPUTE_WER', 'true').lower() == 'true'
    compute_cer = os.environ.get('COMPUTE_CER', 'true').lower() == 'true'
    
    if quick_mode:
        print("⚠️ Running in QUICK TEST MODE (small dataset)")
        train_samples = int(os.environ.get('QUICK_TEST_SAMPLES', '1000'))
        val_samples = int(os.environ.get('QUICK_TEST_VALIDATION', '100'))
        print(f"   Train samples: {train_samples}")
        print(f"   Validation samples: {val_samples}")
    else:
        print("🏃 Running in FULL TRAINING MODE")
        train_samples = None
        val_samples = None
    
    print()

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/final", exist_ok=True)

    # Load model and processor
    print("[1/7] Loading model and processor...")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, using model from Hugging Face...")
        model_path = os.environ.get('MODEL_NAME', 'openai/whisper-small')
    
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print(f"  ✓ Model loaded: {model_path}")
    print(f"  ✓ Parameters: {model.num_parameters() / 1e6:.1f}M")
    print()

    # Load dataset
    print("[2/7] Loading dataset...")
    if quick_mode:
        train_dataset = load_dataset(
            "librispeech_asr", "clean", 
            split=f"train.100[:{train_samples}]",
            trust_remote_code=True
        )
        eval_dataset = load_dataset(
            "librispeech_asr", "clean", 
            split=f"validation[:{val_samples}]",
            trust_remote_code=True
        )
    else:
        train_dataset = load_dataset(
            "librispeech_asr", "clean", 
            split="train.100+train.360",
            trust_remote_code=True
        )
        eval_dataset = load_dataset(
            "librispeech_asr", "clean", 
            split="validation",
            trust_remote_code=True
        )
    print(f"  ✓ Train set: {len(train_dataset):,} samples")
    print(f"  ✓ Validation set: {len(eval_dataset):,} samples")
    print()

    # Preprocess dataset
    print("[3/7] Preprocessing data...")
    def prepare_example(example):
        audio = example["audio"]
        example["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    train_dataset = train_dataset.map(
        prepare_example, 
        remove_columns=train_dataset.column_names,
        num_proc=num_workers
    )
    eval_dataset = eval_dataset.map(
        prepare_example, 
        remove_columns=eval_dataset.column_names,
        num_proc=num_workers
    )
    print("  ✓ Data preprocessed")
    print()

    # Set up trainer
    print("[4/7] Setting up trainer...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(os.environ.get('PER_DEVICE_TRAIN_BATCH_SIZE', '8')),
        per_device_eval_batch_size=int(os.environ.get('PER_DEVICE_EVAL_BATCH_SIZE', '4')),
        gradient_accumulation_steps=int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', '4')),
        learning_rate=float(os.environ.get('LEARNING_RATE', '1e-4')),
        warmup_steps=int(os.environ.get('WARMUP_STEPS', '50')),
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=25,
        fp16=os.environ.get('FP16_ENABLED', 'true').lower() == 'true',
        remove_unused_columns=False,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=os.environ.get('PIN_MEMORY', 'true').lower() == 'true',
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
    )
    print("  ✓ Trainer configured")
    print()

    # Train
    print("[5/7] Starting training...")
    print("-" * 60)
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    print("-" * 60)
    print(f"  ✓ Training complete!")
    print(f"  ✓ Time: {training_time / 60:.1f} minutes")
    print()

    # Save model
    print("[6/7] Saving model...")
    final_model_path = f"{output_dir}/final"
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"  ✓ Model saved to: {final_model_path}")
    print()

    # Evaluate
    print("[7/7] Evaluating model...")
    wer = 0.0
    cer = 0.0
    try:
        # Load evaluation metrics
        if compute_wer:
            wer_metric = evaluate.load("wer")
        if compute_cer:
            cer_metric = evaluate.load("cer")
        
        # Load trained model for evaluation
        eval_model = WhisperForConditionalGeneration.from_pretrained(final_model_path)
        if torch.cuda.is_available():
            eval_model = eval_model.cuda()
        
        # Load original dataset for evaluation (with text)
        eval_samples = min(len(eval_dataset), 100)
        eval_raw_dataset = load_dataset(
            "librispeech_asr", "clean", 
            split=f"validation[:{eval_samples}]",
            trust_remote_code=True
        )
        
        predictions = []
        references = []
        
        print(f"  Running inference on {len(eval_raw_dataset)} samples...")
        for idx, example in enumerate(eval_raw_dataset):
            audio = example["audio"]
            text = example["text"]
            
            input_features = processor.feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"], 
                return_tensors="pt"
            ).input_features
            
            if torch.cuda.is_available():
                input_features = input_features.cuda()
            
            with torch.no_grad():
                predicted_ids = eval_model.generate(
                    input_features, 
                    language="english", 
                    task="transcribe",
                    max_length=225
                )
            
            pred_text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(text)
            
            if (idx + 1) % 20 == 0:
                print(f"    Processed {idx + 1}/{len(eval_raw_dataset)}")
        
        if compute_wer:
            wer = wer_metric.compute(predictions=predictions, references=references) * 100
        if compute_cer:
            cer = cer_metric.compute(predictions=predictions, references=references) * 100
        
        print("\n  Evaluation Results:")
        if compute_wer:
            print(f"    WER: {wer:.2f}%")
        if compute_cer:
            print(f"    CER: {cer:.2f}%")
        if compute_wer:
            print(f"    Word Accuracy: {(100 - wer):.2f}%")
        if compute_cer:
            print(f"    Character Accuracy: {(100 - cer):.2f}%")
    except Exception as e:
        print(f"  ⚠️ Evaluation skipped: {e}")
    
    print()
    
    # Generate report
    print("Generating training report...")
    generate_report(
        output_dir, 
        model, 
        trainer, 
        wer, 
        cer, 
        quick_mode,
        training_time
    )
    
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print()
    print(f"Final model: {final_model_path}")
    print(f"Training report: {output_dir}/training_report.md")
    print()

def generate_report(output_dir, model, trainer, wer, cer, quick_mode, training_time):
    """Generate a simple training report"""
    report_path = os.path.join(output_dir, "training_report.md")
    
    report_content = f"""# Whisper ASR Training Report

## Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Mode**: {'Quick Test' if quick_mode else 'Full Training'}
- **Training Time**: {training_time / 60:.1f} minutes

## Results
- **WER**: {wer:.2f}%
- **CER**: {cer:.2f}%
- **Word Accuracy**: {(100 - wer):.2f}%
- **Character Accuracy**: {(100 - cer):.2f}%

## Model
- **Parameters**: {model.num_parameters() / 1e6:.1f}M

## Usage
Model saved to: {output_dir}/final
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"  ✓ Report saved to: {report_path}")

if __name__ == "__main__":
    main()
