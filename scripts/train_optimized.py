import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "hf_cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(PROJECT_ROOT, "hf_cache", "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(PROJECT_ROOT, "hf_cache", "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(PROJECT_ROOT, "hf_cache", "datasets")
os.environ["TOKENIZERS_CACHE"] = os.path.join(PROJECT_ROOT, "hf_cache", "tokenizers")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_DOWNLOAD_TIMEOUT"] = "600"

import os, sys, argparse, warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch, yaml, numpy as np
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer,
    WhisperFeatureExtractor, Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
import evaluate
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_DOWNLOAD_TIMEOUT"] = "600"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def load_config(path):
    with open(path) as f: return yaml.safe_load(f)

def prepare_dataset_optimized(dataset, processor, sr, num_proc=4):
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))
    
    def prep(batch):
        audio = batch["audio"]
        if isinstance(audio, list):
            audio_inputs = [a["array"] for a in audio]
        else:
            audio_inputs = [audio["array"]]
        
        input_features = processor.feature_extractor(
            audio_inputs,
            sampling_rate=sr
        ).input_features
        
        labels = processor.tokenizer(batch["text"]).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels
        }
    
    dataset = dataset.map(
        prep,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=50,
    )
    return dataset

def compute_metrics(pred, tokenizer):
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    def flatten_seq(seq):
        if isinstance(seq, np.ndarray):
            return [int(x) for x in seq.flatten().tolist()]
        if isinstance(seq, (list, tuple)):
            result = []
            for item in seq:
                if isinstance(item, (np.ndarray, list, tuple)):
                    result.extend(flatten_seq(item))
                else:
                    result.append(int(item))
            return result
        return [int(seq)]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    label_ids = np.where(label_ids == -100, pad_id, label_ids)

    pred_str, label_str = [], []
    for seq in pred_ids:
        tokens = [t for t in flatten_seq(seq) if t >= 0]
        pred_str.append(tokenizer.decode(tokens, skip_special_tokens=True).strip() if tokens else "")

    for seq in label_ids:
        tokens = [t for t in flatten_seq(seq) if t >= 0]
        label_str.append(tokenizer.decode(tokens, skip_special_tokens=True).strip() if tokens else "")

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/optimized_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    val_cfg = config["validation"]
    model_name = config["model"]["name"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    device = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_compute = torch.cuda.get_device_capability(0)

    print("="*60)
    print("ASR Training Pipeline - OPTIMIZED VERSION")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Dataset: {data_cfg['dataset_name']} ({data_cfg['subset']})")
    print(f"Train: {data_cfg['split']} | Eval: {val_cfg.get('eval_subset', data_cfg['split'])}")
    print(f"Device: {device}", end="")
    if torch.cuda.is_available():
        print(f" ({gpu_name}, {gpu_mem:.1f} GB, Compute {gpu_compute[0]}.{gpu_compute[1]})")
    else:
        print()
    print("="*60)

    print("\n[1/6] Loading model and processor...")
    processor = WhisperProcessor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
        language=config["model"]["language"],
        task=config["model"]["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    use_fp16 = config["model"].get("use_fp16", False) and torch.cuda.is_available()
    if use_fp16:
        model = model.half()
        print("  ✓ FP16 enabled")

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    print(f"  ✓ Model loaded: {model.num_parameters()/1e6:.1f}M parameters")

    print("\n[2/6] Loading datasets...")
    train_dataset = load_dataset(data_cfg["dataset_name"], data_cfg["subset"], split=data_cfg["split"])
    print(f"  ✓ Train samples: {len(train_dataset)}")

    print("\n[3/6] Loading eval dataset...")
    eval_split = val_cfg.get("eval_subset", data_cfg["split"])
    eval_dataset = load_dataset(data_cfg["dataset_name"], data_cfg["subset"], split=eval_split)
    max_eval = val_cfg.get("max_eval_samples", len(eval_dataset))
    if max_eval < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(max_eval))
    print(f"  ✓ Eval samples: {len(eval_dataset)}")

    print("\n[4/6] Preprocessing datasets with multiprocessing...")
    num_proc = train_cfg.get("num_workers", 4)
    print(f"  Using {num_proc} processes for preprocessing...")
    train_dataset = prepare_dataset_optimized(train_dataset, processor, data_cfg["sample_rate"], num_proc=num_proc)
    eval_dataset = prepare_dataset_optimized(eval_dataset, processor, data_cfg["sample_rate"], num_proc=num_proc)
    print(f"  ✓ Train preprocessed: {len(train_dataset)}")
    print(f"  ✓ Eval preprocessed: {len(eval_dataset)}")

    print("\n[5/6] Configuring trainer with optimizations...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        logging_dir=train_cfg["logging_dir"],
        dataloader_num_workers=train_cfg["num_workers"],
        dataloader_pin_memory=train_cfg["dataloader_pin_memory"],
        report_to=train_cfg["report_to"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        save_total_limit=3,
        remove_unused_columns=False,
        fp16=use_fp16,
        bf16=False,
        generation_max_length=225,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        optim=train_cfg.get("optim", "adamw_torch"),
        prediction_loss_only=train_cfg.get("prediction_loss_only", False),
        fp16_full_eval=train_cfg.get("fp16_full_eval", False),
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        processing_class=processor.tokenizer,
    )

    eff_bs = train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"]
    print(f"  ✓ Effective batch size: {eff_bs}")
    print(f"  ✓ Max steps: {train_cfg['max_steps']}")
    print(f"  ✓ Gradient checkpointing: {train_cfg.get('gradient_checkpointing', False)}")

    print("\n[6/6] Starting training...")
    print("-"*60)
    train_result = trainer.train()
    print("-"*60)

    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    eval_metrics = trainer.evaluate()
    print(f"\n  WER: {eval_metrics.get('eval_wer','N/A'):.2f}%")
    print(f"  CER: {eval_metrics.get('eval_cer','N/A'):.2f}%")
    print(f"  Eval Loss: {eval_metrics.get('eval_loss','N/A'):.4f}")
    print(f"  Train Loss: {train_result.training_loss:.4f}")
    print(f"  Steps: {train_result.global_step}")
    print(f"  Best model: {trainer.state.best_model_checkpoint}")

    trainer.save_model(os.path.join(train_cfg["output_dir"], "final"))
    processor.save_pretrained(os.path.join(train_cfg["output_dir"], "final"))

    print(f"\n  ✓ Model saved to: {os.path.abspath(train_cfg['output_dir'])}/final")
    print("="*60)

    return eval_metrics

if __name__ == "__main__":
    main()
