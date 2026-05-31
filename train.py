import os
os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/hf_cache/datasets"
os.environ["TOKENIZERS_CACHE"] = "/root/autodl-tmp/hf_cache/tokenizers"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_DOWNLOAD_TIMEOUT"] = "600"

import os, sys, argparse, warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch, yaml, numpy as np
from datasets import load_dataset, Audio, Dataset
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer,
    WhisperFeatureExtractor, Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
import evaluate
warnings.filterwarnings("ignore")

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

def compute_metrics(pred, tokenizer):
    import numpy as np
    metric_wer = evaluate.load("wer"); metric_cer = evaluate.load("cer")
    pred_ids = pred.predictions; label_ids = pred.label_ids
    if isinstance(pred_ids, tuple): pred_ids = pred_ids[0]
    def flatten_seq(seq):
        if isinstance(seq, np.ndarray): return [int(x) for x in seq.flatten().tolist()]
        if isinstance(seq, (list, tuple)):
            result = []
            for item in seq:
                if isinstance(item, (np.ndarray, list, tuple)): result.extend(flatten_seq(item))
                else: result.append(int(item))
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
    parser.add_argument("--config", type=str, default="configs/gpu_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    val_cfg = config["validation"]; model_name = config["model"]["name"]
    data_cfg = config["data"]; train_cfg = config["training"]
    device = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory/1024**3
    print("="*60); print("ASR Training Pipeline"); print("="*60)
    print(f"Model: {model_name} | Dataset: {data_cfg['dataset_name']} ({data_cfg['subset']})")
    print(f"Train: {data_cfg['split']} | Eval: {val_cfg.get('eval_subset', data_cfg['split'])}")
    print(f"Device: {device}", end="")
    if torch.cuda.is_available(): print(f" ({gpu_name}, {gpu_mem:.1f} GB)")
    else: print()
    print("="*60)

    print("\n[1/6] Loading model...")
    processor = WhisperProcessor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=config["model"]["language"], task=config["model"]["task"])
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None; model.config.suppress_tokens = []; model.config.use_cache = False
    use_fp16 = config["model"].get("use_fp16", False) and torch.cuda.is_available()
    if use_fp16: print("  FP16 enabled")
    print(f"  Params: {model.num_parameters()/1e6:.1f}M")

    print("\n[2/6] Loading train dataset...")
    splits = data_cfg["split"].split("+")
    train_dataset_raw = []
    for split in splits:
        ds = load_dataset(data_cfg["dataset_name"], data_cfg["subset"], split=split.strip())
        train_dataset_raw.extend(list(ds))
    train_dataset_raw = train_dataset_raw[:10000]
    print(f"  Train samples: {len(train_dataset_raw)}")

    print("\n[3/6] Loading eval dataset...")
    eval_split = val_cfg.get("eval_subset", data_cfg["split"])
    eval_dataset_raw = load_dataset(data_cfg["dataset_name"], data_cfg["subset"], split=eval_split)
    max_eval = val_cfg.get("max_eval_samples", len(eval_dataset_raw))
    if max_eval < len(eval_dataset_raw): 
        eval_dataset_raw = eval_dataset_raw.select(range(max_eval))
    print(f"  Eval samples: {len(eval_dataset_raw)}")

    print("\n[4/6] Preprocessing...")
    train_list = []
    for i, example in enumerate(train_dataset_raw):
        audio_array = example["audio"]["array"]
        processed = {
            "input_features": processor.feature_extractor(audio_array, sampling_rate=data_cfg["sample_rate"]).input_features[0],
            "labels": processor.tokenizer(example["text"]).input_ids
        }
        train_list.append(processed)
        if (i + 1) % 1000 == 0:
            print(f"  Processed train: {i+1}/{len(train_dataset_raw)}")
    
    eval_list = []
    for i, example in enumerate(eval_dataset_raw):
        audio_array = example["audio"]["array"]
        processed = {
            "input_features": processor.feature_extractor(audio_array, sampling_rate=data_cfg["sample_rate"]).input_features[0],
            "labels": processor.tokenizer(example["text"]).input_ids
        }
        eval_list.append(processed)
    
    train_dataset = Dataset.from_list(train_list)
    eval_dataset = Dataset.from_list(eval_list)
    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    print("\n[5/6] Configuring trainer...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        warmup_steps=train_cfg["warmup_steps"], max_steps=train_cfg["max_steps"],
        eval_strategy="steps", eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"], logging_steps=train_cfg["logging_steps"],
        logging_dir=train_cfg["logging_dir"],
        dataloader_num_workers=train_cfg["num_workers"],
        dataloader_pin_memory=train_cfg["dataloader_pin_memory"],
        report_to=train_cfg["report_to"],
        load_best_model_at_end=False,
        push_to_hub=False, save_total_limit=1, remove_unused_columns=False,
        fp16=use_fp16, bf16=False, generation_max_length=225,
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args, model=model, train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
    )
    eff_bs = train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"]
    print(f"  Effective batch: {eff_bs} | Max steps: {train_cfg['max_steps']}")

    print("\n[6/6] Training..."); print("-"*60)
    train_result = trainer.train()
    print("-"*60)

    print("\n"+"="*60); print("TRAINING COMPLETE"); print("="*60)
    print(f"\n  Train Loss: {train_result.training_loss:.4f}")
    print(f"  Steps: {train_result.global_step}")
    trainer.save_model(os.path.join(train_cfg["output_dir"], "final"))
    processor.save_pretrained(os.path.join(train_cfg["output_dir"], "final"))
    print(f"\n  Model saved to: {os.path.abspath(train_cfg['output_dir'])}/final")
    print("="*60)
    return train_result

if __name__ == "__main__": main()
