import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RepoClsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_type="weighted_ce", pos_weight=1.0, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        class_weights = torch.tensor([1.0, float(self.pos_weight)], device=logits.device)
        ce = F.cross_entropy(logits, labels, weight=class_weights)

        if self.loss_type == "focal":
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            labels_onehot = F.one_hot(labels, num_classes=2).float()
            pt = (probs * labels_onehot).sum(dim=-1)
            focal_factor = (1 - pt).pow(self.focal_gamma)
            ce_each = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
            loss = (focal_factor * ce_each).mean()
        else:
            loss = ce

        return (loss, outputs) if return_outputs else loss


@dataclass
class LLMArtifacts:
    model
    tokenizer


def load_model_and_tokenizer(cfg: Dict):
    model_name = cfg["model"]["model_name"]
    use_4bit = bool(cfg["model"]["use_4bit"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(cfg["model"]["lora_r"]),
        lora_alpha=int(cfg["model"]["lora_alpha"]),
        lora_dropout=float(cfg["model"]["lora_dropout"]),
        target_modules=list(cfg["model"]["lora_target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def fit_one_fold(train_texts, train_labels, val_texts, val_labels, cfg: Dict, fold_out_dir: str):
    set_seed(cfg["seed"])
    os.makedirs(fold_out_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(cfg)

    train_ds = RepoClsDataset(
        train_texts, train_labels, tokenizer, int(cfg["model"]["max_length"])
    )
    val_ds = RepoClsDataset(
        val_texts, val_labels, tokenizer, int(cfg["model"]["max_length"])
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    args = TrainingArguments(
        output_dir=fold_out_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=int(cfg["model"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["model"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["model"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["model"]["learning_rate"]),
        num_train_epochs=float(cfg["model"]["num_train_epochs"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
        warmup_ratio=float(cfg["model"]["warmup_ratio"]),
        lr_scheduler_type=str(cfg["model"]["lr_scheduler_type"]),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        save_total_limit=1,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        loss_type=cfg["loss"]["loss_type"],
        pos_weight=float(cfg["loss"]["positive_class_weight"]),
        focal_gamma=float(cfg["loss"]["focal_gamma"]),
    )
    trainer.train()

    return trainer, model, tokenizer


def predict_proba(trainer: Trainer, texts: List[str], labels: Optional[List[int]], tokenizer, cfg: Dict):
    dummy_labels = labels if labels is not None else [0] * len(texts)
    ds = RepoClsDataset(texts, dummy_labels, tokenizer, int(cfg["model"]["max_length"]))
    out = trainer.predict(ds)
    logits = out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    return probs