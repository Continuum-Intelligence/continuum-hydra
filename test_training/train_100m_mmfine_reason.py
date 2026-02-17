from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ~100M-class model (GPT-2 is ~124M parameters).
DEFAULT_MODEL_ID = "gpt2"
DEFAULT_DATASET_ID = "OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trainer for MMFineReason dataset")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--output-dir", type=str, default="./outputs/mmfine_100m")
    parser.add_argument("--max-samples", type=int, default=2000, help="Limit samples for quick test runs")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path for resuming")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def flatten_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := flatten_to_text(item)))
    if isinstance(value, dict):
        # Common conversational keys are handled naturally by recursive flattening.
        chunks: list[str] = []
        for key in sorted(value.keys()):
            part = flatten_to_text(value[key])
            if part:
                chunks.append(f"{key}: {part}")
        return "\n".join(chunks)
    return str(value)


def example_to_text(example: dict[str, Any]) -> str:
    preferred_keys = [
        "text",
        "prompt",
        "response",
        "question",
        "answer",
        "instruction",
        "output",
        "messages",
        "conversations",
    ]

    parts: list[str] = []
    used = set()

    for key in preferred_keys:
        if key in example:
            txt = flatten_to_text(example[key]).strip()
            if txt:
                parts.append(txt)
                used.add(key)

    if not parts:
        for key in sorted(example.keys()):
            if key in used:
                continue
            txt = flatten_to_text(example[key]).strip()
            if txt:
                parts.append(f"{key}: {txt}")

    return "\n\n".join(parts).strip()


def choose_train_split(dataset_obj: Any) -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj

    # DatasetDict path
    if "train" in dataset_obj:
        return dataset_obj["train"]

    # Fallback to the first available split
    first_split = sorted(dataset_obj.keys())[0]
    return dataset_obj[first_split]


def latest_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    return str(checkpoints[-1])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    train_ds = choose_train_split(ds)

    if args.max_samples > 0:
        capped = min(args.max_samples, len(train_ds))
        train_ds = train_ds.select(range(capped))
        print(f"Using {capped} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    def tokenize_fn(example: dict[str, Any]) -> dict[str, Any]:
        text = example_to_text(example)
        if not text:
            text = "\n"
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.block_size,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = train_ds.map(
        tokenize_fn,
        remove_columns=train_ds.column_names,
        desc="Tokenizing dataset",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[],
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    resume_from = args.resume or latest_checkpoint(output_dir)
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print("Training complete")


if __name__ == "__main__":
    main()
