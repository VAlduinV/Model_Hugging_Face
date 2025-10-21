
"""
tasks/task_a.py

Task (a): Evaluate at least two pretrained text-classification models from Hugging Face
on your FastText dataset (train.ft.txt / test.ft.txt). We use `transformers.pipeline`
for simplicity and speed.

Two default models (can be changed via CLI flags):
- distilbert-base-uncased-finetuned-sst-2-english
- textattack/roberta-base-SST-2
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from transformers import pipeline

from utils.dataset import read_fasttext_file, to_dataframe, save_json

DEFAULT_MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/roberta-base-SST-2",
]

def _predict_batch(nlp, texts: List[str], batch_size: int = 32) -> List[int]:
    """
    Predict labels for a batch of texts using a transformers pipeline.
    Returns list of ints: 1 for POSITIVE, 0 for NEGATIVE.
    """
    preds: List[int] = []
    for i in range(0, len(texts), batch_size):
        sub = texts[i:i+batch_size]
        outputs = nlp(sub, truncation=True)
        # pipeline can return a dict or list of dicts depending on input
        if isinstance(outputs, dict):
            outputs = [outputs]
        for o in outputs:
            label = o["label"].upper()
            if "POS" in label:  # e.g. POSITIVE / LABEL_1
                preds.append(1)
            elif "NEG" in label or "LABEL_0" in label:
                preds.append(0)
            else:
                # Fallback: compare scores if two labels exist
                if isinstance(o.get("score"), float):
                    # If only a single score, assume it is positive class prob
                    preds.append(int(o["score"] >= 0.5))
                else:
                    preds.append(0)
    return preds

def evaluate_model(model_name: str, df_test: "pd.DataFrame", device: int = -1, batch: int = 32):
    """Run the model on df_test and compute metrics."""
    clf = pipeline("sentiment-analysis", model=model_name, device=device)
    y_true = df_test["label"].tolist()
    y_pred = _predict_batch(clf, df_test["text"].tolist(), batch_size=batch)

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=["NEGATIVE", "POSITIVE"], output_dict=True)
    return acc, report, y_pred

def main():
    ap = argparse.ArgumentParser(description="Task (a): Evaluate Hugging Face classifiers on FastText dataset.")
    ap.add_argument("--train", type=str, required=True, help="Path to train.ft.txt")
    ap.add_argument("--test", type=str, required=True, help="Path to test.ft.txt")
    ap.add_argument("--invert-labels", action="store_true", help="Swap mapping: __label__1<->POSITIVE, __label__2<->NEGATIVE")
    ap.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS, help="Model names to evaluate.")
    ap.add_argument("--device", type=int, default=-1, help="Device index for transformers pipeline (-1 = CPU, 0 = first GPU).")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for pipeline calls.")
    ap.add_argument("--outdir", type=str, default="outputs/task_a", help="Directory to save results.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Read datasets
    test_records = read_fasttext_file(args.test, invert_labels=args.invert_labels)
    df_test = to_dataframe(test_records)

    summary = {}
    for model in args.models:
        print(f"Evaluating {model} on {len(df_test)} test samples...")
        acc, report, y_pred = evaluate_model(model, df_test, device=args.device, batch=args.batch)
        summary[model] = {"accuracy": acc, "report": report}

        # Save per-model predictions
        pred_path = os.path.join(args.outdir, f"{Path(model).name}_predictions.csv")
        pd.DataFrame({
            "text": df_test["text"],
            "true_label": df_test["label"],
            "pred_label": y_pred,
        }).to_csv(pred_path, index=False, encoding="utf-8")
        print(f"Saved predictions to {pred_path}")

    # Save overall summary
    save_json(summary, os.path.join(args.outdir, "summary.json"))
    print("\n=== RESULTS ===")
    for m, stats in summary.items():
        print(f"{m}: accuracy={stats['accuracy']:.4f}")
    print(f"\nFull classification reports saved to {os.path.join(args.outdir, 'summary.json')}")

if __name__ == "__main__":
    main()
