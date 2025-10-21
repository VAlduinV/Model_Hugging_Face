
"""
tasks/task_b.py

Task (b): Work with Ukrainian-capable models via `pipeline`:
- Zero-shot classification: XNLI-tuned models support Ukrainian (uk).
  e.g., "joeddav/xlm-roberta-large-xnli" or "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli".
- Summarization: "csebuetnlp/mT5_multilingual_XLSum" supports Ukrainian.
- Translation: Helsinki-NLP OPUS-MT models for uk<->en.

This script demonstrates all three tasks and saves outputs.
"""
from __future__ import annotations

import argparse, os, json
from transformers import pipeline

def main():
    ap = argparse.ArgumentParser(description="Task (b): Ukrainian models demos (zero-shot, summarization, translation).")
    ap.add_argument("--outdir", type=str, default="outputs/task_b", help="Where to save results.")
    # Zero-shot
    ap.add_argument("--zsl-model", type=str, default="joeddav/xlm-roberta-large-xnli", help="Model for zero-shot classification.")
    ap.add_argument("--labels", nargs="+", default=["позитивний", "негативний", "нейтральний"], help="Candidate labels for zero-shot.")
    ap.add_argument("--zsl-hypothesis", type=str, default="Цей текст має тональність {}.", help="Template for zero-shot.")
    # Summarization
    ap.add_argument("--sum-model", type=str, default="csebuetnlp/mT5_multilingual_XLSum", help="Model for summarization (supports uk).")
    ap.add_argument("--sum-max", type=int, default=128, help="Max summary tokens.")
    ap.add_argument("--sum-min", type=int, default=24, help="Min summary tokens.")
    # Translation
    ap.add_argument("--uk2en", type=str, default="Helsinki-NLP/opus-mt-uk-en", help="uk->en translation model")
    ap.add_argument("--en2uk", type=str, default="Helsinki-NLP/opus-mt-en-uk", help="en->uk translation model")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Demo texts (you may replace with your own via editing this script or loading a file).
    text_uk = "Це дуже цікавий альбом з потужними вокалами та чудовою енергією. Рекомендую всім прихильникам якісної музики."
    text_long_uk = (
        "Українські інженери створили новий прототип безпілотника, який здатний виконувати автономні місії в складних умовах. "
        "Під час випробувань апарат продемонстрував стабільний політ та точне наведення на ціль, "
        "а також енергоефективну систему живлення. Проєкт підтримується кількома університетами та стартапами."
    )
    text_en = "This album has stunning vocals and great energy. I highly recommend it to all fans of quality music."

    results = {}

    # 1) Zero-shot classification in Ukrainian
    zsl = pipeline("zero-shot-classification", model=args.zsl_model)
    zsl_out = zsl(text_uk, candidate_labels=args.labels, hypothesis_template=args.zsl_hypothesis)
    results["zero_shot"] = zsl_out

    # 2) Summarization in Ukrainian
    summarizer = pipeline("summarization", model=args.sum_model)
    sum_out = summarizer(text_long_uk, max_length=args.sum_max, min_length=args.sum_min)
    results["summarization"] = sum_out

    # 3) Translation uk->en and en->uk
    trans_uk_en = pipeline("translation", model=args.uk2en)
    trans_en_uk = pipeline("translation", model=args.en2uk)
    uk2en = trans_uk_en(text_uk)[0]["translation_text"]
    en2uk = trans_en_uk(text_en)[0]["translation_text"]
    results["translation"] = {"uk2en": uk2en, "en2uk": en2uk}

    # Save
    with open(os.path.join(args.outdir, "uk_models_demo.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=== Ukrainian models demo saved to:", os.path.join(args.outdir, "uk_models_demo.json"))
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
