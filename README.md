
# Hugging Face – Лабораторна №3 (pipelines)

**Авторський шаблон** для виконання завдань (a)–(c). Кожне завдання — окремий файл у `tasks/`, запуск — через `main.py` або безпосередньо.

```
hf_lab3_project/
├─ main.py
├─ tasks/
│  ├─ task_a.py        # (a) Класифікація на вашому датасеті (__label__1/2)
│  ├─ task_b.py        # (b) Українські моделі: zero-shot, summarization, translation
│  └─ task_c.py        # (c) Diffusers: text2img / img2img / inpaint
├─ utils/
│  └─ dataset.py       # Парсер FastText-файлів
├─ outputs/            # Всі результати зберігаються тут
└─ requirements.txt
```

## Встановлення

1. Python 3.10+  
2. (Бажано) створіть віртуальне середовище
3. Встановіть залежності:

```bash
pip install -r requirements.txt
```

> **GPU** (CUDA) значно прискорює завдання (особливо `task_c`). На CPU також працює, але повільніше.

---

## (a) Класифікація текстів через `pipeline`

Потрібні файли: `train.ft.txt` та `test.ft.txt` у форматі FastText.

**Приклад рядка:**
```
__label__2 Great CD: My lovely Pat has one of the GREAT voices of her generation...
```

**Запуск через `main.py`:**
```bash
python main.py a --train /path/to/train.ft.txt --test /path/to/test.ft.txt   --models distilbert-base-uncased-finetuned-sst-2-english textattack/roberta-base-SST-2   --batch 32 --device -1 --outdir outputs/task_a
```

Або напряму:
```bash
python tasks/task_a.py --train /path/to/train.ft.txt --test /path/to/test.ft.txt
```

### Корисні прапорці
- `--invert-labels` — якщо у вашому датасеті навпаки: `__label__1 = POSITIVE`, `__label__2 = NEGATIVE`.
- `--models ...` — список моделей (мінімум дві). Можна додавати власні.
- `--device` — `-1` (CPU) або індекс GPU (`0`, `1`, …).
- `--batch` — розмір батчу для виклику pipeline.
- `--outdir` — куди зберігати результати.

### Вихідні файли
- `outputs/task_a/summary.json` — accuracy + повний classification report (precision/recall/F1).
- окремі `*_predictions.csv` — тексти, істинні та предиктовані мітки.

**Висновки:** порівняйте accuracy двох (і більше) моделей між собою та з результатами Вашої ЛР2.

---

## (b) Українські моделі через `pipeline`

**Запуск:**
```bash
python main.py b
```
або
```bash
python tasks/task_b.py --zsl-model joeddav/xlm-roberta-large-xnli   --sum-model csebuetnlp/mT5_multilingual_XLSum   --uk2en Helsinki-NLP/opus-mt-uk-en --en2uk Helsinki-NLP/opus-mt-en-uk   --outdir outputs/task_b
```

### Що виконується
1. **Zero-shot-classification (uk):** XNLI-моделі (`xlm-roberta-large-xnli` або `mDeBERTa-v3-base-mnli-xnli`).
   - `--labels` — кандидати класів, за замовчуванням `["позитивний","негативний","нейтральний"]`
   - `--zsl-hypothesis` — шаблон гіпотези (укр).

2. **Summarization (uk):** `csebuetnlp/mT5_multilingual_XLSum`.
   - `--sum-max`, `--sum-min` — обмеження довжини.

3. **Translation:** `Helsinki-NLP/opus-mt-uk-en` та `...-en-uk`.

**Результати:** `outputs/task_b/uk_models_demo.json`.

---

## (c) Diffusers: генерація / перетворення зображень

**Text-to-Image:**
```bash
python main.py c --mode text2img --prompt "Generate a world in the cyberpunk style like in the movie Tron: Legacy"   --model stabilityai/sd-turbo --steps 15 --guidance 2.5 --seed 42   --width 512 --height 512 --outdir outputs/task_c
```

**Image-to-Image:**
```bash
python main.py c --mode img2img --model stabilityai/sd-turbo   --init-image path/to/input.png --prompt "cinematic lighting, detailed"   --strength 0.6 --steps 20 --guidance 3.5 --outdir outputs/task_c
```

**Inpainting (маска біла = що замінюємо):**
```bash
python main.py c --mode inpaint --model stabilityai/sd-turbo   --init-image path/to/base.png --mask-image path/to/mask.png   --prompt "replace the sky with dramatic storm clouds" --steps 25 --outdir outputs/task_c
```

### Пояснення прапорців
- `--mode` — один з `text2img | img2img | inpaint`.
- `--prompt` — текстовий запит.
- `--negative-prompt` — що **не** бажано бачити.
- `--steps` — кількість кроків дифузії (більше = якісніше, повільніше).
- `--guidance` — CFG scale, наскільки сильно слідувати prompt (2–8 адекватно).
- `--width/--height` — розмір зображення.
- `--seed` — фіксація рандому для відтворюваності.
- `--init-image` — вхідне зображення для `img2img`/`inpaint`.
- `--mask-image` — маска для `inpaint` (білим замінюємо, чорним лишаємо).
- `--strength` — наскільки змінювати `init-image` (0..1).

---

## Нотатки
- Якщо під час (a) ярлики даних відрізняються — використайте `--invert-labels`.
- Ви можете додати більше моделей (наприклад, `finiteautomata/bertweet-base-sentiment-analysis`) до списку `--models`.
- Для великих наборів даних додайте `--device 0` (GPU).
- За необхідності створіть окреме `data/` і покладіть туди `train.ft.txt`, `test.ft.txt`.

Успіхів! :)
