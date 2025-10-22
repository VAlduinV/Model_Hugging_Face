# REPORT_FILES_CSV.md

Цей документ описує призначення CSV‑файлів з передбаченнями моделей і пояснює, що саме зараз завантажено.

## Що це за файли

- `distilbert-base-uncased-finetuned-sst-2-english_predictions.csv`
- `roberta-base-SST-2_predictions.csv`

**Призначення:** у цих CSV зазвичай зберігаються *покадрові передбачення* для кожного тексту з тестового набору. Вони потрібні для:
- побудови **confusion matrix**, **precision/recall/F1** без повторного інференсу;
- **помилкоаналізу** (перегляд рядків, де `true_label != pred_label`);
- порівняння моделей на *одному і тому ж* тесті;
- відбору зразків для **активного навчання**/донавчання.

### Очікувана схема CSV
```
text,true_label,pred_label
"I loved the soundtrack...",1,1
"Very slow shipping and bad quality.",0,0
...
```
- `true_label`: 0=NEGATIVE, 1=POSITIVE
- `pred_label`: прогноз відповідної моделі

## Що всередині *поточних* файлів

Завантажені файли виглядають як **Git LFS pointers** (не самі дані, а маленькі вказівники на великі файли). Тому вони не читаються як таблиця.

### distilbert-base-uncased-finetuned-sst-2-english_predictions.csv
- Розмір файла (байт): **134**
- MD5: `117c1bf4e3991f8c091da28f84a60174`
- Перші рядки:
```
version https://git-lfs.github.com/spec/v1
oid sha256:1c713fef2549ed6998627ecd07cb4c16e88485baeaac459cbd21e1e801da1de2
size 175471400
```

### roberta-base-SST-2_predictions.csv
- Розмір файла (байт): **134**
- MD5: `85cb96eb372a2dfe007a6f76bfb9fce3`
- Перші рядки:
```
version https://git-lfs.github.com/spec/v1
oid sha256:e27141092771cc8541fab103d4715ebb66eca6c558793ca64665783e2d1ced8f
size 175471400
```

> Формат LFS-посилань має вигляд:
> ```
> version https://git-lfs.github.com/spec/v1
> oid sha256:<хеш-оригіналу>
> size <розмір_оригінального_файла>
> ```
> Це означає, що замість реальних CSV приїхали тільки «квиточки» Git LFS.

## Як отримати *реальні* CSV

1. **Увімкнути Git LFS локально** і стягнути об’єкти:
   ```bash
   git lfs install
   git lfs pull
   ```
2. **Експортувати з проєкту** повторним запуском (якщо потрібно згенерувати заново):
   ```bash
   python main.py a --train data/train.ft.txt --test data/test.ft.txt      --models distilbert-base-uncased-finetuned-sst-2-english textattack/roberta-base-SST-2      --batch 64 --device 0 --outdir outputs/task_a
   ```
   Після цього в `outputs/task_a/` з’являться повноцінні `*_predictions.csv`.
3. **Перевірити, що це CSV**, відкривши файл — перший рядок має бути `text,true_label,pred_label` (або з іншим порядком колонок).

## Як ці файли використовує код у проєкті

- Скрипт `tasks/task_a.py` після інференсу записує для кожної моделі файл `*_predictions.csv` і агрегує підсумки в `summary.json`.
- Будь‑який подальший аналіз (heatmaps, top‑errors, довжина тексту vs помилки, тощо) робиться поверх цих CSV без повторного прогону моделей.

Якщо надаси саме **повні CSV** (не LFS‑пойнтери), я можу автоматично порахувати точні розподіли, перевірити узгодженість з `summary.json` та зібрати confusion matrices.
