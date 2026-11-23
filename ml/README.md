# Machine Learning Module

Модуль навчання та оцінки baseline-моделі (ResNet) для бінарної класифікації хвороб рослин (`healthy` vs `sick`).

## Структура
- `src/` — вихідний код (`dataset.py`, `train.py`, `evaluate.py`, `check_dataset.py`)
- `reports/` — звіти та візуалізації (`dataset_report.md`, сітка аугментацій, confusion matrix)
- `data/` — датасет (train, val, test, unlabeled) [в `.gitignore`]
- `runs/` — збережені чекпойнти та логи експериментів [в `.gitignore`]

## Налаштування оточення
1. Створення віртуального середовища:
   ```powershell
   python -m venv .venv
   ```
2. Встановлення залежностей (з підтримкою CUDA):
   ```powershell
   .venv\Scripts\python.exe -m pip install -r ml/requirements.txt
   ```
3. Перевірка доступності GPU:
   ```powershell
   .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
   ```

## Команди запуску
- **Перевірка датасету:**
  ```powershell
  .venv\Scripts\python.exe ml/src/check_dataset.py
  ```
- **Навчання baseline ResNet34:**
  ```powershell
  .venv\Scripts\python.exe ml/src/train.py --arch resnet34 --epochs 15 --batch-size 32 --lr 1e-4 --use-class-weights
  ```
- **Оцінка моделі на val / test:**
  ```powershell
  .venv\Scripts\python.exe ml/src/evaluate.py --checkpoint ml/runs/baseline_resnet34/best_model.pt --split val
  ```
