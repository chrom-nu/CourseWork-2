# Plant Disease Classification & Grad-CAM Interpretation

Проєкт курсової роботи з крос-платформного програмування: система детекції хвороб листя рослин за фотографією за допомогою моделі глибокого навчання (ResNet34), візуального пояснення висновків нейромережі (Grad-CAM), FastAPI бекенду та клієнта на Flutter.

---

## 1. Архітектура та компоненти

- **`ml/`** — машинне навчання на PyTorch: датасет, аугментації, навчання базової моделі ResNet34, валідація, генерація звітів та Grad-CAM сіток.
- **`shared/`** — спільний код попередньої обробки зображень (ImageNet нормалізація, resize, center crop) та константи класів.
- **`backend/`** — сервіс FastAPI:
  - Асинхронний інференс моделі у пулі потоків (не блокує event loop).
  - Генерація та накладення карт уваги Grad-CAM.
  - База даних PostgreSQL для історії передбачень та фідбек-циклу користувачів.
  - Панель адміністратора SQLAdmin (`/admin`).
  - Міграції схеми через Alembic.
- **`mobile/`** — крос-платформний застосунок на Flutter з екраном перевірки зв'язку з бекендом.

---

## 2. Швидкий запуск через Docker Compose

### Вимоги:
- Встановлений Docker Desktop
- Чекпойнт моделі у `ml/runs/resnet34_weighted_seed42/best_model.pt`

### Команди:
```bash
# 1. Запуск API та PostgreSQL
docker compose up --build -d

# 2. Перевірка статусу контейнерів
docker compose ps

# 3. Перевірка працездатності сервісу
curl http://localhost:8000/health
```

Після запуску:
- **API Swagger документація:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Адмін-панель SQLAdmin:** [http://localhost:8000/admin](http://localhost:8000/admin)

Зупинка:
```bash
docker compose down
```

---

## 3. Локальний запуск бекенду (без Docker)

```bash
# 1. Активація віртуального середовища
.\.venv\Scripts\activate

# 2. Встановлення залежностей
pip install -r backend/requirements.txt

# 3. Налаштування змінних оточення (використання локального SQLite)
cp .env.example .env

# 4. Запуск міграцій
cd backend
alembic upgrade head
cd ..

# 5. Запуск сервера розробки
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 4. Контракт REST API

Повний опис контракту знаходиться у [`docs/api.md`](docs/api.md).

| Метод | Шлях | Опис |
|---|---|---|
| `GET` | `/health` | Стан сервісу, моделі та бази даних |
| `GET` | `/model-info` | Архітектура (`resnet34`), розмір входу, класи, температура |
| `POST` | `/predict` | Multipart завантаження фото (`file`), повертає діагноз, ймовірності та base64 карту Grad-CAM |
| `POST` | `/feedback` | Фіксація коректної мітки користувача (`prediction_id`, `correct_label`) |
| `GET` | `/admin` | Адмін-панель SQLAdmin для перегляду передбачень і фідбеку |

---

## 5. Запуск тестів

```bash
# Запуск повного набору юніт- та інтеграційних тестів
pytest backend/tests -v

# Перевірка 100% відповідності інференсу API тренувальній оцінці evaluate.py
python backend/tests/verify_val_10.py

# Перевірка відображення передбачень і фідбеку в SQLAdmin
python backend/tests/verify_admin.py
```

---

## 6. Запуск Flutter-клієнта

```bash
cd mobile

# Отримання залежностей
flutter pub get

# Для фізичного Android-пристрою (USB Debugging):
adb reverse tcp:8000 tcp:8000
flutter run

# Для швидкого перегляду в браузері:
flutter run -d chrome
```
