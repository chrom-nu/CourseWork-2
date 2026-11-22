# Контракт REST API (Специфікація)

> **Документ:** `docs/api.md`  
> **Версія:** 1.0.0  
> **Статус:** Очікує погодження з автором (Andrew)  
> **Базовий префікс:** `/api/v1` (або прямі роути згідно з WEEK3.md)

---

## 1. Загальні положення

- Усі відповіді повертаються у форматі `application/json` (крім base64 зображення теплової карти всередині JSON).
- Часові мітки передаються у форматі ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`).
- Ідентифікатори запитів (`prediction_id`, `feedback_id`) є UUID v4.

---

## 2. Формат помилок (Unified Error Format)

Усі помилки повертають статус відмінний від 2xx та уніфікований JSON:

```json
{
  "error": {
    "code": "ERROR_CODE_STRING",
    "message": "Зрозумілий опис помилки для користувача / клієнта",
    "details": null
  }
}
```

### Стандартні коди помилок:

| HTTP Status | Код помилки (`code`) | Опис причини |
|---|---|---|
| `400 Bad Request` | `INVALID_IMAGE_FORMAT` | Файл не є валідним JPEG або PNG. |
| `400 Bad Request` | `CORRUPTED_IMAGE` | Файл неможливо відкрити як зображення (пошкоджені байти). |
| `400 Bad Request` | `INVALID_INPUT` | Помилка валідації полів запиту. |
| `404 Not Found` | `PREDICTION_NOT_FOUND` | Вказаний `prediction_id` не знайдено в базі даних. |
| `413 Payload Too Large` | `FILE_TOO_LARGE` | Розмір файлу перевищує допустимий ліміт (наприклад, 10 МБ). |
| `422 Unprocessable Entity`| `VALIDATION_ERROR` | FastAPI / Pydantic помилка валідації схеми. |
| `500 Internal Server Error`| `INFERENCE_ERROR` | Помилка під час виконання інференсу чи Grad-CAM. |
| `503 Service Unavailable` | `SERVICE_UNHEALTHY` | Модель не ініціалізована або база даних недоступна. |

---

## 3. Специфікація ендпоінтів

### 3.1. `GET /health`
Перевірка працездатності сервісу, стану завантаженої моделі та з'єднання з БД.

- **Метод:** `GET`
- **Шлях:** `/health`
- **Автентифікація:** Не потрібна

#### Відповідь (200 OK):
```json
{
  "status": "ok",
  "model_loaded": true,
  "database_connected": true,
  "version": "1.0.0"
}
```

#### Відповідь (503 Service Unavailable):
```json
{
  "status": "degraded",
  "model_loaded": false,
  "database_connected": true,
  "version": "1.0.0"
}
```

---

### 3.2. `GET /model-info`
Отримання метаданих поточної активної моделі.

- **Метод:** `GET`
- **Шлях:** `/model-info`
- **Автентифікація:** Не потрібна

#### Відповідь (200 OK):
```json
{
  "model_version": "resnet34_weighted_seed42",
  "architecture": "resnet34",
  "input_size": [224, 224],
  "classes": ["healthy", "sick"],
  "temperature": 1.0,
  "device": "cpu"
}
```

---

### 3.3. `POST /predict`
Класифікація завантаженого фото рослини та генерація Grad-CAM карти уваги.

- **Метод:** `POST`
- **Шлях:** `/predict`
- **Контент:** `multipart/form-data`
- **Параметри форми:**
  - `file` (обов'язкове): бінарні байти зображення (тип `image/jpeg` або `image/png`). Максимальний розмір: 10 МБ.
  - `include_heatmap` (опціональне, bool, за замовчуванням `true`): чи генерувати Grad-CAM.

#### Валідація:
1. Перевірка Content-Type (лише `image/jpeg`, `image/png`).
2. Перевірка заголовка / магічних байтів та спроба відкрити через PIL (`verify()`).
3. Перевірка розміру (макс. 10 MB).

#### Відповідь (200 OK):
```json
{
  "prediction_id": "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
  "label": "sick",
  "probabilities": {
    "healthy": 0.0821,
    "sick": 0.9179
  },
  "heatmap": "iVBORw0KGgoAAAANSUhEUgAA...",
  "model_version": "resnet34_weighted_seed42",
  "created_at": "2026-11-23T00:05:00Z"
}
```

*Примітка щодо `heatmap`: повертається чистий base64-рядок PNG-зображення теплової карти Grad-CAM, накладеної на оригінальне зображення (або null, якщо `include_heatmap=false`).*

#### Відповідь з помилкою (400 Bad Request):
```json
{
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "Only JPEG and PNG images are supported. Received: image/webp",
    "details": null
  }
}
```

---

### 3.4. `POST /feedback`
Надсилання відгуку користувача щодо коректності прогнозу для донавчання (feedback loop).

- **Метод:** `POST`
- **Шлях:** `/feedback`
- **Контент:** `application/json`

#### Тіло запиту:
```json
{
  "prediction_id": "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
  "correct_label": "sick"
}
```

#### Валідація:
- `prediction_id`: обов'язковий валідний UUID v4. Повинен існувати в таблиці `predictions`.
- `correct_label`: значення з множини `["healthy", "sick"]`.

#### Відповідь (200 OK):
```json
{
  "status": "ok",
  "feedback_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
  "prediction_id": "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
  "correct_label": "sick",
  "created_at": "2026-11-23T00:06:00Z"
}
```

#### Відповідь з помилкою (404 Not Found):
```json
{
  "error": {
    "code": "PREDICTION_NOT_FOUND",
    "message": "Prediction with ID 'c9bf9e57-1685-4c89-bafb-ff5af830be8a' does not exist.",
    "details": null
  }
}
```
