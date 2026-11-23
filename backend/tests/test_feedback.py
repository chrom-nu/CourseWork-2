import uuid


def test_feedback_success(client, valid_jpeg_bytes):
    # 1. Create prediction first
    pred_res = client.post(
        "/predict",
        files={"file": ("leaf.jpg", valid_jpeg_bytes, "image/jpeg")},
    )
    assert pred_res.status_code == 200
    prediction_id = pred_res.json()["prediction_id"]

    # 2. Submit feedback
    feedback_res = client.post(
        "/feedback",
        json={"prediction_id": prediction_id, "correct_label": "healthy"},
    )
    assert feedback_res.status_code == 200
    data = feedback_res.json()
    assert data["status"] == "ok"
    assert "feedback_id" in data
    assert data["prediction_id"] == prediction_id
    assert data["correct_label"] == "healthy"


def test_feedback_unknown_prediction_id(client):
    random_id = str(uuid.uuid4())
    feedback_res = client.post(
        "/feedback",
        json={"prediction_id": random_id, "correct_label": "sick"},
    )
    assert feedback_res.status_code == 404
    data = feedback_res.json()
    assert "error" in data
    assert data["error"]["code"] == "PREDICTION_NOT_FOUND"


def test_feedback_invalid_label(client, valid_jpeg_bytes):
    pred_res = client.post(
        "/predict",
        files={"file": ("leaf.jpg", valid_jpeg_bytes, "image/jpeg")},
    )
    assert pred_res.status_code == 200
    prediction_id = pred_res.json()["prediction_id"]

    feedback_res = client.post(
        "/feedback",
        json={"prediction_id": prediction_id, "correct_label": "invalid_label"},
    )
    assert feedback_res.status_code == 422
    data = feedback_res.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
