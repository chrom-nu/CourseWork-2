def test_predict_success_jpeg(client, valid_jpeg_bytes):
    response = client.post(
        "/predict",
        files={"file": ("leaf.jpg", valid_jpeg_bytes, "image/jpeg")},
        data={"include_heatmap": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert data["label"] in ["healthy", "sick"]
    assert "healthy" in data["probabilities"]
    assert "sick" in data["probabilities"]
    assert 0.0 <= data["probabilities"]["healthy"] <= 1.0
    assert 0.0 <= data["probabilities"]["sick"] <= 1.0
    assert data["heatmap"] is not None
    assert len(data["heatmap"]) > 100
    assert "model_version" in data


def test_predict_without_heatmap(client, valid_png_bytes):
    response = client.post(
        "/predict",
        files={"file": ("leaf.png", valid_png_bytes, "image/png")},
        data={"include_heatmap": "false"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["heatmap"] is None


def test_predict_invalid_content_type(client, text_file_bytes):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", text_file_bytes, "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_IMAGE_FORMAT"


def test_predict_corrupted_image(client, corrupted_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("corrupted.jpg", corrupted_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "CORRUPTED_IMAGE"


def test_predict_file_too_large(client):
    large_bytes = b"0" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
    response = client.post(
        "/predict",
        files={"file": ("large.jpg", large_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "FILE_TOO_LARGE"
