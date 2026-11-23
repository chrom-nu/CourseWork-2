def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert data["architecture"] == "resnet34"
    assert data["input_size"] == [224, 224]
    assert data["classes"] == ["healthy", "sick"]
    assert "temperature" in data
    assert "device" in data
