import os
import sys

# Ensure workspace root in sys.path
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from backend.app.main import app

def test_admin_flow():
    with TestClient(app) as client:
        # 1. Post a prediction
        test_img_path = "ml/data/val/healthy/val__12.jpg"
        with open(test_img_path, "rb") as f:
            pred_resp = client.post("/predict", files={"file": ("val__12.jpg", f, "image/jpeg")})
        assert pred_resp.status_code == 200, f"Predict failed: {pred_resp.text}"
        pred_data = pred_resp.json()
        pred_id = pred_data["prediction_id"]
        print(f"Prediction created: id={pred_id}, label={pred_data['label']}")

        # 2. Post feedback
        feedback_resp = client.post("/feedback", json={"prediction_id": pred_id, "correct_label": "healthy"})
        assert feedback_resp.status_code == 200, f"Feedback failed: {feedback_resp.text}"
        feedback_data = feedback_resp.json()
        print(f"Feedback created: id={feedback_data['feedback_id']}, pred_id={feedback_data['prediction_id']}")

        # 3. Check SQLAdmin endpoints
        admin_pred_resp = client.get("/admin/prediction/list")
        assert admin_pred_resp.status_code == 200, f"Admin predictions list failed: {admin_pred_resp.status_code}"
        assert pred_id in admin_pred_resp.text, "Prediction ID not found in SQLAdmin prediction list!"
        print("SQLAdmin prediction list: Prediction successfully visible in admin UI!")

        admin_feedback_resp = client.get("/admin/feedback/list")
        assert admin_feedback_resp.status_code == 200, f"Admin feedback list failed: {admin_feedback_resp.status_code}"
        assert pred_id in admin_feedback_resp.text, "Feedback prediction ID not found in SQLAdmin feedback list!"
        print("SQLAdmin feedback list: Feedback successfully visible in admin UI!")

    print("\nTask 4 verification PASSED: Both prediction and feedback are visible in SQLAdmin!")

if __name__ == "__main__":
    test_admin_flow()
