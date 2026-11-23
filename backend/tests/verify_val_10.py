import os
import sys
import glob

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from PIL import Image
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from shared.transforms import preprocess_image, get_model, IDX_TO_CLASS

def run_val_verification():
    val_healthy = sorted(glob.glob("ml/data/val/healthy/*.jpg") + glob.glob("ml/data/val/healthy/*.JPG"))[:5]
    val_sick = sorted(glob.glob("ml/data/val/sick/*.jpg") + glob.glob("ml/data/val/sick/*.JPG"))[:5]
    test_images = val_healthy + val_sick

    print(f"Testing on {len(test_images)} validation images: 5 healthy, 5 sick.")

    # 1. Raw model evaluation
    ckpt_path = "ml/runs/resnet34_weighted_seed42/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_model = get_model("resnet34", num_classes=2, pretrained=False)
    raw_model.load_state_dict(ckpt["state_dict"])
    raw_model.eval()

    with TestClient(app) as client:
        header = f"| {'Image':<30} | {'Raw Label':<10} | {'API Label':<10} | {'Raw P(sick)':<12} | {'API P(sick)':<12} | {'Diff':<8} | {'Status':<6} |"
        sep = "|" + "-"*32 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*10 + "|" + "-"*8 + "|"
        print(header)
        print(sep)

        all_match = True
        for img_path in test_images:
            fname = os.path.basename(img_path)
            with Image.open(img_path) as img:
                tensor = preprocess_image(img, img_size=224)
            
            with torch.no_grad():
                logits = raw_model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                raw_pred_idx = torch.argmax(probs).item()
                raw_label = IDX_TO_CLASS[raw_pred_idx]
                raw_p_sick = probs[1].item()

            # Call API
            with open(img_path, "rb") as f:
                resp = client.post("/predict", files={"file": (fname, f, "image/jpeg")})
            
            assert resp.status_code == 200, f"Failed: {resp.text}"
            data = resp.json()
            api_label = data["label"]
            api_p_sick = data["probabilities"]["sick"]
            diff = abs(raw_p_sick - api_p_sick)
            match = (raw_label == api_label) and (diff < 1e-3)
            if not match:
                all_match = False
            status_str = "PASS" if match else "FAIL"
            short_name = fname[:28]
            print(f"| {short_name:<30} | {raw_label:<10} | {api_label:<10} | {raw_p_sick:<12.4f} | {api_p_sick:<12.4f} | {diff:<8.4f} | {status_str:<6} |")

        print("\nAll 10 validation images matched exactly:", all_match)
        assert all_match, "Mismatch between raw evaluate logic and API!"

if __name__ == "__main__":
    run_val_verification()
