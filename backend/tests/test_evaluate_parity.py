import os
import glob
from PIL import Image
import torch
from shared.transforms import preprocess_image, get_model, IDX_TO_CLASS


def test_evaluate_parity(client):
    val_files = sorted(glob.glob("ml/data/val/*/*.jpg") + glob.glob("ml/data/val/*/*.JPG"))[:4]
    if not val_files:
        return  # If dataset not present (e.g. CI stub), skip gracefully

    ckpt_path = "ml/runs/resnet34_weighted_seed42/best_model.pt"
    if not os.path.exists(ckpt_path):
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_model = get_model("resnet34", num_classes=2, pretrained=False)
    raw_model.load_state_dict(ckpt["state_dict"])
    raw_model.eval()

    for fpath in val_files:
        with Image.open(fpath) as img:
            tensor = preprocess_image(img, img_size=224)

        with torch.no_grad():
            logits = raw_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            raw_pred_idx = torch.argmax(probs).item()
            raw_label = IDX_TO_CLASS[raw_pred_idx]
            raw_p_sick = round(float(probs[1].item()), 4)

        with open(fpath, "rb") as f:
            resp = client.post("/predict", files={"file": (os.path.basename(fpath), f, "image/jpeg")})

        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == raw_label
        assert abs(data["probabilities"]["sick"] - raw_p_sick) < 1e-3
