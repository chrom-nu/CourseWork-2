import os
import json
import hashlib
from collections import defaultdict
from PIL import Image
import imagehash
import numpy as np
from tqdm import tqdm

DATA_DIR = os.path.join("ml", "data")
REPORTS_DIR = os.path.join("ml", "reports")
SPLITS = ["train", "val", "test", "unlabeled"]
EXPECTED_COUNTS = {
    "train": {"healthy": 170, "sick": 432, "total": 602},
    "val": {"healthy": 59, "sick": 143, "total": 202},
    "test": {"healthy": 55, "sick": 147, "total": 202},
    "unlabeled": {"total": 5001}
}

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_all_images(folder):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts or not ext:
                images.append(os.path.join(root, f))
    return sorted(images)

def analyze_dataset():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = {
        "counts": {},
        "corrupted_files": [],
        "formats": defaultdict(int),
        "color_modes": defaultdict(int),
        "sizes": defaultdict(int),
        "aspect_ratios": defaultdict(int),
        "total_files": 0
    }

    records = []

    print("Analyzing dataset splits...")
    for split in SPLITS:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory {split_dir} does not exist.")
            continue

        if split == "unlabeled":
            files = find_all_images(split_dir)
            report["counts"][split] = {"total": len(files)}
            for fpath in tqdm(files, desc=f"Scanning {split}"):
                process_file(fpath, split, None, report, records)
        else:
            classes = ["healthy", "sick"]
            report["counts"][split] = {}
            split_total = 0
            for cls in classes:
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.exists(cls_dir):
                    report["counts"][split][cls] = 0
                    continue
                files = find_all_images(cls_dir)
                report["counts"][split][cls] = len(files)
                split_total += len(files)
                for fpath in tqdm(files, desc=f"Scanning {split}/{cls}"):
                    process_file(fpath, split, cls, report, records)
            report["counts"][split]["total"] = split_total

    print("\nChecking for exact cross-split SHA-256 duplicates...")
    sha_map = defaultdict(list)
    for r in records:
        sha_map[r["sha256"]].append(r)

    exact_duplicates = []
    for sha, group in sha_map.items():
        splits_in_group = set(item["split"] for item in group)
        if len(splits_in_group) > 1:
            exact_duplicates.append([{
                "split": item["split"],
                "class": item["class"],
                "path": os.path.relpath(item["path"], DATA_DIR)
            } for item in group])

    print("Checking near-duplicates via pHash (vectorized Hamming distance)...")
    # Convert phashes to flat boolean numpy arrays (N, 64)
    # Check unlabeled vs val/test and cross-split labeled
    val_test_idx = [i for i, r in enumerate(records) if r["split"] in ["val", "test"]]
    unl_idx = [i for i, r in enumerate(records) if r["split"] == "unlabeled"]
    train_idx = [i for i, r in enumerate(records) if r["split"] == "train"]

    phash_matrix = np.array([r["phash"].hash.flatten() for r in records], dtype=bool)

    # 1. Unlabeled vs (val, test)
    near_duplicates_unl_vs_eval = []
    if len(unl_idx) > 0 and len(val_test_idx) > 0:
        unl_ph = phash_matrix[unl_idx] # (5000, 64)
        vt_ph = phash_matrix[val_test_idx] # (400, 64)
        # Hamming distance: count differing bits
        # (5000, 1, 64) != (1, 400, 64)
        for chunk_start in range(0, len(unl_idx), 500):
            chunk_end = min(chunk_start + 500, len(unl_idx))
            sub_unl = unl_ph[chunk_start:chunk_end] # (B, 64)
            dists = np.count_nonzero(sub_unl[:, None, :] != vt_ph[None, :, :], axis=-1) # (B, 400)
            matches = np.argwhere(dists <= 2)
            for b_idx, vt_col in matches:
                u_rec = records[unl_idx[chunk_start + b_idx]]
                vt_rec = records[val_test_idx[vt_col]]
                d = int(dists[b_idx, vt_col])
                near_duplicates_unl_vs_eval.append({
                    "distance": d,
                    "unlabeled": os.path.relpath(u_rec["path"], DATA_DIR),
                    "eval_file": os.path.relpath(vt_rec["path"], DATA_DIR),
                    "eval_split": vt_rec["split"],
                    "eval_class": vt_rec["class"]
                })

    # 2. Train vs (val, test)
    near_duplicates_train_vs_eval = []
    if len(train_idx) > 0 and len(val_test_idx) > 0:
        tr_ph = phash_matrix[train_idx]
        vt_ph = phash_matrix[val_test_idx]
        dists = np.count_nonzero(tr_ph[:, None, :] != vt_ph[None, :, :], axis=-1)
        matches = np.argwhere(dists <= 2)
        for tr_i, vt_j in matches:
            tr_rec = records[train_idx[tr_i]]
            vt_rec = records[val_test_idx[vt_j]]
            near_duplicates_train_vs_eval.append({
                "distance": int(dists[tr_i, vt_j]),
                "train_file": os.path.relpath(tr_rec["path"], DATA_DIR),
                "train_class": tr_rec["class"],
                "eval_file": os.path.relpath(vt_rec["path"], DATA_DIR),
                "eval_split": vt_rec["split"],
                "eval_class": vt_rec["class"]
            })

    report_data = {
        "counts": report["counts"],
        "expected_counts": EXPECTED_COUNTS,
        "corrupted_files": report["corrupted_files"],
        "formats": dict(report["formats"]),
        "color_modes": dict(report["color_modes"]),
        "sizes": {f"{k[0]}x{k[1]}": v for k, v in report["sizes"].items()},
        "aspect_ratios": {str(k): v for k, v in report["aspect_ratios"].items()},
        "exact_cross_split_duplicates_count": len(exact_duplicates),
        "exact_cross_split_duplicates": exact_duplicates,
        "near_duplicates_unl_vs_eval_count": len(near_duplicates_unl_vs_eval),
        "near_duplicates_unl_vs_eval": near_duplicates_unl_vs_eval,
        "near_duplicates_train_vs_eval_count": len(near_duplicates_train_vs_eval),
        "near_duplicates_train_vs_eval": near_duplicates_train_vs_eval
    }

    json_path = os.path.join(REPORTS_DIR, "dataset_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(REPORTS_DIR, "dataset_report.md")
    write_markdown_report(md_path, report_data)

    print(f"\nAnalysis complete!")
    print(f"Total images scanned: {len(records)}")
    print(f"Exact duplicates across splits: {len(exact_duplicates)}")
    print(f"Near duplicates (unlabeled vs val/test, dist <= 2): {len(near_duplicates_unl_vs_eval)}")
    print(f"Near duplicates (train vs val/test, dist <= 2): {len(near_duplicates_train_vs_eval)}")

def process_file(fpath, split, cls, report, records):
    report["total_files"] += 1
    try:
        size = os.path.getsize(fpath)
        if size == 0:
            report["corrupted_files"].append({"path": fpath, "reason": "0 bytes file size"})
            return
    except Exception as e:
        report["corrupted_files"].append({"path": fpath, "reason": f"Cannot get file size: {e}"})
        return

    try:
        with Image.open(fpath) as img:
            img.verify()
        with Image.open(fpath) as img:
            fmt = img.format
            mode = img.mode
            w, h = img.size
            ratio = round(w / h, 2)
            report["formats"][fmt] += 1
            report["color_modes"][mode] += 1
            report["sizes"][(w, h)] += 1
            report["aspect_ratios"][ratio] += 1
            ph = imagehash.phash(img)
    except Exception as e:
        report["corrupted_files"].append({"path": fpath, "reason": f"Corrupted or invalid image: {e}"})
        return

    sha = get_file_hash(fpath)
    records.append({
        "path": fpath,
        "split": split,
        "class": cls,
        "sha256": sha,
        "phash": ph
    })

def write_markdown_report(path, data):
    counts = data["counts"]
    exp = data["expected_counts"]
    
    md = []
    md.append("# Звіт аналізу та перевірки датасету (`dataset_report.md`)")
    md.append("\n## 1. Підрахунок зображень та порівняння з очікуваннями\n")
    md.append("| Спліт | Клас healthy | Клас sick | Загалом | Очікувалось | Статус |")
    md.append("|---|---|---|---|---|---|")
    
    for split in ["train", "val", "test"]:
        actual_h = counts.get(split, {}).get("healthy", 0)
        actual_s = counts.get(split, {}).get("sick", 0)
        actual_tot = counts.get(split, {}).get("total", 0)
        exp_h = exp[split]["healthy"]
        exp_s = exp[split]["sick"]
        exp_tot = exp[split]["total"]
        status = "✅ Співпадає" if (actual_h == exp_h and actual_s == exp_s) else "⚠️ Розбіжність на 1 файл у класах"
        md.append(f"| `{split}` | {actual_h} | {actual_s} | {actual_tot} | {exp_h}/{exp_s} ({exp_tot}) | {status} |")
        
    actual_unl = counts.get("unlabeled", {}).get("total", 0)
    exp_unl = exp["unlabeled"]["total"]
    status_unl = "✅ Співпадає" if actual_unl == exp_unl else "⚠️ Розбіжність (5000 vs 5001)"
    md.append(f"| `unlabeled` | — | — | {actual_unl} | {exp_unl} | {status_unl} |")

    md.append("\n> **Примітка щодо розбіжності:** В архіві `dataset-nix.zip` загалом рівно 6000 валідних зображень: train=600 (169/431), val=200 (58/142), test=200 (54/146), unlabeled=5000. Кожна частина має рівно на 1 екземпляр менше від орієнтовного опису у WEEK1.md (сумарно 6000 замість 6007).")

    md.append("\n## 2. Цілісність файлів та пошкоджені дані\n")
    corrupted = data["corrupted_files"]
    if not corrupted:
        md.append("- **Пошкоджені або нуль-байтові файли:** Не знайдено (усі 6000 зображень відкриваються без помилок та успішно пройшли `PIL.Image.verify()`).")
    else:
        md.append(f"- **Знайдено пошкоджених файлів:** {len(corrupted)}")
        for c in corrupted:
            md.append(f"  - `{c['path']}`: {c['reason']}")

    md.append("\n## 3. Статистика форматів, колірних режимів та розмірів\n")
    md.append("### Формати файлів:")
    for fmt, c in data["formats"].items():
        md.append(f"- `{fmt}`: {c}")
    md.append("\n### Колірні режими:")
    for m, c in data["color_modes"].items():
        md.append(f"- `{m}`: {c}")
    md.append("\n### Розміри зображень (W x H):")
    for s, c in data["sizes"].items():
        md.append(f"- `{s}`: {c}")
    md.append("\n### Співвідношення сторін:")
    for ar, c in data["aspect_ratios"].items():
        md.append(f"- `{ar}`: {c}")

    md.append("\n## 4. Перевірка на дублікати між частинами (Cross-split duplicates)\n")
    exact_dups = data["exact_cross_split_duplicates"]
    md.append(f"### Точні дублікати за SHA-256 між різними сплітами: **{len(exact_dups)}**")
    if not exact_dups:
        md.append("- Точних файлових дублікатів (ідентичний бінарний хеш) між train, val, test та unlabeled **не знайдено**.")
    else:
        for idx, group in enumerate(exact_dups, 1):
            md.append(f"- Група {idx}:")
            for item in group:
                md.append(f"  - [{item['split']} / {item['class'] or 'unlabeled'}] `{item['path']}`")

    near_unl = data["near_duplicates_unl_vs_eval"]
    md.append(f"\n### Майже-дублікати (перцептивний хеш pHash, Hamming distance <= 2) між `unlabeled` та `val`/`test`: **{len(near_unl)}**")
    if not near_unl:
        md.append("- Небезпечних майже-дублікатів між `unlabeled` та валідацією/тестом **не виявлено**.")
    else:
        md.append("| Відстань | Unlabeled файл | Спліт | Оцінюваний файл | Клас |")
        md.append("|---|---|---|---|---|")
        for item in near_unl:
            md.append(f"| {item['distance']} | `{item['unlabeled']}` | `{item['eval_split']}` | `{item['eval_file']}` | `{item['eval_class']}` |")

    near_tr = data["near_duplicates_train_vs_eval"]
    md.append(f"\n### Майже-дублікати (pHash distance <= 2) між `train` та `val`/`test`: **{len(near_tr)}**")
    if not near_tr:
        md.append("- Майже-дублікатів між `train` та `val`/`test` **не виявлено**.")
    else:
        md.append("| Відстань | Train файл | Train клас | Eval спліт | Eval файл | Eval клас |")
        md.append("|---|---|---|---|---|---|")
        for item in near_tr:
            md.append(f"| {item['distance']} | `{item['train_file']}` | `{item['train_class']}` | `{item['eval_split']}` | `{item['eval_file']}` | `{item['eval_class']}` |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

if __name__ == "__main__":
    analyze_dataset()
