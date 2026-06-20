import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn import metrics


def min_max_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    min_value = values.min()
    max_value = values.max()
    if max_value <= min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def auc(labels, scores):
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return float("nan")
    return metrics.roc_auc_score(labels, min_max_normalize(scores))


def defect_type_from_filename(filename):
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    for token in ["positive", "good", "template"]:
        if token in stem:
            return "good"
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 else "unknown"


def summarize_npz(eval_dir, top_k):
    records = []
    for path in glob.glob(os.path.join(eval_dir, "*.npz")):
        data = np.load(path, allow_pickle=True)
        pred = np.asarray(data["pred"]).reshape(-1)
        mask = np.asarray(data["mask"]).reshape(-1)
        cls_prob = np.asarray(data["cls_prob"]).reshape(-1) if "cls_prob" in data else None
        cls_label = int(np.asarray(data["cls_label"]).reshape(-1)[0])
        cls_pred = int(np.asarray(data["cls_pred"]).reshape(-1)[0])
        records.append(
            {
                "clsname": str(data["clsname"]),
                "filename": str(data["filename"]),
                "label": int(np.asarray(data["label"]).reshape(-1)[0]),
                "score": float(pred.max()),
                "pixel_auc": auc(mask, pred),
                "cls_label": cls_label,
                "cls_pred": cls_pred,
                "cls_conf": float(cls_prob[cls_label]) if cls_prob is not None and cls_label < len(cls_prob) else np.nan,
                "defect_type": defect_type_from_filename(data["filename"]),
            }
        )

    by_class = defaultdict(list)
    for record in records:
        by_class[record["clsname"]].append(record)

    rows = []
    for clsname, items in by_class.items():
        labels = [item["label"] for item in items]
        scores = [item["score"] for item in items]
        good_scores = [item["score"] for item in items if item["label"] == 0]
        bad_scores = [item["score"] for item in items if item["label"] == 1]
        cls_acc = np.mean([item["cls_label"] == item["cls_pred"] for item in items])
        cls_conf = np.nanmean([item["cls_conf"] for item in items])
        pixel_values = [item["pixel_auc"] for item in items if not np.isnan(item["pixel_auc"])]
        confusion = Counter(item["cls_pred"] for item in items if item["cls_label"] != item["cls_pred"])
        defect_scores = defaultdict(list)
        for item in items:
            defect_scores[item["defect_type"]].append(item["score"])
        rows.append(
            {
                "clsname": clsname,
                "obj_auc": auc(labels, scores),
                "pixel_auc": float(np.mean(pixel_values)) if pixel_values else float("nan"),
                "cls_acc": float(cls_acc),
                "cls_conf": float(cls_conf),
                "score_gap": float(np.mean(bad_scores) - np.mean(good_scores)) if good_scores and bad_scores else float("nan"),
                "confusion": confusion.most_common(3),
                "defect_scores": {
                    key: float(np.mean(value)) for key, value in sorted(defect_scores.items())
                },
            }
        )

    rows.sort(key=lambda row: (np.nan_to_num(row["obj_auc"], nan=1.0), np.nan_to_num(row["pixel_auc"], nan=1.0)))
    print("| clsname | obj_auc | pixel_auc | cls_acc | cls_conf | score_gap | top_confusions |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for row in rows[:top_k]:
        print(
            f"| {row['clsname']} | {row['obj_auc']:.4f} | {row['pixel_auc']:.4f} | "
            f"{row['cls_acc']:.4f} | {row['cls_conf']:.4f} | {row['score_gap']:.4f} | {row['confusion']} |"
        )
        print(f"  defect_score_means: {row['defect_scores']}")


def summarize_markdown(markdown_path, top_k):
    text = open(markdown_path, "r", encoding="utf-8").read()
    header_line = next(
        line for line in text.splitlines() if line.startswith("|") and "clsname" in line
    )
    headers = [part.strip() for part in header_line.strip("|").split("|")]
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or "---" in line or "clsname" in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts[0] == "mean" or len(parts) != 4:
            continue
        values = dict(zip(headers, parts))
        rows.append(values)
    rows.sort(key=lambda row: float(row.get("obj-AUROC", "1")))
    print("| clsname | cls-ACC | obj-AUROC | pixel-AUROC |")
    print("|---|---:|---:|---:|")
    for row in rows[:top_k]:
        print(
            f"| {row['clsname']} | {float(row.get('cls-ACC', 0)):.4f} | "
            f"{float(row.get('obj-AUROC', 0)):.4f} | {float(row.get('pixel-AUROC', 0)):.4f} |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", help="Directory containing dumped .npz inference outputs.")
    parser.add_argument("--markdown", help="Fallback result markdown table.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.eval_dir:
        summarize_npz(args.eval_dir, args.top_k)
    elif args.markdown:
        summarize_markdown(args.markdown, args.top_k)
    else:
        raise SystemExit("Provide --eval-dir or --markdown")


if __name__ == "__main__":
    main()
