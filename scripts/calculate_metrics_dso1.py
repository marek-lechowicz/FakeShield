import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# DTE-FDM phrases its "authentic" verdict in several ways. MFLM/test.py only
# checks for the literal "has not been tampered with", but the model also says
# "appears not to have been tampered", "appears to be authentic", "appears to
# be a genuine photograph", etc. Catching all forms is critical for detection
# metrics: with the literal-only match, ~30% of clean images on DSO-1 get
# misclassified as tampered purely from parsing.
NEGATION_REGEX = re.compile(
    r"\bnot\b[^.]{0,40}\btampered\b"
    r"|\bappears?\s+(?:to\s+be\s+)?(?:authentic|genuine)\b"
    r"|\bgenuine\s+photograph\b"
    r"|\bno\s+(?:signs?|evidence|indication)s?\s+of\s+tampering\b",
    re.IGNORECASE,
)


def predict_label(text: str) -> int:
    """Return 1 if DTE-FDM verdict says tampered, else 0."""
    head = text[:500]
    return 0 if NEGATION_REGEX.search(head) else 1


def load_binary_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m > 127


def iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-jsonl", required=True)
    parser.add_argument("--dte-output", required=True)
    parser.add_argument("--mflm-dir", required=True)
    parser.add_argument(
        "--invert-gt-mask", action="store_true",
        help="Invert GT mask before IoU. DSO-1 ships masks with black=tampered, "
             "white=clean, opposite to MFLM output and most other datasets.",
    )
    args = parser.parse_args()

    questions = [json.loads(l) for l in open(args.question_jsonl)]

    dte_outputs = {}
    with open(args.dte_output) as f:
        for line in f:
            o = json.loads(line)
            dte_outputs[o["image"]] = o["outputs"]

    mflm_dir = Path(args.mflm_dir)

    y_true, y_pred, y_score = [], [], []
    ious = []
    missing_dte, missing_mask = 0, 0

    for q in tqdm(questions, desc="Scoring"):
        img = q["image"]
        gt_label = q["gt_label"]

        text = dte_outputs.get(img)
        if text is None:
            missing_dte += 1
            text = ""
        pred_label = predict_label(text)

        # AUC score: fraction of pixels MFLM marked as tampered.
        # 0 when DTE-FDM voted "authentic" (MFLM is then skipped).
        pred_mask = None
        score = 0.0
        if pred_label == 1:
            mask_path = mflm_dir / Path(img).name
            if mask_path.exists():
                pred_mask = load_binary_mask(mask_path)
                if pred_mask is not None:
                    score = float(pred_mask.mean())
            else:
                missing_mask += 1

        y_true.append(gt_label)
        y_pred.append(pred_label)
        y_score.append(score)

        # Pixel-level IoU only on tampered images (where GT mask exists).
        if q.get("gt_mask"):
            gt_mask = load_binary_mask(q["gt_mask"])
            if gt_mask is None:
                continue
            if args.invert_gt_mask:
                gt_mask = ~gt_mask
            if pred_mask is None:
                pred_resized = np.zeros_like(gt_mask)
            elif pred_mask.shape != gt_mask.shape:
                resized = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                pred_resized = resized > 0
            else:
                pred_resized = pred_mask
            ious.append(iou(pred_resized, gt_mask))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    n = len(y_true)
    n_t = int(y_true.sum())
    n_n = n - n_t
    print()
    print(f"Images scored : {n}  (tampered={n_t}, normal={n_n})")
    print(f"DTE-FDM rows missing for {missing_dte} images")
    print(f"MFLM masks missing for {missing_mask} images that DTE-FDM flagged tampered")
    print()
    if ious:
        print(f"Mean IoU (over {len(ious)} tampered): {np.mean(ious):.4f}")
        print(f"Median IoU                          : {np.median(ious):.4f}")
    else:
        print("No IoU samples")
    print()
    print("Image-level detection (tampered=1, normal=0):")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  MCC      : {matthews_corrcoef(y_true, y_pred):.4f}")
    if len(np.unique(y_true)) >= 2:
        print(f"  AUC      : {roc_auc_score(y_true, y_score):.4f}  "
              f"(score = MFLM mask coverage)")
    else:
        print("  AUC      : N/A (only one class in GT)")


if __name__ == "__main__":
    main()
