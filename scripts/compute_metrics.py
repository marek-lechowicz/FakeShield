"""
Unified metric script for FakeShield evaluation runs.

Inputs:
    - question.jsonl  (image, gt_mask, optional gt_label)
    - DTE-FDM output jsonl (image, outputs text)
    - MFLM output directory (PNG mask per filename, may be missing)

Detection metrics:
    test_acc, precision, recall, mcc  -> binary y_pred from DTE-FDM verdict
    test_auc, average_precision       -> continuous score = MFLM mask coverage

Localization metrics (averaged over tampered images with GT mask):
    *_th_0.5  -> threshold the MFLM mask at 0.5
    *_best    -> per-image oracle threshold sweep
                 (MFLM ships binary masks, so on this pipeline *_best == *_th_0.5;
                  the sweep is kept so the script works if MFLM ever emits soft masks)
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


NEGATION_REGEX = re.compile(
    r"\bnot\b[^.]{0,40}\btampered\b"
    r"|\bappears?\s+(?:to\s+be\s+)?(?:authentic|genuine)\b"
    r"|\bgenuine\s+photograph\b"
    r"|\bno\s+(?:signs?|evidence|indication)s?\s+of\s+tampering\b",
    re.IGNORECASE,
)


def predict_label(text: str) -> int:
    return 0 if NEGATION_REGEX.search(text[:500]) else 1


def pixel_f1_iou(pred_bin, gt_bin):
    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    denom_f1 = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    if denom_iou == 0:
        return 1.0, 1.0
    f1 = (2 * tp) / denom_f1 if denom_f1 > 0 else 0.0
    iou = tp / denom_iou
    return float(f1), float(iou)


def sweep_best(pred_soft, gt_bin):
    """Per-image oracle threshold sweep. Returns (best_f1, best_iou)."""
    uniq = np.unique(pred_soft)
    if uniq.size <= 2:
        return pixel_f1_iou(pred_soft > 0.5, gt_bin)
    best_f1 = 0.0
    best_iou = 0.0
    for t in np.linspace(0.05, 0.95, 19):
        f1, iou = pixel_f1_iou(pred_soft > t, gt_bin)
        if f1 > best_f1:
            best_f1 = f1
        if iou > best_iou:
            best_iou = iou
    return best_f1, best_iou


def resolve_gt_label(item, gt_mask_path):
    if "gt_label" in item:
        return int(item["gt_label"])
    # Infer from mask presence + content
    if not gt_mask_path:
        return 0
    p = Path(gt_mask_path)
    if not p.exists():
        return 0
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0
    return 1 if (m > 127).any() else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--question-jsonl", required=True)
    p.add_argument("--dte-output", required=True)
    p.add_argument("--mflm-dir", required=True)
    p.add_argument(
        "--invert-gt-mask", action="store_true",
        help="Invert GT mask before scoring. Required for DSO-1 (black=tampered).",
    )
    p.add_argument(
        "--output", default=None,
        help="Optional path to also write the metrics block.",
    )
    args = p.parse_args()

    questions = [json.loads(l) for l in open(args.question_jsonl)]

    dte = {}
    with open(args.dte_output) as f:
        for line in f:
            o = json.loads(line)
            dte[o["image"]] = o["outputs"]

    mflm_dir = Path(args.mflm_dir)

    y_true, y_pred, y_score = [], [], []
    f1_05, iou_05 = [], []
    f1_best, iou_best = [], []
    missing_dte = 0
    missing_mflm = 0
    n_tampered_with_gt = 0

    for q in tqdm(questions, desc="Scoring"):
        img = q["image"]
        gt_mask_path = q.get("gt_mask")
        gt_label = resolve_gt_label(q, gt_mask_path)

        text = dte.get(img)
        if text is None:
            missing_dte += 1
            text = ""
        pred_label = predict_label(text)

        # Load MFLM mask if it exists. Used both for the continuous detection
        # score and for localization metrics.
        mask_path = mflm_dir / Path(img).name
        pred_soft = None
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                pred_soft = m.astype(np.float32) / 255.0
        elif pred_label == 1:
            missing_mflm += 1

        # Detection score: mask coverage at 0.5. Zero when MFLM did not run
        # (DTE-FDM voted "authentic" -> MFLM is skipped by test.py).
        score = float((pred_soft > 0.5).mean()) if pred_soft is not None else 0.0

        y_true.append(gt_label)
        y_pred.append(pred_label)
        y_score.append(score)

        # Localization: only on tampered images with a GT mask on disk.
        if gt_label != 1 or not gt_mask_path:
            continue
        gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt_bin = gt > 127
        if args.invert_gt_mask:
            gt_bin = ~gt_bin
        n_tampered_with_gt += 1

        if pred_soft is None:
            pred_resized = np.zeros_like(gt_bin, dtype=np.float32)
        elif pred_soft.shape != gt_bin.shape:
            pred_resized = cv2.resize(
                pred_soft,
                (gt_bin.shape[1], gt_bin.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            pred_resized = pred_soft

        f1, iou = pixel_f1_iou(pred_resized > 0.5, gt_bin)
        f1_05.append(f1)
        iou_05.append(iou)
        bf1, biou = sweep_best(pred_resized, gt_bin)
        f1_best.append(bf1)
        iou_best.append(biou)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    metrics = {}
    metrics["test_acc"] = float(accuracy_score(y_true, y_pred))
    metrics["test_auc"] = (
        float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")
    )
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["average_precision"] = (
        float(average_precision_score(y_true, y_score))
        if len(np.unique(y_true)) >= 2 else float("nan")
    )
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    metrics["localization_f1_best"] = _mean(f1_best)
    metrics["localization_f1_th_0.5"] = _mean(f1_05)
    metrics["localization_iou_best"] = _mean(iou_best)
    metrics["localization_iou_th_0.5"] = _mean(iou_05)

    n = len(y_true)
    n_t = int(y_true.sum())
    print()
    print(f"Images scored : {n}  (tampered={n_t}, clean={n - n_t})")
    print(f"DTE-FDM rows missing for {missing_dte} images")
    print(f"MFLM masks missing for {missing_mflm} images that DTE-FDM flagged tampered")
    print(f"Localization averaged over {n_tampered_with_gt} tampered images with GT mask")
    print()

    width = max(len(k) for k in metrics)
    lines = [f"  {k:<{width}}: {v:.4f}" for k, v in metrics.items()]
    print("\n".join(lines))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write(f"Images scored : {n}  (tampered={n_t}, clean={n - n_t})\n")
            f.write(f"Localization averaged over {n_tampered_with_gt} tampered images with GT mask\n\n")
            f.write("\n".join(lines) + "\n")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
