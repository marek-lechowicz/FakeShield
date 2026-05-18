import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-jsonl", type=str, required=True, help="Input question.jsonl with gt_mask field")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory with generated masks from MFLM")
    args = parser.parse_args()

    results = []
    with open(args.question_jsonl, "r") as f:
        for line in f:
            results.append(json.loads(line))

    ious = []
    
    print("Calculating IoU...")
    for item in tqdm(results):
        img_name = Path(item["image"]).name
        gt_mask_path = item["gt_mask"]
        pred_mask_path = Path(args.output_dir) / img_name
        
        if not pred_mask_path.exists():
            # If model didn't generate mask, assume empty mask
            pred_mask = None
        else:
            pred_mask = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE)
            if pred_mask is None:
                # Fallback if image failed to load
                pred_mask = None
            else:
                pred_mask = pred_mask > 127

        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Warning: GT mask not found at {gt_mask_path}")
            continue
        gt_mask = gt_mask > 127
        
        if pred_mask is None:
            pred_mask = np.zeros_like(gt_mask)
            
        # Ensure same size
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_mask = pred_mask > 0

        iou = calculate_iou(pred_mask, gt_mask)
        ious.append(iou)

    if ious:
        mean_iou = np.mean(ious)
        print(f"\nResults for {len(ious)} images:")
        print(f"Mean IoU: {mean_iou:.4f}")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
