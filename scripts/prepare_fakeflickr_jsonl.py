import json
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-path", type=str, required=True, help="Path to fakeflickr_{model}_{split}_list.txt")
    parser.add_argument("--ff-root", type=str, default="/home/marek/FakeFlickr/data", help="FakeFlickr data root")
    parser.add_argument("--output-jsonl", type=str, required=True, help="Output question.jsonl path")
    args = parser.parse_args()

    ff_root = Path(args.ff_root)
    list_path = Path(args.list_path)
    
    questions = []
    with open(list_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            img_rel, mask_rel = line.strip().split(",")
            img_abs = str(ff_root / img_rel)
            
            # Question for DTE-FDM (Stage 1)
            qs = "Was this photo taken directly from the camera without any processing? Has it been tampered with by any artificial photo modification techniques such as ps? Please zoom in on any details in the image, paying special attention to the edges of the objects, capturing some unnatural edges and perspective relationships, some incorrect semantics, unnatural lighting and darkness etc."
            
            questions.append({
                "image": img_abs,
                "text": qs,
                "gt_mask": str(ff_root / mask_rel)
            })

    with open(args.output_jsonl, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
            
    print(f"Created {len(questions)} questions in {args.output_jsonl}")

if __name__ == "__main__":
    main()
