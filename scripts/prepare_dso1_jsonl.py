import argparse
import json
from pathlib import Path

QUESTION = (
    "Was this photo taken directly from the camera without any processing? "
    "Has it been tampered with by any artificial photo modification techniques such as ps? "
    "Please zoom in on any details in the image, paying special attention to the edges of "
    "the objects, capturing some unnatural edges and perspective relationships, some "
    "incorrect semantics, unnatural lighting and darkness etc."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dso-root", type=str,
        default="dataset/tifs-database/tifs-database/DSO-1",
    )
    parser.add_argument(
        "--mask-root", type=str,
        default="dataset/tifs-database/tifs-database/DSO-1-Fake-Images-Masks",
    )
    parser.add_argument("--output-jsonl", type=str, required=True)
    args = parser.parse_args()

    dso_root = Path(args.dso_root).resolve()
    mask_root = Path(args.mask_root).resolve()

    items = []
    for img_path in sorted(dso_root.glob("*.png")):
        name = img_path.name
        if name.startswith("splicing-"):
            gt_mask = str(mask_root / name)
            gt_label = 1
        elif name.startswith("normal-"):
            gt_mask = None
            gt_label = 0
        else:
            continue
        items.append({
            "image": str(img_path),
            "text": QUESTION,
            "gt_mask": gt_mask,
            "gt_label": gt_label,
        })

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    n_tamper = sum(1 for x in items if x["gt_label"] == 1)
    n_normal = sum(1 for x in items if x["gt_label"] == 0)
    print(f"Wrote {len(items)} entries to {args.output_jsonl} "
          f"(tampered={n_tamper}, normal={n_normal})")


if __name__ == "__main__":
    main()
