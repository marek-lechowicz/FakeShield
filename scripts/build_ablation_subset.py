"""Pick 5 normals where the 8-bit run produced false positives (clean image
classified as tampered) plus 5 spliced where the 8-bit run got it right
(true positives). These 10 images are the most informative for testing
whether moving to fp16 fixes detection."""

import argparse
import json
import re
import sys

# Mirror the negation regex from calculate_metrics_dso1.py.
NEGATION_REGEX = re.compile(
    r"\bnot\b[^.]{0,40}\btampered\b"
    r"|\bappears?\s+(?:to\s+be\s+)?(?:authentic|genuine)\b"
    r"|\bgenuine\s+photograph\b"
    r"|\bno\s+(?:signs?|evidence|indication)s?\s+of\s+tampering\b",
    re.IGNORECASE,
)


def predicted_tampered(text: str) -> bool:
    return NEGATION_REGEX.search(text[:500]) is None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-jsonl", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--n-per-class", type=int, default=5)
    args = parser.parse_args()

    questions = [json.loads(l) for l in open(args.question_jsonl)]
    baseline = {}
    with open(args.baseline_output) as f:
        for line in f:
            o = json.loads(line)
            baseline[o["image"]] = o["outputs"]

    normal_fps, spliced_tps = [], []
    for q in questions:
        text = baseline.get(q["image"], "")
        pred_t = predicted_tampered(text)
        if q["gt_label"] == 0 and pred_t:
            normal_fps.append(q)
        elif q["gt_label"] == 1 and pred_t:
            spliced_tps.append(q)

    n = args.n_per_class
    if len(normal_fps) < n or len(spliced_tps) < n:
        print(f"WARN: only {len(normal_fps)} normal FPs / {len(spliced_tps)} "
              f"spliced TPs available", file=sys.stderr)

    subset = normal_fps[:n] + spliced_tps[:n]
    with open(args.output_jsonl, "w") as f:
        for q in subset:
            f.write(json.dumps(q) + "\n")

    print(f"Wrote {len(subset)} entries to {args.output_jsonl}")
    print(f"  - {min(n, len(normal_fps))} normal-FPs (8-bit said tampered, GT=clean)")
    print(f"  - {min(n, len(spliced_tps))} spliced-TPs (8-bit said tampered, GT=tampered)")


if __name__ == "__main__":
    main()
