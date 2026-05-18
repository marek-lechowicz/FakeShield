"""Side-by-side comparison of 8-bit baseline vs fp16+offload on the
ablation subset. Prints a table of GT label / 8-bit verdict / fp16 verdict
plus a short summary."""

import argparse
import json
import re

NEGATION_REGEX = re.compile(
    r"\bnot\b[^.]{0,40}\btampered\b"
    r"|\bappears?\s+(?:to\s+be\s+)?(?:authentic|genuine)\b"
    r"|\bgenuine\s+photograph\b"
    r"|\bno\s+(?:signs?|evidence|indication)s?\s+of\s+tampering\b",
    re.IGNORECASE,
)


def verdict(text: str) -> str:
    if not text:
        return "MISSING"
    return "CLEAN" if NEGATION_REGEX.search(text[:500]) else "TAMPERED"


def first_sentence(text: str, n: int = 110) -> str:
    text = re.sub(r"^\s*1\.\s*[^:]+:\s*", "", text)
    s = re.split(r"\.\s", text, maxsplit=1)[0]
    return s.strip()[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-jsonl", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--fp16-output", required=True)
    args = parser.parse_args()

    subset = [json.loads(l) for l in open(args.subset_jsonl)]
    bit8 = {json.loads(l)["image"]: json.loads(l)["outputs"]
            for l in open(args.baseline_output)}
    fp16 = {json.loads(l)["image"]: json.loads(l)["outputs"]
            for l in open(args.fp16_output)}

    rows = []
    for q in subset:
        img = q["image"]
        gt = "TAMPERED" if q["gt_label"] == 1 else "CLEAN"
        v8 = verdict(bit8.get(img, ""))
        vf = verdict(fp16.get(img, ""))
        rows.append((q["image"].rsplit("/", 1)[-1], gt, v8, vf,
                     first_sentence(bit8.get(img, ""), 50),
                     first_sentence(fp16.get(img, ""), 50)))

    name_w = max(len(r[0]) for r in rows)
    print(f"{'image':<{name_w}}  {'GT':<8}  {'8-bit':<10}  {'fp16':<10}  match")
    print("-" * (name_w + 50))
    eight_correct = fp16_correct = both_correct = 0
    for name, gt, v8, vf, _, _ in rows:
        is8 = v8 == gt
        isf = vf == gt
        eight_correct += is8
        fp16_correct += isf
        both_correct += is8 and isf
        mark = ("✓" if is8 else "✗") + " / " + ("✓" if isf else "✗")
        print(f"{name:<{name_w}}  {gt:<8}  {v8:<10}  {vf:<10}  {mark}")

    n = len(rows)
    print()
    print(f"Correct verdicts:  8-bit = {eight_correct}/{n}, fp16 = {fp16_correct}/{n}")
    if fp16_correct > eight_correct:
        print(f"=> fp16 fixes {fp16_correct - eight_correct} cases that 8-bit got wrong "
              f"(quantization is at least partly to blame)")
    elif fp16_correct == eight_correct:
        print("=> fp16 produces same verdicts; gap to paper is NOT primarily quantization")
    else:
        print("=> fp16 is worse on this subset (unexpected; check setup)")


if __name__ == "__main__":
    main()
