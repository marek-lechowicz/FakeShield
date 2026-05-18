# Unable to reproduce DSO-1 detection numbers from Table 1 (ACC=0.97); released `model_vqa.py` always prepends a "suspected tampered" DTG prefix

Hi, thanks for releasing the code and weights — really appreciate the work.

I'm trying to reproduce the DSO-1 numbers from the paper as a baseline for another project, and I'm seeing a gap that I can't explain. I'd like to confirm whether the released eval pipeline is the same one used to produce the paper's numbers, or whether there's a separate configuration I'm missing.

## Setup

- **Hardware:** RTX 3090 (24 GB), Linux, CUDA 12.0 system / torch 1.13.0+cu116.
- **Code:** current `main`, with only the minimal patches needed to start (transformers API kwargs, CLIPVisionModel `device_map` fallback, `mmcv` version range relaxation, and the input-format fix in `MFLM/test.py` that's already discussed in other issues). No changes to model logic or prompting.
- **Weights:** `zhipeixu/fakeshield-v1-22b` from HF, plus `sam_vit_h_4b8939.pth` for MFLM.
- **Pipeline:** `DTE-FDM/llava/eval/model_vqa.py` → `MFLM/test.py`, exactly as in `scripts/test.sh`.
- **DTE-FDM precision:** `--load-8bit` (bitsandbytes int8). I verified this is *not* the cause of the regression — see ablation below.
- **MFLM precision:** native bf16 (no quantization), fits on 3090 24 GB.
- **Dataset:** DSO-1 (100 spliced + 100 normal) from the original Carvalho et al. 2013 release, with `DSO-1-Fake-Images-Masks` for ground truth (note: GT masks are inverted vs MFLM output; black = tampered, white = clean — handled in my eval script).
- **Detection criterion:** I parse the DTE-FDM text output, mirroring the convention in `MFLM/test.py:261`: an image is predicted "authentic" iff the model's verdict sentence contains negation of "tampered" (regex covers `not been tampered`, `appears authentic`, `appears to be a genuine photograph`, etc., not just the literal `"has not been tampered with"`).

## Observed vs paper numbers on DSO-1

|  | Paper (Tables 1 & 4) | My run (8-bit) |
|---|---|---|
| Detection ACC (Table 1) | **0.97** | **0.66** |
| Detection F1 (Table 1) | **0.98** | ~0.72 (Prec=0.61, Rec=0.87) |
| Localization IoU (Table 4) | **0.48** | **0.37** |
| Localization F1 (Table 4) | **0.52** | not computed |

Localization is in the ballpark (~22% relative gap), which I can imagine attributing to small implementation differences. **Detection is a 31-percentage-point absolute gap on a 200-image benchmark**, which I can't.

The detection error is dominated by **false positives on the 100 authentic ("normal-*") DSO-1 images**: my run gets recall = 0.87 (correctly flags most spliced images) but specificity = 0.42 (about 58/100 authentic images are classified as tampered). The model often confabulates regions in clean group photos with phrasings like:

> *"The picture has been tampered with, specifically in the central region where the faces of two individuals should be."*

## Ablation: ruled out 8-bit quantization

Because 8-bit was an obvious suspect, I ran a controlled ablation on 10 DSO-1 images (5 normals where 8-bit gave a false positive + 5 spliced true positives) comparing:

- DTE-FDM in **8-bit** (bitsandbytes int8), greedy decoding
- DTE-FDM in **fp16 + HF accelerate CPU offload** (max_memory={0: "14GiB", "cpu": "60GiB"}), greedy decoding, same prompt and seed-equivalent settings

Result: **fp16 produced the same verdict as 8-bit on all 10 images**, including all 5 false positives.

```
image            GT        8-bit       fp16       
normal-03.png    CLEAN     TAMPERED    TAMPERED    ✗ / ✗
normal-13.png    CLEAN     TAMPERED    TAMPERED    ✗ / ✗
normal-15.png    CLEAN     TAMPERED    TAMPERED    ✗ / ✗
normal-16.png    CLEAN     TAMPERED    TAMPERED    ✗ / ✗
normal-17.png    CLEAN     TAMPERED    TAMPERED    ✗ / ✗
splicing-01.png  TAMPERED  TAMPERED    TAMPERED    ✓ / ✓
splicing-02.png  TAMPERED  TAMPERED    TAMPERED    ✓ / ✓
splicing-03.png  TAMPERED  TAMPERED    TAMPERED    ✓ / ✓
splicing-04.png  TAMPERED  TAMPERED    TAMPERED    ✓ / ✓
splicing-05.png  TAMPERED  TAMPERED    TAMPERED    ✓ / ✓
```

So quantization is not the bottleneck — fp16 reproduces the same problematic behavior.

## Hypothesis: DTG prefix biases the LLM toward "tampered"

In `DTE-FDM/llava/eval/model_vqa.py:102-109`, every image — *regardless of whether it's actually tampered* — has the prompt prefixed with a deterministic "suspected tampered" framing derived from the 3-class Domain Tag Generator (ResNet50):

```python
label = DTG.predict(image_file)
if label == 0:
    qs = "This is a picture that is suspected to have been tampered with by AIGC inpainting. " + qs
elif label == 1:
    qs = "This is a picture that is suspected to have been tampered with by DeepFake. " + qs
elif label == 2:
    qs = "This is a picture that is suspected to have been tampered with by Photoshop. " + qs
```

Since the DTG has no "authentic" output class, **every clean image still receives a `"This is a picture that is suspected to have been tampered with by …"` prefix** before the actual question. On DSO-1's authentic group photos, the LLM frequently agrees with this prior and confabulates a tampering region.

I would expect a model trained jointly with this prefix to be calibrated to push back when the image actually doesn't show tampering — and the paper's ACC=0.97 on DSO-1 (mostly authentic photos) suggests it does push back in your evaluation. So my best guess is that the actual eval that produced Table 1 either:

1. Did **not** apply the DTG prefix to clean images (perhaps DTG had an "authentic" class internally that's not in the released checkpoint), or
2. Used a different detection criterion (e.g., a mask-coverage threshold from MFLM rather than DTE-FDM text parsing — Section 4.1 mentions "a default threshold of 0.5"), or
3. The DTG prefix was added after the eval that produced the paper numbers.

## Questions

1. **Was the DTG prefix in `model_vqa.py:104-109` used during the Table 1 detection evaluation?** If so, how does the LLM resist its own "suspected tampered" prefix on clean images well enough to reach ACC=0.97?
2. **What exactly is the detection criterion?** Section 4.1 says *"For both detection and localization, a default threshold of 0.5 is applied unless otherwise specified."* — what is the score being thresholded for image-level detection?
3. **Is there an "authentic" class for DTG** that's missing from the released checkpoint, or a code path that disables the DTG prefix when the model is uncertain?
4. **Are the released weights/config the same** ones that produced the paper's Table 1 numbers, or were the paper numbers obtained with a different snapshot?

If there's a known reproducibility configuration for DSO-1 detection, I'd be happy to apply it and update my comparison. Otherwise I'll be reporting the observed 0.66 / 0.37 numbers in my paper as a baseline, with a transparent description of the setup.

Thanks again for the work and for any clarification.

---

Happy to share the exact eval scripts (subset selection, metric calculation including the GT-mask inversion for DSO-1, and the 8-bit-vs-fp16 ablation harness) if that's useful.
