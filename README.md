# CoRe-Prompt: Certifiably Robust Prompts for Medical Vision–Language Models

CoRe-Prompt learns a small text prompt (8 context vectors, 4,096 parameters) for a frozen medical vision–language model (PLIP) so that its **randomized-smoothing certificate** is as large as possible. It adds two terms to the noisy cross-entropy used by PromptSmooth (MICCAI 2024):

- a **consistency** term across Gaussian noise draws (Jeong & Shin, NeurIPS 2020);
- a **certified-radius hinge** (MACER, ICLR 2020).

The prompt is on the text side, so the radius term needs no back-propagation through the image encoder.

## What this repository contains

Everything needed to reproduce every table and figure of the paper:

| Component | Location |
|---|---|
| Complete implementation | `coreprompt/` (models, losses, training, test-time adaptation, certification, attacks, defences, statistics, report) |
| Environment configuration | `requirements.txt`, `environment.yml` (pinned versions, see below) |
| Dataset preparation scripts | `coreprompt/data.py` and the `datastats` stage |
| Trained prompts | `checkpoints/<backbone>_<dataset>/<method>_sigma<σ>.pt` |
| Certification code | `coreprompt/smoothing.py` (CERTIFY, Clopper–Pearson bound, certified accuracy, ACR) |
| Adaptive-attack evaluation scripts | `coreprompt/attacks.py`, `coreprompt/defenses.py` and the `attacks` stage |
| Experimental configurations | `coreprompt/config.py` (presets and per-stage budgets), `configs/*.yaml` (exact configuration of each paper run) |
| Random seeds | `seed` in every configuration (paper: seed 0 for the few-shot draw, the test subset, noise and attacks) |
| Raw results | `results/` (see below) |
| Unit tests | `tests/` |
| Paper sources | `paper/main.tex`, `paper/sn-bibliography.bib` |

### Raw results (`results/`)

| File | Content |
|---|---|
| `certify_<setting>.jsonl` | per-image certificates: method, σ, predicted class, label, p_A lower bound, radius, abstention |
| `fulltest_<setting>.npz` | per-image clean and noisy-vote predictions on the larger test subset |
| `attacks_l2.json`, `attacks_linf.json` | per-image predictions for every attack, ε and defence |
| `ablation.jsonl`, `sensitivity.jsonl` | certificates for each loss variant and hyper-parameter setting |
| `trainlogs/*.json` | per-epoch training losses |
| `efficiency.json`, `timings.json` | training time, memory and certification time |
| `report/` | the tables (CSV and LaTeX) and figures (PDF, vector) generated from the files above |

## Environment

- Python 3.13, PyTorch 2.14.1 (CUDA 13.0), Transformers 4.46.3, MedMNIST 3.0.2
- One NVIDIA T4 GPU (16 GB); 16-bit autocast for image encoding
- Models: `vinid/plip` and `openai/clip-vit-base-patch32` from the Hugging Face Hub, encoders frozen

```bash
conda env create -f environment.yml    # or: pip install -r requirements.txt
python -m pytest tests -q
```

## Data preparation

PathMNIST (KatherColon) and BloodMNIST are read from MedMNIST+ at 224 × 224 px. `coreprompt/data.py` reads only the images that are needed (few-shot training images from the validation split and test images) with HTTP range requests and caches them locally. Few-shot images (16 per class) and the certified test subset (100 images) are drawn with a seeded permutation, so the same seed gives the same images. Gaussian noise is added in [0, 1] pixel space before CLIP normalisation, without clipping; adversarial images are clipped to [0, 1].

```bash
python -m coreprompt.run datastats --config configs/paper.yaml --out_dir outputs
```

## Reproducing the paper

Run all stages with the paper configuration:

```bash
for s in sanity datastats certify fulltest attacks ablation gen_clip gen_blood efficiency report; do
  python -m coreprompt.run $s --config configs/paper.yaml --out_dir outputs
done
```

Each stage saves its results as it goes and resumes after an interruption. To regenerate only the tables and figures from the shipped raw results, run `python -m coreprompt.run report --results_dir results --out_dir outputs`.

| Paper item | Stage | Output in `report/` |
|---|---|---|
| Table 2a–b (PLIP / KatherColon, published values) | `certify` | `table_main.tex`, `bootstrap.csv`, `mcnemar.csv` |
| Table 2c (CLIP / KatherColon) | `gen_clip` | `table_clip.tex` |
| Table 2d (PLIP / BloodMNIST) | `gen_blood` | `table_blood.tex` |
| Figure 2 (certified-accuracy curves) | `certify` | `fig_curves.pdf` |
| Figure 3a (SmoothAdv, ℓ2) | `attacks` | `fig_smoothadv.pdf` |
| Figure 3b–c (ablation, sensitivity) | `ablation` | `fig_ablation.pdf`, `fig_sensitivity.pdf` |
| Figure 3d (transformation defences, ℓ∞) | `attacks` | `fig_transforms.pdf` |
| Clean and noisy-vote accuracy | `fulltest` | `fulltest.csv` |
| Cost | `efficiency` | `efficiency.csv` |

### Paper budget and extended protocols

| Experiment | Setting in the paper |
|---|---|
| Certification | 100 test images, σ ∈ {0.25, 0.5, 1.0}, n₀ = 32, n = 500, α = 0.001 |
| Training | 16 shots per class, 20 epochs, SGD (lr 0.002, momentum 0.9, cosine), batch 16, m = 4, λ = 10, η = 0.5, μ = 1, γ = 4, β = 1, one prompt per σ, seed 0 |
| Clean and noisy-vote accuracy | 2,000 PathMNIST / 1,000 BloodMNIST images, 8 noisy copies |
| Adaptive attacks | 50 images, 10-step PGD, 4 noise draws (EOT/SmoothAdv), PREDICT with n = 50 |
| Ablation and sensitivity | σ = 0.5, 100 images |
| Other backbone and dataset | σ ∈ {0.25, 0.5}, 100 images, n = 500 |

This budget was set by the session and usage limits of the GPU used. PromptSmooth's own protocol uses 500 images, n = 10,000 and σ = 0.1 as well; with n = 500 the largest certifiable radius is about 2.21σ (3.20σ at n = 10,000). Every setting can be changed with `--set`, for example:

```bash
# PromptSmooth-like protocol
python -m coreprompt.run certify --config configs/paper.yaml \
  --set n=10000 n_test=500 "sigmas=[0.1,0.25,0.5,1.0]"
# repeated seeds and a shots sweep
for seed in 0 1 2 3 4; do for k in 1 2 4 8 16 32; do
  python -m coreprompt.run certify --config configs/paper.yaml --set seed=$seed shots=$k
done; done
```

## PromptSmooth reimplementation

PromptSmooth was **reimplemented from its paper**; the official code and checkpoints were not used. The reimplementation follows its prompt learner, objective, initialisation, optimiser and test-time adaptation, but differs in four ways: one prompt per σ instead of one noise-agnostic prompt, 20 instead of 50 epochs, 64 instead of 100 adaptation views, and an assumed adaptation learning rate. Its certified accuracy differs from the published values by 7–8 points on average, partly because of the different protocol, so it should be treated as approximate. CoRe-Prompt and the reimplemented baselines use the same images, noise, shots and per-σ training.

## Results

PLIP / KatherColon unless stated otherwise. CA = certified accuracy (%) at ℓ2 radius r, best over σ; ACR = average certified radius; † = test-time adaptation (the certificate holds for the adapted prompt, which was not adaptively attacked).

| Method | r=0 | 0.5 | 1.0 | 1.5 | ACR |
|---|---|---|---|---|---|
| Zero-shot + RS | 29 | 24 | 15 | 15 | 0.33 |
| CoOp + RS | 29 | 20 | 15 | 15 | 0.33 |
| PromptSmooth few-shot (reimpl.) | 87 | 66 | 35 | 24 | 0.62 |
| CoRe-Prompt | 84 | 63 | 38 | 33 | 0.78 |
| PromptSmooth combined† (reimpl.) | 87 | 85 | 59 | 42 | 0.96 |
| CoRe-Prompt + TTA† | 85 | 84 | 57 | 32 | 0.71 |

How to read these results:

- **Trade-off, not consistent superiority.** CoRe-Prompt has a larger ACR than PromptSmooth few-shot but lower certified accuracy up to radius 0.5. Only the radius gain at σ = 1.0 is statistically supported (paired difference +0.17, 95% CI [0.11, 0.24]); differences at σ = 0.25 and 0.5 are not significant.
- **With test-time adaptation**, CoRe-Prompt is weaker than PromptSmooth combined (ACR 0.71 vs 0.96).
- **Other settings.** With CLIP ViT-B/32 the difference is negligible (ACR 0.47 vs 0.45). On BloodMNIST the ACR rises from 0.29 to 0.41, but accuracy at radius 0 falls from 71% to 61%.
- **Adaptive attacks.** Under SmoothAdv (σ = 0.5, ε = 1.0), CoRe-Prompt keeps 58% and PromptSmooth few-shot 50% (a four-image difference on 50 images, not significant; the attack is weak). Adaptive BPDA/EOT attacks reduce bit-depth reduction, JPEG and resize-and-pad to 0% at ε = 4/255 (ℓ∞), up to 26 points below their non-adaptive estimates.

All results come from a single seed and a reduced certification budget; see the paper's limitations.

## Data and models

- **PathMNIST and BloodMNIST**: MedMNIST v2/MedMNIST+ (Yang et al., *Scientific Data* 2023), derived from NCT-CRC-HE-100K / CRC-VAL-HE-7K (Kather et al., *PLoS Med* 2019) and Acevedo et al. (*Data in Brief* 2020). Patches from the same patient may appear in one split, so the per-image statistics treat correlated images as independent.
- **Models**: PLIP (`vinid/plip`, Huang et al., *Nature Medicine* 2023) and CLIP ViT-B/32 (`openai/clip-vit-base-patch32`).

## Citation

```bibtex
@article{coreprompt,
  title  = {CoRe-Prompt: Certifiably Robust Prompts for Medical Vision--Language Models},
  author = {[Authors]},
  year   = {2026}
}
```

## License

MIT
