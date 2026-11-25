# CoRe-Prompt: Certifiably Robust Prompts for Medical Vision–Language Models

CoRe-Prompt learns a small text prompt for a frozen medical vision–language model (PLIP) so that its **randomized-smoothing certificate** is as large as possible. It adds two terms to the noisy cross-entropy used by PromptSmooth (MICCAI 2024):

- a **consistency** term across Gaussian noise draws (Jeong & Shin, NeurIPS 2020);
- a **certified-radius hinge** (MACER, ICLR 2020).

Only 4,096 prompt parameters are trained. Because the prompt sits on the text side, the radius term needs no back-propagation through the image encoder, and every compared method is certified with the same noisy image features. That makes the comparison paired and costs one image-encoder pass per noise sample for all methods together.

The repository also contains:
- an independent reproduction of PromptSmooth and its baselines;
- an adaptive-attack evaluation (BPDA/EOT and SmoothAdv);
- ablation and sensitivity studies;
- cross-model (CLIP ViT-B/32) and cross-dataset (BloodMNIST) generalisation;
- cost measurements.

Everything runs on a free Google Colab T4 GPU.

## Main results

All results below are on PLIP / KatherColon (PathMNIST test split) unless the setting says otherwise. All methods were certified on the same images with the same noise. In the tables, CA = certified accuracy, ACR = average certified radius, and † marks methods with test-time adaptation, whose certificate holds for the adapted prompt.

**Certified accuracy (%) at L2 radius r, best over σ, and ACR**

| Method | r=0 | 0.5 | 1.0 | 1.5 | ACR |
|---|---|---|---|---|---|
| Zero-shot + RS | 29 | 24 | 15 | 15 | 0.33 |
| CoOp + RS | 29 | 20 | 15 | 15 | 0.33 |
| PromptSmooth few-shot | **87** | **66** | 35 | 24 | 0.62 |
| **CoRe-Prompt (ours)** | 84 | 63 | **38** | **33** | **0.78** |
| PromptSmooth combined† | 87 | 85 | 59 | 42 | 0.96 |
| CoRe-Prompt + TTA† (ours) | 85 | 84 | 57 | 32 | 0.71 |

**Average certified radius: CoRe-Prompt vs PromptSmooth few-shot, by setting**

| Setting | PromptSmooth few-shot | CoRe-Prompt |
|---|---|---|
| PLIP / KatherColon | 0.62 | **0.78** (+26%) |
| CLIP ViT-B/32 / KatherColon | 0.45 | **0.47** |
| PLIP / BloodMNIST | 0.29 | **0.41** (+41%) |

**Other findings**
- **Reproduction:** the reproduced PromptSmooth few-shot and combined variants are within 7.9 and 7.1 percentage points of the published values on average.
- **Adaptive attack on smoothed classifiers:** under SmoothAdv (σ = 0.5, ε = 1.0), CoRe-Prompt keeps 58% accuracy and PromptSmooth few-shot 50%.
- **Input-transformation defences:** adaptive attacks reduce bit-depth reduction, JPEG and random resize-and-pad to 0% accuracy at ε = 4/255, up to 26 points below their non-adaptive estimates.

## Running the experiments

1. Open `notebooks/CoRe_Prompt_colab.ipynb` in Google Colab.
2. Choose **Runtime ▸ Change runtime type ▸ T4 GPU**.
3. Choose **Runtime ▸ Run all**, and allow Google Drive access when asked.

**What the notebook does:**
- **Self-contained code:** it writes the `coreprompt` package itself, installs pinned dependencies (`transformers==4.46.3`, `medmnist`), runs the unit tests and then runs every experiment.
- **Data download:** it reads only the few-shot and test images of MedMNIST+ (224 px) over HTTP range requests, about 2.4 GB in total, and caches them on Drive.
- **Checkpoints:** learned prompts, per-image certificates and attack results are saved to Drive. After a disconnect, *Run all* again resumes where it stopped.
- **Outputs:** all tables (CSV + LaTeX) and figures go to `MyDrive/coreprompt_quick/report/`, and the final cell downloads them as a zip.

### Experimental budget

The default configuration fits the resource limits of the free Colab tier (preset `quick` in `coreprompt/config.py`):

| Experiment | Setting |
|---|---|
| Baseline reproduction + CoRe-Prompt | 100 certified images, σ ∈ {0.25, 0.5, 1.0}, n = 500 Monte-Carlo samples, α = 0.001, 20 epochs |
| Accuracy on the test split | 2,000 PathMNIST / 1,000 BloodMNIST images, majority vote over 8 noisy copies |
| Adaptive attacks | 50 images, 10-step PGD, EOT/SmoothAdv with 4 draws, PREDICT n = 50 |
| Ablation and sensitivity | σ = 0.5, 100 images |
| Generalisation | σ ∈ {0.25, 0.5}, 100 images, n = 500 |

PromptSmooth's own protocol uses 500 images and n = 10,000. Smaller n caps the certifiable radius at about 2.2σ (instead of 3.2σ) and makes the certificates conservative. To run closer to that protocol, use the `colab` preset or pass overrides, for example:

```bash
python -m coreprompt.run certify --preset colab --set n=10000 n_test=500
```

### Command line

```bash
pip install -r requirements.txt
for s in sanity datastats certify fulltest attacks ablation gen_clip gen_blood efficiency report; do
  python -m coreprompt.run $s --preset quick --out_dir outputs
done
python -m pytest tests -q
```

## Repository layout

```
coreprompt/
  config.py     presets and per-stage budgets
  data.py       MedMNIST+ remote reading, few-shot sampling, caching
  models.py     PLIP/CLIP wrapper, CoOp-style prompt learner
  losses.py     noisy CE, consistency, MACER radius hinge
  train.py      prompt training (CoOp / PromptSmooth / CoRe-Prompt)
  tta.py        PromptSmooth test-time adaptation
  methods.py    method factory
  smoothing.py  shared-feature CERTIFY, certified accuracy, ACR
  attacks.py    PGD (L-inf/L2, BPDA, EOT), SmoothAdv, PREDICT
  defenses.py   bit-depth, JPEG, random resize-and-pad
  report.py     tables, figures, bootstrap / Wilson / McNemar statistics
  run.py        experiment stages
notebooks/      CoRe_Prompt_colab.ipynb (complete pipeline)
tests/          unit tests (certification maths, losses, data reader, report)
paper/          main.tex, sn-bibliography.bib (Springer Nature template)
```

## Data and models

- **PathMNIST and BloodMNIST** come from MedMNIST v2/MedMNIST+ (Yang et al., *Scientific Data* 2023). PathMNIST is derived from NCT-CRC-HE-100K / CRC-VAL-HE-7K (Kather et al., *PLoS Med* 2019), and BloodMNIST from Acevedo et al. (*Data in Brief* 2020).
- **Models** are PLIP (`vinid/plip`, Huang et al., *Nature Medicine* 2023) and CLIP ViT-B/32 (`openai/clip-vit-base-patch32`).

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
