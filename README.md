# Retinal OCT Classification with Explainable Deep Learning

A leakage-aware retinal OCT classification project using compact CNNs, transfer learning, Grad-CAM, calibration analysis, patient-level failure analysis, and independent test evaluation.

## Project overview

The project classifies retinal OCT images into four categories:

- **CNV** — Choroidal Neovascularization
- **DME** — Diabetic Macular Edema
- **DRUSEN** — Drusen
- **NORMAL** — Normal retina

The workflow emphasizes experimental reliability rather than accuracy alone:

**dataset audit → leakage assessment → patient-disjoint validation → controlled model experiments → explainability → calibration → independent test evaluation → distribution-shift analysis**

## Dataset integrity

The initial OCT2017 Version 2 package was audited before training. The audit found:

- 84,484 image records
- 83,605 unique filenames
- only 32 supplied validation images
- 546 overlapping patient identifiers between supplied training and test partitions
- 879 filenames repeated across partitions

Because reliable medical-image evaluation requires subject-independent testing, the supplied Version 2 partition was not used for model development.

The final experiments used **OCT2017 Version 3**. Independent checks found:

- 108,309 training images
- 1,000 official test images
- 4,785 unique training patient IDs
- 635 unique test patient IDs
- zero train-test patient overlap
- zero duplicate filenames across train and test

The official test set was kept untouched until final evaluation.

## Patient-disjoint development split

| Split | Images |
|---|---:|
| Training | 97,348 |
| Validation | 10,961 |
| Official Test | 1,000 |

Patient overlap was verified to be zero between all three partitions.

## Experiments

| Experiment | Main modification | Parameters | Val Accuracy | Val Macro F1 |
|---|---|---:|---:|---:|
| E01 | Custom CNN + direct resize | 389,828 | 0.9490 | 0.9141 |
| E02 | Custom CNN + weighted cross-entropy | 389,828 | 0.9244 | 0.8891 |
| E03 | Custom CNN + aspect-preserving preprocessing | 389,828 | 0.9528 | 0.9268 |
| E04 | Pretrained ResNet18 | 11,178,564 | **0.9568** | **0.9351** |

### Main validation findings

- Inverse-frequency class weighting improved minority-class recall but reduced overall Macro F1.
- Aspect-preserving preprocessing improved the compact CNN without increasing parameter count.
- Pretrained ResNet18 achieved the strongest validation performance.

## Independent test evaluation

After all model-development decisions were frozen, E03 and E04 were evaluated once on the untouched official test set.

| Model | Parameters | Test Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| E03 Custom CNN | 389,828 | 0.7510 | 0.7510 | 0.7490 |
| E04 ResNet18 | 11,178,564 | **0.9120** | **0.9120** | **0.9098** |

The independent test changed the interpretation of the validation results. The compact CNN experienced a large validation-to-test decline, while ResNet18 generalized substantially better.

## Class-level ResNet18 test performance

| Class | Correct | Total | Recall |
|---|---:|---:|---:|
| CNV | 248 | 250 | 0.992 |
| DME | 245 | 250 | 0.980 |
| DRUSEN | 169 | 250 | 0.676 |
| NORMAL | 250 | 250 | 1.000 |

The dominant remaining failure mode was **DRUSEN → CNV** confusion.

## Explainability and failure analysis

Grad-CAM was used to inspect spatial regions influencing ResNet18 predictions. In the sampled validation analysis:

- mean Grad-CAM attention inside OCT content: **95.48%**
- mean attention in padding: **4.52%**

Patient-level analysis showed that errors were strongly clustered:

- 95 patients contributed at least one validation error
- top 1 patient accounted for 17.30% of validation errors
- top 5 patients accounted for 44.94%
- top 20 patients accounted for 68.78%

This demonstrates that image-level metrics can hide subject-level variability.

## Confidence calibration

The final ResNet18 achieved:

- **Expected Calibration Error (ECE): 0.023**
- error rate at confidence ≥ 0.99: **0.63%**

The model was reasonably calibrated overall, while still exhibiting some individual high-confidence failures.

## Distribution shift

Post-hoc analysis found substantial differences between validation and official test populations.

Validation mean aspect ratios were approximately **1.08–1.46**, while official test classes were around **2.10**. Differences in image intensity statistics were also observed.

These findings do not prove that geometry or intensity differences alone caused the test-performance decline, but they provide evidence of meaningful distribution shift.

## Repository structure

```text
retinal-oct-classification-xai/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── retinal_oct_classification.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── utils.py
├── data_splits/
│   └── README.md
├── results/
│   ├── validation_experiments.csv
│   └── final_test_results.csv
├── figures/
│   └── README.md
└── docs/
    └── methodology.md
```

## Reproducibility

Key design decisions:

- random seed: **42**
- patient-disjoint validation
- Macro F1 used for model selection
- official OCT2017 Version 3 test set untouched until final evaluation
- best checkpoints selected using validation performance
- no further model tuning after viewing final test results

The raw OCT dataset and model checkpoints are intentionally not redistributed in this repository.

## Limitations

- One public OCT dataset was evaluated.
- The official test set contains 1,000 images.
- Development and test populations show measurable distribution shift.
- Dataset labels were not independently reviewed in this project.
- Grad-CAM provides coarse model-attention visualization and does not establish clinically validated lesion localization.
- Model confidence should not be interpreted as clinical certainty.

This repository is a research/portfolio project and is **not intended for clinical diagnosis or deployment**.

## Kaggle notebook

View the Kaggle notebook [https://www.kaggle.com/code/ahmadraza9112/explainable-retinal-oct-classification-f]

## License

Code and documentation in this repository are released under the MIT License. The OCT dataset is governed by its original source/license and is not redistributed here.
