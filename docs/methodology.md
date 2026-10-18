# Methodology

## 1. Dataset audit

The project began by auditing OCT2017 Version 2 before any model training. The audit examined class counts, image geometry, filename-derived patient IDs, repeated filenames, file hashes, and decoded-pixel similarity.

The supplied V2 split was rejected for final model development because of subject overlap and the extremely small validation partition.

## 2. Final dataset

OCT2017 Version 3 was used for model development and final evaluation. The notebook verifies zero train-test patient overlap and zero duplicate filenames across the official train and test partitions.

## 3. Development split

A patient-disjoint validation subset was created from the official training population using group-aware stratification. The selected split contains:

- 97,348 training images
- 10,961 validation images
- 1,000 untouched official test images

All images belonging to a patient remain in the same partition.

## 4. Experiments

### E01 — compact CNN baseline

A 389,828-parameter custom CNN was trained with direct 224×224 resizing and standard cross-entropy.

### E02 — class-weighted loss

The E01 architecture and preprocessing were retained while standard cross-entropy was replaced by inverse-frequency class-weighted cross-entropy.

### E03 — aspect-preserving preprocessing

The compact CNN and standard cross-entropy were retained, but direct square resizing was replaced by aspect-preserving resize plus symmetric padding.

### E04 — pretrained ResNet18

The best geometry-preserving preprocessing strategy was combined with ImageNet-pretrained ResNet18. Grayscale OCT inputs were replicated to three channels and normalized using ImageNet statistics.

## 5. Model selection

Validation Macro F1 was fixed as the primary model-selection metric. The official test set was not used to select architectures, losses, preprocessing, thresholds, or hyperparameters.

## 6. Explainability

Grad-CAM was applied to the final ResNet18. Qualitative examples included high-confidence correct predictions and high-confidence errors from distinct patients. A quantitative padding analysis measured the fraction of Grad-CAM activation inside the original OCT content region.

## 7. Calibration

Maximum predicted probabilities were compared for correct and incorrect predictions. Expected Calibration Error was calculated using 15 confidence bins.

## 8. Patient-level failure analysis

Validation errors were grouped by patient ID to determine whether mistakes were uniformly distributed or concentrated among particular subjects.

## 9. Final test evaluation

After development decisions were frozen, E03 and E04 were evaluated once on the official OCT2017 Version 3 test set.

## 10. Post-hoc distribution-shift analysis

After final evaluation, validation and test image geometry and grayscale intensity statistics were compared to investigate the observed validation-to-test performance gap. This analysis was diagnostic only; no further model tuning was performed.
