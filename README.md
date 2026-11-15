# Dementia Severity Classification from Brain MRI

Final Year Project, Group 2, B.Tech Computer Science & Engineering (Data Science).

A transfer-learning model that reads a brain MRI slice and predicts one of three levels:
**None**, **Very mild**, **Mild-moderate**.

## Data

OASIS brain MRI slices, taken from the Kaggle copy
[`kanaadlimaye/alzheimers-classification-dataset`](https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset).

- 10,880 images, 224x224 RGB (train 8,960, validation 1,280, test 640)
- Original labels: ND (none), VMD (very mild), MD (mild), MoD (moderate)
- The dataset is not stored in this repository. Download it from the link above.

## Data cleaning

| Step | What was done |
|---|---|
| Integrity check | All 10,880 files open correctly, all are 224x224 RGB |
| Split check | No image, source image or identical file appears in more than one split |
| Label cleaning | 21 CSV rows had all-zero labels and 7 disagreed with the filename. The label is taken from the filename prefix (`ND__`, `VMD__`, `MD__`, `MoD__`) |
| Class merge | Moderate (MoD) has 88 / 15 / 5 images in train / validation / test, so MD and MoD are merged into one class |
| Class imbalance | Class weights during training, per-class recall reported |
| Normalization | Pixels rescaled to [-1, 1], the range InceptionResNetV2 expects |
| Missing values | None: the data is images with one label each |

Figures and the checking script are in [`mri-eda/`](mri-eda/).

## Method

1. Pretrained **InceptionResNetV2** (ImageNet weights) used as a feature extractor: each scan becomes 1,536 numbers.
2. A **softmax classifier** (multinomial logistic regression) is trained on those features with the base frozen.
3. The top 40 layers of the base are **fine-tuned** with a small learning rate.
4. Evaluation on the untouched test set, plus Grad-CAM heatmaps.

## Results (test set, 640 images)

| Class | Recall | Precision |
|---|---|---|
| None | 81.5% | 83.6% |
| Very mild | 70.4% | 71.7% |
| Mild-moderate | 79.1% | 69.9% |
| **Overall accuracy** | **77.2%** | |

These numbers come from the run made before the final label cleaning (28 of 10,880 images were relabelled afterwards), so a rerun may differ slightly.

## Limitations

- The dataset has no patient ID, so slices from one person may appear in several splits. Scores may be somewhat optimistic.
- The moderate class is very small (5 test images before merging), so it cannot be evaluated on its own.
- The model sees images only, no cognitive scores or clinical data.
- Not a diagnostic tool.

## Reproduce

1. Open `MRI-Baseline-Colab.ipynb` in Google Colab and select a GPU runtime.
2. Add your Kaggle API token under Secrets as `KAGGLE_API_TOKEN`.
3. Run all cells. The notebook downloads the data, trains, evaluates and draws the heatmaps.

To regenerate the data figures locally, place the dataset under `data/oasis-mri/x/` and run `python mri-eda/eda_mri.py`.

## Repository contents

- `MRI-Baseline-Colab.ipynb`: training and evaluation notebook
- `mri-eda/`: data analysis script, figures, quality-check table
