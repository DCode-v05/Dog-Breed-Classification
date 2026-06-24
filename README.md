# Dog Breed Classification

**A transfer-learning image classifier that identifies a dog's breed from a photo across 120 breeds — built on the Kaggle Dog Breed Identification dataset.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

## Overview

This is an end-to-end deep learning pipeline that takes a photo of a dog and predicts which of 120 breeds it is. It's built around the [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) competition, which provides 10,222 labeled training images and 10,357 unlabeled test images, and asks for a probability across all 120 breeds for each test image.

The whole thing lives in a single Jupyter notebook (95 cells — 77 code, 18 markdown) that walks from raw image files to a Kaggle-format submission. The approach is transfer learning: instead of training a convolutional network from scratch on a fairly small dataset, it takes a MobileNetV2 backbone pre-trained on ImageNet and fine-tunes it for the 120-breed problem. I built and ran it in Google Colab to get a GPU, with the dataset and model checkpoints stored on Google Drive.

This is a learning project — it follows the standard "dog vision" transfer-learning workflow — but it's complete and runs the full loop: load and tensorize images, batch them, train, evaluate, predict on the test set, format a submission, and classify your own custom photos. The full-dataset run reaches roughly 87% accuracy on the training data.

## Key Features

- Classifies a dog image into one of **120 breeds** with a softmax probability distribution.
- **Transfer learning with MobileNetV2** from TensorFlow Hub (`mobilenet_v2_130_224`), fine-tuned end-to-end for this dataset.
- Full image pipeline: file path → decoded JPEG → normalized float tensor → resized to 224×224 → batched (`tf.data`, batch size 32).
- One-hot label encoding driven by `labels.csv`, with a unique-breed vocabulary of 120 classes.
- Train/validation split via scikit-learn's `train_test_split` for the subset experiments.
- Two training modes built into the notebook: a fast **1,000-image subset** for quick iteration, and a **full 10,222-image** run.
- Training callbacks: **EarlyStopping** (monitors `val_accuracy`, patience 3), plus **TensorBoard** logging to `logs/` so you can watch loss and accuracy curves.
- Model persistence to `.h5` (HDF5), with reload support that re-registers the custom `hub.KerasLayer`.
- Batch prediction over all 10,357 test images, exported to `predictions.csv` and `predictions.xlsx`.
- Kaggle-format submission file (`final_predictions.csv`): an `id` column plus one probability column per breed.
- **Predict on your own images** — drop photos into `Dog Images/` and get the top-5 breeds with confidence scores.
- Visualization helpers: accuracy/loss curves, top-prediction bar charts, and side-by-side actual-vs-predicted displays.

## How It Works

The notebook is organized as a linear pipeline, mirrored by its markdown section headers: get data ready → tensors → batches → model → callbacks → fit → predict → evaluate → full-dataset training → save/load → test predictions → submission → custom images.

### Data preparation

`labels.csv` maps each image ID to a breed string (e.g. `golden_retriever`, `boston_bull`, `dingo`). The notebook builds a sorted list of the 120 unique breeds and turns each label into a one-hot vector of length 120 — that's the `OUTPUT_SHAPE`. Image file paths are assembled from the IDs in `train/`.

### Preprocessing into tensors and batches

Each image is read from disk, JPEG-decoded into a 3-channel tensor, cast to float and scaled to the 0–1 range, then resized to a fixed **224×224×3** (`IMG_SIZE = 224`, the input size MobileNetV2 expects). Images and labels are zipped into a `tf.data` pipeline and grouped into batches of 32 (`BATCH_SIZE = 32`) so they stream efficiently to the GPU during training.

### Model

The model is a small `tf_keras.Sequential`:

```python
model = tf_keras.Sequential([
    hub.KerasLayer(MODEL_URL, trainable=True),
    tf_keras.layers.Dense(output_shape, activation="softmax"),
])
```

`MODEL_URL` is the TF-Hub MobileNetV2 classification model `mobilenet_v2_130_224/classification/4`. It's loaded with `trainable=True`, so the whole backbone is fine-tuned rather than frozen — the pre-trained ImageNet weights are the starting point and they keep updating during training. A `Dense(120, softmax)` head sits on top to produce the per-breed probabilities. The model compiles with **categorical cross-entropy** loss, the **Adam** optimizer, and accuracy as the tracked metric.

### Callbacks

Two callbacks wire into `model.fit`:

- **EarlyStopping** on `val_accuracy` with `patience=3` — stops training once validation accuracy stops improving for three epochs, which avoids wasting epochs and overfitting on the subset runs.
- **TensorBoard** — writes per-run event files into timestamped folders under `logs/`, so training and validation curves are inspectable after the fact.

`NUM_EPOCHS` is set up to 100, but EarlyStopping usually cuts the subset runs off well before that.

### Training: subset then full

The notebook first trains on a **1,000-image subset** (`NUM_IMAGES = 1000`) with a held-out validation split — useful for sanity-checking the pipeline and iterating quickly. It then retrains on the **full 10,222-image** set (320 batches per epoch) to get the model that's actually used for predictions. Both runs produce a saved `.h5` checkpoint under `model/`, named with a timestamp and the run type (`...1000-images-mobilenetv2-Adam.h5` and `...full-images-mobilenetv2-Adam.h5`).

### Prediction and submission

After training, the saved model is reloaded (passing `custom_objects={"KerasLayer": hub.KerasLayer}` so Keras knows how to rebuild the TF-Hub layer). It predicts across all 10,357 test images. Each prediction is a 120-length probability vector, which gets reshaped into the Kaggle submission table: an `id` column plus one column per breed. Those go out to `predictions.csv`/`predictions.xlsx` and the final `final_predictions.csv`.

### Custom images

The last section runs inference on arbitrary photos placed in `Dog Images/` (the repo ships three samples — a basset, a Bedlington terrier, and a pug). For each one it prints the top-5 most likely breeds with their confidence percentages.

## Results / Highlights

A few honest notes on the numbers, since the README should reflect what the code actually measures:

- The **full-dataset run reaches ~87% accuracy** (peaking around 0.874) — but this is **training accuracy**. The full run fits on all labeled images with no held-out validation split, so it's the model's accuracy on data it has seen, not a generalization estimate.
- The **1,000-image subset run** climbs to ~80% *training* accuracy over a handful of epochs before EarlyStopping triggers. Its small held-out validation set is noisy (high val-loss), so the subset is best read as a pipeline check rather than a performance result.
- The competition's actual scoring metric is **multi-class log-loss on the hidden Kaggle test set**, which isn't computed locally here — the test images are unlabeled by design.
- **Scale:** 120 breeds, 10,222 training images, 10,357 test images, MobileNetV2 backbone at 224×224, batches of 32.

In short: the pipeline trains and predicts end-to-end and the full model fits the training data well; treat the accuracy figures as training-set numbers, not a validated test score.

## Tech Stack

- **Language:** Python 3.x
- **Frameworks / libraries:** TensorFlow, TensorFlow Hub, `tf_keras`, scikit-learn (`train_test_split`)
- **Data / numerics:** pandas, NumPy, Matplotlib
- **Model:** MobileNetV2 (`mobilenet_v2_130_224`) via TF-Hub, fine-tuned with a Dense(120) softmax head
- **Tooling / infra:** Jupyter Notebook, Google Colab (GPU runtime + Google Drive mounts), TensorBoard, Git LFS for the large data/model files

## Getting Started

### Prerequisites

- Python 3.x (a GPU runtime is strongly recommended — training on CPU is slow)
- The Kaggle [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) dataset (`train/`, `test/`, `labels.csv`, `sample_submission.csv`)
- Git LFS installed, since the notebook, model checkpoints, and prediction files are stored with LFS

### Installation

```bash
git lfs install
git clone https://github.com/DCode-v05/Dog-Breed-Classification.git
cd Dog-Breed-Classification
pip install tensorflow tensorflow_hub tf_keras pandas numpy matplotlib scikit-learn
```

### Running

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook "Dog Breed Classification.ipynb"
```

The recommended path is Google Colab with a GPU runtime, mounting Drive for the dataset and checkpoints:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

- **Train:** run the data-prep and training cells. Start with the 1,000-image subset to verify the pipeline, then run the full-dataset training cell for the real model.
- **Monitor:** point TensorBoard at the `logs/` directory to watch loss/accuracy curves across runs.
- **Predict on the test set:** the prediction cells load the saved model and write `final_predictions.csv` in Kaggle submission format — upload that to the competition.
- **Classify your own photos:** put images in `Dog Images/` and run the custom-image section to see the top-5 predicted breeds with confidence scores.
- **Reuse the model:** load a saved checkpoint with
  ```python
  from tensorflow.keras.models import load_model
  import tensorflow_hub as hub
  model = load_model("model/...full-images-mobilenetv2-Adam.h5",
                     custom_objects={"KerasLayer": hub.KerasLayer})
  ```

## Project Structure

```
Dog-Breed-Classification/
├── Dog Breed Classification.ipynb   # The full pipeline: data -> train -> predict -> submit (95 cells)
├── labels.csv                       # 10,222 rows mapping image id -> breed (120 unique breeds)
├── sample_submission.csv            # Kaggle reference format (id + 120 breed columns)
├── train/                           # 10,222 labeled training images
├── test/                            # 10,357 unlabeled test images for prediction
├── Dog Images/                      # 3 sample photos for custom inference (basset, bedlington_terrier, pug)
├── model/                           # Saved MobileNetV2 checkpoints (.h5): 1000-image and full-image runs
├── logs/                            # TensorBoard event files (7 timestamped training runs)
├── predictions.csv                  # Raw test-set predictions
├── predictions.xlsx                 # Same predictions in Excel
├── final_predictions.csv            # Kaggle-format submission file
└── README.md
```

---

## Contact

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
