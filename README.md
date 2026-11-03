# Multi-Class Texture Classification

**Patch-Level Texture Classification in Remote Sensing Using Transfer Learning on TILDA-400**

A desktop app (Tkinter GUI) that classifies fabric texture patches from the TILDA-400 dataset into three classes. It uses a pretrained **VGG19** network as a feature extractor and classical machine-learning classifiers on top of the extracted features. The proposed model is VGG19 + Extra Trees.

## How it works

1. **Feature extraction:** each image is resized to 224×224, preprocessed, and passed through VGG19 (ImageNet weights, no top layer, average pooling). This produces a 512-dimensional feature vector per image.
2. **Train/test split:** an 80/20 split of the shuffled features (`random_state=42`). The shuffle order is saved in `model/shuffled_indices.npy` so results are reproducible.
3. **Classification:** the features are fed to one of these classifiers:
   - Linear Discriminant Analysis (LDA), existing baseline
   - Quadratic Discriminant Analysis (QDA), existing baseline
   - Support Vector Machine (RBF kernel, C=100, gamma=0.001)
   - **Extra Trees Classifier (200 trees, max depth 32), the proposed model**
4. **Evaluation:** accuracy, macro precision, recall and F1-score, a classification report and a confusion-matrix heatmap are shown for each model.
5. **Prediction:** pick a single image and the app shows it with its predicted class.

## Dataset

TILDA-400 patches, in `TILDA/`, split into three classes:

| Class | Images |
|---|---|
| `good` | 1001 |
| `hole` | 1101 |
| `objects` | 1049 |

## Project structure

```
.
├── Main.py                  # GUI application (feature extraction, training, evaluation, prediction)
├── run.bat                  # Windows launcher (python main.py)
├── background.jpg           # GUI background image
├── TILDA/                   # Dataset (good / hole / objects)
├── model/
│   ├── X_compressed.npz     # Cached VGG19 features
│   ├── Y.npy                # Cached labels
│   ├── shuffled_indices.npy # Saved shuffle order for the split
│   ├── LDA_model.pkl        # Trained models
│   ├── QDA_model.pkl
│   ├── SVM_model.pkl
│   ├── ETC_model.pkl
│   └── XGB_model.pkl        # Saved model, not used by the GUI
├── FINAL DOCUMENTATION.pdf  # Project report
└── PPT.pdf                  # Presentation
```

## Requirements

- Python 3.8+
- Packages: `tensorflow` (includes Keras), `scikit-learn`, `numpy`, `pandas`, `scipy`, `opencv-python`, `scikit-image`, `matplotlib`, `seaborn`, `pillow`, `joblib`
- Tkinter (included with most Python installers)

```bash
pip install tensorflow scikit-learn numpy pandas scipy opencv-python scikit-image matplotlib seaborn pillow joblib
```

The first run downloads the VGG19 ImageNet weights, so an internet connection is needed once.

## Usage

Run from the project folder:

```bash
python Main.py
```

or double-click `run.bat` on Windows.

Then click the buttons in order:

1. **Upload Dataset:** select the `TILDA` folder.
2. **VGG19 Feature extraction:** loads the cached features from `model/` if present, otherwise extracts them (slow on CPU).
3. **Train Test Splitting:** creates the 80/20 split.
4. **Existing LDA / Existing QDA / Train SVM / Train ETC:** evaluate each model. Saved models in `model/` are loaded instead of retraining. Delete a `.pkl` file to retrain that model.
5. **Upload test image:** choose an image to classify. This uses the saved ETC model.
6. **Exit:** close the app.

## Notes

- `Main.py` loads `background.jpg` and the `model/` folder using relative paths, so run it from the project root.
- Run steps 1 to 3 before training or evaluating a model.

## Documentation

See `FINAL DOCUMENTATION.pdf` for the full report and `PPT.pdf` for the presentation.
