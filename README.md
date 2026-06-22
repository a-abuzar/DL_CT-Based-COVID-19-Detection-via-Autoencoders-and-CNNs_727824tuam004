# Hybrid COVID-19 Detection using CT Scans

## Project Objective
Develop a hybrid deep‑learning pipeline that combines auto‑encoders and convolutional neural networks (CNNs) to detect COVID‑19 infection from chest CT scans. The solution aims to achieve high sensitivity and specificity while being lightweight enough for deployment in clinical settings.

## Dataset Source
The primary dataset is the **[Kaggle SARS‑CoV‑2 CT Scan Dataset](https://www.kaggle.com/datasets/ananthu2000/sars-cov2-ct-scan-dataset)**, which contains labeled CT images of patients with COVID‑19, other pneumonia, and healthy controls.

## Expected Outcomes
- **Model Architecture**: An auto‑encoder for unsupervised feature learning followed by a fine‑tuned CNN classifier.
- **Performance Metrics**: Target ≥ 95 % accuracy, ROC‑AUC ≥ 0.97, and balanced precision/recall across classes.
- **Deliverables**:
  - Training scripts (`src/`)
  - Jupyter notebooks for exploratory analysis (`notebooks/`)
  - Documentation and usage guide (`docs/`)
  - A concise README (this file) outlining the project.

## Repository Structure
```
├─ data/          # Raw and processed datasets (excluded from VCS)
├─ docs/          # Project documentation
├─ notebooks/     # Exploratory notebooks
├─ src/           # Source code (models, training, utilities)
├─ .gitignore     # Ignores large data files and caches
└─ README.md      # Project overview (you are looking at it)
```

---
*Feel free to adjust the architecture or experiment with additional datasets as the project evolves.*
