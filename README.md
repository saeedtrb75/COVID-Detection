

ResNet50 + SVM Image Classification
This project implements an image classification pipeline using a combination of ResNet50 for feature extraction and Support Vector Machine (SVM) for classification.
The model is trained on SARS-COV-2 Ct-Scan dataset and performs classification, evaluates accuracy, and plots performance metrics such as ROC and Precision-Recall curves.

Project Structure
ResNet50 Model: A pretrained ResNet50 model (without the top layer) is used as a feature extractor.
SVM Classifier: After extracting features from the images, an SVM classifier is trained to perform the final classification.
Performance Evaluation: Accuracy, confusion matrix, ROC curve, and Precision-Recall curve are plotted to evaluate the model’s performance.

Parameters
batch_size: Number of samples per gradient update (default: 100).
epochs: Number of training epochs (default: 10).
lr: Learning rate for Adam optimizer (default: 0.001).

Evaluation
Accuracy: The model’s accuracy is printed after the SVM classification.
Confusion Matrix: A confusion matrix is displayed for performance evaluation.
ROC Curve: The Receiver Operating Characteristic (ROC) curve is plotted.
Precision-Recall Curve: Precision-Recall curve is plotted.
SVM Decision Boundary: A visualization of the SVM decision boundary on extracted features.

## Dataset

This project uses the **SARS-COV-2 Ct-Scan Dataset**. Proper citation and usage are in accordance with the dataset's terms of use.

You can access the dataset here: [SARS-COV-2 Ct-Scan Dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset))

Please make sure to review and comply with any licensing or usage restrictions related to this dataset.
