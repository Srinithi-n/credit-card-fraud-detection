# Credit Card Fraud Detection

## Week 1 Task

Implemented two Machine Learning algorithms:

1. Logistic Regression
2. Random Forest Classifier

## Dataset

Credit Card Fraud Detection Dataset

- Total Records: 284,807
- Features: 30
- Target:
  - 0 = Genuine Transaction
  - 1 = Fraudulent Transaction

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

## Model Performance

### Logistic Regression

- Accuracy: 99.92%
- Precision: 83%
- Recall: 64%
- F1-Score: 72%

### Random Forest

- Accuracy: 99.96%
- Precision: 94%
- Recall: 82%
- F1-Score: 87%

## Conclusion

Random Forest outperformed Logistic Regression and provided better fraud detection performance.

## Author

SRINITHI N
## Week 2: Handling Imbalanced Data using SMOTE

### Objective
To address the class imbalance problem in the Credit Card Fraud Detection dataset.

### Implementation
- Applied SMOTE (Synthetic Minority Oversampling Technique)
- Balanced fraud and genuine transaction classes
- Trained Logistic Regression and Random Forest models
- Evaluated model performance using Precision, Recall and F1-Score

### Results

Original Class Distribution:
- Genuine: 284315
- Fraud: 492

After SMOTE:
- Genuine: 227451
- Fraud: 227451

### Conclusion

SMOTE successfully balanced the dataset.

Random Forest provided the best balance between fraud detection and false positive reduction.

# Week 3 - Autoencoder for Deep Anomaly Detection

## Objectives
- Implemented an Autoencoder using TensorFlow.
- Trained only on normal transactions.
- Detected fraudulent transactions using reconstruction error.

## Technologies
- Python
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn

## Results

Training completed for 10 epochs.

Accuracy: 95%

Fraud Recall: 87%

The Autoencoder successfully detected most fraudulent transactions using reconstruction error.

# Week 4 - NLP using BERT

## Objective
Apply a pretrained BERT model to transaction descriptions and perform fraud classification.

## Work Completed
- Generated transaction descriptions for demonstration.
- Extracted text embeddings using DistilBERT.
- Applied SMOTE to balance the training data.
- Trained a Logistic Regression classifier.
- Evaluated the model using Precision, Recall and F1-score.

## Technologies Used
- Python
- Transformers (Hugging Face)
- PyTorch
- Scikit-learn
- imbalanced-learn

## Results
- Successfully implemented a BERT-based fraud detection pipeline.
- Generated embeddings from transaction descriptions.
- Balanced the dataset using SMOTE.
- Evaluated the classifier on a held-out test set.