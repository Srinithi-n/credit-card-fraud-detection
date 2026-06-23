import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:", df.shape)

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Check class distribution
print("\nOriginal Class Distribution:")
print(y.value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# Logistic Regression
lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_smote, y_train_smote)

lr_pred = lr.predict(X_test)

print("\n===== Logistic Regression =====")
print(classification_report(y_test, lr_pred))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_smote, y_train_smote)

rf_pred = rf.predict(X_test)

print("\n===== Random Forest =====")
print(classification_report(y_test, rf_pred))