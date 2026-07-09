import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("data/creditcard.csv")

# Keep all fraud cases
fraud = df[df["Class"] == 1]

# Randomly select 5000 normal transactions
normal = df[df["Class"] == 0].sample(n=5000, random_state=42)

# Combine them
df = pd.concat([normal, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)

print(df["Class"].value_counts())

print("Dataset Shape:", df.shape)

# ===========================
# Generate Transaction Descriptions
# ===========================

def generate_description(amount):
    if amount < 20:
        return "Small grocery purchase"

    elif amount < 100:
        return "Restaurant payment"

    elif amount < 500:
        return "Online shopping"

    elif amount < 2000:
        return "Electronics purchase"

    elif amount < 10000:
        return "Travel booking"

    else:
        return "International money transfer"

df["Description"] = df["Amount"].apply(generate_description)

print(df[["Amount", "Description"]].head())

# ===========================
# Load BERT
# ===========================

print("\nLoading BERT model...")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModel.from_pretrained("distilbert-base-uncased")

# ===========================
# Encode Text
# ===========================

descriptions = df["Description"].tolist()

embeddings = []

print("Generating embeddings...")

for text in descriptions:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    )

    with torch.no_grad():

        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:,0,:].numpy()

    embeddings.append(embedding[0])

X = np.array(embeddings)

y = df["Class"]

# ===========================
# Train/Test Split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===========================
# Apply SMOTE
# ===========================

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# ===========================
# Logistic Regression
# ===========================

clf = LogisticRegression(max_iter=1000)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("\nClassification Report")

print(classification_report(y_test, pred))