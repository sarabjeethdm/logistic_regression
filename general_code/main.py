import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # change URI if needed
db = client["dev_v2"]
collection = db["pharmacy_claims"]

# Load data into a DataFrame
data = pd.DataFrame(list(collection.find()))
print(data.head())


# Convert dates
data["Fill Date"] = pd.to_datetime(data["Fill Date"])

# Extract useful date features
data["Fill Month"] = data["Fill Date"].dt.month
data["Fill Year"] = data["Fill Date"].dt.year

# Aggregate per member
features = (
    data.groupby("Member ID")
    .agg(
        {
            "Metric Quantity": "sum",
            "Days Supply": "mean",
            "Total Billed": "sum",
            "NDC": lambda x: list(x),
        }
    )
    .reset_index()
)

# One-hot encode NDCs
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
ndc_encoded = mlb.fit_transform(features["NDC"])
ndc_df = pd.DataFrame(ndc_encoded, columns=mlb.classes_)
features = pd.concat([features.drop(columns="NDC"), ndc_df], axis=1)

diagnosis_collection = db['diagnosis_claims']
diagnosis_data = pd.DataFrame(list(diagnosis_collection.find()))

# Multi-label target: aggregate per member
target = diagnosis_data.groupby('Member ID')['Diagnosis Code'].apply(list).reset_index()

# Merge with features
df = features.merge(target, on='Member ID')

# Multi-hot encode target
y_mlb = MultiLabelBinarizer()
y = y_mlb.fit_transform(df['Diagnosis Code'])
X = df.drop(columns=['Member ID', 'Diagnosis Code'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression for multi-label classification
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("F1 Score (micro):", f1_score(y_test, y_pred, average='micro'))

