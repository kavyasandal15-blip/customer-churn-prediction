import pandas as pd

# Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# First 5 rows
print(df.head())

# Dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# Dataset shape
print("Shape:", df.shape)

# Column names
print(df.columns)

# Churn distribution
print(df["Churn"].value_counts())

# Remove customerID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column in df.columns:
    try:
        df[column] = le.fit_transform(df[column])
    except:
        pass
# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print(X.dtypes)

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

print(y_pred[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)

import pickle

# Save Random Forest model
pickle.dump(rf_model, open("churn_model.pkl", "wb"))

print("Model saved successfully!")
