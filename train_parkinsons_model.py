import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "parkinsons.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "parkinsons_model.pkl")

print(f"📂 Looking for dataset at: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {DATA_PATH}")

# -----------------------
# Load dataset
# -----------------------
data = pd.read_csv(DATA_PATH)

# Drop rows with missing target
data = data.dropna(subset=['status'])

# Features & target
X = data.drop(columns=['status', 'name'], errors='ignore')
y = data['status']

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Pipeline (impute → scale → model)
# -----------------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),       # handle any missing numeric values
    ("scaler", StandardScaler()),                      # scale numeric values
    ("model", RandomForestClassifier(random_state=42)) # classifier
])

# -----------------------
# Train model
# -----------------------
pipeline.fit(X_train, y_train)

# -----------------------
# Save model
# -----------------------
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Parkinson's model saved at {MODEL_PATH}")
