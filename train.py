import pandas as pd

# Try reading with error handling
try:
    df = pd.read_csv("data/training_data.csv", quoting=1)  # quoting=1 = QUOTE_ALL
    print("✅ CSV loaded successfully")
    print(df.head())
except Exception as e:
    print("❌ Error reading CSV:", e)

# main.py
import pandas as pd
import os
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# File paths
DATA_PATH = "data/training_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.pkl")

# ----------------- Clean and load data -----------------
df = pd.read_csv(DATA_PATH).dropna(subset=["message", "intent"])
X = df["message"]
y = df["intent"]

# ----------------- Train the model -----------------
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)
pipeline.fit(X, y)

# ----------------- Save the model -----------------
os.makedirs(MODEL_DIR, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Model trained and saved to {MODEL_PATH}")


# Save model and vectorizer
os.makedirs("models", exist_ok=True)
with open("models/intent_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)



print("✅ Model trained and saved to models/intent_model.pkl")
