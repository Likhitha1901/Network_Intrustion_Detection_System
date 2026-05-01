import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("📥 Loading dataset...")
df = pd.read_csv("dataset.csv")

print("📊 Dataset shape:", df.shape)

# Drop ID column if exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Target column (for UNSW dataset it's usually 'label' or 'attack_cat')
if 'label' in df.columns:
    y = df['label']
    X = df.drop(columns=['label'])
elif 'attack_cat' in df.columns:
    y = df['attack_cat']
    X = df.drop(columns=['attack_cat'])
else:
    raise Exception("❌ No target column found")

print("🔧 Encoding categorical features...")
encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

print("🎯 Encoding target...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
encoders['target'] = label_encoder

print("⚖️ Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("🤖 Training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
import pandas as pd

importance_df = pd.DataFrame({
    'Feature Name': X.columns,
    'Importance Score': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance Score',
    ascending=False
)

importance_df.to_csv("feature_importance.csv", index=False)

print("💾 Saving files...")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))
pickle.dump((X_test, y_test), open("test_data.pkl", "wb"))

print("🔥 SUCCESS: Model trained and files saved!")