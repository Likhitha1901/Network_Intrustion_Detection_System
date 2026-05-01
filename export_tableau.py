import pandas as pd
import pickle
import numpy as np
import os

def export_for_tableau():
    print("Starting Tableau Export Process...")
    
    # 1. Load the trained model and preprocessing objects
    try:
        model = pickle.load(open("model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        encoders = pickle.load(open("encoders.pkl", "rb"))
        feature_cols = pickle.load(open("columns.pkl", "rb"))
        print("Model and preprocessing objects loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Required file missing. Please run train_model.py first. ({e})")
        return

    # 2. Load the dataset
    if not os.path.exists("dataset.csv"):
        print("Error: dataset.csv not found.")
        return
    
    df = pd.read_csv("dataset.csv")
    print(f"Loaded dataset with {df.shape[0]} rows.")

    # 3. Preprocess for prediction
    # Use only columns required by the model
    # We must ensure all columns in feature_cols exist in df
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns from dataset: {missing_cols}")
        return

    X = df[feature_cols].copy()

    # Encode categorical features
    for col, le in encoders.items():
        if col != 'target' and col in X.columns:
            print(f"Encoding {col}...")
            X[col] = X[col].astype(str)
            # Handle unseen labels by mapping them to the first known class
            known_classes = set(le.classes_)
            X[col] = X[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            X[col] = le.transform(X[col])

    # Scale features
    X_scaled = scaler.transform(X)

    # 4. Generate Predictions
    print("Generating predictions...")
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Get the human-readable labels from target encoder
    target_encoder = encoders['target']
    predicted_labels = target_encoder.inverse_transform(predictions)
    
    # Map 0/1 to Normal/Attack if necessary (based on classes [0, 1])
    # However, inverse_transform should already handle this if they were strings in training.
    # Since we saw [0 1], it returns 0 or 1. Let's make it friendly for Tableau.
    friendly_labels = ["Normal" if p == 0 else "Attack" for p in predicted_labels]
    
    # Get confidence (max probability)
    confidence_scores = np.max(probabilities, axis=1)

    # 5. Enrich the original dataframe
    df['AI_Predicted_Label'] = friendly_labels
    df['AI_Prediction_Probability'] = confidence_scores
    
    # If we have ground truth 'label', compare it
    if 'label' in df.columns:
        df['Is_AI_Correct'] = df['label'] == predictions

    # 6. Export to CSV for Tableau
    output_file = "nids_tableau_export.csv"
    df.to_csv(output_file, index=False)
    
    print(f"SUCCESS: Exported {df.shape[0]} rows to '{output_file}'.")
    print("Columns added: 'AI_Predicted_Label', 'AI_Prediction_Probability', 'Is_AI_Correct'")
    print("You can now import this file into Tableau for visualization.")

if __name__ == "__main__":
    export_for_tableau()
