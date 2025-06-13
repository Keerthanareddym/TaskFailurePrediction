import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Bidirectional
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter
import psutil
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Bidirectional, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


# Load and preprocess dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    
   # print("🔍 Raw Columns:", df.columns.tolist())  # Check original columns
    df.columns = df.columns.str.strip()
   # print("🔍 Cleaned Columns:", df.columns.tolist())  # Check after stripping spaces
    # Drop unnecessary columns
    drop_cols = ["Unnamed: 0", "user", "collection_name", "collection_logical_name"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore", inplace=True)
    if 'failed' not in df.columns:
        raise ValueError("🚨 Column 'failed' not found in dataset!")

    y = df['failed']
    X = df.drop(columns=['failed'])
     # Convert non-numeric columns to numeric
    # Select only the 4 features needed for training
    #selected_features = ["feature_1", "feature_2", "feature_3", "feature_4"]
    
    # Ensure the selected features exist in the dataset
    #missing_features = [feat for feat in selected_features if feat not in X.columns]
    #if missing_features:
        #raise ValueError(f"🚨 Missing required features: {missing_features}")

    #X = X[selected_features]  # Keep only these 4 features
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(float)  # Convert if possible
            except ValueError:
                X.drop(columns=[col], inplace=True)  # Drop if conversion fails
        # Handle missing values (NaN)
    imputer = SimpleImputer(strategy="mean")  # Fill NaNs with mean
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

   # print("✅ Numeric Columns Used:", X.columns.tolist())
    #print(f"✅ Missing values after imputation: {X.isna().sum().sum()}")

    return X, y
import matplotlib.pyplot as plt

def plot_metrics(metrics_dict):
    models = list(metrics_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]

    for metric in metrics:
        values = [metrics_dict[model][metric] for model in models]
        plt.figure(figsize=(8, 5))
        plt.bar(models, values, color='skyblue')
        plt.title(f"{metric.title()} Comparison")
        plt.ylabel(metric.title())
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
        plt.tight_layout()
        plt.show()

  

    os.makedirs("models", exist_ok=True)
    best_model, best_accuracy = None, 0
    metrics = {}  # Store all metrics for plotting
    for name, model in models.items():
        print(f"🛠️ Training {name} model...")  # Debugging print
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        

        if accuracy > best_accuracy:

            best_accuracy = accuracy
            best_model = model

        #print(f"{name} Accuracy: {accuracy:.4f}")
        #print("Classification Report:\n", classification_report(y_test, y_pred))
        #print("hii")
        #print(f"🛠️ Saving {name} model...")
        joblib.dump(model, f"models/{name}.pkl")
        #print(f"✅ {name} model saved successfully.")
    print("✅ Finished training ml models.")
    #print("🛠️ Checking memory before VotingClassifier...")
      #  print(f"Saved {name} model.")
     # Train Voting Classifie
    print("🔹 Training VotingClassifier...")
    voting_clf = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
        ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
        
    ], voting='soft')
    import gc
    gc.collect()  # Force garbage collection
    print("✅ Memory cleared. Starting VotingClassifier...")
    voting_clf.fit(X_train, y_train)
    print("✅ VotingClassifier training completed!")
    y_pred_voting = voting_clf.predict(X_test)

    
    

    voting_accuracy = accuracy_score(y_test, y_pred_voting)
    print(f"VotingClassifier Accuracy: {voting_accuracy:.4f}")
    print("Classification Report (Voting Classifier):\n", classification_report(y_test, y_pred_voting))
    
    # Save the best ML model + Voting Classifier
    joblib.dump(best_model, "model.sav")
    joblib.dump(voting_clf, "models/VotingClassifier.sav")
    print(f"✅ Best Model Saved with Accuracy: {best_accuracy:.4f}")
    print("✅ Voting Classifier Model Saved.")
    

import os
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

def train_cnn_bilstm(X, y):
    model_path = "models/CNN_BiLSTM.h5"

    if os.path.exists(model_path):
        print("✅ Loading pre-trained CNN + BiLSTM model...")
        model = load_model(model_path)

        # Evaluate the model on test data
        X = np.array(X).reshape(X.shape[0], X.shape[1], 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ Loaded Model Accuracy: {accuracy:.4f}")  # Print accuracy
        print("🔹 Classification Report:\n", classification_report(y_test, y_pred))

    else:
        print("🚀 Training new CNN + BiLSTM model from scratch...")
        # (Training code remains the same)

    return model  # Return model for further use



if __name__ == "__main__":
    X, y = load_data("dataset/dataset.csv")
    train_ml_models(X, y)
    train_cnn_bilstm(X, y)