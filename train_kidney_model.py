# train_kidney_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def train_kidney_model():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(r'D:\GUVI-Project\project 4\data\kidney_disease.csv')
    
    # Display basic info
    print("Dataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDataset Info:")
    print(df.info())
    
    # Handle missing values (you can modify this as per dataset)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=[object]).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert categorical variables
    le_dict = {}
    for col in df.select_dtypes(include=[object]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # Save each encoder
    
    # Define features and target
    X = df.drop('classification', axis=1)  # Replace 'classification' with actual target column
    y = df['classification']
    
    # For binary classification, map target to 0 and 1 if needed
    y = y.map({'ckd': 1, 'notckd': 0})  # Modify mapping based on dataset
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model, scaler, and encoders
    model_path = r'D:\GUVI-Project\project 4\models\kidney_model.pkl'
    scaler_path = r'D:\GUVI-Project\project 4\models\kidney_scaler.pkl'
    encoder_path = r'D:\GUVI-Project\project 4\models\kidney_encoders.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(le_dict, f)
    
    print("\nModel, scaler, and encoders saved successfully!")
    print(f"Model saved at: {model_path}")
    print(f"Scaler saved at: {scaler_path}")
    print(f"Encoders saved at: {encoder_path}")
    
    return model, scaler, le_dict, accuracy

if __name__ == "__main__":
    train_kidney_model()
