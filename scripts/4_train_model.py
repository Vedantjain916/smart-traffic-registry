import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

FINAL_CSV = os.path.join("processed", "datasets", "final_audio_dataset.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("===============================================")
print("🧠 TRAINING ADVANCED AI: TUNED XGBOOST")
print("===============================================")

# 1. Load Data
print("Loading dataset (this may take a moment)...")
df = pd.read_csv(FINAL_CSV)

# 2. Separate Features (X) and Labels (y)
possible_label_names = ['label', 'class', 'category', 'target', 'labels']
label_col = next((col for col in df.columns if col.lower() in possible_label_names), None)
file_col = next((col for col in df.columns if 'file' in col.lower() or 'name' in col.lower()), None)

X = df.drop(columns=[label_col, file_col], errors='ignore')
y = df[label_col]

# 3. Encode text labels into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = encoder.classes_

# 4. Split Data (80% for learning, 20% for testing)
print("Splitting data into 80% Training and 20% Testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 5. Scale the Data
print("Applying StandardScaler to normalize features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. MEMORY-SAFE Hyperparameter Tuning
print("\n⚙️ Tuning XGBoost Hyperparameters (Memory-Safe Mode)...")

# Take a small 5% slice of the TRAINING data just for tuning (prevents RAM crashes)
X_tune, _, y_tune, _ = train_test_split(X_train_scaled, y_train, train_size=0.05, random_state=42, stratify=y_train)

# Define the grid of parameters to test
param_grid = {
    'max_depth': [3, 5, 7],                  
    'learning_rate': [0.05, 0.1, 0.15],    
    'n_estimators': [100, 150, 200],            
    'subsample': [0.7, 0.8, 0.9],               
    'colsample_bytree': [0.7, 0.8, 0.9]         
}

# Base model
xgb_base = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist', 
    random_state=42,
    n_jobs=1 # Use 1 core to prevent parallel memory overhead
)

# Set up the random search with fewer iterations
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=5, 
    scoring='accuracy',
    cv=2, 
    verbose=1,
    random_state=42,
    n_jobs=1 # No parallel search
)

# Fit on the SMALLER dataset to find the best rules
print("Searching for the best hyperparameters...")
random_search.fit(X_tune, y_tune)

# Extract the winning model configuration
best_params = random_search.best_params_
print(f"\n🏆 Best Parameters Found: {best_params}")

# 7. Train Final Model with early stopping to save time/memory
print("\n🚀 Training final model with best parameters on ALL training data...")
best_model = xgb.XGBClassifier(
    **best_params,
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist', 
    random_state=42,
    n_jobs=1
)
best_model.fit(X_train_scaled, y_train)

# 8. Test the AI
print("\n🧪 Testing Best Model on unseen data...")
y_pred = best_model.predict(X_test_scaled)

print("\n===============================================")
print("📊 FINAL EXAM RESULTS (ACCURACY)")
print("===============================================")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=class_names))

# 9. Visualizing the Mistakes: Confusion Matrix
print("\n🎨 Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Vehicle Audio Classification')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

plot_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(plot_path)
print(f"✅ Saved confusion matrix plot to: {plot_path}")

# 10. Save Model, Encoder, and Scaler
print("\n💾 Saving Model, Scaler, and Label Encoder...")
model_path = os.path.join(MODEL_DIR, "vehicle_classifier_xgb.json")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

best_model.save_model(model_path)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)
print(f"✅ Saved model to: {model_path}")
print(f"✅ Saved scaler to: {scaler_path}")
print(f"✅ Saved label encoder to: {encoder_path}")

# 11. Feature Importance
print("\n===============================================")
print("🏆 TOP 10 MOST IMPORTANT FEATURES")
print("===============================================")
importances = best_model.feature_importances_
feature_names = X.columns
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
print(feat_imp.head(10).to_string(index=False))
print("===============================================")
