import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Load Data ---
print("Loading dataset...")
# The path is relative to the root of the project where you run the script
data_path = 'suzuki_reaction_dataset.csv' 
df = pd.read_csv(data_path)

# --- 2. Feature Engineering & Preprocessing ---
print("Preprocessing data...")
# Convert categorical features into numerical using One-Hot Encoding
# This is a crucial step for most ML models
df_processed = pd.get_dummies(df, columns=['catalyst', 'solvent'], drop_first=True)

# Separate features (X) and target (y)
X = df_processed.drop('yield', axis=1)
y = df_processed['yield']

# --- 3. Split Data into Training and Testing sets ---
# We use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# --- 4. Model Training ---
print("Training the XGBoost Regressor model...")
# Initialize the XGBoost model for regression
# n_estimators: number of boosting rounds
# learning_rate: step size shrinkage to prevent overfitting
# random_state: for reproducibility
model = xgb.XGBRegressor(objective='reg:squarederror', 
                         n_estimators=100, 
                         learning_rate=0.1, 
                         random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)
print("Model training completed.")

# --- 5. Model Evaluation ---
print("\n--- Model Evaluation ---")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# --- 6. Visualization: Actual vs. Predicted Yield ---
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, ci=None, color='blue', line_kws={'color': 'red', 'linestyle': '--'})
plt.xlabel("Actual Yield (%)", fontsize=12)
plt.ylabel("Predicted Yield (%)", fontsize=12)
plt.title("Model Performance: Actual vs. Predicted Yield", fontsize=14)
plt.grid(True)
plt.savefig("actual_vs_predicted_yield.png") # Save the plot as a file
print("\nSaved 'actual_vs_predicted_yield.png' plot.")

# --- 7. Feature Importance ---
# This shows which reaction conditions are most important for the prediction
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette='viridis')
plt.xlabel("Feature Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Feature Importance for Yield Prediction", fontsize=14)
plt.tight_layout()
plt.savefig("feature_importance.png") # Save the plot as a file
print("Saved 'feature_importance.png' plot.")

print("\nProject script finished successfully!")