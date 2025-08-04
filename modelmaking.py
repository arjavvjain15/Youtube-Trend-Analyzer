import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import uniform, randint

# --------------------------------------------------------------------------
# 1. DATA LOADING AND INITIAL PREPARATION
# --------------------------------------------------------------------------
print("Step 1: Loading and preparing the dataset...")
df = pd.read_csv("viewdataset.csv")

# Convert stringified lists into real lists
df['prev_5_video_views'] = df['prev_5_video_views'].apply(ast.literal_eval)
df['prev_5_video_likes'] = df['prev_5_video_likes'].apply(ast.literal_eval)
df['prev_5_video_comments'] = df['prev_5_video_comments'].apply(ast.literal_eval)
df['days_since_prev_5_uploads'] = df['days_since_prev_5_uploads'].apply(ast.literal_eval)

# Split each list column into 5 separate columns for individual processing
for i in range(5):
    df[f'views_{i+1}'] = df['prev_5_video_views'].apply(lambda x: x[i])
    df[f'likes_{i+1}'] = df['prev_5_video_likes'].apply(lambda x: x[i])
    df[f'comments_{i+1}'] = df['prev_5_video_comments'].apply(lambda x: x[i])
    df[f'days_since_upload_{i+1}'] = df['days_since_prev_5_uploads'].apply(lambda x: x[i])

# --------------------------------------------------------------------------
# 2. DYNAMIC WEIGHT CALCULATION
# --------------------------------------------------------------------------
print("Step 2: Calculating dynamic weights based on video age...")

def calculate_dynamic_weights(row):
    """Calculates weights based on the inverse of video age."""
    ages = [row[f'days_since_upload_{i+1}'] for i in range(5)]
    raw_weights = [1 / (age + 1) for age in ages]
    sum_of_weights = sum(raw_weights)
    if sum_of_weights == 0:
        return [0.2] * 5
    return [w / sum_of_weights for w in raw_weights]

df['dynamic_weights'] = df.apply(calculate_dynamic_weights, axis=1)

# --------------------------------------------------------------------------
# 3. ADVANCED FEATURE ENGINEERING
# --------------------------------------------------------------------------
print("Step 3: Engineering advanced and weighted features...")

# 3a. Create weighted average features for previous videos
df['weighted_prev_views'] = df.apply(
    lambda row: np.dot([row[f'views_{i+1}'] for i in range(5)], row['dynamic_weights']), axis=1
)
df['weighted_prev_likes'] = df.apply(
    lambda row: np.dot([row[f'likes_{i+1}'] for i in range(5)], row['dynamic_weights']), axis=1
)
df['weighted_prev_comments'] = df.apply(
    lambda row: np.dot([row[f'comments_{i+1}'] for i in range(5)], row['dynamic_weights']), axis=1
)

# 3b. **NEW**: Create interaction features (engagement ratios)
# Add 1 to the denominator to prevent division by zero
df['likes_per_view_ratio'] = df['current_likes'] / (df['current_views'] + 1)
df['comments_per_view_ratio'] = df['current_comments'] / (df['current_views'] + 1)

# --------------------------------------------------------------------------
# 4. APPLY LOG TRANSFORMATION
# --------------------------------------------------------------------------
print("Step 4: Applying log transformation to features and target...")
# We use np.log1p which calculates log(1+x) to handle zero values gracefully.
features_to_transform = [
    'current_views', 'current_comments', 'current_likes',
    'weighted_prev_views', 'weighted_prev_likes', 'weighted_prev_comments',
    'likes_per_view_ratio', 'comments_per_view_ratio' # Also transform the new features
]
# Store original current_views before transforming for the final check
df['original_current_views'] = df['current_views']

for feat in features_to_transform:
    df[feat] = np.log1p(df[feat])

# IMPORTANT: Also transform the target variable
df['views_after_30_days'] = np.log1p(df['views_after_30_days'])

# --------------------------------------------------------------------------
# 5. MODEL TRAINING WITH HYPERPARAMETER TUNING
# --------------------------------------------------------------------------
print("Step 5: Defining features and starting advanced model training...")

# Use the new, expanded feature set
X = df[features_to_transform + ['original_current_views']]
y = df['views_after_30_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate the original views from the test set for the final adjustment
current_views_test = X_test['original_current_views']
X_test_final = X_test.drop(columns=['original_current_views'])
X_train_final = X_train.drop(columns=['original_current_views'])

# **NEW**: Use a more powerful GradientBoostingRegressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# **NEW**: Define a search space for the model's hyperparameters.
# This tells RandomizedSearchCV which settings to try.
param_distributions = {
    'regressor__n_estimators': randint(100, 500),
    'regressor__learning_rate': uniform(0.01, 0.2),
    'regressor__max_depth': randint(3, 10),
    'regressor__subsample': uniform(0.7, 0.3)
}

# **NEW**: Set up RandomizedSearchCV to find the best model settings.
# It will try 50 different combinations (n_iter=50) using 5-fold cross-validation (cv=5).
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1 # Prints progress
)

print("Searching for the best hyperparameters...")
search.fit(X_train_final, y_train)

print(f"\nBest parameters found: {search.best_params_}")
best_model = search.best_estimator_

# --------------------------------------------------------------------------
# 6. EVALUATION AND SAVING
# --------------------------------------------------------------------------
print("\nStep 6: Evaluating best model performance...")
y_pred_log = best_model.predict(X_test_final)

# Convert predictions and test data back from log scale to their original scale.
y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

# Ensure the prediction is not less than the current views
y_pred_adjusted = np.maximum(y_pred_orig, current_views_test)

mse = mean_squared_error(y_test_orig, y_pred_adjusted)
r2 = r2_score(y_test_orig, y_pred_adjusted)

print(f"\n--- Model Evaluation (on original scale) ---")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"R2 Score: {r2:.4f}")

joblib.dump(best_model, "predictive_view_modelGBR.pkl")
print("\nBest model saved successfully as 'predictive_view_model.pkl'")
