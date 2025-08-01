import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("viewdataset.csv")

# Step 2: Convert stringified lists into real lists
df['prev_5_video_views'] = df['prev_5_video_views'].apply(ast.literal_eval)
df['prev_5_video_likes'] = df['prev_5_video_likes'].apply(ast.literal_eval)
df['prev_5_video_comments'] = df['prev_5_video_comments'].apply(ast.literal_eval)
df['days_since_prev_5_uploads'] = df['days_since_prev_5_uploads'].apply(ast.literal_eval)

# Step 3: Split each list column into 5 separate columns
for i in range(5):
    df[f'views_{i+1}'] = df['prev_5_video_views'].apply(lambda x: x[i])
    df[f'likes_{i+1}'] = df['prev_5_video_likes'].apply(lambda x: x[i])
    df[f'comments_{i+1}'] = df['prev_5_video_comments'].apply(lambda x: x[i])
    df[f'days_since_upload_{i+1}'] = df['days_since_prev_5_uploads'].apply(lambda x: x[i])

# Drop original list columns
df.drop(['prev_5_video_views', 'prev_5_video_likes', 'prev_5_video_comments', 'days_since_prev_5_uploads'], axis=1, inplace=True)

# Step 4: Define input features and output
X = df[['current_views', 'current_comments', 'current_likes'] +
       [f'views_{i+1}' for i in range(5)] +
       [f'likes_{i+1}' for i in range(5)] +
       [f'comments_{i+1}' for i in range(5)] +
       [f'days_since_upload_{i+1}' for i in range(5)]]

y = df['views_after_30_days']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the model pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Step 9: Save model
joblib.dump(model, "predictive_view_model.pkl")

# Step 10: Predict on new input
new_data = pd.DataFrame([{
    'current_views': 250000,
    'current_comments': 4000,
    'current_likes': 15000,
    'views_1': 310000,
    'views_2': 295000,
    'views_3': 280000,
    'views_4': 270000,
    'views_5': 260000,
    'likes_1': 18000,
    'likes_2': 17500,
    'likes_3': 16000,
    'likes_4': 15000,
    'likes_5': 14000,
    'comments_1': 3600,
    'comments_2': 3450,
    'comments_3': 3300,
    'comments_4': 3200,
    'comments_5': 3100,
    'days_since_upload_1': 5,
    'days_since_upload_2': 6,
    'days_since_upload_3': 7,
    'days_since_upload_4': 5,
    'days_since_upload_5': 4
}])

# Load and predict
loaded_model = joblib.load("predictive_view_model.pkl")
predicted_views = loaded_model.predict(new_data)
print(f"Predicted views after 30 days: {predicted_views[0]:,.0f}")
