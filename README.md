# YouTube Trend Analyzer
An AI-powered web application that helps content creators and casual viewers understand what’s trending on YouTube and predict how videos will perform — all in one place.

It combines machine learning predictions,real-time YouTube Data API analytics, and natural language summarization to save hours of manual research.

# Features
📈 30-Day View Prediction – Predict future views of any YouTube video using a trained ML regression model.
📰 AI Summaries – Instantly summarize trending videos so you can grasp the content without watching it.
🔍 Category & Filter Search – Browse trending videos by category, with an option to exclude Shorts.
⚡ Real-time Analytics – Fetch live statistics (views, likes, comments, etc.) directly from YouTube.
🎯 Content Creators's Help – Understand why videos trend and plan content strategically.

# Tech Stack
Frontend: HTML5, CSS3, JavaScript
Backend: Flask (Python)
Machine Learning: Scikit-learn, Joblib, Pandas, NumPy
NLP: Hugging Face Transformers, NLTK
API: YouTube Data API v3 (Google API Python Client)
Models:
sshleifer/distilbart-cnn-12-6 (fine-tuned for summarization)
Gradient Boosting Regressor (view prediction)

# Usage
To Analyze a Video:
Paste the YouTube video URL → Fetch stats → Get AI summary + 30-day view prediction.

To Browse Trending Content:
Select category → View summaries + performance estimates for top trending videos.


# 📊 Model Performance
Summarization: Fine-tuned DistilBART model evaluated using ROUGE scores.
Prediction Model: Gradient Boosting Regressor with R² = 0.7984 (explains ~80% of variation).
