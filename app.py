from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from googleapiclient.discovery import build
import joblib
import os
import re
from datetime import datetime, timezone
from dateutil import parser

# Import the summarization functions
try:
    from summarize_text import summarize as custom_summarize, process_and_summarize_list
except ImportError:
    print("CRITICAL WARNING: 'summarize_text.py' not found. Summarization will not work.")
    def custom_summarize(text): return "(Error: The summarization script is missing.)"
    def process_and_summarize_list(items): return "(Error: The summarization script is missing.)"

# --- App Setup ---
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'a-default-secret-key-that-is-long-and-random')
app = Flask(__name__)
app.secret_key = SECRET_KEY

# --- API & Model Setup ---
# It's recommended to use environment variables for API keys in production
API_KEY = os.environ.get('YOUTUBE_API_KEY', 'AIzaSyC4ZwDopoXxUEMWjUCjrnKO7yCQ4ZcB5Gw') # Replace with your key if not set as env var
youtube = build("youtube", "v3", developerKey=API_KEY)
model = joblib.load("predictive_view_model.pkl")

CATEGORY_MAP = {
    "All": None, "Film & Animation": "1", "Autos & Vehicles": "2", "Music": "10",
    "Pets & Animals": "15", "Sports": "17", "Short Movies": "18", "Travel & Events": "19",
    "Gaming": "20", "People & Blogs": "22", "Comedy": "23", "Entertainment": "24",
    "News & Politics": "25", "Howto & Style": "26", "Education": "27", "Science & Technology": "28"
}

# ============ HELPER FUNCTIONS ============

def get_video_id_from_url(url):
    """Extracts the YouTube video ID from various URL formats."""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_details(video_ids):
    """Fetches details for a list of video IDs."""
    if not video_ids:
        return []
    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(video_ids)
        )
        response = request.execute()
        return response.get('items', [])
    except Exception as e:
        print(f"Error fetching video details: {e}")
        return []

def get_channel_uploads(channel_id, current_video_id, max_results=6):
    """Fetches the latest videos from a channel's upload playlist."""
    try:
        channel_request = youtube.channels().list(part="contentDetails", id=channel_id)
        channel_response = channel_request.execute()
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        playlist_request = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=max_results
        )
        playlist_response = playlist_request.execute()
        
        video_ids = [
            item['snippet']['resourceId']['videoId']
            for item in playlist_response.get('items', [])
            if item['snippet']['resourceId']['videoId'] != current_video_id
        ]
        return video_ids[:max_results-1]
    except Exception as e:
        print(f"Error fetching channel uploads: {e}")
        return []

def calculate_days_since(published_at_string):
    """Calculates the number of days since a video was published."""
    if not published_at_string:
        return 0
    published_date = parser.isoparse(published_at_string)
    now = datetime.now(timezone.utc)
    delta = now - published_date
    return delta.days

def safe_summarize(text):
    try:
        return custom_summarize(text)
    except Exception as e:
        print(f"Error in safe_summarize wrapper: {e}")
        return "(An error occurred during summarization)"

def get_trending_videos(region_code='IN', max_results=5, category_id=None):
    try:
        request = youtube.videos().list(
            part="snippet,statistics", chart="mostPopular",
            regionCode=region_code, maxResults=max_results,
            videoCategoryId=category_id if category_id else None
        )
        response = request.execute()
        videos = []
        for item in response.get('items', []):
            snippet = item['snippet']
            stats = item['statistics']
            videos.append({
                'title': snippet.get('title', 'N/A'),
                'description': snippet.get('description', ''),
                'channelTitle': snippet.get('channelTitle', 'N/A'),
                'viewCount': int(stats.get('viewCount', 0)),
                'likeCount': int(stats.get('likeCount', 0)),
                'commentCount': int(stats.get('commentCount', 0))
            })
        return pd.DataFrame(videos)
    except Exception as e:
        print(f"An error occurred while fetching YouTube videos: {e}")
        return pd.DataFrame()

# ============ ROUTES ============ #

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('analyzer'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if 'user' in session:
        return redirect(url_for('analyzer'))
    if request.method == 'POST':
        # Using simple user/pass for demo. Use a proper auth system for production.
        if request.form['username'] == 'admin' and request.form['password'] == 'password':
            session['user'] = request.form['username']
            return redirect(url_for('analyzer'))
        else:
            error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/analyzer', methods=['GET', 'POST'])
def analyzer():
    if 'user' not in session:
        return redirect(url_for('login'))

    result = []
    topic_summary = ''
    # Set default form data for GET request
    form_data = {
        'region': 'IN',
        'count': 5,
        'category': 'All'
    }

    if request.method == 'POST':
        form_data['region'] = request.form.get('region', 'IN')
        form_data['count'] = request.form.get('count', 5)
        form_data['category'] = request.form.get('category', 'All')
        
        category_id = CATEGORY_MAP.get(form_data['category'])
        
        try:
            count = int(form_data['count'])
        except (ValueError, TypeError):
            count = 5

        df = get_trending_videos(form_data['region'], count, category_id)
        if not df.empty:
            df['summary'] = df['description'].apply(safe_summarize)
            titles = df['title'].dropna().tolist()
            topic_summary = process_and_summarize_list(titles)
            result = df.to_dict(orient='records')
        else:
            topic_summary = "Could not fetch trending videos. Please check the API key and network connection."

    return render_template("index.html", result=result, topic_summary=topic_summary,
                           categories=list(CATEGORY_MAP.keys()), form_data=form_data)


@app.route('/manual', methods=['GET', 'POST'])
def manual():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    error = None
    video_data = {
        'current_views': '', 'current_likes': '', 'current_comments': '',
        'prev_views': [''] * 5, 'prev_likes': [''] * 5,
        'prev_comments': [''] * 5, 'prev_days': [''] * 5,
    }

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'fetch_data':
            video_link = request.form.get('video_link')
            video_id = get_video_id_from_url(video_link)

            if not video_id:
                error = "Invalid YouTube link. Please enter a valid video URL."
            else:
                current_video_details = get_video_details([video_id])
                if not current_video_details:
                    error = "Could not fetch details for the provided video link."
                else:
                    current_video = current_video_details[0]
                    stats = current_video.get('statistics', {})
                    video_data['current_views'] = int(stats.get('viewCount', 0))
                    video_data['current_likes'] = int(stats.get('likeCount', 0))
                    video_data['current_comments'] = int(stats.get('commentCount', 0))
                    
                    channel_id = current_video['snippet']['channelId']
                    prev_video_ids = get_channel_uploads(channel_id, video_id)
                    prev_video_details = get_video_details(prev_video_ids)

                    # Reset lists before populating
                    video_data.update({
                        'prev_views': [], 'prev_likes': [], 'prev_comments': [], 'prev_days': []
                    })
                    for video in prev_video_details:
                        prev_stats = video.get('statistics', {})
                        video_data['prev_views'].append(int(prev_stats.get('viewCount', 0)))
                        video_data['prev_likes'].append(int(prev_stats.get('likeCount', 0)))
                        video_data['prev_comments'].append(int(prev_stats.get('commentCount', 0)))
                        video_data['prev_days'].append(calculate_days_since(video['snippet'].get('publishedAt')))
                    
                    # Pad with 0 if fewer than 5 videos were found
                    while len(video_data['prev_views']) < 5:
                        video_data['prev_views'].append(0)
                        video_data['prev_likes'].append(0)
                        video_data['prev_comments'].append(0)
                        video_data['prev_days'].append(0)

        elif action == 'predict':
            try:
                input_data = {
                    'current_views': float(request.form['current_views']),
                    'current_comments': float(request.form['current_comments']),
                    'current_likes': float(request.form['current_likes']),
                }
                for i in range(1, 6):
                    input_data[f'views_{i}'] = float(request.form[f'view{i}'])
                    input_data[f'likes_{i}'] = float(request.form[f'like{i}'])
                    input_data[f'comments_{i}'] = float(request.form[f'comment{i}'])
                    input_data[f'days_since_upload_{i}'] = float(request.form[f'days{i}'])

                expected_feature_order = [
                    'current_views', 'current_comments', 'current_likes',
                    'views_1', 'views_2', 'views_3', 'views_4', 'views_5',
                    'likes_1', 'likes_2', 'likes_3', 'likes_4', 'likes_5',
                    'comments_1', 'comments_2', 'comments_3', 'comments_4', 'comments_5',
                    'days_since_upload_1', 'days_since_upload_2', 'days_since_upload_3',
                    'days_since_upload_4', 'days_since_upload_5'
                ]
                
                input_df = pd.DataFrame([input_data])
                input_df_ordered = input_df[expected_feature_order]

                prediction_value = model.predict(input_df_ordered)[0]
                prediction = f"{prediction_value:,.2f}"
                
                # **FIX:** Keep the form data populated after prediction
                video_data = {
                    'current_views': request.form['current_views'],
                    'current_likes': request.form['current_likes'],
                    'current_comments': request.form['current_comments'],
                    'prev_views': [request.form.get(f'view{i}') for i in range(1,6)],
                    'prev_likes': [request.form.get(f'like{i}') for i in range(1,6)],
                    'prev_comments': [request.form.get(f'comment{i}') for i in range(1,6)],
                    'prev_days': [request.form.get(f'days{i}') for i in range(1,6)],
                }

            except Exception as e:
                error = f"Error during prediction: {str(e)}"
                # Keep user-entered data on error
                video_data = {
                    'current_views': request.form.get('current_views', ''),
                    'current_likes': request.form.get('current_likes', ''),
                    'current_comments': request.form.get('current_comments', ''),
                    'prev_views': [request.form.get(f'view{i}', '') for i in range(1,6)],
                    'prev_likes': [request.form.get(f'like{i}', '') for i in range(1,6)],
                    'prev_comments': [request.form.get(f'comment{i}', '') for i in range(1,6)],
                    'prev_days': [request.form.get(f'days{i}', '') for i in range(1,6)],
                }


    return render_template("manual.html", prediction=prediction, video_data=video_data, error=error)

# ============ MAIN ============ #
if __name__ == '__main__':
    # Use debug=False in a production environment
    app.run(host='0.0.0.0', port=5000, debug=True)