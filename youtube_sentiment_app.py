import streamlit as st
from googleapiclient.discovery import build
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
import re

# Replace with your actual API Key
YOUTUBE_API_KEY = "AIzaSyDKxJz7lkOuESqq04RH9HxHw4SBHMPtKs4"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Streamlit setup
st.set_page_config(page_title="YouTube Comment Sentiment", layout="centered")
st.title("📺 YouTube Comment Sentiment Analyzer")

# Helper to extract video ID
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

# Fetch comments
def get_comments(video_id, max_comments=50):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(100, max_comments),
        textFormat="plainText"
    )
    response = request.execute()

    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

# Sentiment analyzer
def analyze_comments(comments):
    data = []
    pos, neg, neu = 0, 0, 0

    for comment in comments:
        blob = TextBlob(comment)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            sentiment = "Positive"
            pos += 1
        elif polarity < 0:
            sentiment = "Negative"
            neg += 1
        else:
            sentiment = "Neutral"
            neu += 1

        data.append({
            "Comment": comment,
            "Polarity": polarity,
            "Sentiment": sentiment
        })

    return pd.DataFrame(data), [pos, neg, neu]

# UI
video_url = st.text_input("Enter YouTube Video URL:")
max_count = st.slider("Number of comments", 10, 100, 50)

if st.button("Analyze Comments"):
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL")
    else:
        st.info("Fetching and analyzing comments...")
        comments = get_comments(video_id, max_comments=max_count)

        if not comments:
            st.warning("No comments found.")
        else:
            df, sentiment_counts = analyze_comments(comments)

            # Pie Chart
            st.subheader("📊 Sentiment Distribution")
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=["Positive", "Negative", "Neutral"],
                   autopct="%1.1f%%", colors=["green", "red", "gray"])
            ax.axis("equal")
            st.pyplot(fig)

            # Table
            st.subheader("💬 Comments & Sentiment")
            st.dataframe(df)

            # Download button
            st.download_button("Download CSV", df.to_csv(index=False),
                               file_name="youtube_comments_sentiment.csv",
                               mime="text/csv")
