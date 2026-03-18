import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------------------
# Preprocessing function
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ------------------------------
# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")  # Update path
    df['overview'] = df['overview'].fillna('')
    df['clean_text'] = df['overview'].apply(preprocess_text)
    return df

df = load_data()

# ------------------------------
# TF-IDF and cosine similarity
@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_similarity(df)

# Mapping from title to index
title_to_index = pd.Series(df.index, index=df['title']).to_dict()

# Recommendation function
def recommend(item_name, top_n=5):
    if item_name not in title_to_index:
        return ["Movie not found."]
    idx = title_to_index[item_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_idx = [i[0] for i in sim_scores[1:top_n+1]]
    return df['title'].iloc[top_idx].tolist()

# ------------------------------
# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Select a movie to get top similar recommendations:")

# Dropdown menu
selected_movie = st.selectbox("Choose a movie", df['title'].values)

# Button to generate recommendations
if st.button("Recommend"):
    recommendations = recommend(selected_movie, top_n=5)
    st.subheader("Top Recommendations:")
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")