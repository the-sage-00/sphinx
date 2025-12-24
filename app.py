# ======================================================
# 🎬 MOVIE RECOMMENDATION SYSTEM (WITH POSTERS)
# ======================================================
# Uses:
# - Sentence-BERT embeddings
# - FAISS for fast similarity search
# - TMDB API for movie posters
# ======================================================

import streamlit as st
import pickle
import faiss
import numpy as np
import requests
import os

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write(
    "Semantic movie recommendations using **Sentence-BERT + FAISS**, with posters from TMDB."
)

# ------------------------------------------------------
# TMDB CONFIG (USE STREAMLIT SECRETS)
# ------------------------------------------------------
# ------------------------------------------------------
# OMDB CONFIG (USE STREAMLIT SECRETS)
# ------------------------------------------------------
try:
    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
except:
    OMDB_API_KEY = "trilogy" # Fallback/Demo key often used in tutorials

def fetch_poster(title):
    """
    Fetch movie poster from OMDB API using title
    """
    url = "http://www.omdbapi.com/"
    params = {
        "apikey": OMDB_API_KEY,
        "t": title
    }
    try:
        response = requests.get(url, params=params, timeout=2)
        response.raise_for_status()
        data = response.json()
        
        if data.get("Response") == "True":
            poster_path = data.get("Poster")
            if poster_path and poster_path != "N/A":
                return poster_path
    
        print(f"Warning: No poster found for '{title}'")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch poster for '{title}'. Error: {e}")
        return "https://via.placeholder.com/500x750?text=Error"
        
    return "https://via.placeholder.com/500x750?text=No+Poster"

# ------------------------------------------------------
# LOAD ARTIFACTS (CACHED)
# ------------------------------------------------------
@st.cache_resource
def load_artifacts():
    movies = pickle.load(open("artifacts/movies.pkl", "rb"))
    embeddings = pickle.load(open("artifacts/embeddings.pkl", "rb"))
    index = faiss.read_index("artifacts/faiss_index.index")
    return movies, embeddings, index

movies, embeddings, index = load_artifacts()

# ------------------------------------------------------
# RECOMMENDATION FUNCTION (FAISS)
# ------------------------------------------------------
def recommend(movie_name, top_n=5):
    if movie_name not in movies['title'].values:
        return [], []

    # Find the positional index of the movie (row number)
    try:
        movie_idx = movies[movies['title'] == movie_name].index[0]
        # Use simple list index to guarantee positional index if dataframe index is not RangeIndex
        # (This assumes movies and embeddings are aligned by row position)
        movie_idx = list(movies['title']).index(movie_name)
    except IndexError:
        return [], []

    query_vector = embeddings[movie_idx].reshape(1, -1)

    _, indices = index.search(query_vector, top_n + 1)

    recommended_names = []
    recommended_posters = []

    print(f"DEBUG: Found similar movies indices: {indices}")
    for i in indices[0][1:]:
        try:
            # For OMDB we use title, not movie_id
            movie_title = movies.iloc[i].title
            
            print(f"DEBUG: Fetching poster for: {movie_title}")
            poster_url = fetch_poster(movie_title)
            print(f"DEBUG: Poster URL: {poster_url}")
            
            recommended_names.append(movie_title)
            recommended_posters.append(poster_url)
        except Exception as e:
            print(f"ERROR in loop: {e}")

    return recommended_names, recommended_posters

# ------------------------------------------------------
# UI CONTROLS
# ------------------------------------------------------
movie_list = sorted(movies['title'].values)

selected_movie = st.selectbox(
    "🎥 Select a movie",
    movie_list
)

num_recs = st.slider(
    "🔢 Number of recommendations",
    3, 10, 5
)

if st.button("🚀 Show Recommendations"):
    with st.spinner("Finding similar movies..."):
        names, posters = recommend(selected_movie, num_recs)

    if names:
        cols = st.columns(len(names))
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.text(name)
                if poster:
                    st.image(poster)
                else:
                    st.write("No poster available")

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using ML, NLP, FAISS & TMDB API")
