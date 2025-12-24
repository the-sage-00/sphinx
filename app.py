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

# Original title removed for custom UI


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

import random
import time

# ------------------------------------------------------
# CUSTOM CSS & STYLING (THE "CRAZY" PART)
# ------------------------------------------------------
st.markdown("""
    <style>
    /* Import funky font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Raleway:wght@300&display=swap');

    /* Global darker theme override just in case */
    .stApp {
        background-color: #0e1117;
        background-image: linear-gradient(315deg, #0e1117 0%, #1a1c29 74%);
    }

    /* NEON Title Styles */
    .title-text {
        font-family: 'Orbitron', sans-serif;
        color: #fff;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 0 0 10px #ff00de, 0 0 20px #ff00de, 0 0 40px #ff00de, 0 0 80px #ff00de;
        animation: glow 1.5s ease-in-out infinite alternate;
        font-size: 50px;
        margin-bottom: 20px;
    }
    
    @keyframes glow {
        from {
            text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #e60073, 0 0 40px #e60073, 0 0 50px #e60073, 0 0 60px #e60073, 0 0 70px #e60073;
        }
        to {
            text-shadow: 0 0 20px #fff, 0 0 30px #ff4da6, 0 0 40px #ff4da6, 0 0 50px #ff4da6, 0 0 60px #ff4da6, 0 0 70px #ff4da6, 0 0 80px #ff4da6;
        }
    }

    /* Custom Selection Box */
    .stSelectbox label {
        color: #00e5ff !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.2rem;
    }

    /* Cards for Movies */
    .movie-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
        text-align: center;
        transition: transform 0.3s;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .movie-card:hover {
        transform: scale(1.05);
        border-color: #00e5ff;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.3);
    }
    .movie-title {
        color: #fff;
        font-family: 'Raleway', sans-serif;
        font-weight: bold;
        margin-top: 10px;
        font-size: 1rem;
    }
    
    /* Button Styling */
    .stButton>button {
        color: white;
        background: linear-gradient(45deg, #ff00de, #00e5ff);
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        background: linear-gradient(45deg, #00e5ff, #ff00de);
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------
# HEADER & INTRODUCTION
# ------------------------------------------------------
st.markdown('<div class="title-text">🍿 CINE-GENIUS 🍿</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #aaa; margin-bottom: 30px; font-family: Raleway;'>
    <i>"Because scrolling through Netflix takes 2 hours and you still watch The Office."</i>
    </div>
    """, unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # ------------------------------------------------------
    # UI CONTROLS
    # ------------------------------------------------------
    movie_list = sorted(movies['title'].values)

    selected_movie = st.selectbox(
        "🎬 Pick a movie you actually liked:",
        movie_list
    )

    st.write("") # Spacer

    num_recs = st.slider(
        "🌶️ How many spicy recommendations?",
        3, 10, 5
    )

    st.write("") # Spacer
    st.write("") # Spacer

    if st.button("🔮 SUMMON RECOMMENDATIONS 🔮"):
        
        # FUNNY LOADING MESSAGES
        loading_texts = [
            "Consulting the Ancient Scripts of Hollywood...",
            "Bribing the Rotten Tomatoes critics...",
            "Asking Leonardo DiCaprio for advice...",
            "Calculating the perfect level of drama...",
            "Loading subtitles for languages you don't speak...",
            "Stealing data from the matrix...",
            "Finding movies better than your ex's taste..."
        ]
        
        progress_text = st.empty()
        for _ in range(3):
            progress_text.text(f"⏳ {random.choice(loading_texts)}")
            time.sleep(0.4)
        progress_text.empty()

        # FETCH RECOMMENDATIONS
        names, posters = recommend(selected_movie, num_recs)

        if names:
            st.balloons()
            st.markdown(f"<h3 style='text-align: center; color: #00e5ff; font-family: Orbitron;'>✨ HERE IS YOUR NEXT OBSESSION ✨</h3>", unsafe_allow_html=True)
            st.write("")
            
            # Display rows of 3 to avoid cramping
            cols = st.columns(len(names))
            
            # Simple list usage since st.columns returns a list
            for i in range(len(names)):
                with cols[i]:
                    img_html = ""
                    if posters[i]:
                        img_html = f'<img src="{posters[i]}" style="width:100%; border-radius:10px;">'
                    else:
                        img_html = '<div style="height:300px; background:#333; color:#fff; display:flex; align-items:center; justify-content:center;">No Poster</div>'
                    
                    st.markdown(f"""
                    <div class="movie-card">
                        {img_html}
                        <div class="movie-title">{names[i]}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
    Made with 💖, ☕, and a lot of 🎬 by your favorite developer.<br>
    <i>Do not blame us if you binge-watch until 3 AM.</i>
    </div>
    """, unsafe_allow_html=True
)
