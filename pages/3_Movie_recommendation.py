import os
from difflib import get_close_matches
import pandas as pd, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "data", "movies_slim.csv"),                    # pages/data/
    os.path.join(os.path.dirname(HERE), "data", "movies_slim.csv"),   # project_root/data/
    os.path.join(os.getcwd(), "data", "movies_slim.csv"),             # folder streamlit was run from
]
DATA = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[0])

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")
st.write("Content-based: top 3 actors, director, genres and keywords, ranked by TF-IDF cosine similarity.")

if not os.path.exists(DATA):
    st.error("DATA/movies_slim.csv not found. Run: python tools/make_slim_data.py <path to Data folder>")
    st.stop()


@st.cache_resource(show_spinner="Building the recommender...")
def build():
    df = pd.read_csv(DATA).dropna(subset=["title", "tags"]).reset_index(drop=True)
    for c in ("genre_names", "director_name", "actor_names"):   # older slim files may not have these
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("")
    X = TfidfVectorizer(min_df=2, stop_words="english").fit_transform(df["tags"])
    # if two movies share a title, the one with more votes wins (it is written last)
    idx = {df.title[i].lower(): i for i in df.vote_count.sort_values().index}
    return df, X, idx


df, X, idx = build()
st.caption(f"{len(df):,} movies indexed (movies with at least 50 votes).")
title = st.selectbox("Pick a movie", sorted(df.title.unique()), index=None, placeholder="Start typing a title")
typed = st.text_input("Or type a title (typos are fine)")
n = st.slider("Number of recommendations", 5, 20, 10)

query = typed.strip() or title      # typed text takes priority over the dropdown
if query:
    key = query.lower()
    if key not in idx:
        close = get_close_matches(key, list(idx), n=5, cutoff=0.6)
        st.warning(f"'{query}' not found. Did you mean: {', '.join(df.title[idx[c]] for c in close)}" if close
                   else f"'{query}' not found.")
    else:
        i = idx[key]
        sims = cosine_similarity(X[i], X).ravel()
        sims[i] = -1
        top = sims.argsort()[::-1][:n]
        out = pd.DataFrame({
            "Title": df.title[top].values,
            "Year": df.release_date[top].astype(str).str[:4].values,
            "Genres": df.genre_names[top].values,
            "Director": df.director_name[top].values,
            "Similarity": sims[top].round(3),
        })
        st.subheader(f"Because you watched {df.title[i]}")
        st.dataframe(out, use_container_width=True, hide_index=True)
