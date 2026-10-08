from pathlib import Path
from difflib import get_close_matches

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

# Streamlit Cloud runs on Linux, so folder names are case-sensitive.
# Look for DATA/ and data/ in the project root, next to this page, and in the working folder.
ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [
    ROOT / "DATA" / "movies_slim.csv",
    ROOT / "data" / "movies_slim.csv",
    Path(__file__).resolve().parent / "DATA" / "movies_slim.csv",
    Path(__file__).resolve().parent / "data" / "movies_slim.csv",
    Path.cwd() / "DATA" / "movies_slim.csv",
    Path.cwd() / "data" / "movies_slim.csv",
]
DATA = next((p for p in CANDIDATES if p.exists()), None)

st.title("Movie Recommendation System")
st.write("Content-based: top 3 actors, director, genres and keywords, ranked by TF-IDF cosine similarity.")

if DATA is None:
    st.error("movies_slim.csv not found. Put it in the DATA folder of your GitHub repo (DATA/movies_slim.csv).")
    st.stop()


@st.cache_resource(show_spinner="Building the recommender...")
def build(path):
    df = pd.read_csv(path).dropna(subset=["title", "tags"]).reset_index(drop=True)
    for c in ("genre_names", "director_name", "actor_names", "release_date"):
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("")
    if "vote_count" not in df:
        df["vote_count"] = 0
    X = TfidfVectorizer(min_df=2, stop_words="english").fit_transform(df["tags"])
    # if two movies share a title, the one with more votes wins (it is written last)
    idx = {df.title[i].lower(): i for i in df.vote_count.sort_values().index}
    return df, X, idx


df, X, idx = build(str(DATA))
st.caption(f"{len(df):,} movies indexed (movies with at least 50 votes).")

title = st.selectbox("Pick a movie", sorted(df.title.unique()), index=None, placeholder="Start typing a title")
typed = st.text_input("Or type a title (typos are fine)")
n = st.slider("Number of recommendations", 5, 20, 10)

query = typed.strip() or title  # typed text takes priority over the dropdown
if query:
    key = query.lower()
    if key not in idx:
        close = get_close_matches(key, list(idx), n=5, cutoff=0.6)
        if close:
            st.warning(f"'{query}' not found. Did you mean: {', '.join(df.title[idx[c]] for c in close)}")
        else:
            st.warning(f"'{query}' not found.")
    else:
        i = idx[key]
        sims = cosine_similarity(X[i], X).ravel()
        sims[i] = -1  # a movie must not recommend itself
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
