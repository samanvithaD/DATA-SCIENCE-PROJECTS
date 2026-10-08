import os
import streamlit as st

st.set_page_config(page_title="Samanvitha | Data Science Projects", page_icon="📊", layout="wide")

# ---------- find the page files, whatever they are numbered / capitalised ----------
def find_page(stem):
    for folder in ("pages", "Pages"):
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(stem.lower() + ".py"):
                    return f"{folder}/{f}"
    return None

# ---------- project content (edit text here) ----------
PROJECTS = [
    dict(stem="Customer_Attrition", color="#3F51B5", title="Customer Attrition",
         text="Find out which telecom customers are likely to leave, and score any customer live.",
         tags=["Logistic Regression", "Random Forest", "XGBoost"]),
    dict(stem="Retail_Forecasting", color="#0F7C80", title="Retail Sales Forecasting",
         text="Forecast daily demand with prediction intervals, then work out how much safety stock to hold.",
         tags=["Baselines", "ARIMA", "Prediction intervals", "Safety stock"]),
    dict(stem="Movie_recommendation", color="#C98A1B", title="Movie Recommender",
         text="Pick a movie you like and get similar ones, matched on cast, director, genres and keywords.",
         tags=["TF-IDF", "Cosine similarity", "9,141 movies"]),
]

# ---------- styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"], .stMarkdown, p { font-family: 'Source Sans 3', sans-serif; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 1rem !important; padding-bottom: 0.5rem !important; max-width: 1150px; }
header[data-testid="stHeader"] { background: transparent; height: 2.5rem; }

.hero { padding: 1.4rem 2.4rem; border-radius: 22px; margin-bottom: 1rem;
        background: #14283A; color: #F4F7F9; position: relative; overflow: hidden; }
.hero:before { content:""; position:absolute; right:-70px; top:-70px; width:260px; height:260px;
        border-radius:50%; background:#0F7C80; opacity:.55; }
.hero:after  { content:""; position:absolute; right:90px; bottom:-90px; width:180px; height:180px;
        border-radius:50%; background:#C98A1B; opacity:.45; }
.hero h1 { font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:2.6rem;
        line-height:1.08; margin:0 0 .6rem 0; color:#F4F7F9; position:relative; z-index:1; }
.hero p  { font-size:1.2rem; max-width:34rem; margin:0; color:#C9D6E0; position:relative; z-index:1; }

.card { background:#fff; border-radius:16px; padding:1.4rem 1.4rem 1.2rem 1.4rem;
        border:1px solid #DCE5EC; border-top-width:6px; min-height:395px; margin-bottom:0; box-sizing:border-box; }
.card h3 { font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.45rem;
        margin:0 0 .6rem 0; color:#14283A; min-height:3.6rem; }
.card p  { font-size:1.05rem; line-height:1.5; color:#3B4F60; margin:0 0 1rem 0; }
.chip { display:inline-block; font-size:.85rem; font-weight:600; padding:.18rem .65rem;
        border-radius:999px; margin:0 .35rem .4rem 0; }

a[data-testid="stPageLink-NavLink"] { background:#14283A; border-radius:10px; padding:.55rem 1rem;
        margin-top:.7rem; justify-content:center; transition: background .15s; }
a[data-testid="stPageLink-NavLink"] p { color:#F4F7F9 !important; font-weight:600; }
a[data-testid="stPageLink-NavLink"]:hover { background:#0F7C80; }

@media (max-width: 700px) { .hero h1 { font-size:2.1rem; } .card { min-height:0; } .card h3 { min-height:0; } }
</style>
""", unsafe_allow_html=True)

# ---------- hero ----------
st.markdown("""
<div class="hero">
  <h1>Dharmavarapu Sita Samanvitha</h1>
  <p>Data science projects you can try yourself: predict who will leave, forecast what will sell, and find what to watch next.</p>
</div>
""", unsafe_allow_html=True)

# ---------- project cards ----------
cols = st.columns(len(PROJECTS), gap="large")
for col, p in zip(cols, PROJECTS):
    chips = "".join(
        f'<span class="chip" style="background:{p["color"]}1A;color:{p["color"]}">{t}</span>' for t in p["tags"])
    with col:
        st.markdown(
            f'<div class="card" style="border-top-color:{p["color"]}">'
            f'<h3>{p["title"]}</h3><p>{p["text"]}</p>{chips}</div>',
            unsafe_allow_html=True)
        page = find_page(p["stem"])
        if page:
            st.page_link(page, label=f'Open {p["title"]}', use_container_width=True)
        else:
            st.caption(f'Page file for "{p["title"]}" not found in the pages folder.')